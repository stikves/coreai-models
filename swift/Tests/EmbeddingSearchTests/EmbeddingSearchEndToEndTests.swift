// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIEmbeddings
import CoreAIShared
import CoreAIVectorIndex
import Foundation
import TestUtilities
import Testing

@testable import embedding_search

@Suite("embedding-search end to end (fake model)")
struct EmbeddingSearchEndToEndTests {
    let silent = ProgressReporter(write: { _ in })

    /// red/green/blue images (nested, png + jpeg), one undecodable file, and a jsonl corpus.
    func makeCorpus() throws -> (dir: URL, images: [URL], texts: [TextDocument]) {
        let dir = try TestFiles.temporaryDirectory()
        let imageDir = dir.appending(path: "images")
        try TestFiles.writeImage(imageDir.appending(path: "red.png"), red: 1, green: 0, blue: 0)
        try TestFiles.writeImage(imageDir.appending(path: "sub/green.jpg"), red: 0, green: 1, blue: 0, type: .jpeg)
        try TestFiles.writeImage(imageDir.appending(path: "sub/deeper/blue.png"), red: 0, green: 0, blue: 1)
        try TestFiles.write("not an image", to: imageDir.appending(path: "broken.png"))
        let jsonl = dir.appending(path: "docs.jsonl")
        try TestFiles.write(
            """
            {"id": "apple", "text": "a red apple on the table"}
            {"id": "grass", "text": "fresh green grass"}
            {"id": "sea", "text": "the deep blue sea", "title": "Ocean"}
            """, to: jsonl)
        return (dir, try ImageDiscovery.discover(at: imageDir), try TextCorpus.load(from: jsonl))
    }

    @Test("index a temp dir, save, reopen, query by text and image with kind filters")
    func indexAndQuery() async throws {
        let (dir, images, texts) = try makeCorpus()
        defer { try? FileManager.default.removeItem(at: dir) }
        #expect(images.count == 4)

        let fake = try SearchTestModel.make()
        let builder = IndexBuilder(
            model: fake.model, bundle: "bundle", dimensions: 8, batch: 2,
            documentPrompt: fake.model.resolvePrompt(requested: nil, defaultName: "document"), progress: silent)
        let (index, summary) = try await builder.build(images: images, texts: texts)
        #expect(summary.images == 3)
        #expect(summary.texts == 3)
        #expect(summary.skippedImages.count == 1)
        #expect(summary.skippedImages[0].hasSuffix("broken.png"))
        #expect(index.count == 6)

        // Prompts: untitled docs get the 'document' prompt; the titled one uses the template, no prompt.
        let textRows = await fake.text.calls.map(SearchTestModel.rows)
        #expect(textRows.contains(["document: a red apple on the table", "document: fresh green grass"]))
        #expect(textRows.contains(["title: Ocean | text: the deep blue sea"]))
        #expect(await fake.imageEncoder.calls.count == 3)  // one image per call, 3 decodable images

        let url = dir.appending(path: "idx.cvi")
        try index.save(to: url)

        let runner = QueryRunner(model: fake.model, bundle: "bundle")
        let reopened = try runner.openIndex(at: url)
        #expect(reopened.count == 6)
        #expect(reopened.descriptor == VectorIndexDescriptor(space: fake.model.space, dimension: 8))

        let all = try await runner.search(.text("red"), in: reopened, k: 10, kind: .all, promptName: "query")
        #expect(all.count == 6)
        #expect(all.map(\.rank) == Array(1...6))
        #expect(Set(all.prefix(2).map(\.id)) == [images.first { $0.lastPathComponent == "red.png" }!.path, "apple"])
        #expect(await fake.text.calls.last.map(SearchTestModel.rows) == ["query: red"])

        let onlyImages = try await runner.search(.text("green"), in: reopened, k: 10, kind: .image, promptName: nil)
        #expect(onlyImages.count == 3)
        #expect(onlyImages.allSatisfy { $0.kind == "image" && $0.path != nil && $0.text == nil })
        #expect(onlyImages[0].path?.hasSuffix("sub/green.jpg") == true)

        let onlyTexts = try await runner.search(.text("blue"), in: reopened, k: 2, kind: .text, promptName: "query")
        #expect(onlyTexts.count == 2)
        #expect(onlyTexts[0].id == "sea")
        #expect(onlyTexts[0].title == "Ocean")
        #expect(onlyTexts[0].text == "the deep blue sea")
        #expect(onlyTexts[0].score >= onlyTexts[1].score)

        // Image query: the prompt is never applied even if one is passed.
        let textCallsBefore = await fake.text.calls.count
        let blue = try CGImageUtils.load(from: images.first { $0.lastPathComponent == "blue.png" }!)
        let byImage = try await runner.search(.image(blue), in: reopened, k: 1, kind: .image, promptName: "query")
        #expect(byImage.first?.path?.hasSuffix("blue.png") == true)
        #expect(byImage.first.map { abs($0.score - 1) < 1e-4 } == true)
        #expect(await fake.text.calls.count == textCallsBefore)  // the image query ran no text call
    }

    @Test("Matryoshka index: query width follows the index; explicit mismatch fails closed")
    func matryoshkaWidth() async throws {
        let (dir, images, texts) = try makeCorpus()
        defer { try? FileManager.default.removeItem(at: dir) }
        let fake = try SearchTestModel.make()
        let builder = IndexBuilder(
            model: fake.model, bundle: "bundle", dimensions: 4, batch: 8, documentPrompt: "document",
            elementType: .float16, progress: silent)
        let (index, _) = try await builder.build(images: images, texts: texts)
        let url = dir.appending(path: "idx4.cvi")
        try index.save(to: url)

        let runner = QueryRunner(model: fake.model, bundle: "bundle")
        let reopened = try runner.openIndex(at: url)
        #expect(reopened.descriptor.dimension == 4)
        #expect(reopened.descriptor.elementType == .float16)
        let hits = try await runner.search(.text("red apple"), in: reopened, k: 3, kind: .text, promptName: "query")
        #expect(hits.first?.id == "apple")
        #expect(hits.count == 3)

        #expect(throws: VectorIndexError.self) { try runner.openIndex(at: url, dimensions: 8) }
        let wrongBuilder = IndexBuilder(
            model: fake.model, bundle: "bundle", dimensions: 5, batch: 8, documentPrompt: nil, progress: silent)
        await #expect(throws: EmbeddingError.unsupportedDimension(5, supported: [8, 4])) {
            _ = try await wrongBuilder.build(images: [], texts: texts)
        }
    }

    @Test("an index from another model's space is refused with a typed descriptor mismatch")
    func spaceMismatch() async throws {
        let (dir, _, texts) = try makeCorpus()
        defer { try? FileManager.default.removeItem(at: dir) }
        let builder = IndexBuilder(
            model: try SearchTestModel.make().model, bundle: "bundle", dimensions: 8, batch: 8, documentPrompt: nil,
            progress: silent)
        let (index, _) = try await builder.build(images: [], texts: texts)
        let url = dir.appending(path: "idx.cvi")
        try index.save(to: url)

        let other = try SearchTestModel.make(space: ("test/other", "2"))
        do {
            _ = try QueryRunner(model: other.model, bundle: "other").openIndex(at: url)
            Issue.record("expected a descriptor mismatch")
        } catch let error as VectorIndexError {
            guard case .descriptorMismatch(let expected, let actual) = error else {
                Issue.record("unexpected error \(error)")
                return
            }
            #expect(expected.space == VectorSpace(id: "test/other", revision: "2"))
            #expect(actual.space == VectorSpace(id: "test/fake", revision: "1"))
            #expect(error.localizedDescription.contains("test/fake@1"))
        }
    }

    @Test("bundles without an image tower fail clearly for image indexing and image queries")
    func missingImageTower() async throws {
        let (dir, images, texts) = try makeCorpus()
        defer { try? FileManager.default.removeItem(at: dir) }
        let textOnly = try SearchTestModel.make(image: false)
        let builder = IndexBuilder(
            model: textOnly.model, bundle: "text-only", dimensions: 8, batch: 8, documentPrompt: nil, progress: silent)
        await #expect(throws: EmbeddingSearchError.missingTower(.image, bundle: "text-only", available: ["text"])) {
            _ = try await builder.build(images: images, texts: texts)
        }
        #expect(await textOnly.text.calls.isEmpty)  // fails before any embedding work

        let (index, _) = try await builder.build(images: [], texts: texts)
        let runner = QueryRunner(model: textOnly.model, bundle: "text-only")
        let image = try CGImageUtils.load(
            from: try #require(images.first { $0.pathExtension == "png" && $0.lastPathComponent != "broken.png" }))
        await #expect(throws: EmbeddingSearchError.self) {
            _ = try await runner.search(.image(image), in: index, k: 1, kind: .all, promptName: nil)
        }
        #expect(
            EmbeddingSearchError.missingTower(.image, bundle: "b", available: ["text"]).localizedDescription
                == "the bundle at b has no image tower (towers: text)")
    }

    @Test("empty corpus and all-broken images are errors")
    func emptyInputs() async throws {
        let dir = try TestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        let broken = dir.appending(path: "x.png")
        try TestFiles.write("nope", to: broken)
        let builder = IndexBuilder(
            model: try SearchTestModel.make().model, bundle: "bundle", dimensions: 8, batch: 8, documentPrompt: nil,
            progress: silent)
        await #expect(throws: EmbeddingSearchError.emptyCorpus) { _ = try await builder.build(images: [], texts: []) }
        await #expect(throws: EmbeddingSearchError.allImagesFailed(1)) {
            _ = try await builder.build(images: [broken], texts: [])
        }
    }
}

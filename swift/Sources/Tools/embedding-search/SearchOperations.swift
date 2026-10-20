// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIEmbeddings
import CoreAIShared
import CoreAIVectorIndex
import Foundation

/// Errors raised by the CLI layer; library errors pass through.
enum EmbeddingSearchError: Error, Equatable, CustomStringConvertible, LocalizedError {
    case missingTower(EmbeddingModality, bundle: String, available: [String])
    case notFound(String)
    case notAnImage(String)
    case unreadableImage(String)
    case invalidTextRecord(file: String, line: Int, reason: String)
    case emptyCorpus
    case allImagesFailed(Int)
    case notWritable(String)
    case recordsFailed(Int)
    case invalidQueries(String)

    var description: String {
        switch self {
        case .missingTower(let modality, let bundle, let available):
            return
                "the bundle at \(bundle) has no \(modality.rawValue) tower (towers: \(available.joined(separator: ", ")))"
        case .notFound(let path): return "cannot read \(path)"
        case .notAnImage(let path):
            return "\(path) is not a supported image (\(ImageDiscovery.extensions.sorted().joined(separator: ", ")))"
        case .unreadableImage(let path): return "cannot decode image \(path)"
        case .invalidTextRecord(let file, let line, let reason): return "\(file):\(line): \(reason)"
        case .emptyCorpus: return "nothing to index: no images found and no text documents"
        case .allImagesFailed(let n): return "all \(n) images failed to decode"
        case .notWritable(let path): return "cannot write \(path)"
        case .recordsFailed(let n): return "\(n) record(s) failed (see their \"error\")"
        case .invalidQueries(let reason): return "--queries-json: \(reason)"
        }
    }

    var errorDescription: String? { description }
}

/// Progress + throughput lines on stderr.
struct ProgressReporter: Sendable {
    var write: @Sendable (String) -> Void = { message in
        FileHandle.standardError.write(Data((message + "\n").utf8))
    }

    func report(_ label: String, done: Int, total: Int, elapsed: TimeInterval) {
        let rate = elapsed > 0 ? Double(done) / elapsed : 0
        write(String(format: "%@ %d/%d  %.1f/s  (%.1fs)", label, done, total, rate, elapsed))
    }
}

private func seconds(since start: ContinuousClock.Instant) -> TimeInterval {
    (ContinuousClock.now - start).inSeconds
}

extension EmbeddingModel {
    /// Fails with a clear message when the bundle has no tower for `modality`.
    nonisolated func require(_ modality: EmbeddingModality, bundle: String) throws {
        guard supportedModalities.contains(modality) else {
            throw EmbeddingSearchError.missingTower(
                modality, bundle: bundle, available: supportedModalities.map(\.rawValue).sorted())
        }
    }

    /// An explicitly requested prompt name is kept (the library throws when unknown); the CLI's default
    /// name is dropped when the bundle does not define it.
    nonisolated func resolvePrompt(requested: String?, defaultName: String) -> String? {
        if let requested { return requested.isEmpty ? nil : requested }
        return promptNames.contains(defaultName) ? defaultName : nil
    }
}

/// Summary of an `index` run.
struct IndexSummary: Equatable, Sendable {
    var images = 0
    var texts = 0
    var skippedImages: [String] = []
    var imageSeconds: TimeInterval = 0
    var textSeconds: TimeInterval = 0
}

/// Embeds images and texts into a new ``VectorIndex``.
struct IndexBuilder {
    var model: EmbeddingModel
    var bundle: String
    var dimensions: Int
    var batch = 32
    var documentPrompt: String?
    var elementType: VectorElementType = .float32
    var progress = ProgressReporter()

    /// Checks that need no model load: non-empty corpus, towers present, supported width.
    func preflight(imageCount: Int, textCount: Int) throws -> VectorIndexDescriptor {
        guard imageCount > 0 || textCount > 0 else { throw EmbeddingSearchError.emptyCorpus }
        if imageCount > 0 { try model.require(.image, bundle: bundle) }
        if textCount > 0 { try model.require(.text, bundle: bundle) }
        return try VectorIndexDescriptor(model: model, dimensions: dimensions, elementType: elementType)
    }

    func build(images: [URL], texts: [TextDocument]) async throws -> (VectorIndex, IndexSummary) {
        var index = try VectorIndex(descriptor: preflight(imageCount: images.count, textCount: texts.count))
        let space = model.space
        var summary = IndexSummary()

        // Images
        var start = ContinuousClock.now
        var processed = 0
        for chunk in images.chunked(by: batch) {
            var urls: [URL] = []
            var inputs: [EmbeddingInput] = []
            for url in chunk {
                do {
                    inputs.append(.image(try CGImageUtils.load(from: url)))
                    urls.append(url)
                } catch {
                    summary.skippedImages.append(url.path)
                    progress.write("warning: skipping \(url.path): \(error.localizedDescription)")
                }
            }
            if !inputs.isEmpty {
                let vectors = try await model.embed(inputs, promptName: nil, dimensions: dimensions)
                let entries = try zip(urls, vectors).map {
                    try IndexEntryFactory.imageEntry(url: $0, vector: $1.vector)
                }
                try index.append(contentsOf: entries, space: space)
                summary.images += entries.count
            }
            processed += chunk.count
            progress.report("images", done: processed, total: images.count, elapsed: seconds(since: start))
        }
        summary.imageSeconds = seconds(since: start)
        if !images.isEmpty && summary.images == 0 { throw EmbeddingSearchError.allImagesFailed(images.count) }

        // Texts. Titled documents use the bundle's document template instead of the prompt prefix,
        // so they run in their own batches.
        start = ContinuousClock.now
        processed = 0
        for chunk in texts.chunked(by: batch) {
            for group in [chunk.filter { !usesTemplate($0) }, chunk.filter(usesTemplate)] where !group.isEmpty {
                let templated = usesTemplate(group[0])
                let inputs = group.map { doc in
                    EmbeddingInput.text(
                        templated ? model.formatDocument(title: doc.title, text: doc.text) : doc.text)
                }
                let vectors = try await model.embed(
                    inputs, promptName: templated ? nil : documentPrompt, dimensions: dimensions)
                let entries = try zip(group, vectors).map {
                    try IndexEntryFactory.textEntry(document: $0, vector: $1.vector)
                }
                try index.append(contentsOf: entries, space: space)
                summary.texts += entries.count
            }
            processed += chunk.count
            progress.report("texts", done: processed, total: texts.count, elapsed: seconds(since: start))
        }
        summary.textSeconds = seconds(since: start)
        return (index, summary)
    }

    private func usesTemplate(_ doc: TextDocument) -> Bool { doc.title != nil && model.documentTemplate != nil }
}

/// One query hit as printed by `query`.
struct QueryHit: Codable, Equatable, Sendable {
    var rank: Int
    var score: Float
    var id: String
    var kind: String?
    var path: String?
    var text: String?
    var title: String?
}

/// The query side: open an index compatibly with a model, embed a query, search.
struct QueryRunner {
    var model: EmbeddingModel
    var bundle: String

    /// Opens `url` and verifies it was built in this model's space (fails closed with
    /// `VectorIndexError.descriptorMismatch`). The query width defaults to the index's width.
    func openIndex(at url: URL, dimensions: Int? = nil) throws -> VectorIndex {
        let index = try VectorIndex.open(at: url)
        let width = dimensions ?? index.descriptor.dimension
        let expected = VectorIndexDescriptor(
            space: model.space, dimension: width, elementType: index.descriptor.elementType)
        guard expected == index.descriptor else {
            throw VectorIndexError.descriptorMismatch(expected: expected, actual: index.descriptor)
        }
        guard model.matryoshkaDimensions.contains(width) else {
            throw EmbeddingError.unsupportedDimension(width, supported: model.matryoshkaDimensions)
        }
        return index
    }

    func embedQuery(_ input: EmbeddingInput, promptName: String?, dimensions: Int) async throws -> Embedding {
        let isImage = { if case .image = input { true } else { false } }()
        try model.require(isImage ? .image : .text, bundle: bundle)
        return try await model.embed(input, promptName: isImage ? nil : promptName, dimensions: dimensions)
    }

    func search(
        _ input: EmbeddingInput, in index: VectorIndex, k: Int, kind: QueryKind, promptName: String?
    ) async throws -> [QueryHit] {
        let query = try await embedQuery(input, promptName: promptName, dimensions: index.descriptor.dimension)
        return try search(query, in: index, k: k, kind: kind)
    }

    func search(_ query: Embedding, in index: VectorIndex, k: Int, kind: QueryKind) throws -> [QueryHit] {
        let results = try index.search(query, k: k, filter: IndexEntryFactory.filter(for: kind))
        return results.enumerated().map { offset, result in
            let payload = IndexEntryFactory.payload(of: result)
            return QueryHit(
                rank: offset + 1, score: result.score, id: result.id,
                kind: result.attributes[IndexEntryFactory.kindAttribute] ?? payload?.kind.rawValue,
                path: payload?.path, text: payload?.text, title: payload?.title)
        }
    }
}

/// `--kind` for `query`.
enum QueryKind: String, CaseIterable, Sendable {
    case image
    case text
    case all
}

// MARK: - Output formatting

enum OutputFormatter {
    /// Header row plus one row per hit; every column is padded to its widest cell.
    static func table(_ hits: [QueryHit]) -> String {
        let width = 100
        guard !hits.isEmpty else { return "no results" }
        let rows = hits.map { hit -> [String] in
            var item = hit.path ?? hit.text.map { "[\(hit.id)] \(oneLine($0))" } ?? hit.id
            if item.count > width { item = String(item.prefix(width - 1)) + "…" }
            return [String(hit.rank), String(format: "%.4f", hit.score), hit.kind ?? "?", item]
        }
        let header = ["rank", "score", "kind", "item"]
        let widths = (0..<header.count).map { column in
            ([header] + rows).map { $0[column].count }.max() ?? 0
        }
        return ([header] + rows).map { row in
            zip(row, widths).map { $0.padding(toLength: $1, withPad: " ", startingAt: 0) }
                .joined(separator: "  ")
                .trimmingCharacters(in: .whitespaces)
        }.joined(separator: "\n")
    }

    private static func oneLine(_ s: String) -> String {
        s.split(whereSeparator: \.isNewline).joined(separator: " ")
    }
}

extension Array {
    func chunked(by size: Int) -> [ArraySlice<Element>] {
        precondition(size > 0)
        return stride(from: 0, to: count, by: size).map { self[$0..<Swift.min($0 + size, count)] }
    }
}

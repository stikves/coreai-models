// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation
import Testing

@testable import CoreAIEmbeddings

@Suite struct EmbeddingModelTests {
    @Test func clipTextBatchesAndOrder() async throws {
        let log = CallLog()
        var model = try FakeModels.model("clip-vit-b32", log: log) { _, _, b, inputs in
            // The row's first word id, so the output keeps the input order visible.
            let ids = inputs["input_ids"]!.ints!
            return [Float(ids[b * 77 + 1])] + [Float](repeating: 0, count: 511)
        }
        let texts = (0..<70).map { String(repeating: "x", count: $0 % 5 + 1) }
        let vectors = try await model.embed(texts: texts)
        #expect(vectors.map { $0.vector[0] } == texts.map { Float($0.count + 10) })
        let calls = await log.calls(of: "text_encoder")
        #expect(calls.map { $0.inputs["input_ids"]!.shape } == [[64, 77], [6, 77]])
        let first = calls[0].inputs
        #expect(Array(first["input_ids"]!.ints!.prefix(4)) == [49406, 11, 49407, 49407])
        #expect(Array(first["attention_mask"]!.ints!.prefix(4)) == [1, 1, 1, 0])
        #expect(vectors.allSatisfy { $0.space.id == "openai/clip-vit-base-patch32" })
    }

    @Test func clipImageCrops() async throws {
        let log = CallLog()
        var model = try FakeModels.model("clip-vit-b32", log: log)
        let red = try Fixtures.image(width: 300, height: 200) { _, _ in (255, 0, 0) }
        _ = try await model.embed(images: [red, red])
        let input = try #require(await log.calls(of: "image_encoder").first?.inputs["pixel_values"])
        #expect(input.shape == [2, 3, 224, 224])
        let r = (1 - 0.48145466) / Float(0.26862954)
        #expect(abs(input.floats[0] - r) < 1e-5)
        #expect(abs(input.floats[224 * 224] - (0 - 0.4578275) / 0.26130258) < 1e-5)
    }

    @Test func embeddingGemma2NeuralEngineImage() async throws {
        let log = CallLog()
        var model = try FakeModels.model("embeddinggemma-2-ios", log: log) { component, output, _, _ in
            output == "soft_tokens" ? (0..<(280 * 512)).map { Float($0 / 512 + 1) } : nil
        }
        let image = try Fixtures.image(width: 64, height: 64) { x, y in (UInt8(x), UInt8(y), 0) }
        _ = try await model.embed(images: [image])
        let pixels = try #require(await log.calls(of: "image_encoder").first?.inputs)
        #expect(pixels["pixel_values"]?.shape == [1, 2520, 768])
        // 64x64 resizes to 768x768: 48x48 patches, 256 soft tokens.
        let positions = try #require(pixels["position_ids"]?.ints)
        #expect(Array(positions.prefix(4)) == [0, 0, 1, 0])
        #expect(positions[2304 * 2] == -1)
        let table = try #require(await log.calls(of: "token_table").first)
        let ids = try #require(table.inputs["input_ids"]?.ints)
        #expect(ids.count == 260 && ids.prefix(2) == [2, 255999] && ids.suffix(2) == [258882, 1])
        let encoder = try #require(await log.calls(of: "encoder").first)
        #expect(encoder.function == "embeds_len512_b1")
        let embeds = try #require(encoder.inputs["inputs_embeds"]?.floats)
        #expect(embeds[2 * 512] == 1 && embeds[257 * 512] == 256)
        #expect(embeds[1 * 512] == 1 && embeds[258 * 512] == 1)
        #expect(encoder.inputs["bias"]?.shape == [1, 512, 1, 512])
        let pool = try #require(encoder.inputs["pool_weights"]?.floats)
        #expect(abs(pool.reduce(0, +) - 1) < 1e-5 && pool[260] == 0)
    }

    @Test func embeddingGemma2GPUImageUsesSharedAsset() async throws {
        let log = CallLog()
        var model = try FakeModels.model("embeddinggemma-2", log: log)
        let image = try Fixtures.image(width: 64, height: 64) { _, _ in (9, 9, 9) }
        _ = try await model.embed(images: [image])
        let calls = await log.calls
        #expect(calls.map(\.component) == ["image_encoder", "token_table", "encoder"])
        #expect(calls[2].function == "embeds_len288_b1")
        let mask = try #require(calls[2].inputs["attention_mask"]?.ints)
        #expect(mask.prefix(260).allSatisfy { $0 == 1 } && mask.dropFirst(260).allSatisfy { $0 == 0 })
    }

    @Test func widthsAndCosine() async throws {
        var model = try FakeModels.model("embeddinggemma-2") { _, _, _, _ in [3, 4] + [Float](repeating: 0, count: 766)
        }
        let full = try await model.embed(["a"])[0]
        let small = try await model.embed(["a"], dimensions: 128)[0]
        #expect(small.dimension == 128 && abs(small.vector[0] - 0.6) < 1e-6 && abs(small.vector[1] - 0.8) < 1e-6)
        #expect(try model.truncate(full, to: 128) == small)
        #expect(throws: EmbeddingError.self) { try model.truncate(small, to: 256) }
        #expect(throws: EmbeddingError.self) { try model.truncate(full, to: 100) }
        await #expect(throws: EmbeddingError.self) { try await model.embed(["a"], dimensions: 100) }

        #expect(try full.cosine(full) > 0.9999)
        let width = #expect(throws: EmbeddingError.self) { try full.cosine(small) }
        #expect(width?.description.contains("[768] vs") == true)
        let other = Embedding(vector: full.vector, space: EmbeddingSpace(id: "other/model", revision: "abc"))
        #expect(throws: EmbeddingError.self) { try full.cosine(other) }
        let truncated = #expect(throws: EmbeddingError.self) { try model.truncate(other, to: 128) }
        guard case .incompatibleEmbeddings = truncated else {
            Issue.record("truncate threw \(String(describing: truncated)), not incompatibleEmbeddings")
            return
        }
    }

    @Test func mixedInputsKeepTheirOrder() async throws {
        let log = CallLog()
        var model = try FakeModels.model("clip-vit-b32", log: log) { component, _, b, _ in
            [component == "text_encoder" ? Float(b + 1) : Float(-(b + 1))] + [Float](repeating: 0, count: 511)
        }
        let image = try Fixtures.image(width: 8, height: 8) { _, _ in (0, 0, 0) }
        let vectors = try await model.embed([.image(image), "a", .image(image), "b c"])
        #expect(vectors.map { $0.vector[0] } == [-1, 1, -2, 2])
        #expect(await log.calls.map(\.component) == ["text_encoder", "image_encoder"])
    }

    @Test func warmupRunsEveryFunction() async throws {
        let gpuLog = CallLog()
        var gpu = try FakeModels.model("embeddinggemma-2", log: gpuLog)
        try await gpu.warmup()
        let gpuCalls = await gpuLog.calls
        let text = Set(gpuCalls.filter { $0.component == "text_encoder" }.map(\.function))
        #expect(
            text == [
                "len128_b1", "len128_b8", "len128_b32", "len512_b1", "len512_b8", "len512_b32", "len2048_b1",
                "len2048_b8",
            ])
        #expect(gpuCalls.contains { $0.component == "image_encoder" })

        let neuralEngineLog = CallLog()
        var neuralEngine = try FakeModels.model("embeddinggemma-2-ios", log: neuralEngineLog)
        try await neuralEngine.warmup([.text])
        let encoder = Set(await neuralEngineLog.calls(of: "encoder").map(\.function))
        #expect(encoder == ["embeds_len128_b1", "embeds_len512_b1", "embeds_len1024_b1"])
        #expect(await neuralEngineLog.calls(of: "image_encoder").isEmpty)
    }

    @Test func cancelledEmbedThrowsBeforeAnyProgramCall() async throws {
        let log = CallLog()
        let image = try Fixtures.image(width: 8, height: 8) { _, _ in (0, 0, 0) }
        let task = Task {
            var model = try FakeModels.model("clip-vit-b32", log: log)
            withUnsafeCurrentTask { $0?.cancel() }
            return try await model.embed(["a", .image(image)])
        }
        await #expect(throws: CancellationError.self) { try await task.value }
        #expect(await log.calls.isEmpty)
    }

    @Test func embeddingRoundTripsThroughCodable() async throws {
        var model = try FakeModels.model("clip-vit-b32")
        let e = try await model.embed(["a"])[0]
        let decoded = try JSONDecoder().decode(Embedding.self, from: JSONEncoder().encode(e))
        #expect(decoded == e)
        #expect(decoded.space.id == "openai/clip-vit-base-patch32")
    }

    @Test func truncationKeepsEosAndSetsStatus() async throws {
        let log = CallLog()
        let block = try Fixtures.block("clip-vit-b32")
        let text = try await FakeModels.clipText(block.text)
        var model = try FakeModels.model("clip-vit-b32", log: log, text: text)
        let long = Array(repeating: "w", count: 100).joined(separator: " ")
        let image = try Fixtures.image(width: 8, height: 8) { _, _ in (0, 0, 0) }
        let vectors = try await model.embed([.text(long), "a b", .image(image)])
        #expect(vectors.map(\.status) == [.truncated(tokens: 102, limit: 77), .complete, .complete])
        // The cut and the in-limit ids are those of `CLIPTokenizer.encodeWithMask`.
        let clip = try CLIPTokenizer(vocab: FakeModels.clipVocab, merges: [])
        let inputs = try #require(await log.calls(of: "text_encoder").first?.inputs)
        let ids = try #require(inputs["input_ids"]?.ints)
        #expect(Array(ids[0..<77]) == clip.encodeWithMask(long).ids)
        #expect(ids.count == 2 * 77 && ids.last == 49407)
        let short = clip.encodeWithMask("a b")
        #expect(text.tokenize("a b").ids == Array(short.ids.prefix(Int(short.attentionMask.reduce(0, +)))))
        #expect(inputs["attention_mask"]?.ints?.prefix(77).reduce(0, +) == 77)

        var gemma = try FakeModels.model("embeddinggemma-2") { _, _, _, _ in [3, 4] + [Float](repeating: 0, count: 766)
        }
        let cut = try await gemma.embed(texts: [Array(repeating: "w", count: 3000).joined(separator: " ")])[0]
        #expect(cut.status == .truncated(tokens: 3002, limit: 2048))
        #expect(try gemma.truncate(cut, to: 128).status == cut.status)
        #expect(try cut.cosine(Embedding(vector: cut.vector, space: cut.space)) > 0.9999)
    }

    @Test func defaultPromptIsPrepended() throws {
        let block = try Fixtures.block("embeddinggemma-2") { $0.edit("text") { $0["default_prompt_name"] = "query" } }
        #expect(block.text.prompt == "task: search result | query: ")
        let text = FakeModels.text(block.text)
        #expect(text.tokenize("cats").ids == text.encode("task: search result | query: cats").map(Int32.init))
    }

    @Test func missingTowerAndNonFiniteOutput() async throws {
        var text = try FakeModels.model("clip-vit-b32-text")
        let image = try Fixtures.image(width: 8, height: 8) { _, _ in (0, 0, 0) }
        let error = await #expect(throws: EmbeddingError.self) { try await text.embed(images: [image]) }
        #expect(error?.description == "the bundle has no image tower")
        var broken = try FakeModels.model("clip-vit-b32") { _, _, _, _ in [.nan] }
        await #expect(throws: EmbeddingError.self) { try await broken.embed(texts: ["a"]) }
    }
}

@Suite struct SharedEmbeddingModelTests {
    @Test func servesCallersOneAtATimeInOrder() async throws {
        let log = CallLog()
        let shared = try SharedEmbeddingModel(
            FakeModels.model("clip-vit-b32", log: log) { _, _, b, inputs in
                [Float(inputs["input_ids"]!.ints![b * 77 + 1])] + [Float](repeating: 0, count: 511)
            })
        await log.hold()
        var calls: [Task<[Embedding], Error>] = []
        for words in 1...4 {
            calls.append(Task { try await shared.embed(texts: [String(repeating: "x", count: words)]) })
            // The first call runs and is held; each later one waits behind the ones before it.
            while await log.calls.isEmpty || shared.waitingCount < words - 1 { await Task.yield() }
        }
        await log.release()
        for (words, call) in calls.enumerated() {
            #expect(try await call.value.map { $0.vector[0] } == [Float(words + 11)])
        }
        #expect(await log.calls.map { $0.inputs["input_ids"]!.ints![1] } == [11, 12, 13, 14])
        #expect(await log.mostRunning == 1)

        // A call that throws puts the model back.
        await #expect(throws: EmbeddingError.self) { try await shared.embed(["a"], dimensions: 100) }
        #expect(try await shared.embed(["a"]).count == 1)
    }
}

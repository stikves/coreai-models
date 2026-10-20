// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation
import TestUtilities
import Testing

@testable import CoreAIEmbeddings

/// Fake components for the Neural Engine layout: a token table, an additive-bias encoder and an image encoder.
private struct FakeNeuralEngineComponents {
    static let hidden = 8
    let table: FakeEmbeddingComponent
    let encoder: FakeEmbeddingComponent
    let imageEncoder: FakeEmbeddingComponent

    init(dimension: Int = 768) {
        let hidden = Self.hidden
        // Row t of the table output is `id_t` repeated, so the encoder input shows which ids it saw.
        table = FakeEmbeddingComponent(dimension: hidden) { _, inputs in
            guard case .int32(let ids) = inputs["input_ids"]?.storage else { return [] }
            return ids.flatMap { [Float](repeating: Float($0), count: hidden) }
        }
        encoder = FakeEmbeddingComponent.constantEncoder(dimension: dimension)
        imageEncoder = FakeEmbeddingComponent.softTokens(hidden: hidden)
    }

    var byName: [String: FakeEmbeddingComponent] {
        ["token_table": table, "encoder": encoder, "image_encoder": imageEncoder]
    }
}

private func fakeNeuralEngineModel(metadata: Data? = nil) throws -> (EmbeddingModel, FakeNeuralEngineComponents) {
    let components = FakeNeuralEngineComponents()
    let model = try EmbeddingFixtures.model(
        metadata: metadata ?? EmbeddingFixtures.aneMetadata(), components: components.byName)
    return (model, components)
}

/// Asserts the additive-bias encoder inputs for `valid` real tokens in a `length` bucket.
private func expectAdditiveBias(_ call: FakeEmbeddingComponent.Call, length: Int, valid: Int) {
    #expect(call.shapes["bias"] == [1, length, 1, length])
    #expect(call.shapes["pool_weights"] == [1, length])
    #expect(call.inputs["attention_mask"] == nil)
    guard case .float32(let bias) = call.inputs["bias"]?.storage,
        case .float32(let weights) = call.inputs["pool_weights"]?.storage
    else {
        Issue.record("bias / pool_weights are not float tensors")
        return
    }
    // Layout [1, key, 1, query]: every query sees the same per-key value.
    for key in [0, valid - 1, valid, length - 1] where key < length {
        let row = bias[(key * length)..<((key + 1) * length)]
        let expected: Float = key < valid ? 0 : -30000
        #expect(row.allSatisfy { $0 == expected }, "key \(key)")
    }
    #expect(bias.filter { $0 == 0 }.count == valid * length)
    let n = Float(valid)
    #expect(weights.prefix(valid).allSatisfy { abs($0 - 1 / n) < 1e-7 })
    #expect(weights.dropFirst(valid).allSatisfy { $0 == 0 })
}

@Suite("Neural Engine bundle")
struct NeuralEngineBundleTests {
    @Test("decodes the Neural Engine bundle metadata: per-component compute units and mask encoding")
    func decodesFixture() throws {
        let c = try EmbeddingFixtures.aneConfig()
        #expect(c.space == EmbeddingFixtures.space)
        #expect(c.precision?.compute == "float16")
        #expect(c.precision?.forbidden == [])
        #expect(c.precision?.computeUnits == ["neural_engine", "gpu"])
        #expect(c.component("text_encoder") == nil)
        #expect(c.component("token_table")?.computeUnits == ["gpu"])
        #expect(c.component("token_table")?.maskEncoding == nil)
        let encoder = try #require(c.component("encoder"))
        #expect(encoder.computeUnits == ["neural_engine"])
        #expect(encoder.maskEncoding == .additiveBias)
        #expect(encoder.inputs == ["inputs_embeds", "bias", "pool_weights"])
        #expect(encoder.output == "embedding")
        #expect(encoder.functions?.map(\.length) == [128, 512, 1024])
        #expect(c.component("image_encoder")?.computeUnits == ["neural_engine"])
        #expect(c.tower(for: .text)?.pipeline == ["token_table", "encoder"])
        #expect(c.tower(for: .image)?.pipeline == ["image_encoder", "token_table", "encoder"])
        #expect(c.tower(for: .text)?.output == "embedding")
        #expect(c.tower(for: .image)?.output == "embedding")
        #expect(c.text?.truncation?.maxTokens == 1024)
    }

    @Test("compute units resolve to the GPU when listed, else the Neural Engine")
    func computeUnitResolution() throws {
        #expect(ComputeUnit.resolve(nil) == .gpu)
        #expect(ComputeUnit.resolve(["gpu"]) == .gpu)
        #expect(ComputeUnit.resolve(["neural_engine"]) == .neuralEngine)
        #expect(ComputeUnit.resolve(["ANE"]) == .neuralEngine)
        #expect(ComputeUnit.resolve(["neural_engine", "gpu"]) == .gpu)
        #expect(ComputeUnit.resolve(["cpu"]) == nil)
        #expect(ComputeUnit.resolve([]) == nil)

        let url = URL(fileURLWithPath: "bundle/x.aimodel")
        #expect(
            try CoreAIEmbeddingComponent(name: "x", modelURL: url, computeUnits: ["neural_engine"]).computeUnit
                == .neuralEngine)
        #expect(try CoreAIEmbeddingComponent(name: "x", modelURL: url, computeUnits: ["gpu"]).computeUnit == .gpu)
        #expect(throws: EmbeddingError.self) {
            try CoreAIEmbeddingComponent(name: "x", modelURL: url, computeUnits: ["cpu"])
        }
    }

    @Test("a component's units come from its own entry, else the bundle's precision block")
    func computeUnitsOfComponent() throws {
        let c = try EmbeddingFixtures.aneConfig()
        #expect(c.computeUnits(of: "token_table") == ["gpu"])
        #expect(c.computeUnits(of: "encoder") == ["neural_engine"])
        #expect(c.computeUnits(of: "image_encoder") == ["neural_engine"])
        #expect(c.computeUnits(of: "undeclared") == ["neural_engine", "gpu"])
        let text = try EmbeddingFixtures.textConfig()
        #expect(text.computeUnits(of: "text_encoder") == ["gpu"])
    }

    @Test("loads when every component resolves; float16 compute with an empty forbidden list is allowed")
    func loads() async throws {
        let (model, _) = try fakeNeuralEngineModel()
        try await model.load()
        #expect(model.name == "embeddinggemma-2-ios")
        #expect(model.supportedModalities == [.text, .image])
    }

    @Test("text runs token_table, then the encoder at the smallest batch-1 bucket with additive bias")
    func twoStageText() async throws {
        let (model, components) = try fakeNeuralEngineModel()
        let ids = try await model.tokenize("hello world", promptName: "query")
        let e = try await model.embed(.text("hello world"), promptName: "query")
        #expect(e.dimension == 768)
        #expect(e.tokenCount == ids.count)
        #expect(ids.count < 128)

        let tableCall = try #require(await components.table.calls.first)
        #expect(tableCall.function == "main")
        #expect(tableCall.shapes["input_ids"] == [1, 128])
        guard case .int32(let fed) = tableCall.inputs["input_ids"]?.storage else {
            Issue.record("token table ids are not int32")
            return
        }
        #expect(Array(fed.prefix(ids.count)) == ids)
        #expect(fed.dropFirst(ids.count).allSatisfy { $0 == 0 })  // pad id from metadata

        let call = try #require(await components.encoder.calls.first)
        #expect(call.function == "len128_b1")
        #expect(call.output == "embedding")
        #expect(call.shapes["inputs_embeds"] == [1, 128, FakeNeuralEngineComponents.hidden])
        guard case .float32(let embeds) = call.inputs["inputs_embeds"]?.storage else {
            Issue.record("inputs_embeds is not float")
            return
        }
        #expect(embeds.prefix(FakeNeuralEngineComponents.hidden).allSatisfy { $0 == Float(ids[0]) })
        expectAdditiveBias(call, length: 128, valid: ids.count)
    }

    @Test("bucket choice: 7 tokens -> 128, 300 -> 512, 1024 -> 1024; longer token inputs are refused")
    func bucketChoice() async throws {
        let (model, components) = try fakeNeuralEngineModel()
        let rows = [7, 300, 1024].map { n in (0..<n).map { Int32(10 + $0 % 50) } }
        let out = try await model.embed(tokens: rows, dimensions: nil)
        #expect(out.map(\.tokenCount) == [7, 300, 1024])
        let calls = await components.encoder.calls
        #expect(calls.map(\.function) == ["len128_b1", "len512_b1", "len1024_b1"])
        expectAdditiveBias(calls[1], length: 512, valid: 300)
        #expect(await components.table.calls.map { $0.shapes["input_ids"]! } == [[1, 128], [1, 512], [1, 1024]])

        await #expect(throws: EmbeddingError.inputTooLong(tokens: 1500, maximum: 1024)) {
            try await model.embed(tokens: [[Int32](repeating: 10, count: 1500)], dimensions: nil)
        }
    }

    @Test("long text is truncated to the largest bucket and keeps EOS")
    func longTextTruncates() async throws {
        let (model, components) = try fakeNeuralEngineModel()
        let text = String(repeating: "hello world ", count: 1500)
        let e = try await model.embed(.text(text), dimensions: 256)
        #expect(e.truncatedInput)
        #expect(e.tokenCount == 1024)
        #expect(abs(EmbeddingMath.l2Norm(e.vector) - 1) < 1e-5)  // renormalized after the Matryoshka cut
        let call = try #require(await components.encoder.calls.first)
        #expect(call.function == "len1024_b1")
        expectAdditiveBias(call, length: 1024, valid: 1024)
        let tableCall = try #require(await components.table.calls.first)
        guard case .int32(let fed) = tableCall.inputs["input_ids"]?.storage else {
            Issue.record("token table ids are not int32")
            return
        }
        #expect(fed.last == 1)  // eos id from metadata
    }

    @Test("image runs image_encoder, token_table, then the additive-bias encoder")
    func imagePipeline() async throws {
        let (model, components) = try fakeNeuralEngineModel()
        let image = try #require(ImageTestSupport.gradientImage(width: 512, height: 512))
        let e = try await model.embed(.image(image))
        #expect(e.tokenCount == 256 + 4)

        let imageCall = try #require(await components.imageEncoder.calls.first)
        #expect(imageCall.output == "soft_tokens")
        #expect(imageCall.shapes["pixel_values"] == [1, 2520, 768])
        #expect(imageCall.shapes["position_ids"] == [1, 2520, 2])
        guard case .int32(let positions) = imageCall.inputs["position_ids"]?.storage else {
            Issue.record("position_ids are not int32")
            return
        }
        #expect(positions.suffix(2) == [-1, -1])  // padding patches

        let call = try #require(await components.encoder.calls.first)
        #expect(call.function == "len512_b1")
        #expect(call.output == "embedding")
        expectAdditiveBias(call, length: 512, valid: 260)
        guard case .float32(let embeds) = call.inputs["inputs_embeds"]?.storage else {
            Issue.record("inputs_embeds is not float")
            return
        }
        let h = FakeNeuralEngineComponents.hidden
        // Soft-token values come from `FakeEmbeddingComponent.softTokens` (`% 7`).
        #expect(Array(embeds[(2 * h)..<(3 * h)]) == (0..<h).map { Float($0 % 7) })  // soft token 0
    }

    @Test("encoder inputs follow the declared mask encoding")
    func encoderInputsByMaskEncoding() {
        let embeds = [Float](repeating: 1, count: 4 * 2)
        let mask: [Int32] = [1, 1, 1, 0]
        let plain = EmbeddingModel.encoderInputs(
            component: .init(functions: [.init(function: "len4_b1", batch: 1, length: 4)]), embeds: embeds,
            mask: mask, hidden: 2)
        #expect(Set(plain.keys) == ["inputs_embeds", "attention_mask"])
        #expect(plain["attention_mask"] == EmbeddingTensor(int32: mask, shape: [1, 4]))

        let biased = EmbeddingModel.encoderInputs(
            component: .init(function: "main", maskEncoding: .additiveBias), embeds: embeds, mask: mask, hidden: 2)
        #expect(Set(biased.keys) == ["inputs_embeds", "bias", "pool_weights"])
        #expect(biased["pool_weights"] == EmbeddingTensor(float32: [1 / 3, 1 / 3, 1 / 3, 0], shape: [1, 4]))
        #expect(biased["bias"]?.shape == [1, 4, 1, 4])
    }

    @Test("int32 position ids fed to a float16 input keep the -1 padding value")
    func float16PositionIds() throws {
        var array = NDArray(shape: [1, 3, 2], scalarType: .float16)
        try CoreAIEmbeddingComponent.fill(&array, with: EmbeddingTensor(int32: [0, 0, 1, 0, -1, -1], shape: [1, 3, 2]))
        #expect(flattenNDArray(array, as: Float16.self) == [0, 0, 1, 0, -1, -1])
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIEmbeddings

// MARK: - Function planning

@Suite struct FunctionPlanTests {
    @Test func clipSplitsDynamicBatches() throws {
        let text = try #require(try Fixtures.block("clip-vit-b32").components["text_encoder"])
        let calls = try text.plan(lengths: Array(repeating: 10, count: 70), name: "text_encoder")
        #expect(calls.map(\.rows.count) == [64, 6])
        #expect(calls.map(\.batch) == [64, 6])
        #expect(calls.allSatisfy { $0.length == 77 && $0.function == "text_encoder" })
    }

    @Test func embeddingGemma2Buckets() throws {
        let text = try #require(try Fixtures.block("embeddinggemma-2").components["text_encoder"])
        let nine = try text.plan(lengths: Array(repeating: 100, count: 9), name: "text_encoder")
        #expect(nine.map(\.function) == ["len128_b8", "len128_b1"])
        let mixed = try text.plan(lengths: [600, 100, 12], name: "text_encoder")
        #expect(mixed.map(\.function) == ["len128_b1", "len128_b1", "len2048_b1"])
        #expect(mixed.map(\.rows) == [[1], [2], [0]])
        #expect(throws: EmbeddingError.self) { try text.plan(lengths: [3000], name: "text_encoder") }
        let image = try #require(try Fixtures.block("embeddinggemma-2").components["image_encoder"])
        let calls = try image.plan(lengths: [260, 12], name: "image_encoder")
        #expect(calls.map(\.rows) == [[0], [1]])
        #expect(calls.allSatisfy { $0.length == nil && $0.function == "main" })
    }

    @Test func dynamicLengthPadsToLongestRow() throws {
        let table = try #require(try Fixtures.block("embeddinggemma-2").components["token_table"])
        let calls = try table.plan(lengths: [5, 1], name: "token_table")
        #expect(calls.map(\.length) == [5, 2])
        #expect(calls.map(\.batch) == [1, 1])
    }

    @Test func padsBelowMinimumBatch() throws {
        let component = Component(
            asset: "a", inputs: [:], outputs: [:], computeUnits: [.gpu],
            functions: [FunctionEntry(function: "b4", batch: 4...4, length: nil)])
        let calls = try component.plan(lengths: [0, 0, 0, 0, 0], name: "c")
        #expect(calls.map(\.rows.count) == [4, 1])
        #expect(calls.map(\.batch) == [4, 4])
    }
}

// MARK: - Driver planning

@Suite struct PipelineDriverPlanTests {
    static func driver(_ name: String, _ modality: EmbeddingModality) throws -> PipelineDriver {
        let bundle = try FakeModels.bundle(name)
        let tower = try #require(bundle.block.tower(for: modality))
        return try PipelineDriver(
            tower: tower, block: bundle.block,
            programs: ProgramCache(bundle: bundle.bundle, factory: ProgramCache.coreAI))
    }

    @Test(arguments: Fixtures.names)
    func plansEveryTower(_ name: String) throws {
        let bundle = try FakeModels.bundle(name)
        for tower in bundle.block.towers {
            let driver = try Self.driver(name, tower.modality)
            #expect(driver.stages.map(\.name) == tower.pipeline)
        }
    }

    @Test func seedsAndDerivations() throws {
        #expect(try Self.driver("clip-vit-b32", .image).seed == .croppedPixels)
        #expect(try Self.driver("clip-vit-b32", .text).stages[0].derived.isEmpty)
        let gpuImage = try Self.driver("embeddinggemma-2", .image)
        #expect(gpuImage.seed == .patches)
        #expect(gpuImage.stages[2].derived == ["inputs_embeds": .mergeSoftTokens(imageTokenId: 258880)])
        let neuralEngineText = try Self.driver("embeddinggemma-2-ios", .text)
        #expect(
            neuralEngineText.stages[1].derived == [
                "bias": .additiveBias(AdditiveBias(masked: -30000, unmasked: 0)), "pool_weights": .poolWeights,
            ])
        let neuralEngineImage = try Self.driver("embeddinggemma-2-ios", .image)
        #expect(Set(neuralEngineImage.stages[2].derived.keys) == ["inputs_embeds", "bias", "pool_weights"])
    }

    @Test func refusesAnInputNothingProvides() throws {
        let programs = ProgramCache(bundle: try FakeModels.bundle("clip-vit-b32").bundle, factory: ProgramCache.coreAI)
        let text = try Fixtures.block("clip-vit-b32") {
            $0.edit("components.text_encoder") {
                $0["inputs"] = [
                    "input_ids": "int32[B,L]", "attention_mask": "int32[B,L]", "token_type_ids": "int32[B,L]",
                ]
            }
        }
        let error = #expect(throws: EmbeddingError.self) {
            try PipelineDriver(tower: text.tower(for: .text)!, block: text, programs: programs)
        }
        #expect(error?.description.contains("token_type_ids") == true)
        let image = try Fixtures.block("clip-vit-b32") {
            $0.edit("components.image_encoder") { $0["inputs"] = ["pixel_values": "float32[B,196,768]"] }
        }
        #expect(throws: EmbeddingError.self) {
            try PipelineDriver(tower: image.tower(for: .image)!, block: image, programs: programs)
        }
    }

    @Test func derivations() throws {
        let driver = try Self.driver("embeddinggemma-2-ios", .image)
        var row = driver.rows(tokens: [[2, 255999, 258880, 258880, 258882, 1]])[0]
        row.tensors["inputs_embeds"] = HostTensor(float32: [Float](repeating: -1, count: 6 * 2), shape: [6, 2])
        row.tensors["soft_tokens"] = HostTensor(float32: [10, 11, 20, 21, 30, 31], shape: [3, 2])
        let merged = try driver.derive(.mergeSoftTokens(imageTokenId: 258880), row: row, length: 8)
        #expect(merged.floats == [-1, -1, -1, -1, 10, 11, 20, 21, -1, -1, -1, -1])

        let bias = try driver.derive(.additiveBias(AdditiveBias(masked: -9, unmasked: 0)), row: row, length: 8)
        #expect(bias.shape == [8, 1, 8])
        #expect(Array(bias.floats[0..<8]) == [Float](repeating: 0, count: 8))
        #expect(Array(bias.floats[48..<56]) == [Float](repeating: -9, count: 8))

        let pool = try driver.derive(.poolWeights, row: row, length: 8)
        #expect(pool.floats == [1, 1, 1, 1, 1, 1, 0, 0].map { $0 / 6 })
    }
}

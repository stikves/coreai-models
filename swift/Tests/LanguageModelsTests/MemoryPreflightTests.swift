// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAILanguageModels

@Suite("Memory preflight")
struct MemoryPreflightTests {
    private static let gb: UInt64 = 1_000_000_000

    // MARK: State type names (as the asset summary prints them)

    @Test("a growing KV state is sized at the context")
    func growingKVState() {
        let bytes = MemoryPreflight.stateBytes(
            typeName: "NDArray (Float16, 28 × 1 × 8 × ? × 128)", contextTokens: 256)
        #expect(bytes == 2 * 28 * 1 * 8 * 256 * 128)
    }

    @Test("element size follows the scalar type")
    func scalarTypes() {
        #expect(
            MemoryPreflight.stateBytes(typeName: "NDArray (Int8, 36 × 1 × 8 × ? × 128)", contextTokens: 10)
                == 36 * 8 * 10 * 128)
        #expect(
            MemoryPreflight.stateBytes(typeName: "NDArray (BFloat16, 18 × 1 × 16 × 128 × 128)", contextTokens: 10)
                == 2 * 18 * 16 * 128 * 128)
        #expect(
            MemoryPreflight.stateBytes(typeName: "NDArray (Float32, 4 × 4)", contextTokens: 1) == 4 * 16)
    }

    @Test("with a dynamic batch too, one dynamic dimension counts at the context")
    func dynamicBatch() {
        // As in a batched export: [layers, batch?, heads, sequence?, head dim].
        #expect(
            MemoryPreflight.stateBytes(typeName: "NDArray (Float16, 36 × ? × 8 × ? × 128)", contextTokens: 100)
                == 2 * 36 * 8 * 100 * 128)
    }

    @Test("an unreadable type name gives no size")
    func unreadable() {
        #expect(MemoryPreflight.stateBytes(typeName: "NDArray (Complex64, 4 × 4)", contextTokens: 8) == nil)
        #expect(MemoryPreflight.stateBytes(typeName: "PixelBuffer", contextTokens: 8) == nil)
        #expect(MemoryPreflight.stateBytes(typeName: "NDArray (Float16, 4 × n)", contextTokens: 8) == nil)
    }

    @Test("a state shared by several functions counts once, at its largest")
    func sharedStatesCountOnce() {
        let bytes = MemoryPreflight.stateBytes(
            typeNames: [
                "keyCache": ["NDArray (Float16, 2 × 4)", "NDArray (Float16, 2 × 8)"],
                "valueCache": ["NDArray (Float16, 2 × 8)"],
            ],
            contextTokens: 1)
        #expect(bytes == 2 * 16 + 2 * 16)
    }

    @Test("one unreadable state makes the state size unknown")
    func unreadableStateIsUnknown() {
        let bytes = MemoryPreflight.stateBytes(
            typeNames: ["keyCache": ["NDArray (Float16, 2 × 4)"], "odd": ["PixelBuffer"]], contextTokens: 1)
        #expect(bytes == nil)
    }

    // MARK: KV context

    @Test("a fixed-size cache is sized for its full capacity")
    func fixedSizeContext() {
        #expect(
            MemoryPreflight.kvContextTokens(
                strategy: .fixedSize, initialCapacity: nil, maxTokens: 50, maxContextLength: 32768) == 32768)
        #expect(
            MemoryPreflight.kvContextTokens(
                strategy: .fixedSize, initialCapacity: 4096, maxTokens: 50, maxContextLength: 32768) == 4096)
    }

    @Test("a growing cache is sized for its initial capacity or the new tokens")
    func growingContext() {
        #expect(
            MemoryPreflight.kvContextTokens(
                strategy: .auto, initialCapacity: nil, maxTokens: 50, maxContextLength: 32768) == 256)
        #expect(
            MemoryPreflight.kvContextTokens(
                strategy: .growing, initialCapacity: nil, maxTokens: 2000, maxContextLength: 32768) == 2000)
        #expect(
            MemoryPreflight.kvContextTokens(
                strategy: .growing, initialCapacity: nil, maxTokens: 100_000, maxContextLength: 32768) == 32768)
    }

    // MARK: Verdict

    private static func device(_ bytes: UInt64) -> MemoryPreflight.Budget {
        MemoryPreflight.Budget(deviceBytes: bytes, processBytes: nil)
    }

    @Test("unknown availability never blocks")
    func unknownAvailability() {
        let estimate = MemoryEstimate(weightBytes: 100 * Self.gb, stateBytes: 50 * Self.gb, contextTokens: 1)
        let none = MemoryPreflight.Budget(deviceBytes: nil, processBytes: nil)
        #expect(MemoryPreflight.verdict(for: estimate, budget: none) == .fits)
        #expect(MemoryPreflight.verdict(for: estimate, budget: .init(deviceBytes: 0, processBytes: 0)) == .fits)
    }

    @Test("weights and state above the device's memory are too large")
    func tooLargeForDevice() {
        let estimate = MemoryEstimate(weightBytes: 5 * Self.gb, stateBytes: 2 * Self.gb, contextTokens: 1)
        #expect(estimate.totalBytes == 7 * Self.gb)
        #expect(
            MemoryPreflight.verdict(for: estimate, budget: Self.device(6 * Self.gb))
                == .tooLarge(limit: .device, neededBytes: 7 * Self.gb, availableBytes: 6 * Self.gb))
        #expect(MemoryPreflight.verdict(for: estimate, budget: Self.device(7 * Self.gb)) == .fits)
    }

    @Test("only the state is held against the process limit")
    func processLimitCountsState() {
        // Weights are memory-mapped and may not count against the process limit.
        let estimate = MemoryEstimate(weightBytes: 5 * Self.gb, stateBytes: 1 * Self.gb, contextTokens: 1)
        let phone = MemoryPreflight.Budget(deviceBytes: 8 * Self.gb, processBytes: 3 * Self.gb)
        #expect(MemoryPreflight.verdict(for: estimate, budget: phone) == .fits)

        let bigCache = MemoryEstimate(weightBytes: 1 * Self.gb, stateBytes: 4 * Self.gb, contextTokens: 1)
        #expect(
            MemoryPreflight.verdict(for: bigCache, budget: phone)
                == .tooLarge(limit: .process, neededBytes: 4 * Self.gb, availableBytes: 3 * Self.gb))
    }

    @Test("without state sizes the weights get the documented margin, and no process check")
    func unknownStateMargin() {
        let estimate = MemoryEstimate(weightBytes: 10 * Self.gb, stateBytes: nil, contextTokens: 1)
        #expect(estimate.totalBytes == UInt64(Double(10 * Self.gb) * MemoryEstimate.unknownStateMargin))
        let phone = MemoryPreflight.Budget(deviceBytes: 16 * Self.gb, processBytes: 1)
        #expect(MemoryPreflight.verdict(for: estimate, budget: phone) == .fits)
    }

    /// Measured export sizes of models that run on Macs with the listed RAM. The check must
    /// never refuse them, even with the KV cache at a generous context.
    @Test(
        "models that run on Macs today pass",
        arguments: [
            // (export bytes, layers, KV heads, head dim, Mac RAM GB)
            (UInt64(4_632_530_944), 36, 8, 128, UInt64(32)),  // qwen3-8b, peaks ~8.9 GB
            (UInt64(6_662_656_000), 48, 8, 256, UInt64(36)),  // gemma3-12b, peaks ~12.8 GB
        ])
    func todaysModelsPass(exportBytes: UInt64, layers: Int, heads: Int, headDim: Int, ramGB: UInt64) {
        let context = 8192
        let kv = UInt64(2 * layers * heads * headDim * 2 * context)  // K and V, fp16
        let estimate = MemoryEstimate(weightBytes: exportBytes, stateBytes: kv, contextTokens: context)
        let ram = ramGB * 1_073_741_824
        #expect(MemoryPreflight.verdict(for: estimate, budget: Self.device(ram)) == .fits)
    }

    @Test("an 8B model with a full 32K fp16 KV cache does not fit an 8 GB phone")
    func eightBillionAtFullContextOnAPhone() {
        let kv = UInt64(2 * 36 * 8 * 128 * 2 * 32768)  // ~4.8 GB
        let estimate = MemoryEstimate(weightBytes: 4_632_530_944, stateBytes: kv, contextTokens: 32768)
        let phone = MemoryPreflight.Budget(deviceBytes: 8 * 1_073_741_824, processBytes: nil)
        guard case .tooLarge(.device, _, _) = MemoryPreflight.verdict(for: estimate, budget: phone) else {
            Issue.record("expected tooLarge for the device")
            return
        }
    }

    @Test("the device message names the need, the breakdown, what's available and the opt-out")
    func deviceMessage() {
        let estimate = MemoryEstimate(weightBytes: 4_600_000_000, stateBytes: 4_800_000_000, contextTokens: 32768)
        let text = MemoryPreflight.tooLargeMessage(
            modelName: "qwen3-8b", estimate: estimate, limit: .device, neededBytes: estimate.totalBytes,
            availableBytes: 8_000_000_000)
        #expect(
            text
                == "qwen3-8b needs ~9.4 GB of memory (weights ~4.6 GB + KV cache ~4.8 GB at 32768 tokens), "
                + "~8.0 GB available. Pass --skip-memory-check to try anyway.")
    }

    @Test("the process message names the KV cache")
    func processMessage() {
        let estimate = MemoryEstimate(weightBytes: 4_600_000_000, stateBytes: 4_800_000_000, contextTokens: 32768)
        let text = MemoryPreflight.tooLargeMessage(
            modelName: "qwen3-8b", estimate: estimate, limit: .process, neededBytes: 4_800_000_000,
            availableBytes: 3_000_000_000)
        #expect(
            text
                == "qwen3-8b needs ~4.8 GB for its KV cache at 32768 tokens, ~3.0 GB available to this process. "
                + "Pass --skip-memory-check to try anyway.")
    }

    // MARK: Bundle

    @Test("the estimate counts the run's assets and skips the draft head")
    func bundleWeights() throws {
        let dir = FileManager.default.temporaryDirectory.appending(
            path: "MemoryPreflightTests-\(UUID().uuidString)/bundle")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent()) }
        try """
        {
          "metadata_version": "0.2",
          "kind": "llm",
          "name": "tiny",
          "assets": { "main": "main.aimodel", "drafter": "drafter.aimodel" },
          "language": { "tokenizer": "x/y", "vocab_size": 10, "max_context_length": 64 }
        }
        """.write(to: dir.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        for (name, size) in [("main.aimodel", 1000), ("drafter.aimodel", 5000)] {
            let asset = dir.appending(path: name)
            try FileManager.default.createDirectory(at: asset, withIntermediateDirectories: true)
            try Data(count: size).write(to: asset.appending(path: "weights.bin"))
        }

        let bundle = try LanguageModelBundle(at: dir)
        let estimate = MemoryPreflight.estimate(bundle: bundle, contextTokens: 64)
        #expect(estimate.weightBytes == 1000)
        // Not a real asset, so its states can't be read.
        #expect(estimate.stateBytes == nil)
        #expect(estimate.contextTokens == 64)
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

#if os(iOS)
import os
#endif

// MARK: - Estimate

/// A lower bound on the memory a language model bundle needs, computed before loading it.
///
/// The estimate is the model assets' size on disk plus the model's state (the KV cache, and any
/// fixed recurrent state) at the context the run asks for, with shapes and scalar types read from
/// the asset's summary. It leaves out activations and framework scratch, so it is an
/// under-estimate on purpose: measured peaks on macOS are about twice the export size, and the
/// check must never refuse a model that runs today.
package struct MemoryEstimate: Sendable, Equatable {
    /// Bytes of the model assets and tensor data the run loads, as stored on disk.
    package var weightBytes: UInt64
    /// Bytes of the model's state at `contextTokens`, or nil when the asset's state shapes
    /// could not be read.
    package var stateBytes: UInt64?
    /// Tokens the KV cache is sized for.
    package var contextTokens: Int

    /// Factor on the weights when the state size is unknown, standing in for the KV cache.
    package static let unknownStateMargin = 1.1

    package init(weightBytes: UInt64, stateBytes: UInt64?, contextTokens: Int) {
        self.weightBytes = weightBytes
        self.stateBytes = stateBytes
        self.contextTokens = contextTokens
    }

    package var totalBytes: UInt64 {
        if let stateBytes { return weightBytes + stateBytes }
        return UInt64(Double(weightBytes) * Self.unknownStateMargin)
    }
}

// MARK: - Preflight

/// Checks, before a model loads, that the device has room for it.
///
/// A model that doesn't fit fails late and badly: the GPU runs out of memory part-way through
/// generation. Comparing a lower-bound estimate with the memory available turns the clear cases
/// into an early, readable error.
package enum MemoryPreflight {
    /// The memory a run can use, as far as the system tells.
    package struct Budget: Equatable, Sendable {
        /// The device's physical memory: weights and state together can't exceed it.
        package var deviceBytes: UInt64?
        /// What is left before this process hits its memory limit (`os_proc_available_memory()`
        /// on iOS), or nil where the system sets no such limit.
        package var processBytes: UInt64?

        package init(deviceBytes: UInt64?, processBytes: UInt64?) {
            self.deviceBytes = deviceBytes
            self.processBytes = processBytes
        }
    }

    /// Which limit an estimate exceeds.
    package enum Limit: Equatable, Sendable {
        /// Weights and state together exceed the device's memory.
        case device
        /// The state alone exceeds what this process may still allocate.
        case process
    }

    /// The outcome of comparing an estimate with the memory available.
    package enum Verdict: Equatable, Sendable {
        /// The estimate fits, or nothing is known about the memory available.
        case fits
        /// The estimate exceeds `limit`: `neededBytes` against `availableBytes`.
        case tooLarge(limit: Limit, neededBytes: UInt64, availableBytes: UInt64)
    }

    /// Compare `estimate` with `budget`. An unknown limit never blocks.
    ///
    /// Only the state is held against the process limit. The weights are memory-mapped from the
    /// asset, and a process's footprint (what the limit counts) may leave such file-backed pages
    /// out, so counting them there could refuse a model that runs today.
    package static func verdict(for estimate: MemoryEstimate, budget: Budget) -> Verdict {
        if let device = budget.deviceBytes, device > 0, estimate.totalBytes > device {
            return .tooLarge(limit: .device, neededBytes: estimate.totalBytes, availableBytes: device)
        }
        if let process = budget.processBytes, process > 0, let state = estimate.stateBytes, state > process {
            return .tooLarge(limit: .process, neededBytes: state, availableBytes: process)
        }
        return .fits
    }

    /// The memory this process can use: physical memory everywhere, and on iOS also what is
    /// left before the process's limit (`os_proc_available_memory()`; 0 means no limit is known).
    package static func availableMemory() -> Budget {
        let device = ProcessInfo.processInfo.physicalMemory
        #if os(iOS)
        let available = os_proc_available_memory()
        return Budget(deviceBytes: device, processBytes: available > 0 ? UInt64(available) : nil)
        #else
        return Budget(deviceBytes: device, processBytes: nil)
        #endif
    }

    /// The context the KV cache is sized for in a run of up to `maxTokens` new tokens.
    ///
    /// A fixed-size cache is allocated at its full size. A growing one starts at its initial
    /// capacity and grows to hold the prompt and the new tokens; the prompt isn't tokenized yet,
    /// so it counts as empty, which keeps the estimate a lower bound.
    package static func kvContextTokens(
        strategy: KVCacheStrategy,
        initialCapacity: Int?,
        maxTokens: Int,
        maxContextLength: Int
    ) -> Int {
        switch strategy {
        case .fixedSize, .chunked:
            return min(initialCapacity ?? maxContextLength, maxContextLength)
        case .auto, .growing:
            let initial =
                initialCapacity ?? KVCacheStrategy.growing.defaultSize(maxContextLength: maxContextLength) ?? 0
            return min(max(initial, maxTokens), maxContextLength)
        }
    }

    /// Estimate the memory `bundle` needs with its KV cache sized for `contextTokens`.
    package static func estimate(bundle: LanguageModelBundle, contextTokens: Int) -> MemoryEstimate {
        // Every component the run loads; a draft head loads only for speculative decoding.
        let components = bundle.modelBundle.assets
            .filter { $0.key != ModelBundle.ComponentKey.drafter }
            .map { bundle.bundlePath.appending(path: $0.value) }
        let files = components + Array(bundle.tensorData.values)
        let weightBytes = files.reduce(UInt64(0)) { $0 + UInt64($1.recursiveFileSizeInBytes() ?? 0) }

        var stateBytes: UInt64?
        if let mainURL = bundle.modelBundle.modelURL(for: ModelBundle.ComponentKey.main),
            let states = try? stateTypeNames(of: mainURL), !states.isEmpty
        {
            // No states at all means the summary told us nothing: keep the size unknown.
            stateBytes = self.stateBytes(typeNames: states, contextTokens: contextTokens)
        }
        return MemoryEstimate(weightBytes: weightBytes, stateBytes: stateBytes, contextTokens: contextTokens)
    }

    /// The type names of the asset's states, one per state name, from its summary (no
    /// specialization). A state shared by several functions is listed once, at its largest.
    static func stateTypeNames(of assetURL: URL) throws -> [String: [String]] {
        let asset = try AIModelAsset(contentsOf: assetURL)
        guard let summary = try asset.summary(includingStatistics: false) else { return [:] }
        var byName: [String: [String]] = [:]
        for function in summary.functions {
            for state in function.states {
                byName[state.name, default: []].append(state.typeName)
            }
        }
        return byName
    }

    /// Total bytes of the states named in `typeNames` at `contextTokens`, each state at the
    /// largest size any function declares for it. Nil if any state's type name can't be read.
    static func stateBytes(typeNames: [String: [String]], contextTokens: Int) -> UInt64? {
        var total: UInt64 = 0
        for (_, names) in typeNames {
            var largest: UInt64?
            for name in names {
                guard let bytes = stateBytes(typeName: name, contextTokens: contextTokens) else { return nil }
                largest = max(largest ?? 0, bytes)
            }
            total += largest ?? 0
        }
        return total
    }

    /// Bytes of one state from its summary type name, such as
    /// `NDArray (Float16, 28 × 1 × 8 × ? × 128)`. Nil when the name doesn't parse.
    ///
    /// A dynamic (`?`) dimension is the sequence or the batch. One of them is counted at
    /// `contextTokens` and any other at 1: whichever is the sequence, that is a lower bound
    /// for a single-sequence run.
    static func stateBytes(typeName: String, contextTokens: Int) -> UInt64? {
        guard let open = typeName.firstIndex(of: "("), let close = typeName.lastIndex(of: ")"),
            open < close
        else { return nil }
        let body = typeName[typeName.index(after: open)..<close]
        let parts = body.split(separator: ",", maxSplits: 1).map { $0.trimmingCharacters(in: .whitespaces) }
        guard parts.count == 2, let elementSize = scalarByteSize(parts[0]) else { return nil }

        let dims = parts[1].split(whereSeparator: { $0 == "×" || $0 == "x" })
            .map { $0.trimmingCharacters(in: .whitespaces) }
        guard !dims.isEmpty else { return nil }

        var count = UInt64(elementSize)
        var sequenceCounted = false
        for dim in dims {
            let value: Int
            if dim == "?" {
                value = sequenceCounted ? 1 : contextTokens
                sequenceCounted = true
            } else if let fixed = Int(dim), fixed >= 0 {
                value = fixed
            } else {
                return nil
            }
            let (product, overflow) = count.multipliedReportingOverflow(by: UInt64(value))
            guard !overflow else { return nil }
            count = product
        }
        return count
    }

    /// Bytes per element of a summary scalar type name.
    static func scalarByteSize(_ name: String) -> Int? {
        switch name.lowercased() {
        case "int8", "uint8", "bool", "float8", "float8e4m3", "float8e5m2": return 1
        case "float16", "bfloat16", "int16", "uint16": return 2
        case "float32", "int32", "uint32": return 4
        case "float64", "int64", "uint64": return 8
        default: return nil
        }
    }

    /// `bytes` as gigabytes for a message, e.g. `9.3 GB`.
    package static func formatGB(_ bytes: UInt64) -> String {
        String(format: "%.1f GB", Double(bytes) / 1_000_000_000)
    }

    /// The error a run prints when the estimate doesn't fit.
    package static func tooLargeMessage(
        modelName: String, estimate: MemoryEstimate, limit: Limit, neededBytes: UInt64, availableBytes: UInt64
    ) -> String {
        let optOut = "Pass --skip-memory-check to try anyway."
        switch limit {
        case .device:
            let breakdown: String
            if let state = estimate.stateBytes {
                breakdown =
                    "weights ~\(formatGB(estimate.weightBytes)) + KV cache ~\(formatGB(state)) "
                    + "at \(estimate.contextTokens) tokens"
            } else {
                breakdown = "weights ~\(formatGB(estimate.weightBytes)) × \(MemoryEstimate.unknownStateMargin)"
            }
            return "\(modelName) needs ~\(formatGB(neededBytes)) of memory (\(breakdown)), "
                + "~\(formatGB(availableBytes)) available. \(optOut)"
        case .process:
            return "\(modelName) needs ~\(formatGB(neededBytes)) for its KV cache at "
                + "\(estimate.contextTokens) tokens, ~\(formatGB(availableBytes)) available to this process. "
                + optOut
        }
    }
}

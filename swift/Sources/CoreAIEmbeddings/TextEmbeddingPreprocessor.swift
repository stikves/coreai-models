// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

/// Prompt prefix, tokenization, BOS/EOS, truncation, length bucketing, batch planning and padding for a text tower.
struct TextEmbeddingPreprocessor: Sendable {
    /// Ids for one input after prompt, special tokens and truncation.
    struct Tokenized: Sendable, Equatable {
        var ids: [Int32]
        var truncated: Bool
    }

    /// One graph call: `indices` into the request, run at a static `(batch, length)`.
    struct BatchPlan: Sendable, Equatable {
        var indices: [Int]
        var batch: Int
        var length: Int
        /// Function to call, when the component lists static functions.
        var function: String?
    }

    let prompts: [String: String]
    let defaultPromptName: String?
    let addBOS: Bool
    let addEOS: Bool
    let bosTokenId: Int32?
    let eosTokenId: Int32?
    let padTokenId: Int32
    let paddingSide: String
    let keepEOSOnTruncation: Bool
    /// Static `(batch, length)` traces, sorted by length then batch. Empty when the graph is dynamic.
    let functions: [EmbeddingConfig.FunctionSpec]
    /// Sorted length buckets.
    let lengthBuckets: [Int]
    /// Allowed batch sizes for a dynamic graph, sorted.
    let batchSizes: [Int]
    let maxBatchTokens: Int?
    /// Longest sequence the tower accepts (including special tokens).
    let maximumSequenceLength: Int

    init(
        configuration: EmbeddingConfig,
        tower: EmbeddingConfig.Tower,
        bosTokenId: Int?,
        eosTokenId: Int?,
        padTokenId: Int?,
        fallbackFunctions: [EmbeddingConfig.FunctionSpec] = []
    ) throws {
        let text = configuration.text
        prompts = text?.prompts ?? [:]
        defaultPromptName = text?.defaultPromptName
        let special = text?.specialTokens
        addBOS = special?.bos ?? false
        addEOS = special?.eos ?? false
        let bos = special?.bosTokenId ?? bosTokenId
        let eos = special?.eosTokenId ?? eosTokenId
        if addBOS && bos == nil { throw EmbeddingError.invalidBundle("no BOS token id in metadata or tokenizer") }
        if addEOS && eos == nil { throw EmbeddingError.invalidBundle("no EOS token id in metadata or tokenizer") }
        self.bosTokenId = bos.map { Int32($0) }
        self.eosTokenId = eos.map { Int32($0) }
        self.padTokenId = Int32(special?.padTokenId ?? padTokenId ?? 0)
        paddingSide = text?.paddingSide ?? "right"
        keepEOSOnTruncation = text?.truncation?.keepEos ?? true

        let component = tower.pipeline.last.flatMap { configuration.component($0) }
        let declared = component?.functions ?? []
        let fns = (declared.isEmpty ? fallbackFunctions : declared).sorted {
            ($0.length, $0.batch) < ($1.length, $1.batch)
        }
        functions = fns
        var buckets = Set(tower.lengthBuckets ?? [])
        buckets.formUnion(fns.map(\.length))
        lengthBuckets = buckets.sorted()
        if let sizes = tower.batchSizes, !sizes.isEmpty {
            batchSizes = Array(Set(sizes)).sorted()
        } else {
            batchSizes = [1]
        }
        maxBatchTokens = tower.maxBatchTokens

        let limit = Self.maximumSequenceLength(text: text, lengthBuckets: lengthBuckets)
        maximumSequenceLength = limit
        guard limit >= (addBOS ? 1 : 0) + (addEOS ? 1 : 0) + 1 else {
            throw EmbeddingError.invalidBundle("maximum sequence length \(limit) leaves no room for content")
        }
    }

    /// Longest accepted sequence: `truncation.max_tokens`, capped by the largest bucket and `model_max_tokens`.
    static func maximumSequenceLength(text: EmbeddingConfig.Text?, lengthBuckets: [Int]) -> Int {
        var limit = text?.truncation?.maxTokens ?? lengthBuckets.last ?? text?.modelMaxTokens ?? 512
        if let bucketMax = lengthBuckets.last { limit = min(limit, bucketMax) }
        if let modelMax = text?.modelMaxTokens { limit = min(limit, modelMax) }
        return limit
    }

    /// Prompt prefix for `name` (falls back to `default_prompt_name`; none means no prefix).
    func prompt(named name: String?) throws -> String {
        guard let key = name ?? defaultPromptName else { return "" }
        guard let prefix = prompts[key] else {
            throw EmbeddingError.unknownPromptName(key, available: prompts.keys.sorted())
        }
        return prefix
    }

    /// Prompt + tokenize + BOS/EOS + truncation (right side, keeping EOS when configured).
    func tokenize(_ text: String, promptName: String?, tokenizer: any Tokenizer) throws -> Tokenized {
        let full = try prompt(named: promptName) + text
        let content = tokenizer.encode(text: full, addSpecialTokens: false).map { Int32($0) }
        return try assemble(content: content)
    }

    /// Add BOS/EOS around `content` and truncate to ``maximumSequenceLength``.
    func assemble(content: [Int32]) throws -> Tokenized {
        let reserved = (addBOS ? 1 : 0) + (addEOS ? 1 : 0)
        let room = maximumSequenceLength - reserved
        var body = content
        var truncated = false
        if body.count > room {
            truncated = true
            body = Array(body.prefix(room))
        }
        var ids: [Int32] = []
        ids.reserveCapacity(body.count + reserved)
        if addBOS, let bos = bosTokenId { ids.append(bos) }
        ids += body
        if addEOS, let eos = eosTokenId { ids.append(eos) }
        if truncated && addEOS && !keepEOSOnTruncation {
            // Without keep_eos the cut drops the EOS slot too; refill it with content.
            ids.removeLast()
            if body.count < content.count { ids.append(content[body.count]) }
        }
        guard !ids.isEmpty else { throw EmbeddingError.emptyInput }
        return Tokenized(ids: ids, truncated: truncated)
    }

    /// Validate fully formed ids: never truncated.
    func validateTokens(_ ids: [Int32]) throws -> Tokenized {
        guard !ids.isEmpty else { throw EmbeddingError.emptyInput }
        guard ids.count <= maximumSequenceLength else {
            throw EmbeddingError.inputTooLong(tokens: ids.count, maximum: maximumSequenceLength)
        }
        return Tokenized(ids: ids, truncated: false)
    }

    /// Smallest length bucket that fits `count` tokens (or `count` itself for a bucket-less dynamic graph).
    func bucket(for count: Int) throws -> Int {
        guard count <= maximumSequenceLength else {
            throw EmbeddingError.inputTooLong(tokens: count, maximum: maximumSequenceLength)
        }
        if lengthBuckets.isEmpty { return count }
        guard let fit = lengthBuckets.first(where: { $0 >= count }) else {
            throw EmbeddingError.inputTooLong(tokens: count, maximum: lengthBuckets.last ?? 0)
        }
        return fit
    }

    /// Group inputs by length bucket (first-seen order) and split each group into graph calls.
    ///
    /// Static functions: each call uses the smallest listed batch that holds the rest of the
    /// group, else the largest listed batch for that length; short calls are padded with dummy rows.
    /// Dynamic graph: calls hold at most `max(batchSizes)` rows and `maxBatchTokens / length` rows.
    func plan(lengths: [Int]) throws -> [BatchPlan] {
        var order: [Int] = []
        var groups: [Int: [Int]] = [:]
        for (i, n) in lengths.enumerated() {
            let b = try bucket(for: n)
            if groups[b] == nil { order.append(b) }
            groups[b, default: []].append(i)
        }
        var plans: [BatchPlan] = []
        for length in order {
            var pending = groups[length]![...]
            let specs = functions.filter { $0.length == length }
            while !pending.isEmpty {
                if !specs.isEmpty {
                    let spec = specs.first { $0.batch >= pending.count } ?? specs.last!
                    let take = min(spec.batch, pending.count)
                    plans.append(
                        BatchPlan(
                            indices: Array(pending.prefix(take)), batch: spec.batch, length: length,
                            function: spec.function))
                    pending = pending.dropFirst(take)
                } else {
                    var cap = batchSizes.last ?? 1
                    if let tokens = maxBatchTokens { cap = min(cap, max(tokens / max(length, 1), 1)) }
                    let take = min(cap, pending.count)
                    plans.append(BatchPlan(indices: Array(pending.prefix(take)), batch: take, length: length))
                    pending = pending.dropFirst(take)
                }
            }
        }
        return plans
    }

    /// Pack rows into `int32[batch, length]` ids and mask. Rows past `rows.count` are dummy rows
    /// (one attended pad token, so pooling never divides by zero); their outputs are discarded.
    func pack(_ rows: [[Int32]], batch: Int, length: Int) -> (ids: [Int32], mask: [Int32]) {
        var ids = [Int32](repeating: padTokenId, count: batch * length)
        var mask = [Int32](repeating: 0, count: batch * length)
        let left = paddingSide == "left"
        for r in 0..<batch {
            let base = r * length
            if r < rows.count {
                let row = rows[r].prefix(length)
                let start = left ? base + length - row.count : base
                for (j, id) in row.enumerated() {
                    ids[start + j] = id
                    mask[start + j] = 1
                }
            } else {
                mask[left ? base + length - 1 : base] = 1
            }
        }
        return (ids, mask)
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

// MARK: - Speculative Decoding Configuration

/// Configuration for speculative decoding, parsed from metadata.json.
public struct SpeculativeConfig: Sendable {
    public let drafterAsset: String
    public let draftLength: Int

    public init(drafterAsset: String, draftLength: Int = 16) {
        self.drafterAsset = drafterAsset
        self.draftLength = draftLength
    }
}

// MARK: - Verification Result

public struct VerificationResult: Sendable {
    public let acceptedCount: Int
    public let correctionToken: Int32
}

// MARK: - Speculative Runner

/// Coordinates speculative decoding between a target engine and a drafter engine.
///
/// The runner implements a draft-verify-accept loop using the target engine's
/// existing `forcedContinuation` + `includeLogits` capability for batch verification,
/// and `reset(to:)` for KV cache rollback on rejection.
///
/// Discovery: if the bundle's metadata.json contains `"assets": {"drafter": "..."}`,
/// the factory creates a SpeculativeRunner wrapping the target engine. No CLI flags
/// needed — speculation is automatic when a drafter asset is present.
///
/// The drafter shares embed_tokens and lm_head with the target (DFlash architecture).
/// At export time, shared weights are deduplicated; at runtime, both .aimodel files
/// are loaded independently (the CoreAI runtime handles weight sharing at the
/// inference function level if the tensors are identical).
public struct SpeculativeRunner<Target: InferenceEngine>: Sendable
where Target.OutputSequence: Sendable {
    public let target: Target
    public let drafter: Target
    public let draftLength: Int

    public init(target: Target, drafter: Target, draftLength: Int = 16) {
        self.target = target
        self.drafter = drafter
        self.draftLength = draftLength
    }

    /// Run one draft-verify-accept cycle starting from the current cache state.
    ///
    /// Both target and drafter must have already processed the shared prefix
    /// (via prior generate() calls or prefill). This method:
    /// 1. Generates K draft tokens from the drafter (sequential forwards)
    /// 2. Verifies all K drafts against the target in one batched forward
    /// 3. Returns the accepted tokens and a correction/bonus token
    /// 4. Rolls back both caches to the accepted position
    ///
    /// - Parameters:
    ///   - lastToken: The last accepted token (needed as drafter input).
    ///   - sampling: Sampling configuration for drafting.
    ///   - maxDraft: Maximum draft tokens this cycle (may be less than draftLength
    ///     if approaching maxTokens limit).
    /// - Returns: Array of accepted token IDs (including the correction/bonus token).
    ///   Empty if the drafter produced nothing.
    public func speculateCycle(
        lastToken: Int32,
        sampling: SamplingConfiguration,
        maxDraft: Int? = nil
    ) async throws -> [Int32] {
        let k = min(maxDraft ?? draftLength, draftLength)

        // Phase 1: Draft K tokens from drafter (sequential AR forwards)
        let draftOutput = try await drafter.generate(
            with: [lastToken],
            samplingConfiguration: sampling,
            inferenceOptions: InferenceOptions(maxTokens: k)
        )
        var draftTokens: [Int32] = []
        for try await output in draftOutput {
            draftTokens.append(output.tokenId)
        }

        guard !draftTokens.isEmpty else { return [] }

        // Phase 2: Verify via target's forcedContinuation + includeLogits.
        // Feed [lastToken] + draftTokens as forced continuation.
        // The engine processes them in one batched forward and returns
        // per-position logits. logits[i] predicts the token at position i+1.
        let verifyInput = [lastToken] + draftTokens
        let verifyOutput = try await target.generate(
            with: verifyInput,
            samplingConfiguration: sampling,
            inferenceOptions: InferenceOptions(
                maxTokens: verifyInput.count,
                includeLogits: true,
                forcedContinuation: verifyInput
            )
        )

        // Collect target logits for each position
        var targetLogits: [[LogitsScalarType]] = []
        for try await output in verifyOutput {
            if let logits = output.logits {
                targetLogits.append(logits)
            }
        }

        // Phase 3: Greedy verification — compare argmax(target_logits[i]) vs draftTokens[i]
        var accepted = 0
        for i in 0..<draftTokens.count {
            guard i < targetLogits.count else { break }
            let targetPrediction = argmax(targetLogits[i])
            if targetPrediction == draftTokens[i] {
                accepted += 1
            } else {
                break
            }
        }

        // Correction/bonus token: target's prediction at the first mismatch position
        let correctionToken: Int32
        if accepted < draftTokens.count, accepted < targetLogits.count {
            correctionToken = argmax(targetLogits[accepted])
        } else if accepted == draftTokens.count, accepted < targetLogits.count {
            correctionToken = argmax(targetLogits[accepted])
        } else {
            correctionToken = draftTokens.last ?? lastToken
        }

        // Phase 4: Rollback rejected tokens from both caches
        let rejected = draftTokens.count - accepted
        if rejected > 0 {
            let drafterRollback = drafter.processedTokenCount - rejected
            try await drafter.reset(to: max(0, drafterRollback))
        }

        // Target processed all verify tokens; rollback to accepted + 1 position
        // (keep the correction token's position in cache)
        let targetKeep = target.processedTokenCount - rejected
        if rejected > 0 {
            try await target.reset(to: max(0, targetKeep))
        }

        // Build result: accepted draft tokens + correction
        var result = Array(draftTokens.prefix(accepted))
        result.append(correctionToken)
        return result
    }
}

// MARK: - Helpers

private func argmax(_ logits: [LogitsScalarType]) -> Int32 {
    guard !logits.isEmpty else { return 0 }
    var maxIdx = 0
    var maxVal = logits[0]
    for i in 1..<logits.count {
        if logits[i] > maxVal {
            maxVal = logits[i]
            maxIdx = i
        }
    }
    return Int32(maxIdx)
}

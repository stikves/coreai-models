// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Protocol for engines that support speculative decoding with a companion drafter model.
///
/// Discovery is bundle-driven: if metadata.json contains `"assets": {"drafter": "..."}`,
/// the engine reports `hasDrafter == true`. Call `loadAndEnableDrafter()` to allocate
/// drafter memory and activate speculative decoding. Call `disableDrafter()` to release.
///
/// When enabled, `generate()` internally runs draft-verify-accept cycles instead of
/// single-token decode. The public API contract is unchanged — callers iterate the
/// same `OutputSequence` and receive tokens in bursts (accepted prefix + correction).
public protocol SpeculativeCapable: InferenceEngine {
    /// Whether the loaded bundle includes a drafter model asset.
    var hasDrafter: Bool { get }

    /// Whether speculative decoding is currently active.
    var isDrafterEnabled: Bool { get }

    /// Load the drafter model and enable speculative decoding.
    ///
    /// This allocates the drafter's weight memory and KV cache.
    /// Subsequent `generate()` calls use draft-verify-accept internally.
    /// No-op if already enabled.
    func loadAndEnableDrafter() async throws

    /// Disable speculative decoding and release drafter memory.
    /// Subsequent `generate()` calls use standard single-token decode.
    /// No-op if already disabled.
    func disableDrafter() async
}

/// Default implementation for engines that don't support speculation.
extension InferenceEngine {
    public var hasDrafter: Bool { false }
    public var isDrafterEnabled: Bool { false }
}

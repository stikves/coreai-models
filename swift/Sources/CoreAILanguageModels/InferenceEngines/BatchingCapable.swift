// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// An engine's concurrent-request width, surfaced to the serving scheduler.
///
/// Batching is a property of the engine × queue, not of the KV-cache backend. Engines that can
/// advance more than one independent request in a single generation step conform to this; the
/// serving `BatchManager`/`RequestQueue` reads ``InferenceEngine/batchCapability`` to decide how
/// many admitted sessions to advance per step.
///
/// Today's engines do not conform, so `batchCapability` is 1 (serialize; prefixes are still shared
/// across turns via the cache). A future paged engine that runs a batched decode step reports `N`,
/// and the serving path does not change — only the width does.
public protocol BatchingCapable {
    /// Maximum number of independent requests the engine can advance in one generation step.
    var maxConcurrentRequests: Int { get }
}

extension InferenceEngine {
    /// Batch width for the serving scheduler: `maxConcurrentRequests` when the engine is
    /// ``BatchingCapable``, otherwise 1 (serialize). Never less than 1.
    public var batchCapability: Int {
        max(1, (self as? BatchingCapable)?.maxConcurrentRequests ?? 1)
    }
}

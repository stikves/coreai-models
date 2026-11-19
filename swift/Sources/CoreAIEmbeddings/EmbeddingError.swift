// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Errors from reading or running an embedding bundle.
public enum EmbeddingError: Error, CustomStringConvertible, LocalizedError {
    /// The `embedding` block breaks the contract.
    case invalidMetadata(String)
    /// The call needs a tower the bundle does not have.
    case modalityNotInBundle(EmbeddingModality)
    /// A tower's pipeline needs an input no earlier stage or rule provides.
    case unsupportedPipeline(EmbeddingModality, String)
    /// A width not in `matryoshkaDimensions`.
    case unsupportedDimension(Int, supported: [Int])
    /// Embeddings from different spaces (or, for `cosine`, of different widths).
    case incompatibleEmbeddings(String)
    case imageConversionFailed
    /// A program returned something other than its declared outputs.
    case unexpectedOutput(component: String, String)
    /// Core AI failed to load or run a program on `unit`: `gpu`, `neural_engine` or `cpu`.
    case programFailed(asset: String, function: String?, unit: String, reason: String)
    /// A `SharedEmbeddingModel` already has its `maxQueueDepth` callers waiting.
    case queueFull(depth: Int)

    public var description: String {
        switch self {
        case .invalidMetadata(let reason):
            return "invalid embedding metadata: \(reason)"
        case .modalityNotInBundle(let modality):
            return "the bundle has no \(modality.rawValue) tower"
        case .unsupportedPipeline(let modality, let reason):
            return "cannot run the \(modality.rawValue) pipeline: \(reason)"
        case .unsupportedDimension(let width, let supported):
            return "dimension \(width) is not one of \(supported)"
        case .incompatibleEmbeddings(let pair):
            return "embeddings from different spaces or widths: \(pair)"
        case .imageConversionFailed:
            return "cannot convert the image to RGB"
        case .unexpectedOutput(let component, let reason):
            return "\(component) returned an unexpected output: \(reason)"
        case .programFailed(let asset, let function, let unit, let reason):
            let name = function.map { "\(asset)/\($0)" } ?? asset
            return "\(name) failed on \(unit): \(reason)"
        case .queueFull(let depth):
            return "\(depth) calls are already waiting for the model"
        }
    }

    public var errorDescription: String? { description }
}

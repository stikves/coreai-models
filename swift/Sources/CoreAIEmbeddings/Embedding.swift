// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIVectorIndex
import CoreGraphics
import Foundation

/// The kind of input an embedding model accepts.
///
/// A bundle exposes one tower per modality; `supportedModalities` on ``EmbeddingModel`` lists which
/// ones it carries, so callers can adapt before sending an input.
///
/// ## Usage
/// ```swift
/// if model.supportedModalities.contains(.image) {
///     let photo = try await model.embed(.image(cgImage))
/// }
/// ```
///
/// ## See also
/// ``EmbeddingInput`` pairs each modality with its payload; ``EmbeddingModel`` runs the towers.
public enum EmbeddingModality: String, Sendable, CaseIterable {
    case text
    case image
}

/// One text or image to embed.
///
/// Both kinds travel through the same `embed` call, so one batch can mix captions and photos and
/// still land in a single vector space.
///
/// ## Usage
/// ```swift
/// let vectors = try await model.embed([.text(caption), .image(photo)])
/// ```
///
/// ## See also
/// `EmbeddingModel.embed(_:promptName:dimensions:)` consumes inputs and returns ``Embedding`` values.
public enum EmbeddingInput: Sendable {
    /// Raw text; the prompt prefix and special tokens come from the bundle metadata.
    case text(String)
    /// An image, routed to the image tower.
    case image(CGImage)
}

/// A normalized vector together with the space it belongs to.
///
/// The space stops vectors from different models or revisions from being compared by accident:
/// similarity and index operations check it and throw on a mismatch.
///
/// ## Usage
/// ```swift
/// let a = try await model.embed(.text("a cat asleep on a sofa"), promptName: "query")
/// let b = try await model.embed(.image(photo))
/// let similarity = try a.cosineSimilarity(to: b)
/// let compact = try a.truncated(to: 128)
/// ```
///
/// ## See also
/// Produced by ``EmbeddingModel``. Append and search embeddings with `VectorIndex`
/// (`VectorIndex.append(id:embedding:payload:attributes:)`).
public struct Embedding: Sendable, Equatable {
    /// The space this vector belongs to.
    public let space: VectorSpace
    /// The vector (float32).
    public let vector: [Float]
    /// Whether the input was cut to fit the model's maximum sequence length.
    public let truncatedInput: Bool
    /// Tokens the encoder saw for this input, including special and prompt tokens.
    public let tokenCount: Int

    /// Number of components in ``vector``.
    public var dimension: Int { vector.count }

    package init(space: VectorSpace, vector: [Float], truncatedInput: Bool = false, tokenCount: Int = 0) {
        self.space = space
        self.vector = vector
        self.truncatedInput = truncatedInput
        self.tokenCount = tokenCount
    }

    /// Cosine similarity with another embedding, in `[-1, 1]`.
    ///
    /// Throws ``EmbeddingError/spaceMismatch(_:_:)`` when the spaces differ and
    /// ``EmbeddingError/dimensionMismatch(_:_:)`` when the widths differ.
    public func cosineSimilarity(to other: Embedding) throws -> Float {
        guard space == other.space else { throw EmbeddingError.spaceMismatch(space, other.space) }
        guard dimension == other.dimension else {
            throw EmbeddingError.dimensionMismatch(dimension, other.dimension)
        }
        return EmbeddingMath.cosineSimilarity(vector, other.vector)
    }

    /// The first `dimensions` components, re-normalized to unit length (Matryoshka prefix).
    public func truncated(to dimensions: Int) throws -> Embedding {
        guard dimensions > 0, dimensions <= vector.count else {
            throw EmbeddingError.unsupportedDimension(dimensions, supported: [vector.count])
        }
        return Embedding(
            space: space, vector: EmbeddingMath.truncate(vector, to: dimensions), truncatedInput: truncatedInput,
            tokenCount: tokenCount)
    }
}

/// Errors thrown by CoreAIEmbeddings.
///
/// ## Usage
/// ```swift
/// do {
///     _ = try await model.embed(.text(note), promptName: "query")
/// } catch EmbeddingError.unknownPromptName(let name, let available) {
///     print("\(name) is not one of \(available)")
/// }
/// ```
///
/// ## See also
/// Thrown by ``EmbeddingModel``, ``Embedding`` and the `VectorIndex` glue.
public enum EmbeddingError: Error, Equatable, LocalizedError {
    /// Two embeddings come from different model spaces.
    case spaceMismatch(VectorSpace, VectorSpace)
    /// Two embeddings have different widths.
    case dimensionMismatch(Int, Int)
    /// The requested width is not one of the model's Matryoshka widths.
    case unsupportedDimension(Int, supported: [Int])
    /// The prompt name is not defined by the bundle.
    case unknownPromptName(String, available: [String])
    /// The bundle has no tower for this modality.
    case unsupportedModality(EmbeddingModality)
    /// The text has more tokens than the largest length the text tower accepts.
    case inputTooLong(tokens: Int, maximum: Int)
    /// The text produced no tokens.
    case emptyInput
    /// The image could not be decoded to RGB pixels.
    case invalidImage
    /// The bundle metadata, tokenizer or assets are missing or inconsistent.
    case invalidBundle(String)
    /// A model component failed to specialize, load or run.
    case componentFailed(component: String, function: String?, computeUnit: String, cached: Bool, reason: String)

    public var errorDescription: String? {
        switch self {
        case .spaceMismatch(let a, let b):
            return "embeddings come from different spaces (\(a) vs \(b)) and cannot be compared"
        case .dimensionMismatch(let a, let b):
            return "embedding dimensions differ (\(a) vs \(b)); truncate the longer one first"
        case .unsupportedDimension(let d, let supported):
            return "dimension \(d) is not supported (supported: \(supported))"
        case .unknownPromptName(let name, let available):
            return "unknown prompt name '\(name)' (available: \(available.sorted()))"
        case .unsupportedModality(let m):
            return "this embedding bundle has no '\(m.rawValue)' tower"
        case .inputTooLong(let tokens, let maximum):
            return "input has \(tokens) tokens; the maximum is \(maximum)"
        case .emptyInput:
            return "input produced no tokens"
        case .invalidImage:
            return "could not convert the image to RGB pixels"
        case .invalidBundle(let detail):
            return "invalid embedding bundle: \(detail)"
        case .componentFailed(let component, let function, let computeUnit, let cached, let reason):
            let target = function.map { "\(component) function '\($0)'" } ?? component
            let device = computeUnit == "neural_engine" ? "the Neural Engine" : "the GPU"
            let hint =
                cached
                ? " Clear the cached specialization with `embedding-search --clear-coreai-cache` "
                    + "or `PreparedModel.clearCache(at:)`."
                : ""
            return "\(target) failed on \(device): \(reason).\(hint)"
        }
    }
}

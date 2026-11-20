// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// A model that turns texts and images into vectors of one space.
///
/// `CoreAIEmbeddingModel` runs a bundle for one owner; `SharedEmbeddingModel` serves many callers
/// from one loaded model. Code that takes either one can be generic:
///
/// ```swift
/// func index<M: EmbeddingModel & ~Copyable>(_ photos: [CGImage], with model: inout M) async throws -> [Embedding] {
///     try await model.embed(images: photos)
/// }
/// ```
public protocol EmbeddingModel: ~Copyable {
    /// Vectors are comparable only within one space.
    var space: EmbeddingSpace { get }
    /// The native vector width.
    var dimension: Int { get }
    /// The widths `embed` can return, native first.
    var matryoshkaDimensions: [Int] { get }
    var modalities: Set<EmbeddingModality> { get }
    /// `logits = logitScale * cosine`, for models trained with one (CLIP).
    var logitScale: Double? { get }

    /// Embeds `inputs`, texts and images in any mix, and returns one vector per input in order.
    ///
    /// - Parameter dimensions: one of `matryoshkaDimensions`; the native width when `nil`.
    mutating func embed(_ inputs: [EmbeddingInput], dimensions: Int?) async throws -> [Embedding]

    /// `embedding` cut to `width` leading values, L2-normalized again when the bundle says so: the
    /// vector `embed` returns at that width, without running the model. The status stays.
    ///
    /// - Throws: `EmbeddingError.unsupportedDimension` unless `width` is one of
    ///   `matryoshkaDimensions` and at most `embedding.dimension`; `.incompatibleEmbeddings` for an
    ///   embedding of another space.
    func truncate(_ embedding: Embedding, to width: Int) throws -> Embedding
}

extension EmbeddingModel where Self: ~Copyable {
    /// Embeds `inputs` at the native width.
    public mutating func embed(_ inputs: [EmbeddingInput]) async throws -> [Embedding] {
        try await embed(inputs, dimensions: nil)
    }

    /// Embeds `texts`.
    public mutating func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(texts.map { .text($0) }, dimensions: dimensions)
    }

    /// Embeds `images`.
    public mutating func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(images.map { .image($0) }, dimensions: dimensions)
    }

    /// The body of `truncate(_:to:)`: checks `embedding` and `width`, then cuts, L2-normalizing
    /// again when `renormalize` is set.
    func checkedCut(_ embedding: Embedding, to width: Int, renormalize: Bool) throws -> Embedding {
        guard embedding.space == space else {
            throw EmbeddingError.incompatibleEmbeddings("the embedding is from \(embedding.space.id), not \(space.id)")
        }
        guard matryoshkaDimensions.contains(width), width <= embedding.dimension else {
            throw EmbeddingError.unsupportedDimension(
                width, supported: matryoshkaDimensions.filter { $0 <= embedding.dimension })
        }
        return embedding.cut(to: width, renormalize: renormalize)
    }

    /// Throws `unsupportedDimension` unless `dimensions` is `nil` or one of `matryoshkaDimensions`.
    func check(dimensions: Int?) throws {
        if let dimensions, !matryoshkaDimensions.contains(dimensions) {
            throw EmbeddingError.unsupportedDimension(dimensions, supported: matryoshkaDimensions)
        }
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#if canImport(CreateMLComponents)
import CoreImage
import CoreML
import CreateMLComponents
import Foundation

// A bundle's towers as CreateMLComponents transformers, e.g.
// `CoreAIEmbeddingImageExtractor(model:).appending(LogisticRegressionClassifier())` where
// `ImageFeaturePrint()` would go. The extractors take a `SharedEmbeddingModel`, so a text and an
// image extractor can share one loaded model.

/// `String` -> `MLShapedArray<Float>` of shape `[dimensions]`. Reports each input to `eventHandler` as one
/// item.
public struct CoreAIEmbeddingTextExtractor: CreateMLComponents.Transformer, Sendable {
    public let model: SharedEmbeddingModel
    /// One of `model.matryoshkaDimensions`; the native width when `nil`.
    public let dimensions: Int?

    /// - Throws: `EmbeddingError.unsupportedDimension`.
    public init(model: SharedEmbeddingModel, dimensions: Int? = nil) throws {
        try model.check(dimensions: dimensions)
        self.model = model
        self.dimensions = dimensions
    }

    public func applied(to input: String, eventHandler: EventHandler? = nil) async throws -> MLShapedArray<Float> {
        let embedding = try await model.embed(texts: [input], dimensions: dimensions)[0]
        eventHandler?(Event(origin: "CoreAIEmbeddingTextExtractor", itemCount: 1, metrics: [:]))
        return MLShapedArray(scalars: embedding.vector, shape: [embedding.dimension])
    }
}

/// `CIImage` -> `MLShapedArray<Float>` of shape `[dimensions]`. An `ImageFeatureExtractor`, so it
/// fits wherever `ImageFeaturePrint` does. Reports each input to `eventHandler` as one item.
public struct CoreAIEmbeddingImageExtractor: ImageFeatureExtractor, @unchecked Sendable {
    public let model: SharedEmbeddingModel
    /// One of `model.matryoshkaDimensions`; the native width when `nil`.
    public let dimensions: Int?
    private let context: CIContext

    /// - Throws: `EmbeddingError.modalityNotInBundle` for a bundle without an image tower,
    ///   `.unsupportedDimension`.
    public init(model: SharedEmbeddingModel, dimensions: Int? = nil, context: CIContext = CIContext()) throws {
        guard model.modalities.contains(.image) else { throw EmbeddingError.modalityNotInBundle(.image) }
        try model.check(dimensions: dimensions)
        self.model = model
        self.dimensions = dimensions
        self.context = context
    }

    public func applied(to input: CIImage, eventHandler: EventHandler? = nil) async throws -> MLShapedArray<Float> {
        guard !input.extent.isInfinite, let image = context.createCGImage(input, from: input.extent) else {
            throw EmbeddingError.imageConversionFailed
        }
        let embedding = try await model.embed(images: [image], dimensions: dimensions)[0]
        eventHandler?(Event(origin: "CoreAIEmbeddingImageExtractor", itemCount: 1, metrics: [:]))
        return MLShapedArray(scalars: embedding.vector, shape: [embedding.dimension])
    }
}
#endif

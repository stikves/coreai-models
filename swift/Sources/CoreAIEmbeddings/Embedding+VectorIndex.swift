// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIVectorIndex
import Foundation

extension VectorIndexDescriptor {
    /// Describes an index that holds `model`'s embeddings at `dimensions` (default: the native width).
    ///
    /// Building the descriptor from the model keeps the index in the model's space and width, so later
    /// appends and searches with that model's embeddings are accepted and anything else is refused.
    /// Throws ``EmbeddingError/unsupportedDimension(_:supported:)`` for a width the model does not offer.
    ///
    /// ## Usage
    /// ```swift
    /// var index = try VectorIndex(descriptor: VectorIndexDescriptor(model: model, elementType: .float16))
    /// for (id, photo) in photos { try index.append(id: id, embedding: try await model.embed(.image(photo))) }
    /// let hits = try index.search(try await model.embed(.text("beach at sunset"), promptName: "query"), k: 10)
    /// ```
    ///
    /// ## See also
    /// ``EmbeddingModel`` produces the embeddings; `VectorIndex` stores and searches them.
    public init(
        model: EmbeddingModel,
        dimensions: Int? = nil,
        elementType: VectorElementType = .float32
    ) throws {
        let width = dimensions ?? model.dimension
        guard model.matryoshkaDimensions.contains(width) else {
            throw EmbeddingError.unsupportedDimension(width, supported: model.matryoshkaDimensions)
        }
        self.init(space: model.space, dimension: width, elementType: elementType)
    }
}

extension VectorIndex {
    /// Appends an embedding; the index refuses a different space or width.
    public mutating func append(
        id: String,
        embedding: Embedding,
        payload: Data? = nil,
        attributes: [String: String] = [:]
    ) throws {
        try append(
            id: id, vector: embedding.vector, space: embedding.space, payload: payload, attributes: attributes)
    }

    /// Searches with an embedding as the query; its space and width must match the index.
    public func search(_ query: Embedding, k: Int, filter: [String: String] = [:]) throws -> [SearchResult] {
        try search(query.vector, space: query.space, k: k, filter: filter)
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// One input's vector, in the space of the bundle that made it. `Codable`: the space travels with
/// the vector when it is stored.
public struct Embedding: Sendable, Equatable, Codable {
    public let vector: [Float]
    /// Vectors are comparable only within one space.
    public let space: EmbeddingSpace

    public init(vector: [Float], space: EmbeddingSpace) {
        self.vector = vector
        self.space = space
    }

    public var dimension: Int { vector.count }

    /// Cosine similarity with `other`, from -1 to 1. Two zero vectors score 1; one scores 0.
    ///
    /// - Throws: `EmbeddingError.incompatibleEmbeddings` when `other` is from another space or has
    ///   another width.
    public func cosine(_ other: Embedding) throws -> Float {
        guard space == other.space, dimension == other.dimension else {
            throw EmbeddingError.incompatibleEmbeddings(
                "\(space.id)@\(space.revision.prefix(8)) [\(dimension)] vs "
                    + "\(other.space.id)@\(other.space.revision.prefix(8)) [\(other.dimension)]")
        }
        return cosineSimilarity(vector, other.vector)
    }
}

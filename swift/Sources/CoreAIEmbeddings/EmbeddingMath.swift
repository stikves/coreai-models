// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// Vector math for embeddings, backed by vDSP.
enum EmbeddingMath {
    static func l2Norm(_ a: [Float]) -> Float {
        guard !a.isEmpty else { return 0 }
        var sumSquares: Float = 0
        vDSP_svesq(a, 1, &sumSquares, vDSP_Length(a.count))
        return sumSquares.squareRoot()
    }

    /// Unit-length copy of `a`; a zero or non-finite-norm vector is returned unchanged.
    static func l2Normalize(_ a: [Float]) -> [Float] {
        let norm = l2Norm(a)
        guard norm > 0, norm.isFinite else { return a }
        var scale = 1 / norm
        var out = [Float](repeating: 0, count: a.count)
        vDSP_vsmul(a, 1, &scale, &out, 1, vDSP_Length(a.count))
        return out
    }

    /// The first `dimensions` components of `a`, re-normalized.
    static func truncate(_ a: [Float], to dimensions: Int) -> [Float] {
        l2Normalize(Array(a.prefix(max(dimensions, 0))))
    }

    /// Cosine similarity in `[-1, 1]`; 0 when either norm is zero or the lengths differ.
    static func cosineSimilarity(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        let denominator = l2Norm(a) * l2Norm(b)
        guard denominator > 0 else { return 0 }
        var dot: Float = 0
        vDSP_dotpr(a, 1, b, 1, &dot, vDSP_Length(a.count))
        return min(max(dot / denominator, -1), 1)
    }

    static func isFinite(_ a: [Float]) -> Bool { a.allSatisfy { $0.isFinite } }
}

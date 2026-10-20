// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

@testable import CoreAIVectorIndex

/// Deterministic SplitMix64 so failures reproduce.
struct SeededGenerator: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

enum Fixtures {
    static let space = VectorSpace(id: "test/model", revision: "1")
    static let otherSpace = VectorSpace(id: "test/model", revision: "2")
    static let elementTypes: [VectorElementType] = [.float16, .float32]

    static func randomVectors(count: Int, dimension: Int, seed: UInt64) -> [[Float]] {
        var rng = SeededGenerator(seed: seed)
        return (0..<count).map { _ in (0..<dimension).map { _ in Float.random(in: -1...1, using: &rng) } }
    }

    static func makeIndex(
        _ vectors: [[Float]],
        elementType: VectorElementType,
        tileRows: Int? = nil,
        backend: TopKBackend = .bnns
    ) throws -> VectorIndex {
        let descriptor = VectorIndexDescriptor(
            space: space, dimension: vectors.first?.count ?? 8, elementType: elementType)
        var index = try VectorIndex(descriptor: descriptor)
        if let tileRows { index.tileRows = tileRows }
        index.topKBackend = backend
        try index.append(
            contentsOf: vectors.enumerated().map { VectorIndexEntry(id: "v\($0.offset)", vector: $0.element) },
            space: space)
        return index
    }
}

/// Naive scalar reference: Double accumulation over the vectors the index actually stores.
enum NaiveSearch {
    static func scores(query: [Float], stored: [[Float]]) -> [Double] {
        stored.map { row in
            var dot = 0.0
            var qq = 0.0
            var xx = 0.0
            for i in row.indices {
                let q = Double(query[i])
                let x = Double(row[i])
                dot += q * x
                qq += q * q
                xx += x * x
            }
            return dot / (qq.squareRoot() * xx.squareRoot())
        }
    }

    /// Indices of the top `k` by score descending, ties by index ascending.
    static func topK(_ scores: [Double], k: Int) -> [Int] {
        Array(scores.indices.sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }.prefix(k))
    }
}

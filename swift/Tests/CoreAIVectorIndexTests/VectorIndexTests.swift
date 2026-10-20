// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIVectorIndex

private typealias F = Fixtures

/// Asserts the index's top-k equals the naive scalar top-k up to `tolerance` (relative to max(1, |score|)):
/// every rank's score matches the reference rank's score, and every returned id's score is its true score.
private func expectMatchesNaive(
    _ index: VectorIndex,
    query: [Float],
    reference stored: [[Float]],
    k: Int,
    tolerance: Double,
    sourceLocation: SourceLocation = #_sourceLocation
) throws {
    let truth = NaiveSearch.scores(query: query, stored: stored)
    let expected = NaiveSearch.topK(truth, k: k)
    let results = try index.search(query, space: F.space, k: k)
    #expect(results.count == expected.count, sourceLocation: sourceLocation)
    for (rank, result) in results.enumerated() {
        let row = Int(result.id.dropFirst())!
        let scale = max(1, abs(truth[row]))
        #expect(abs(Double(result.score) - truth[row]) <= tolerance * scale, sourceLocation: sourceLocation)
        #expect(abs(Double(result.score) - truth[expected[rank]]) <= tolerance * scale, sourceLocation: sourceLocation)
    }
    for (a, b) in zip(results, results.dropFirst()) {
        #expect(a.score >= b.score, sourceLocation: sourceLocation)
    }
}

@Suite("VectorIndex search")
struct VectorIndexSearchTests {
    @Test(
        "matches naive scalar search on random data (small tiles, both top-k backends)",
        arguments: F.elementTypes)
    func exactnessVsNaive(elementType: VectorElementType) throws {
        let vectors = F.randomVectors(count: 517, dimension: 48, seed: 7)
        let queries = F.randomVectors(count: 6, dimension: 48, seed: 99)
        for backend in [TopKBackend.bnns, .scalar] {
            // 37-row tiles force many tiles and a partial last tile, exercising the merge.
            let index = try F.makeIndex(vectors, elementType: elementType, tileRows: 37, backend: backend)
            let stored = (0..<vectors.count).map { index.vector(id: "v\($0)")! }
            for query in queries {
                // Against the stored (normalized / rounded) vectors: only accumulation order differs.
                try expectMatchesNaive(index, query: query, reference: stored, k: 10, tolerance: 1e-4)
                // Against the caller's original vectors: fp16 storage adds rounding error.
                let tolerance = elementType == .float16 ? 5e-3 : 1e-4
                let truth = NaiveSearch.scores(query: query, stored: vectors)
                for result in try index.search(query, space: F.space, k: 10) {
                    let row = Int(result.id.dropFirst())!
                    #expect(abs(Double(result.score) - truth[row]) <= tolerance * max(1, abs(truth[row])))
                }
            }
        }
    }

    @Test("default tiling matches small tiling exactly")
    func tilingInvariance() throws {
        let vectors = F.randomVectors(count: 300, dimension: 32, seed: 3)
        let query = F.randomVectors(count: 1, dimension: 32, seed: 4)[0]
        let whole = try F.makeIndex(vectors, elementType: .float16)
        let tiled = try F.makeIndex(vectors, elementType: .float16, tileRows: 7)
        let a = try whole.search(query, space: F.space, k: 25).map(\.id)
        let b = try tiled.search(query, space: F.space, k: 25).map(\.id)
        #expect(a == b)
    }

    @Test("BNNS and scalar top-k agree exactly, including heavy ties")
    func backendsAgree() throws {
        // Small integer components produce many exactly equal scores.
        var rng = SeededGenerator(seed: 11)
        func randomVector() -> [Float] {
            var v = (0..<8).map { _ in Float(Int.random(in: -2...2, using: &rng)) }
            if v.allSatisfy({ $0 == 0 }) { v[0] = 1 }
            return v
        }
        let vectors = (0..<400).map { _ in randomVector() }
        let queries = (0..<5).map { _ in randomVector() }
        for tile in [16, 400] {
            let bnns = try F.makeIndex(vectors, elementType: .float32, tileRows: tile)
            let scalar = try F.makeIndex(vectors, elementType: .float32, tileRows: tile, backend: .scalar)
            for k in [1, 7, 50, 400, 1000] {
                let a = try bnns.search(batch: queries, space: F.space, k: k)
                let b = try scalar.search(batch: queries, space: F.space, k: k)
                #expect(a == b)
            }
        }
    }

    @Test("ties are broken by insertion order and k > count returns everything")
    func tiesAndLargeK() throws {
        let vectors: [[Float]] = [[1, 0], [2, 0], [1, 0], [2, 0], [0, 1], [1, 0]]
        let index = try F.makeIndex(vectors, elementType: .float32)
        let top3 = try index.search([1, 0], space: F.space, k: 3)
        #expect(top3.map(\.id) == ["v0", "v1", "v2"])
        let all = try index.search([1, 0], space: F.space, k: 100)
        #expect(all.map(\.id) == ["v0", "v1", "v2", "v3", "v5", "v4"])
        #expect(all.map(\.score) == [1, 1, 1, 1, 1, 0])
        #expect(try index.search([1, 0], space: F.space, k: 0).isEmpty)
        #expect(throws: VectorIndexError.invalidK(-1)) { try index.search([1, 0], space: F.space, k: -1) }
    }

    @Test("batched search equals per-query search")
    func batchEqualsSingle() throws {
        let vectors = F.randomVectors(count: 1000, dimension: 64, seed: 21)
        let queries = F.randomVectors(count: 300, dimension: 64, seed: 22)  // > 256 exercises query chunking
        for elementType in F.elementTypes {
            let index = try F.makeIndex(vectors, elementType: elementType)
            let batched = try index.search(batch: queries, space: F.space, k: 8)
            #expect(batched.count == queries.count)
            for (query, batch) in zip(queries, batched) {
                let single = try index.search(query, space: F.space, k: 8)
                #expect(single.count == batch.count)
                for (s, b) in zip(single, batch) {
                    // sgemm vs sgemv may round differently; scores agree to float precision.
                    #expect(abs(s.score - b.score) <= 1e-5)
                }
            }
        }
    }

    @Test("empty index returns no results but still validates the query")
    func emptyIndex() throws {
        let index = try VectorIndex(descriptor: VectorIndexDescriptor(space: F.space, dimension: 4))
        #expect(index.count == 0)
        #expect(try index.search([1, 2, 3, 4], space: F.space, k: 5).isEmpty)
        #expect(try index.search(batch: [[1, 2, 3, 4], [0, 0, 1, 0]], space: F.space, k: 5) == [[], []])
        #expect(try index.search(batch: [], space: F.space, k: 5).isEmpty)
        #expect(throws: VectorIndexError.self) { try index.search([1, 2, 3], space: F.space, k: 5) }
        #expect(throws: VectorIndexError.self) { try index.search([1, 2, 3, 4], space: F.otherSpace, k: 5) }
    }

    @Test("vectors are stored unit length")
    func normalizeOnInsert() throws {
        let vectors = F.randomVectors(count: 50, dimension: 16, seed: 5).map { $0.map { $0 * 7 } }
        let index = try F.makeIndex(vectors, elementType: .float32)
        for id in index.allIDs {
            let norm = VectorIndex.squaredNorm(index.vector(id: id)!).squareRoot()
            #expect(abs(norm - 1) < 1e-6)
        }
    }
}

@Suite("VectorIndex fail-closed validation")
struct VectorIndexValidationTests {
    private func makeEmpty(dimension: Int = 3) throws -> VectorIndex {
        try VectorIndex(descriptor: VectorIndexDescriptor(space: F.space, dimension: dimension))
    }

    @Test("space mismatch is refused on append and search")
    func spaceMismatch() throws {
        var index = try makeEmpty()
        let mismatch = VectorIndexError.spaceMismatch(expected: F.space, actual: F.otherSpace)
        #expect(throws: mismatch) { try index.append(id: "a", vector: [1, 0, 0], space: F.otherSpace) }
        try index.append(id: "a", vector: [1, 0, 0], space: F.space)
        #expect(throws: mismatch) { try index.search([1, 0, 0], space: F.otherSpace, k: 1) }
        #expect(index.count == 1)
    }

    @Test("dimension mismatch is refused on append and search")
    func dimensionMismatch() throws {
        var index = try makeEmpty()
        #expect(throws: VectorIndexError.dimensionMismatch(expected: 3, actual: 2)) {
            try index.append(id: "a", vector: [1, 0], space: F.space)
        }
        try index.append(id: "a", vector: [1, 0, 0], space: F.space)
        #expect(throws: VectorIndexError.dimensionMismatch(expected: 3, actual: 4)) {
            try index.search([1, 0, 0, 0], space: F.space, k: 1)
        }
        #expect(throws: VectorIndexError.dimensionMismatch(expected: 3, actual: 2)) {
            try index.search(batch: [[1, 0, 0], [1, 0]], space: F.space, k: 1)
        }
    }

    @Test("non-finite and zero vectors are refused")
    func badVectors() throws {
        var index = try makeEmpty()
        #expect(throws: VectorIndexError.invalidVector(id: "n")) {
            try index.append(id: "n", vector: [.nan, 0, 0], space: F.space)
        }
        #expect(throws: VectorIndexError.invalidVector(id: "i")) {
            try index.append(id: "i", vector: [.infinity, 0, 0], space: F.space)
        }
        #expect(throws: VectorIndexError.invalidVector(id: "z")) {
            try index.append(id: "z", vector: [0, 0, 0], space: F.space)
        }
        try index.append(id: "a", vector: [1, 0, 0], space: F.space)
        #expect(throws: VectorIndexError.invalidVector(id: nil)) { try index.search([0, 0, 0], space: F.space, k: 1) }
        #expect(throws: VectorIndexError.invalidVector(id: nil)) {
            try index.search([.nan, 0, 0], space: F.space, k: 1)
        }
        #expect(index.count == 1)
    }

    @Test("duplicate ids are refused and a failing batch append changes nothing")
    func duplicatesAndAtomicAppend() throws {
        var index = try makeEmpty()
        try index.append(id: "a", vector: [1, 0, 0], space: F.space)
        #expect(throws: VectorIndexError.duplicateID("a")) {
            try index.append(id: "a", vector: [0, 1, 0], space: F.space)
        }
        let batch = [
            VectorIndexEntry(id: "b", vector: [0, 1, 0]),
            VectorIndexEntry(id: "c", vector: [0, 0, 1]),
            VectorIndexEntry(id: "b", vector: [1, 1, 0]),
        ]
        #expect(throws: VectorIndexError.duplicateID("b")) { try index.append(contentsOf: batch, space: F.space) }
        let badDimension = [VectorIndexEntry(id: "d", vector: [0, 1, 0]), VectorIndexEntry(id: "e", vector: [1])]
        #expect(throws: VectorIndexError.self) { try index.append(contentsOf: badDimension, space: F.space) }
        #expect(index.count == 1)
        #expect(index.allIDs == ["a"])
    }

    @Test("invalid descriptors are refused")
    func invalidDescriptor() {
        #expect(throws: VectorIndexError.self) {
            try VectorIndex(descriptor: VectorIndexDescriptor(space: F.space, dimension: 0))
        }
        #expect(throws: VectorIndexError.self) {
            try VectorIndex(descriptor: VectorIndexDescriptor(space: VectorSpace(id: "", revision: "1"), dimension: 4))
        }
    }

    @Test("every error has a readable description")
    func errorDescriptions() {
        let descriptor = VectorIndexDescriptor(space: F.space, dimension: 4)
        let errors: [VectorIndexError] = [
            .invalidDescriptor("x"), .spaceMismatch(expected: F.space, actual: F.otherSpace),
            .dimensionMismatch(expected: 3, actual: 2), .descriptorMismatch(expected: descriptor, actual: descriptor),
            .invalidVector(id: "a"), .invalidVector(id: nil), .duplicateID("a"), .invalidK(-1),
            .corruptFile("x"), .ioFailure("x"),
        ]
        for error in errors { #expect(error.errorDescription?.isEmpty == false) }
    }
}

@Suite("VectorIndex filter, remove, payload")
struct VectorIndexMutationTests {
    private func makeAttributed() throws -> VectorIndex {
        var index = try VectorIndex(descriptor: VectorIndexDescriptor(space: F.space, dimension: 2))
        let kinds = ["image", "text", "image", "text", "image"]
        for i in 0..<5 {
            try index.append(
                id: "v\(i)", vector: [Float(i + 1), 1], space: F.space,
                payload: Data("img/\(i)".utf8),
                attributes: ["kind": kinds[i], "even": String(i % 2 == 0), "page": String(i / 2)])
        }
        return index
    }

    @Test("attribute filters restrict results before ranking")
    func filters() throws {
        let index = try makeAttributed()
        let query: [Float] = [1, 0]
        // Empty filter matches every row.
        #expect(try index.search(query, space: F.space, k: 10, filter: [:]).count == 5)
        #expect(
            try index.search(query, space: F.space, k: 10, filter: ["kind": "text"]).map(\.id) == ["v3", "v1"])
        let images = try index.search(query, space: F.space, k: 2, filter: ["kind": "image"])
        #expect(images.map(\.id) == ["v4", "v2"])
        // Multi-key filter: every pair must match.
        #expect(
            try index.search(query, space: F.space, k: 10, filter: ["kind": "image", "page": "1"]).map(\.id) == ["v2"])
        #expect(try index.search(query, space: F.space, k: 10, filter: ["kind": "text", "even": "true"]).isEmpty)
        // A missing key or value never matches.
        #expect(try index.search(query, space: F.space, k: 10, filter: ["kind": "audio"]).isEmpty)
        #expect(try index.search(query, space: F.space, k: 10, filter: ["missing": "x"]).isEmpty)
        // Filtering also works across tiles with the scalar backend.
        var tiled = index
        tiled.tileRows = 2
        tiled.topKBackend = .scalar
        #expect(
            try tiled.search(query, space: F.space, k: 10, filter: ["kind": "image"]).map(\.id) == ["v4", "v2", "v0"])
    }

    @Test("remove deletes by id, keeps order of the rest, and allows re-adding the id")
    func remove() throws {
        var index = try makeAttributed()
        let removed1 = index.remove(id: "v4")
        #expect(removed1)
        let removed2 = index.remove(id: "v4")
        #expect(!removed2)
        let removed3 = index.remove(ids: ["v0", "missing", "v2"])
        #expect(removed3 == 2)
        #expect(index.count == 2)
        #expect(index.allIDs == ["v1", "v3"])
        #expect(!index.allIDs.contains("v0"))
        let results = try index.search([1, 0], space: F.space, k: 10)
        #expect(results.map(\.id) == ["v3", "v1"])
        #expect(results[0].score > results[1].score)
        let stored = try #require(index.vector(id: "v3"))
        #expect(abs(stored[0] - 4 / Float(17).squareRoot()) < 1e-6)
        #expect(results[0].payload == Data("img/3".utf8))
        #expect(results[0].attributes["kind"] == "text")
        try index.append(id: "v4", vector: [10, 0], space: F.space)
        #expect(try index.search([1, 0], space: F.space, k: 1).first?.id == "v4")
        let removed4 = index.remove(ids: index.allIDs)
        #expect(removed4 == 3)
        #expect(index.count == 0)
        #expect(try index.search([1, 0], space: F.space, k: 3).isEmpty)
    }

    @Test("BNNS top-k constraints: Int32 indices, K must be clamped to the row length")
    func bnnsConstraints() {
        // Two rows of six; K == row length is the largest legal K.
        var scores: [Float] = [1, 5, 5, 3, 5, 0, 2, 2, 2, 2, 9, 2]
        var values = [Float](repeating: 0, count: 12)
        var indices = [Int32](repeating: 0, count: 12)
        let ok = VectorSearchKernel.bnnsTopK(
            scores: &scores, queryCount: 2, tileLength: 6, k: 6, bestValues: &values, bestIndices: &indices)
        #expect(ok)
        #expect(indices == [1, 2, 4, 3, 0, 5, 4, 0, 1, 2, 3, 5])
        #expect(values == [5, 5, 5, 3, 1, 0, 9, 2, 2, 2, 2, 2])
    }

    @Test("entry(id:) returns the stored vector, payload and attributes; nil once removed")
    func entryByID() throws {
        var index = try makeAttributed()
        let entry = try #require(index.entry(id: "v3"))
        #expect(entry.id == "v3")
        #expect(entry.payload == Data("img/3".utf8))
        #expect(entry.attributes == ["kind": "text", "even": "false", "page": "1"])
        #expect(index.entry(id: "missing") == nil)

        try index.append(id: "bare", vector: [0, 2], space: F.space)
        let bare = try #require(index.entry(id: "bare"))
        #expect(bare.payload == nil)
        #expect(bare.attributes.isEmpty)

        index.remove(id: "v3")
        #expect(index.entry(id: "v3") == nil)
        #expect(index.entry(id: "v4")?.payload == Data("img/4".utf8))
    }

    @Test("a copy is unaffected by later mutation of the original")
    func copyOnWrite() throws {
        var index = try makeAttributed()
        let copy = index
        try index.append(id: "extra", vector: [1, 1], space: F.space)
        index.remove(id: "v0")
        #expect(copy.count == 5)
        #expect(copy.allIDs.contains("v0"))
        #expect(!copy.allIDs.contains("extra"))
        #expect(index.count == 5)
    }
}

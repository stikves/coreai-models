// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// One vector to append, with its id, optional payload and string attributes.
///
/// Use it with `VectorIndex.append(contentsOf:space:)` to add many vectors all or nothing.
///
/// ## Usage
/// ```swift
/// let entries = [
///     VectorIndexEntry(id: "a", vector: va, attributes: ["kind": "text"]),
///     VectorIndexEntry(id: "b", vector: vb),
/// ]
/// try index.append(contentsOf: entries, space: space)
/// ```
///
/// ## See also
/// ``VectorIndex`` and ``SearchResult``.
public struct VectorIndexEntry: Sendable, Hashable {
    /// Unique id within the index.
    public var id: String
    /// The vector; normalized on insert.
    public var vector: [Float]
    /// Opaque bytes returned with search results.
    public var payload: Data?
    /// String attributes that search filters match against.
    public var attributes: [String: String]

    /// Creates an entry.
    public init(id: String, vector: [Float], payload: Data? = nil, attributes: [String: String] = [:]) {
        self.id = id
        self.vector = vector
        self.payload = payload
        self.attributes = attributes
    }
}

/// One ranked match.
///
/// Carries what was stored with the vector, so callers need no second lookup.
///
/// ## Usage
/// ```swift
/// for result in try index.search(query, space: space, k: 5) {
///     print(result.id, result.score, result.attributes, result.payload as Any)
/// }
/// ```
///
/// ## See also
/// ``VectorIndex/search(_:space:k:filter:)``.
public struct SearchResult: Sendable, Hashable {
    /// Id of the matching vector.
    public let id: String
    /// Cosine similarity against the stored vector; higher is better.
    public let score: Float
    /// Payload stored with the vector, if any.
    public let payload: Data?
    /// Attributes stored with the vector.
    public let attributes: [String: String]
}

/// An exact on-device vector index with attribute filtering and single-file persistence.
///
/// Use it to search embeddings without a server: results are exact cosine similarities (no
/// approximate recall loss), and an index saved to disk reopens instantly with its vectors
/// memory-mapped. Every write and query names the ``VectorSpace`` it belongs to, so vectors from
/// different models or model revisions never mix.
///
/// ## Usage
/// ```swift
/// let space = VectorSpace(id: "my-model", revision: "1")
/// var index = try VectorIndex(descriptor: VectorIndexDescriptor(space: space, dimension: 768, elementType: .float16))
/// try index.append(id: "photo-1", vector: v, space: space, attributes: ["kind": "image"])
/// let top = try index.search(q, space: space, k: 5, filter: ["kind": "image"])
/// try index.save(to: url)
/// let reopened = try VectorIndex.open(at: url, expecting: index.descriptor)
/// ```
///
/// ## See also
/// Works with any producer of `Float` vectors. `CoreAIEmbeddings` adds `Embedding` overloads of
/// `append` and `search`.
public struct VectorIndex: Sendable {
    /// The space, dimension and element type this index was created with.
    public let descriptor: VectorIndexDescriptor

    /// `count x dimension` vectors in `descriptor.elementType`; may be a slice of a mapped file.
    var storage: Data
    var storageIsMapped: Bool
    /// L2 norm of each stored row (after rounding to the storage type).
    var norms: [Float]
    /// `1 / norm` per row, kept in sync with `norms` by append/remove/open.
    var rowFactors: [Float] = []
    var ids: [String]
    var payloads: [Data?]
    var attributeRows: [[String: String]]
    var rowByID: [String: Int]

    /// Search tiling and top-k backend; internal so tests can force small tiles and the scalar path.
    var tileRows: Int
    var topKBackend: TopKBackend = .bnns

    /// Creates an empty index; throws `invalidDescriptor` for an empty space id or unsupported dimension.
    public init(descriptor: VectorIndexDescriptor) throws {
        guard !descriptor.space.id.isEmpty else {
            throw VectorIndexError.invalidDescriptor("space id must not be empty")
        }
        guard descriptor.dimension > 0, descriptor.dimension <= VectorIndexDescriptor.maximumDimension else {
            throw VectorIndexError.invalidDescriptor(
                "dimension \(descriptor.dimension) must be in 1...\(VectorIndexDescriptor.maximumDimension)")
        }
        self.descriptor = descriptor
        self.storage = Data()
        self.storageIsMapped = false
        self.norms = []
        self.ids = []
        self.payloads = []
        self.attributeRows = []
        self.rowByID = [:]
        self.tileRows = VectorSearchKernel.defaultTileRows(
            dimension: descriptor.dimension, elementType: descriptor.elementType)
    }

    /// Number of stored vectors.
    public var count: Int { ids.count }
    var allIDs: [String] { ids }

    /// The stored vector (after normalization and storage rounding), widened to `Float`.
    func vector(id: String) -> [Float]? {
        guard let row = rowByID[id] else { return nil }
        let d = descriptor.dimension
        return storage.withUnsafeBytes { raw in
            switch descriptor.elementType {
            case .float32:
                let base = raw.baseAddress!.assumingMemoryBound(to: Float.self) + row * d
                return Array(UnsafeBufferPointer(start: base, count: d))
            case .float16:
                let base = raw.baseAddress!.assumingMemoryBound(to: Float16.self) + row * d
                return UnsafeBufferPointer(start: base, count: d).map { Float($0) }
            }
        }
    }

    /// Everything stored for `id`: the stored vector, payload and attributes.
    func entry(id: String) -> VectorIndexEntry? {
        guard let row = rowByID[id], let vector = vector(id: id) else { return nil }
        return VectorIndexEntry(id: id, vector: vector, payload: payloads[row], attributes: attributeRows[row])
    }

    // MARK: - Mutation

    /// Appends one vector; throws on a space or dimension mismatch, a duplicate id, or an invalid vector.
    public mutating func append(
        id: String,
        vector: [Float],
        space: VectorSpace,
        payload: Data? = nil,
        attributes: [String: String] = [:]
    ) throws {
        try append(
            contentsOf: [VectorIndexEntry(id: id, vector: vector, payload: payload, attributes: attributes)],
            space: space)
    }

    /// Appends all entries or none: every entry is validated before the index changes.
    public mutating func append(contentsOf entries: [VectorIndexEntry], space: VectorSpace) throws {
        try checkSpace(space)
        let d = descriptor.dimension
        var seen = Set<String>()
        var prepared: [(values: [Float], norm: Float)] = []
        prepared.reserveCapacity(entries.count)
        for entry in entries {
            guard rowByID[entry.id] == nil, seen.insert(entry.id).inserted else {
                throw VectorIndexError.duplicateID(entry.id)
            }
            guard entry.vector.count == d else {
                throw VectorIndexError.dimensionMismatch(expected: d, actual: entry.vector.count)
            }
            prepared.append(try prepare(entry.vector, id: entry.id))
        }

        makeStorageOwned()
        storage.reserveCapacity(storage.count + entries.count * d * descriptor.elementType.byteCount)
        for (entry, item) in zip(entries, prepared) {
            switch descriptor.elementType {
            case .float32:
                item.values.withUnsafeBytes { storage.append(contentsOf: $0) }
            case .float16:
                let halves = item.values.map { Float16($0) }
                halves.withUnsafeBytes { storage.append(contentsOf: $0) }
            }
            rowByID[entry.id] = ids.count
            ids.append(entry.id)
            norms.append(item.norm)
            rowFactors.append(contentsOf: Self.rowFactors(norms: [item.norm]))
            payloads.append(entry.payload)
            attributeRows.append(entry.attributes)
        }
    }

    /// Removes the vector with `id`; returns whether it was present.
    @discardableResult
    public mutating func remove(id: String) -> Bool {
        remove(ids: [id]) == 1
    }

    /// Removes every listed id that is present (preserving the order of the rest); returns how many.
    @discardableResult
    mutating func remove<S: Sequence>(ids idsToRemove: S) -> Int where S.Element == String {
        let rows = Set(idsToRemove.compactMap { rowByID[$0] })
        guard !rows.isEmpty else { return 0 }
        makeStorageOwned()
        let rowBytes = descriptor.dimension * descriptor.elementType.byteCount
        storage.withUnsafeMutableBytes { raw in
            let base = raw.baseAddress!
            var write = 0
            for read in 0..<ids.count where !rows.contains(read) {
                if write != read {
                    memmove(base + write * rowBytes, base + read * rowBytes, rowBytes)
                }
                write += 1
            }
        }
        let kept = ids.count - rows.count
        storage.removeSubrange((kept * rowBytes)..<storage.count)
        func compact<T>(_ array: [T]) -> [T] {
            array.enumerated().filter { !rows.contains($0.offset) }.map(\.element)
        }
        ids = compact(ids)
        norms = compact(norms)
        rowFactors = Self.rowFactors(norms: norms)
        payloads = compact(payloads)
        attributeRows = compact(attributeRows)
        rowByID = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($0.element, $0.offset) })
        return rows.count
    }

    // MARK: - Search

    /// Top-`k` cosine matches for one query, sorted by score descending with ties going to the
    /// earlier-inserted row.
    ///
    /// A row matches `filter` when every `(key, value)` pair equals its attribute; an empty filter matches
    /// all rows. Returns fewer than `k` results when fewer rows match.
    public func search(
        _ query: [Float], space: VectorSpace, k: Int, filter: [String: String] = [:]
    ) throws -> [SearchResult] {
        try search(batch: [query], space: space, k: k, filter: filter)[0]
    }

    /// Top-`k` matches for each query, ordered as in ``search(_:space:k:filter:)``.
    public func search(
        batch queries: [[Float]], space: VectorSpace, k: Int, filter: [String: String] = [:]
    ) throws -> [[SearchResult]] {
        try checkSpace(space)
        guard k >= 0 else { throw VectorIndexError.invalidK(k) }
        let d = descriptor.dimension
        var flat = [Float]()
        flat.reserveCapacity(queries.count * d)
        for query in queries {
            guard query.count == d else { throw VectorIndexError.dimensionMismatch(expected: d, actual: query.count) }
            flat.append(contentsOf: try Self.unit(query, id: nil))
        }
        guard k > 0, !ids.isEmpty, !queries.isEmpty else {
            return Array(repeating: [], count: queries.count)
        }

        let eligible: [Bool]? =
            filter.isEmpty
            ? nil : attributeRows.map { row in filter.allSatisfy { row[$0.key] == $0.value } }
        if let eligible, !eligible.contains(true) {
            return Array(repeating: [], count: queries.count)
        }
        let kernel = VectorSearchKernel(
            dimension: d, elementType: descriptor.elementType, tileRows: tileRows, backend: topKBackend)

        // Bound the score matrix by scoring at most `queryChunk` queries per pass.
        let queryChunk = 256
        var results: [[SearchResult]] = []
        results.reserveCapacity(queries.count)
        var start = 0
        while start < queries.count {
            let end = min(start + queryChunk, queries.count)
            let ranked = flat.withUnsafeBufferPointer { flatBuffer in
                storage.withUnsafeBytes { raw in
                    kernel.search(
                        queries: UnsafeBufferPointer(rebasing: flatBuffer[(start * d)..<(end * d)]),
                        queryCount: end - start,
                        storage: raw, rowCount: ids.count, rowFactors: rowFactors, eligible: eligible, k: k)
                }
            }
            for list in ranked {
                results.append(list.map(makeResult))
            }
            start = end
        }
        return results
    }

    // MARK: - Helpers

    func checkSpace(_ space: VectorSpace) throws {
        guard space == descriptor.space else {
            throw VectorIndexError.spaceMismatch(expected: descriptor.space, actual: space)
        }
    }

    private func makeResult(_ scored: ScoredRow) -> SearchResult {
        SearchResult(
            id: ids[scored.row], score: scored.score, payload: payloads[scored.row],
            attributes: attributeRows[scored.row])
    }

    /// Validates, normalizes, rounds to the storage type, and measures the stored norm.
    private func prepare(_ vector: [Float], id: String) throws -> (values: [Float], norm: Float) {
        var values = try Self.unit(vector, id: id)
        if descriptor.elementType == .float16 {
            values = values.map { Float(Float16($0)) }
            guard values.allSatisfy(\.isFinite) else { throw VectorIndexError.invalidVector(id: id) }
        }
        let norm = Float(Self.squaredNorm(values).squareRoot())
        guard norm > 0 else { throw VectorIndexError.invalidVector(id: id) }
        return (values, norm)
    }

    /// `v` scaled to unit length; throws for non-finite or zero vectors.
    private static func unit(_ v: [Float], id: String?) throws -> [Float] {
        guard v.allSatisfy(\.isFinite) else { throw VectorIndexError.invalidVector(id: id) }
        let squared = squaredNorm(v)
        guard squared > 0 else { throw VectorIndexError.invalidVector(id: id) }
        let inverse = Float(1 / squared.squareRoot())
        return v.map { $0 * inverse }
    }

    static func rowFactors(norms: [Float]) -> [Float] {
        norms.map { $0 > 0 ? 1 / $0 : 0 }
    }

    static func squaredNorm(_ values: [Float]) -> Double {
        var sum = 0.0
        for value in values { sum += Double(value) * Double(value) }
        return sum
    }

    /// Copies a memory-mapped (read-only) or sliced vector block into a heap buffer before mutation.
    mutating func makeStorageOwned() {
        guard storageIsMapped || storage.startIndex != 0 else { return }
        storage = storage.withUnsafeBytes { raw in
            raw.isEmpty ? Data() : Data(bytes: raw.baseAddress!, count: raw.count)
        }
        storageIsMapped = false
    }
}

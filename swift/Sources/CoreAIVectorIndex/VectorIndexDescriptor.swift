// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Identifies the producer and revision of a set of vectors.
///
/// Vectors from different spaces are not comparable even with equal dimensions, so appends or queries
/// that name another space throw.
///
/// ## Usage
/// ```swift
/// let space = VectorSpace(id: "my-model", revision: "1")
/// ```
///
/// ## See also
/// ``VectorIndexDescriptor`` pairs a space with a dimension and element type.
public struct VectorSpace: Hashable, Codable, Sendable, CustomStringConvertible {
    /// Stable identifier of the producer, for example `"google/embeddinggemma-2"`.
    public let id: String
    /// Revision of the producer; changes whenever vectors stop being comparable with older ones.
    public let revision: String

    /// Creates a space from a producer id and revision.
    public init(id: String, revision: String) {
        self.id = id
        self.revision = revision
    }

    /// `id@revision`.
    public var description: String { "\(id)@\(revision)" }
}

/// Element type used to store vectors.
///
/// `float16` halves memory; scores are still computed in float32.
///
/// ## Usage
/// ```swift
/// let descriptor = VectorIndexDescriptor(space: space, dimension: 768, elementType: .float16)
/// ```
///
/// ## See also
/// ``VectorIndexDescriptor`` carries the element type.
public enum VectorElementType: String, Codable, Sendable {
    /// 16-bit storage.
    case float16
    /// 32-bit storage.
    case float32

    /// Size in bytes of one stored element.
    var byteCount: Int {
        switch self {
        case .float16: return 2
        case .float32: return 4
        }
    }
}

/// The space, dimension and storage type that an index, its vectors and its queries must agree on.
///
/// An index saved to disk records its descriptor, so reopening it can verify it matches.
///
/// ## Usage
/// ```swift
/// let descriptor = VectorIndexDescriptor(space: space, dimension: 768)
/// var index = try VectorIndex(descriptor: descriptor)
/// let reopened = try VectorIndex.open(at: url, expecting: descriptor)
/// ```
///
/// ## See also
/// ``VectorIndex``, ``VectorSpace`` and ``VectorElementType``.
public struct VectorIndexDescriptor: Hashable, Codable, Sendable {
    /// Space the stored vectors belong to.
    public let space: VectorSpace
    /// Number of components per vector.
    public let dimension: Int
    /// Storage element type of the vector block.
    public let elementType: VectorElementType

    /// Creates a descriptor.
    public init(space: VectorSpace, dimension: Int, elementType: VectorElementType = .float32) {
        self.space = space
        self.dimension = dimension
        self.elementType = elementType
    }

    /// Largest supported dimension; keeps tile math in `Int32`.
    static let maximumDimension = 1 << 16
}

/// Errors thrown by ``VectorIndex``.
///
/// Every compatibility check fails closed with one of these and leaves the index unchanged.
///
/// ## Usage
/// ```swift
/// do {
///     try index.append(id: "a", vector: v, space: otherSpace)
/// } catch VectorIndexError.spaceMismatch(let expected, let actual) {
///     print("index holds \(expected), got \(actual)")
/// }
/// ```
///
/// ## See also
/// ``VectorIndex`` throws these from every write, query and persistence call.
public enum VectorIndexError: Error, Equatable, Sendable, LocalizedError {
    /// The descriptor is not usable (empty space id or unsupported dimension).
    case invalidDescriptor(String)
    /// A write or query named a different space than the index.
    case spaceMismatch(expected: VectorSpace, actual: VectorSpace)
    /// A vector's length differs from the index dimension.
    case dimensionMismatch(expected: Int, actual: Int)
    /// A saved index does not match the descriptor passed to `open`.
    case descriptorMismatch(expected: VectorIndexDescriptor, actual: VectorIndexDescriptor)
    /// A vector is zero or contains a non-finite value (`id` is nil for queries).
    case invalidVector(id: String?)
    /// An id is already present in the index or repeated in one batch.
    case duplicateID(String)
    /// `k` is negative.
    case invalidK(Int)
    /// The file is not a readable index of a supported version.
    case corruptFile(String)
    /// Reading or writing the file failed.
    case ioFailure(String)

    /// One readable sentence per case.
    public var errorDescription: String? {
        switch self {
        case .invalidDescriptor(let reason):
            return "Invalid index descriptor: \(reason)."
        case .spaceMismatch(let expected, let actual):
            return "The index holds vectors from space \(expected), not \(actual)."
        case .dimensionMismatch(let expected, let actual):
            return "Expected a vector of dimension \(expected) but got \(actual)."
        case .descriptorMismatch(let expected, let actual):
            return "The saved index (\(actual.space), dimension \(actual.dimension), \(actual.elementType)) "
                + "does not match the expected (\(expected.space), dimension \(expected.dimension), "
                + "\(expected.elementType))."
        case .invalidVector(let id):
            return "Vector \(id.map { "\"\($0)\"" } ?? "query") is zero or contains a non-finite value."
        case .duplicateID(let id):
            return "The id \"\(id)\" is already in the index."
        case .invalidK(let k):
            return "k must not be negative (got \(k))."
        case .corruptFile(let reason):
            return "The index file is not readable: \(reason)."
        case .ioFailure(let reason):
            return "Index file I/O failed: \(reason)."
        }
    }
}

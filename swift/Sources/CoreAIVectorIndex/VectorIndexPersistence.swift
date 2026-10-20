// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Darwin
import Foundation

/// Single-file persistence (all integers little-endian).
///
/// | Offset                | Content                                                                         |
/// | --------------------- | ------------------------------------------------------------------------------- |
/// | 0                     | Magic `"CAIVIDX\0"` (8 bytes)                                                   |
/// | 8                     | `UInt64` header length `H`                                                      |
/// | 16                    | Header JSON (`formatVersion, descriptor, count, byteOrder, metric, normalized`) |
/// | `align(16 + H, 4096)` | Vector block: `count x dimension` elements of `elementType`                     |
/// | `align(end, 64)`      | Norm block: `count x Float32`                                                   |
/// | After norms           | Metadata JSON (`ids`, `payloads`, `attributes`) to end of file                  |
///
/// The vector block is page aligned so `open` can memory-map it; `save` replaces the file atomically.
extension VectorIndex {
    static let fileMagic = Data("CAIVIDX\0".utf8)
    static let fileFormatVersion = 1
    static let vectorAlignment = 4096
    static let normAlignment = 64
    static let prefixLength = 16

    struct FileHeader: Codable {
        var formatVersion: Int
        var descriptor: VectorIndexDescriptor
        var count: Int
        var byteOrder: String
        var metric: String
        var normalized: Bool
    }

    struct FileMetadata: Codable {
        var ids: [String]
        var payloads: [Data?]
        var attributes: [[String: String]]
    }

    static func align(_ value: Int, to alignment: Int) -> Int {
        (value + alignment - 1) / alignment * alignment
    }

    /// Atomically writes the index to `url`, replacing any existing file.
    public func save(to url: URL) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let header = try encoder.encode(
            FileHeader(
                formatVersion: Self.fileFormatVersion, descriptor: descriptor, count: count, byteOrder: "little",
                metric: "cosine", normalized: true))
        let metadata = try encoder.encode(FileMetadata(ids: ids, payloads: payloads, attributes: attributeRows))

        let vectorOffset = Self.align(Self.prefixLength + header.count, to: Self.vectorAlignment)
        let vectorEnd = vectorOffset + storage.count
        let normOffset = Self.align(vectorEnd, to: Self.normAlignment)

        let directory = url.deletingLastPathComponent()
        let temporary = directory.appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp")
        guard FileManager.default.createFile(atPath: temporary.path, contents: nil) else {
            throw VectorIndexError.ioFailure("cannot create \(temporary.path)")
        }
        do {
            let handle = try FileHandle(forWritingTo: temporary)
            defer { try? handle.close() }
            var prefix = Self.fileMagic
            withUnsafeBytes(of: UInt64(header.count).littleEndian) { prefix.append(contentsOf: $0) }
            try handle.write(contentsOf: prefix)
            try handle.write(contentsOf: header)
            try handle.write(contentsOf: Data(count: vectorOffset - Self.prefixLength - header.count))
            try storage.withUnsafeBytes { raw in
                // Chunked so a large mapped block is never duplicated in memory.
                let chunk = 64 << 20
                var start = 0
                while start < raw.count {
                    let end = min(start + chunk, raw.count)
                    try handle.write(contentsOf: UnsafeRawBufferPointer(rebasing: raw[start..<end]))
                    start = end
                }
            }
            try handle.write(contentsOf: Data(count: normOffset - vectorEnd))
            try norms.withUnsafeBytes { try handle.write(contentsOf: $0) }
            try handle.write(contentsOf: metadata)
            try handle.synchronize()
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw VectorIndexError.ioFailure("write failed: \(error)")
        }
        guard rename(temporary.path, url.path) == 0 else {
            let reason = String(cString: strerror(errno))
            try? FileManager.default.removeItem(at: temporary)
            throw VectorIndexError.ioFailure("rename to \(url.path) failed: \(reason)")
        }
    }

    /// Opens an index saved with ``save(to:)``, memory-mapping its vector block.
    ///
    /// - Parameter expected: when given, the stored descriptor must equal it exactly or the open fails
    ///   with ``VectorIndexError/descriptorMismatch(expected:actual:)``.
    public static func open(at url: URL, expecting expected: VectorIndexDescriptor? = nil) throws -> VectorIndex {
        let file: Data
        do {
            file = try Data(contentsOf: url, options: .alwaysMapped)
        } catch {
            throw VectorIndexError.ioFailure("cannot read \(url.path): \(error)")
        }
        guard file.count >= prefixLength, file.prefix(8) == fileMagic else {
            throw VectorIndexError.corruptFile("bad magic")
        }
        let headerLength = file.withUnsafeBytes { raw in
            UInt64(littleEndian: raw.loadUnaligned(fromByteOffset: 8, as: UInt64.self))
        }
        guard headerLength <= UInt64(file.count - prefixLength) else {
            throw VectorIndexError.corruptFile("header length \(headerLength) exceeds file size")
        }
        let headerEnd = prefixLength + Int(headerLength)
        let header: FileHeader
        do {
            header = try JSONDecoder().decode(FileHeader.self, from: file.subdata(in: prefixLength..<headerEnd))
        } catch {
            throw VectorIndexError.corruptFile("header: \(error)")
        }
        guard header.formatVersion == fileFormatVersion else {
            throw VectorIndexError.corruptFile("unsupported format version \(header.formatVersion)")
        }
        guard header.metric == "cosine", header.normalized else {
            throw VectorIndexError.corruptFile("unsupported metric \(header.metric) / normalized \(header.normalized)")
        }
        guard header.byteOrder == "little" else { throw VectorIndexError.corruptFile("byte order \(header.byteOrder)") }
        if let expected, expected != header.descriptor {
            throw VectorIndexError.descriptorMismatch(expected: expected, actual: header.descriptor)
        }
        var index = try VectorIndex(descriptor: header.descriptor)
        let count = header.count
        let d = header.descriptor.dimension
        let rowBytes = d * header.descriptor.elementType.byteCount
        guard count >= 0, count <= (Int.max / 2) / rowBytes else {
            throw VectorIndexError.corruptFile("count \(count)")
        }
        let vectorOffset = align(headerEnd, to: vectorAlignment)
        let vectorEnd = vectorOffset + count * rowBytes
        let normOffset = align(vectorEnd, to: normAlignment)
        let metadataOffset = normOffset + count * 4
        guard metadataOffset <= file.count else {
            throw VectorIndexError.corruptFile("file truncated: \(file.count) < \(metadataOffset) bytes")
        }

        let metadata: FileMetadata
        do {
            metadata = try JSONDecoder().decode(FileMetadata.self, from: file.subdata(in: metadataOffset..<file.count))
        } catch {
            throw VectorIndexError.corruptFile("metadata: \(error)")
        }
        guard metadata.ids.count == count, metadata.payloads.count == count, metadata.attributes.count == count
        else {
            throw VectorIndexError.corruptFile("metadata counts do not match header count \(count)")
        }
        var rowByID = [String: Int](minimumCapacity: count)
        for (row, id) in metadata.ids.enumerated() {
            guard rowByID.updateValue(row, forKey: id) == nil else {
                throw VectorIndexError.corruptFile("duplicate id")
            }
        }
        let norms = file.withUnsafeBytes { raw in
            (0..<count).map { raw.loadUnaligned(fromByteOffset: normOffset + $0 * 4, as: Float.self) }
        }
        guard norms.allSatisfy({ $0.isFinite && $0 >= 0 }) else { throw VectorIndexError.corruptFile("bad norms") }

        // A slice of the mapped file shares its mapping; nothing is copied until the first mutation.
        index.storage = file[vectorOffset..<vectorEnd]
        index.storageIsMapped = true
        index.norms = norms
        index.rowFactors = rowFactors(norms: norms)
        index.ids = metadata.ids
        index.payloads = metadata.payloads
        index.attributeRows = metadata.attributes
        index.rowByID = rowByID
        return index
    }
}

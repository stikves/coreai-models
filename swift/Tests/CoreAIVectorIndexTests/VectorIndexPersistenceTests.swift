// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIVectorIndex

private typealias F = Fixtures

@Suite("VectorIndex persistence")
struct VectorIndexPersistenceTests {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("vector-index-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    @Test(
        "save/open round trip preserves vectors, ids, payloads, attributes and results",
        arguments: F.elementTypes)
    func roundTrip(elementType: VectorElementType) throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let vectors = F.randomVectors(count: 1234, dimension: 40, seed: 31)
        var index = try F.makeIndex(vectors, elementType: elementType)
        try index.append(
            id: "with-payload", vector: vectors[0].map { -$0 }, space: F.space,
            payload: Data("a cat".utf8), attributes: ["kind": "image", "n": "3", "ok": "true"])
        let url = directory.appendingPathComponent("index.caivi")
        try index.save(to: url)

        let reopened = try VectorIndex.open(at: url, expecting: index.descriptor)
        #expect(reopened.storageIsMapped)
        #expect(reopened.descriptor == index.descriptor)
        #expect(reopened.count == index.count)
        #expect(reopened.allIDs == index.allIDs)
        #expect(reopened.norms == index.norms)
        for id in ["v0", "v617", "v1233", "with-payload"] {
            #expect(reopened.vector(id: id) == index.vector(id: id))
        }
        let queries = F.randomVectors(count: 4, dimension: 40, seed: 32) + [vectors[0].map { -$0 }]
        let before = try index.search(batch: queries, space: F.space, k: 7)
        let after = try reopened.search(batch: queries, space: F.space, k: 7)
        #expect(before == after)
        let hit = try #require(after.last?.first)
        #expect(hit.id == "with-payload")
        #expect(hit.payload == Data("a cat".utf8))
        #expect(hit.attributes == ["kind": "image", "n": "3", "ok": "true"])
        #expect(reopened.entry(id: "with-payload") == index.entry(id: "with-payload"))

        // The vector block is page aligned for mmap.
        let file = try Data(contentsOf: url)
        let headerLength = file.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 8, as: UInt64.self) }
        #expect(VectorIndex.align(16 + Int(headerLength), to: 4096) % 4096 == 0)
        // No temporary files are left behind.
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["index.caivi"])
    }

    @Test("an opened (mapped) index can be mutated and saved over its own file")
    func mutateAfterOpen() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("index.caivi")
        let vectors = F.randomVectors(count: 100, dimension: 16, seed: 41)
        try F.makeIndex(vectors, elementType: .float16).save(to: url)

        var index = try VectorIndex.open(at: url)
        let pristine = index  // value semantics: must not observe later mutations
        try index.append(id: "new", vector: vectors[5], space: F.space)
        #expect(!index.storageIsMapped)
        let removed1 = index.remove(id: "v5")
        #expect(removed1)
        try index.save(to: url)  // replaces the file `pristine` is still mapped from

        #expect(pristine.count == 100)
        #expect(pristine.vector(id: "v5") != nil)
        #expect(try pristine.search(vectors[5], space: F.space, k: 1).first?.id == "v5")
        let reopened = try VectorIndex.open(at: url)
        #expect(reopened.count == 100)
        #expect(!reopened.allIDs.contains("v5"))
        let top = try #require(try reopened.search(vectors[5], space: F.space, k: 1).first)
        #expect(top.id == "new")
        #expect(top.score > 0.999)
    }

    @Test("empty index round trips")
    func emptyRoundTrip() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("empty.caivi")
        let descriptor = VectorIndexDescriptor(space: F.space, dimension: 8, elementType: .float16)
        try VectorIndex(descriptor: descriptor).save(to: url)
        var reopened = try VectorIndex.open(at: url, expecting: descriptor)
        #expect(reopened.count == 0)
        #expect(try reopened.search([Float](repeating: 1, count: 8), space: F.space, k: 3).isEmpty)
        try reopened.append(id: "a", vector: [Float](repeating: 1, count: 8), space: F.space)
        #expect(reopened.count == 1)
    }

    @Test("open fails closed on descriptor mismatch, bad magic, truncation and unknown versions")
    func failClosed() throws {
        let directory = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("index.caivi")
        let index = try F.makeIndex(
            F.randomVectors(count: 10, dimension: 8, seed: 51), elementType: .float32)
        try index.save(to: url)

        let other = VectorIndexDescriptor(space: F.otherSpace, dimension: 8, elementType: .float32)
        #expect(throws: VectorIndexError.descriptorMismatch(expected: other, actual: index.descriptor)) {
            try VectorIndex.open(at: url, expecting: other)
        }
        let file = try Data(contentsOf: url)

        let badMagic = directory.appendingPathComponent("magic.caivi")
        var corrupted = file
        corrupted[0] = 0x58
        try corrupted.write(to: badMagic)
        #expect(throws: VectorIndexError.corruptFile("bad magic")) { try VectorIndex.open(at: badMagic) }

        let truncated = directory.appendingPathComponent("truncated.caivi")
        try file.prefix(4096 + 100).write(to: truncated)
        #expect(throws: VectorIndexError.self) { try VectorIndex.open(at: truncated) }

        let future = directory.appendingPathComponent("future.caivi")
        let versionRange = try #require(file.range(of: Data("\"formatVersion\":1".utf8)))
        var bumped = file
        bumped[versionRange.upperBound - 1] = UInt8(ascii: "9")
        try bumped.write(to: future)
        #expect(throws: VectorIndexError.corruptFile("unsupported format version 9")) {
            try VectorIndex.open(at: future)
        }

        let otherMetric = directory.appendingPathComponent("metric.caivi")
        let metricRange = try #require(file.range(of: Data("\"metric\":\"cosine\"".utf8)))
        var changed = file
        changed[metricRange.upperBound - 2] = UInt8(ascii: "X")
        try changed.write(to: otherMetric)
        #expect(throws: VectorIndexError.self) { try VectorIndex.open(at: otherMetric) }

        #expect(throws: VectorIndexError.self) {
            try VectorIndex.open(at: directory.appendingPathComponent("missing.caivi"))
        }
    }
}

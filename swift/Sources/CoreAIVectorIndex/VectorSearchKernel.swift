// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

/// How per-tile top-k is selected. Both produce identical results (score descending, row ascending).
enum TopKBackend: Sendable {
    // BNNS needs Int32 indices and K <= row length; K is clamped per tile.
    case bnns
    /// Scalar bounded insertion; used if BNNS reports failure and as the test oracle for the BNNS path.
    case scalar
}

struct ScoredRow: Sendable {
    var row: Int
    var score: Float

    /// Total order used everywhere: higher score first, then lower (earlier-inserted) row first.
    @inline(__always)
    static func ranksBefore(_ a: ScoredRow, _ b: ScoredRow) -> Bool {
        a.score != b.score ? a.score > b.score : a.row < b.row
    }
}

/// Tiled exact cosine search: `cblas_sgemm` scores per tile, top-k per tile, then a merge.
struct VectorSearchKernel {
    var dimension: Int
    var elementType: VectorElementType
    var tileRows: Int
    var backend: TopKBackend

    // Tile sizes keep the score matrix and fp16 scratch cache-resident.
    static func defaultTileRows(dimension: Int, elementType: VectorElementType) -> Int {
        switch elementType {
        case .float32: return 16384
        case .float16: return max(64, min(16384, (8 << 20) / (dimension * 4)))
        }
    }

    /// - Parameters:
    ///   - queries: `queryCount x dimension` row-major fp32; the caller passes unit queries.
    ///   - storage: `rowCount x dimension` row-major vectors in `elementType`.
    ///   - rowFactors: per stored row, `1 / norm`.
    ///   - eligible: optional per-row mask; ineligible rows are never returned.
    func search(
        queries: UnsafeBufferPointer<Float>,
        queryCount: Int,
        storage: UnsafeRawBufferPointer,
        rowCount: Int,
        rowFactors: [Float],
        eligible: [Bool]?,
        k: Int
    ) -> [[ScoredRow]] {
        guard queryCount > 0, rowCount > 0, k > 0, let storageBase = storage.baseAddress,
            let queryBase = queries.baseAddress
        else {
            return Array(repeating: [], count: queryCount)
        }
        let d = dimension
        let tile = min(tileRows, rowCount)
        let kTileMax = min(k, tile)
        var candidates = [[ScoredRow]](repeating: [], count: queryCount)
        var scores = [Float](repeating: 0, count: queryCount * tile)
        var converted = [Float](repeating: 0, count: elementType == .float16 ? tile * d : 0)
        var bestValues = [Float](repeating: 0, count: queryCount * kTileMax)
        var bestIndices = [Int32](repeating: 0, count: queryCount * kTileMax)

        var r0 = 0
        while r0 < rowCount {
            let t = min(tile, rowCount - r0)
            converted.withUnsafeMutableBufferPointer { convertedBuffer in
                let rows: UnsafePointer<Float>
                switch elementType {
                case .float32:
                    rows = (storageBase + r0 * d * 4).assumingMemoryBound(to: Float.self)
                case .float16:
                    var src = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: storageBase + r0 * d * 2), height: 1,
                        width: vImagePixelCount(t * d), rowBytes: t * d * 2)
                    var dst = vImage_Buffer(
                        data: convertedBuffer.baseAddress!, height: 1, width: vImagePixelCount(t * d),
                        rowBytes: t * d * 4)
                    let error = vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
                    precondition(error == kvImageNoError, "vImageConvert_Planar16FtoPlanarF failed: \(error)")
                    rows = UnsafePointer(convertedBuffer.baseAddress!)
                }
                scores.withUnsafeMutableBufferPointer { scoreBuffer in
                    let c = scoreBuffer.baseAddress!
                    // C[q, t] = Q[q, :] . X[t, :]  (row-major, X transposed).
                    cblas_sgemm(
                        CblasRowMajor, CblasNoTrans, CblasTrans,
                        Int32(queryCount), Int32(t), Int32(d),
                        1.0, queryBase, Int32(d),
                        rows, Int32(d),
                        0.0, c, Int32(t))
                    finishScores(
                        c, queryCount: queryCount, tileLength: t, rowOffset: r0,
                        rowFactors: rowFactors, eligible: eligible)
                }
            }
            let kTile = min(k, t)
            selectTopK(
                scores: &scores, queryCount: queryCount, tileLength: t, k: kTile, rowOffset: r0,
                bestValues: &bestValues, bestIndices: &bestIndices, into: &candidates)
            r0 += t
        }

        return candidates.map { list in
            if list.count <= 1 { return list }
            let sorted = list.sorted(by: ScoredRow.ranksBefore)
            return Array(sorted.prefix(k))
        }
    }

    /// Turns raw dot products into cosines and masks ineligible rows.
    private func finishScores(
        _ c: UnsafeMutablePointer<Float>,
        queryCount: Int,
        tileLength t: Int,
        rowOffset r0: Int,
        rowFactors: [Float],
        eligible: [Bool]?
    ) {
        let n = vDSP_Length(t)
        // Queries are unit length; divide by each stored row's norm for an exact cosine.
        rowFactors.withUnsafeBufferPointer { inv in
            for q in 0..<queryCount {
                let row = c + q * t
                vDSP_vmul(row, 1, inv.baseAddress! + r0, 1, row, 1, n)
            }
        }
        if let eligible {
            for j in 0..<t where !eligible[r0 + j] {
                for q in 0..<queryCount {
                    c[q * t + j] = -Float.infinity
                }
            }
        }
    }

    private func selectTopK(
        scores: inout [Float],
        queryCount: Int,
        tileLength t: Int,
        k: Int,
        rowOffset r0: Int,
        bestValues: inout [Float],
        bestIndices: inout [Int32],
        into candidates: inout [[ScoredRow]]
    ) {
        var usedBNNS = false
        if backend == .bnns {
            usedBNNS = Self.bnnsTopK(
                scores: &scores, queryCount: queryCount, tileLength: t, k: k,
                bestValues: &bestValues, bestIndices: &bestIndices)
        }
        for q in 0..<queryCount {
            if usedBNNS {
                for i in 0..<k {
                    let value = bestValues[q * k + i]
                    guard value != -Float.infinity else { continue }
                    candidates[q].append(ScoredRow(row: r0 + Int(bestIndices[q * k + i]), score: value))
                }
            } else {
                scores.withUnsafeBufferPointer { buffer in
                    let row = UnsafeBufferPointer(rebasing: buffer[(q * t)..<(q * t + t)])
                    for entry in Self.scalarTopK(row, k: k) {
                        candidates[q].append(ScoredRow(row: r0 + entry.row, score: entry.score))
                    }
                }
            }
        }
    }

    /// Returns `false` when BNNS rejects the call so the caller can fall back to the scalar path.
    static func bnnsTopK(
        scores: inout [Float],
        queryCount: Int,
        tileLength t: Int,
        k: Int,
        bestValues: inout [Float],
        bestIndices: inout [Int32]
    ) -> Bool {
        precondition(k <= t, "BNNSDirectApplyTopK reads past the row when K > row length")
        let status = scores.withUnsafeMutableBufferPointer { inputBuffer in
            bestValues.withUnsafeMutableBufferPointer { valueBuffer in
                bestIndices.withUnsafeMutableBufferPointer { indexBuffer -> Int32 in
                    guard
                        var inputDescriptor = BNNSNDArrayDescriptor(
                            data: UnsafeMutableBufferPointer(rebasing: inputBuffer[0..<t]), shape: .vector(t)),
                        var valueDescriptor = BNNSNDArrayDescriptor(
                            data: UnsafeMutableBufferPointer(rebasing: valueBuffer[0..<k]), shape: .vector(k)),
                        var indexDescriptor = BNNSNDArrayDescriptor(
                            data: UnsafeMutableBufferPointer(rebasing: indexBuffer[0..<k]), shape: .vector(k))
                    else { return -1 }
                    return BNNSDirectApplyTopK(
                        k, 0, queryCount, &inputDescriptor, t, &valueDescriptor, k, &indexDescriptor, k, nil)
                }
            }
        }
        return status == 0
    }

    /// Exact top-k of one row by bounded binary insertion (score descending, index ascending).
    /// `-infinity` entries (masked rows) are skipped.
    static func scalarTopK(_ row: UnsafeBufferPointer<Float>, k: Int) -> [ScoredRow] {
        var best: [ScoredRow] = []
        best.reserveCapacity(k + 1)
        for (j, score) in row.enumerated() where score != -Float.infinity {
            let candidate = ScoredRow(row: j, score: score)
            if best.count == k, let last = best.last, !ScoredRow.ranksBefore(candidate, last) { continue }
            var lo = 0
            var hi = best.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if ScoredRow.ranksBefore(best[mid], candidate) { lo = mid + 1 } else { hi = mid }
            }
            best.insert(candidate, at: lo)
            if best.count > k { best.removeLast() }
        }
        return best
    }
}

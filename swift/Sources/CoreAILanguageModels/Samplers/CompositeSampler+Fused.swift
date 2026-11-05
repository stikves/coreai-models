// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import Foundation

// Fused CPU passes behind `CompositeSampler`. They return the reference path's exact result (same
// filters, CDF order, arithmetic and RNG draws) without its full-vocab conversion, division, sort
// or heap, and without per-call allocation. Rows are read in L1-sized chunks; fp16 is widened chunk
// by chunk. When a pass cannot settle the answer (NaN, +inf, a topP cutoff outside the candidate
// band) it returns nil and the caller runs the reference path.

extension CompositeSampler {
    /// One logit row, read as `Float` in chunks.
    struct Row {
        static let chunkSize = 4096

        let base: UnsafeRawPointer?
        let count: Int
        let isHalf: Bool
        let chunk: UnsafeMutablePointer<Float>

        init(_ bytes: UnsafeRawBufferPointer, isHalf: Bool, chunk: UnsafeMutablePointer<Float>) {
            base = bytes.baseAddress
            count = bytes.count / (isHalf ? 2 : 4)
            self.isHalf = isHalf
            self.chunk = chunk
        }

        /// Entries `[offset, offset + n)`, `n <= chunkSize`. fp16 is widened into `chunk`, valid until the next call.
        @inline(__always)
        func floats(_ offset: Int, _ n: Int) -> UnsafePointer<Float> {
            guard isHalf else { return base!.assumingMemoryBound(to: Float.self) + offset }
            widen(base! + 2 * offset, into: chunk, n)
            return UnsafePointer(chunk)
        }

        func value(_ i: Int) -> Float {
            guard isHalf else { return base!.load(fromByteOffset: 4 * i, as: Float.self) }
            var v: Float = 0
            widen(base! + 2 * i, into: &v, 1)
            return v
        }

        /// The whole row widened into `out`.
        func copy(into out: UnsafeMutablePointer<Float>) {
            guard count > 0 else { return }
            if isHalf {
                widen(base!, into: out, count)
            } else {
                out.update(from: base!.assumingMemoryBound(to: Float.self), count: count)
            }
        }

        func array() -> [Float] {
            [Float](unsafeUninitializedCapacity: count) { buffer, initialized in
                if let out = buffer.baseAddress { copy(into: out) }
                initialized = count
            }
        }

        /// Largest entry, or NaN. `vDSP_maxv` can miss a NaN, so callers still detect NaN on their own scan.
        func max() -> Float {
            guard isHalf else {
                var m: Float = 0
                vDSP_maxv(base!.assumingMemoryBound(to: Float.self), 1, &m, vDSP_Length(count))
                return m
            }
            // fp16 on raw bits: sign-magnitude to a monotone signed key (key(-0) < key(+0)).
            var top = Int16.min
            var largestMagnitude: UInt16 = 0
            base!.withMemoryRebound(to: UInt16.self, capacity: count) { h in
                for j in 0..<count {
                    let s = Int16(bitPattern: h[j])
                    top = Swift.max(top, s ^ ((s &>> 15) & 0x7FFF))
                    largestMagnitude = Swift.max(largestMagnitude, h[j] & 0x7FFF)
                }
            }
            if largestMagnitude > 0x7C00 { return .nan }
            var bits = UInt16(truncatingIfNeeded: top ^ ((top &>> 15) & 0x7FFF))
            var m: Float = 0
            widen(&bits, into: &m, 1)
            return m
        }
    }
}

@inline(__always)
private func widen(_ source: UnsafeRawPointer, into destination: UnsafeMutablePointer<Float>, _ n: Int) {
    var src = vImage_Buffer(
        data: UnsafeMutableRawPointer(mutating: source), height: 1, width: vImagePixelCount(n), rowBytes: 2 * n)
    var dst = vImage_Buffer(data: destination, height: 1, width: vImagePixelCount(n), rowBytes: 4 * n)
    _ = vImageConvert_Planar16FtoPlanarF(&src, &dst, 0)
}

extension CompositeSampler {
    /// A kept-set candidate: `z` is the raw logit until converted to `logit / temperature`.
    struct Candidate {
        var z: Float
        var i: Int32

        /// Selection order: higher z first, equal z by lower index.
        @inline(__always)
        static func better(_ a: Self, _ b: Self) -> Bool {
            a.z > b.z || (a.z == b.z && a.i < b.i)
        }
    }

    /// Per-thread buffers reused across calls. `CompositeSampler`'s API is static and callable from
    /// any thread, so nothing here is shared. Buffers grow to the largest row seen and live as long
    /// as the thread.
    final class Scratch {
        let chunk = UnsafeMutablePointer<Float>.allocate(capacity: Row.chunkSize)
        let offsets = UnsafeMutablePointer<Int>.allocate(capacity: 256)
        var row = Buffer<Float>()
        var floats = Buffer<Float>()
        var candidates = Buffer<Candidate>()
        var spare = Buffer<Candidate>()

        struct Buffer<T> {
            private(set) var base: UnsafeMutablePointer<T>?
            private(set) var capacity = 0

            /// At least `n` elements; contents are not preserved.
            mutating func reserve(_ n: Int) -> UnsafeMutablePointer<T> {
                if n > capacity {
                    base?.deallocate()
                    capacity = Swift.max(n, 2 * capacity, 64)
                    base = .allocate(capacity: capacity)
                }
                return base!
            }

            /// Doubles the capacity, keeping the first `count` elements.
            mutating func grow(keeping count: Int) -> UnsafeMutablePointer<T> {
                let bigger = UnsafeMutablePointer<T>.allocate(capacity: 2 * capacity)
                if let base {
                    bigger.moveInitialize(from: base, count: count)
                    base.deallocate()
                }
                base = bigger
                capacity *= 2
                return bigger
            }

            func free() { base?.deallocate() }
        }

        deinit {
            chunk.deallocate()
            offsets.deallocate()
            row.free()
            floats.free()
            candidates.free()
            spare.free()
        }

        static var current: Scratch {
            if let p = pthread_getspecific(scratchKey) {
                return Unmanaged<CompositeSampler.Scratch>.fromOpaque(p).takeUnretainedValue()
            }
            let scratch = Scratch()
            pthread_setspecific(scratchKey, Unmanaged.passRetained(scratch).toOpaque())
            return scratch
        }
    }
}

private let scratchKey: pthread_key_t = {
    var key = pthread_key_t()
    pthread_key_create(&key) { Unmanaged<CompositeSampler.Scratch>.fromOpaque($0).release() }
    return key
}()

extension CompositeSampler {
    // MARK: - Temperature only

    /// Softmax over the whole row and an inverse-CDF scan in vocab order: `softmaxVectorized` +
    /// `multinomialSample` on `x / T`, fused per chunk. `vDSP_sve` still runs over the whole row so the
    /// sum rounds the same. Nil when the sum is NaN (a NaN, or +inf, in the row).
    static func sampleTemperature(
        _ row: Row,
        temperature: Float,
        scratch: Scratch,
        using rng: inout some RandomNumberGenerator
    ) -> Int32? {
        let n = row.count
        let e = scratch.row.reserve(n)
        // max(x / T) == max(x) / T: division by T > 0 is monotone.
        var t = temperature
        var maxX = row.max()
        var maxZ: Float = 0
        vDSP_vsdiv(&maxX, 1, &t, &maxZ, 1, 1)
        var negMax = -maxZ
        var offset = 0
        while offset < n {
            let c = min(Row.chunkSize, n - offset)
            vDSP_vsdiv(row.floats(offset, c), 1, &t, e + offset, 1, vDSP_Length(c))
            vDSP_vsadd(e + offset, 1, &negMax, e + offset, 1, vDSP_Length(c))
            var c32 = Int32(c)
            vvexpf(e + offset, e + offset, &c32)
            offset += c
        }
        var sumExp: Float = 0
        vDSP_sve(e, 1, &sumExp, vDSP_Length(n))
        guard !sumExp.isNaN else { return nil }
        let invSum = 1.0 / max(sumExp, .leastNormalMagnitude)

        let r = Float.random(in: 0..<1, using: &rng)
        var cum: Float = 0
        for i in 0..<n {
            cum += e[i] * invSum
            if r < cum { return Int32(i) }
        }
        return Int32(n - 1)
    }

    // MARK: - topK / topP / minP

    /// The active set of `selectActiveIndices`, sorted by (z descending, index ascending), in
    /// `scratch.candidates`. Returns its length, or nil when the reference path must decide.
    static func selectActive(
        _ row: Row,
        temperature: Float,
        topK: Int?,
        topP: Float?,
        minP: Float?,
        scratch: Scratch
    ) -> Int? {
        if let k = topK {
            guard k > 0, let n = topKWindow(row, k: min(k, row.count), temperature: temperature, scratch: scratch)
            else { return nil }
            return keptPrefix(n, topP: topP, minP: minP, scratch: scratch)
        }

        var t = temperature
        var maxX = row.max()
        var maxZ: Float = 0
        vDSP_vsdiv(&maxX, 1, &t, &maxZ, 1, 1)
        guard !maxZ.isNaN, maxZ < .infinity else { return nil }
        guard maxZ > -.infinity else { return 0 }

        if let minP {
            // Every entry the minP filter keeps.
            guard
                let n = collect(
                    row, floorZ: min(maxZ + logf(minP), maxZ), temperature: t, sorted: true, scratch: scratch)
            else { return nil }
            return keptPrefix(n.count, topP: topP, minP: minP, scratch: scratch)
        }

        // topP alone: the reference sums exp(z - max) over the whole sorted vocab. Entries more than
        // 17 below the max add exp < 2^-24, under half an ulp of a sum that is >= 1 from its first
        // term on, so they cannot change it; the band above them settles the sum and the cutoff.
        guard let band = collect(row, floorZ: maxZ - 17, temperature: t, sorted: false, scratch: scratch)
        else { return nil }
        var lowX = band.lowX
        var zOut: Float = 0
        vDSP_vsdiv(&lowX, 1, &t, &zOut, 1, 1)
        guard expf(zOut - maxZ) < 0x1p-24 / 1.25 else { return nil }
        return topPBand(band.count, maxZ: maxZ, topP: topP!, zOut: zOut, scratch: scratch)
    }

    /// The topP cutoff over the band `c[0..<n]` (index order; every entry with z > `zOut`), as the
    /// reference computes it over the whole sorted vocab. Only a head needs sorting: once the
    /// running sum's binade no longer changes, each addition rounds to a multiple of that ulp on its
    /// own, so the order of the tail does not matter. A tail that crosses a binade, or lands exactly
    /// halfway between multiples, retries with a deeper head. Nil when the cutoff falls outside the band.
    private static func topPBand(
        _ n: Int, maxZ: Float, topP: Float, zOut: Float, scratch: Scratch
    ) -> Int? {
        let c = scratch.candidates.base!
        let e = scratch.floats.reserve(n)
        let spare = scratch.spare.reserve(n)
        for depth: Float in [8, 12, .infinity] {
            // Stable partition, head (z >= maxZ - depth) first. Equal z stay together, in index order.
            let floorZ = maxZ - depth
            var head = 0
            var tail = 0
            for q in 0..<n {
                if c[q].z >= floorZ {
                    c[head] = c[q]
                    head += 1
                } else {
                    spare[tail] = c[q]
                    tail += 1
                }
            }
            (c + head).update(from: spare, count: tail)
            sortCollected(c, head, scratch: scratch)

            var sumExp: Float = 0
            for q in 0..<head {
                e[q] = expf(c[q].z - maxZ)
                sumExp += e[q]
            }
            if tail > 0 {
                let ulp = sumExp.ulp
                let perUlp = 1 / ulp
                var steps = 0
                var tie = false
                for q in head..<n {
                    let units = expf(c[q].z - maxZ) * perUlp
                    if units - units.rounded(.down) == 0.5 {
                        tie = true
                        break
                    }
                    steps += Int(units.rounded())
                }
                let total = sumExp + Float(steps) * ulp
                if tie || total.exponent != sumExp.exponent { continue }
                sumExp = total
            }

            let invSumExp = 1.0 / max(sumExp, .leastNormalMagnitude)
            var cumProb: Float = 0
            for q in 0..<head {
                cumProb += e[q] * invSumExp
                if cumProb >= topP { return c[q].z > zOut ? q + 1 : nil }
            }
            if tail == 0 { return nil }
        }
        return nil
    }

    /// minP then topP over the first `n` candidates, sorted: `selectActiveIndices` after its sort.
    private static func keptPrefix(_ n: Int, topP: Float?, minP: Float?, scratch: Scratch) -> Int {
        let c = scratch.candidates.base!
        guard n > 0, c[0].z > -.infinity else { return 0 }
        let maxLogit = c[0].z
        var n = n
        if let m = minP {
            let threshold = maxLogit + logf(m)
            var kept = 0
            for q in 0..<n where c[q].z >= threshold {
                c[kept] = c[q]
                kept += 1
            }
            if kept == 0 { return 1 }
            n = kept
        }
        guard let p = topP else { return n }

        let e = scratch.floats.reserve(n)
        var sumExp: Float = 0
        for q in 0..<n {
            e[q] = expf(c[q].z - maxLogit)
            sumExp += e[q]
        }
        let invSumExp = 1.0 / max(sumExp, .leastNormalMagnitude)
        var cumProb: Float = 0
        for q in 0..<n {
            cumProb += e[q] * invSumExp
            if cumProb >= p { return q + 1 }
        }
        return n
    }

    /// The `selectActiveIndices` topK window: the k best z by (z descending, index ascending),
    /// sorted. One streaming pass over raw logits keeps candidates above the running k-th best,
    /// pruned with a quickselect; only candidates are divided by T. Nil when the row holds a NaN.
    private static func topKWindow(
        _ row: Row, k: Int, temperature: Float, scratch: Scratch
    ) -> Int? {
        let n = row.count
        let capacity = min(max(8 * k, 1024), n)
        let c = scratch.candidates.reserve(capacity)
        var count = 0
        var converted = 0  // c[0..<converted] hold z, the rest raw logits
        // Raw logit of the k-th best. An entry at or below it has z at or below the k-th best z
        // (division by T is monotone), and loses a z tie on index, so it cannot enter the window.
        var threshold = -Float.infinity

        func prune() {
            toZ(c + converted, count - converted, temperature: temperature, scratch: scratch)
            nth(c, count, k - 1)
            count = k
            converted = k
            threshold = row.value(Int(c[k - 1].i))
        }

        var offset = 0
        while offset < n {
            let length = min(Row.chunkSize, n - offset)
            let p = row.floats(offset, length)
            var tv = SIMD16<Float>(repeating: threshold)
            var j = 0
            while j < length {
                let end = min(j + 16, length)
                if end - j == 16 {
                    // `!(v <= t)` also flags NaN.
                    let v = UnsafeRawPointer(p + j).loadUnaligned(as: SIMD16<Float>.self)
                    if !any(.!(v .<= tv)) {
                        j = end
                        continue
                    }
                }
                for q in j..<end where !(p[q] <= threshold) {
                    if p[q].isNaN { return nil }
                    c[count] = Candidate(z: p[q], i: Int32(offset + q))
                    count += 1
                    if count == capacity {
                        prune()
                        tv = SIMD16(repeating: threshold)
                    }
                }
                j = end
            }
            offset += length
        }
        if count > k {
            prune()
        } else {
            toZ(c + converted, count - converted, temperature: temperature, scratch: scratch)
        }

        if count < k {
            // Fewer than k entries above -inf: the window also takes the lowest-index -inf entries.
            for q in 0..<count where c[q].z == -.infinity { return nil }
            offset = 0
            while count < k {
                let length = min(Row.chunkSize, n - offset)
                let p = row.floats(offset, length)
                for q in 0..<length where count < k && p[q] == -.infinity {
                    c[count] = Candidate(z: -.infinity, i: Int32(offset + q))
                    count += 1
                }
                offset += length
            }
        }
        sortCandidates(c, count)
        return count
    }

    /// Every entry with z >= `floorZ`, plus a few just below, in index order or `sorted`. `lowX` is
    /// the raw-logit cut: entries below it are left out and have z <= lowX / T. Nil when the row
    /// holds a NaN.
    private static func collect(
        _ row: Row, floorZ: Float, temperature: Float, sorted: Bool, scratch: Scratch
    ) -> (count: Int, lowX: Float)? {
        // Raw cut with a margin that covers any rounding of the division by T.
        let tx = Double(floorZ) * Double(temperature)
        let lowX = floorZ == -.infinity ? -.infinity : Float(tx - abs(tx) * 0x1p-16 - 0x1p-100)
        let n = row.count
        var c = scratch.candidates.reserve(min(n, 4096))
        var capacity = scratch.candidates.capacity
        var count = 0
        let lv = SIMD16<Float>(repeating: lowX)
        var offset = 0
        while offset < n {
            let length = min(Row.chunkSize, n - offset)
            let p = row.floats(offset, length)
            var j = 0
            while j < length {
                let end = min(j + 16, length)
                if end - j == 16 {
                    // `!(v < low)` also flags NaN.
                    let v = UnsafeRawPointer(p + j).loadUnaligned(as: SIMD16<Float>.self)
                    if !any(.!(v .< lv)) {
                        j = end
                        continue
                    }
                }
                if capacity - count < 16 {
                    c = scratch.candidates.grow(keeping: count)
                    capacity = scratch.candidates.capacity
                }
                for q in j..<end {
                    let x = p[q]
                    c[count] = Candidate(z: x, i: Int32(offset + q))
                    count += x < lowX ? 0 : 1
                }
                j = end
            }
            offset += length
        }
        for q in 0..<count where c[q].z.isNaN { return nil }
        toZ(c, count, temperature: temperature, scratch: scratch)
        if sorted { sortCollected(c, count, scratch: scratch) }
        return (count, lowX)
    }

    /// Sorts candidates whose equal-z entries are already in index order.
    private static func sortCollected(
        _ c: UnsafeMutablePointer<Candidate>, _ n: Int, scratch: Scratch
    ) {
        if n <= 64 {
            sortCandidates(c, n)
        } else {
            radixSortDescending(c, n, spare: scratch.spare.reserve(n), offsets: scratch.offsets)
        }
    }

    /// Raw logits to z with the reference's `vDSP_vsdiv`, which is elementwise.
    private static func toZ(
        _ c: UnsafeMutablePointer<Candidate>, _ n: Int, temperature: Float, scratch: Scratch
    ) {
        guard n > 0 else { return }
        let z = scratch.floats.reserve(n)
        for q in 0..<n { z[q] = c[q].z }
        var t = temperature
        vDSP_vsdiv(z, 1, &t, z, 1, vDSP_Length(n))
        for q in 0..<n { c[q].z = z[q] }
    }

    private static func sortCandidates(_ c: UnsafeMutablePointer<Candidate>, _ n: Int) {
        var buffer = UnsafeMutableBufferPointer(start: c, count: n)
        buffer.sort(by: Candidate.better)
    }

    /// Quickselect: afterwards c[k] is the k-th best (0-based) and c[0..<k] are better.
    private static func nth(_ c: UnsafeMutablePointer<Candidate>, _ n: Int, _ k: Int) {
        var lo = 0
        var hi = n - 1
        while lo < hi {
            let pivot = c[(lo + hi) >> 1]
            var i = lo
            var j = hi
            while i <= j {
                while Candidate.better(c[i], pivot) { i += 1 }
                while Candidate.better(pivot, c[j]) { j -= 1 }
                if i <= j {
                    (c[i], c[j]) = (c[j], c[i])
                    i += 1
                    j -= 1
                }
            }
            if k <= j {
                hi = j
            } else if k >= i {
                lo = i
            } else {
                return
            }
        }
    }

    /// Stable LSD radix sort by z descending. Input in ascending index order keeps equal z by index.
    private static func radixSortDescending(
        _ c: UnsafeMutablePointer<Candidate>, _ n: Int, spare: UnsafeMutablePointer<Candidate>,
        offsets: UnsafeMutablePointer<Int>
    ) {
        @inline(__always) func key(_ z: Float) -> UInt32 {
            let bits = (z + 0).bitPattern  // -0 sorts with +0
            return bits & 0x8000_0000 != 0 ? bits : ~bits & 0x7FFF_FFFF
        }
        var src = c
        var dst = spare
        for shift in stride(from: 0 as UInt32, to: 32, by: 8) {
            offsets.initialize(repeating: 0, count: 256)
            for q in 0..<n { offsets[Int(key(src[q].z) >> shift & 0xFF)] += 1 }
            if offsets[Int(key(src[0].z) >> shift & 0xFF)] == n { continue }
            var sum = 0
            for d in 0..<256 {
                let count = offsets[d]
                offsets[d] = sum
                sum += count
            }
            for q in 0..<n {
                let d = Int(key(src[q].z) >> shift & 0xFF)
                dst[offsets[d]] = src[q]
                offsets[d] += 1
            }
            swap(&src, &dst)
        }
        if src != c { c.update(from: src, count: n) }
    }
}

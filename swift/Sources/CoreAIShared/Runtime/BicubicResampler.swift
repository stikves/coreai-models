// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// Antialiased bicubic resize that matches torch's uint8 bicubic antialias resize bit-exactly (see BicubicResamplerTests).
package enum BicubicResampler {
    /// Quantized taps for one axis.
    struct Taps {
        var start: [Int]
        var count: [Int]
        /// `out * stride` int weights, zero padded.
        var weights: [Int32]
        var stride: Int
        var precision: Int
    }

    @inline(__always) static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1.0 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2.0 { return ((a * x - 5 * a) * x + 8 * a) * x - 4 * a }
        return 0.0
    }

    static func taps(inputSize: Int, outputSize: Int) -> Taps {
        let scale = Double(inputSize) / Double(outputSize)
        let support = scale >= 1.0 ? 2.0 * scale : 2.0
        let invScale = scale >= 1.0 ? 1.0 / scale : 1.0
        let stride = Int(support.rounded(.up)) * 2 + 1
        var start = [Int](repeating: 0, count: outputSize)
        var count = [Int](repeating: 0, count: outputSize)
        var real = [Double](repeating: 0, count: outputSize * stride)
        var maxWeight = 0.0
        for i in 0..<outputSize {
            let center = scale * (Double(i) + 0.5)
            let xmin = max(Int(center - support + 0.5), 0)
            let xsize = min(Int(center + support + 0.5), inputSize) - xmin
            start[i] = xmin
            count[i] = xsize
            var total = 0.0
            for j in 0..<xsize {
                let w = cubic((Double(j + xmin) - center + 0.5) * invScale)
                real[i * stride + j] = w
                total += w
            }
            if total != 0.0 {
                for j in 0..<xsize {
                    real[i * stride + j] /= total
                    maxWeight = max(maxWeight, real[i * stride + j])
                }
            }
        }
        var precision = 0
        while precision < 22 {
            if Int(0.5 + maxWeight * Double(1 << (precision + 1))) >= (1 << 15) { break }
            precision += 1
        }
        let one = Double(1 << precision)
        let weights = real.map { w -> Int32 in
            let v = w * one
            return Int32(truncatingIfNeeded: Int16(truncatingIfNeeded: v < 0 ? Int(-0.5 + v) : Int(0.5 + v)))
        }
        return Taps(start: start, count: count, weights: weights, stride: stride, precision: precision)
    }

    /// Resize interleaved 8-bit pixels (`channels` per pixel, rows top to bottom).
    package static func resize(
        _ pixels: [UInt8], width: Int, height: Int, channels: Int, toWidth outWidth: Int, toHeight outHeight: Int
    ) -> [UInt8] {
        precondition(width > 0 && height > 0 && outWidth > 0 && outHeight > 0 && channels > 0)
        precondition(pixels.count == width * height * channels)
        var current = pixels
        var currentWidth = width
        if outWidth != width {
            current = horizontal(current, width: width, height: height, channels: channels, outWidth: outWidth)
            currentWidth = outWidth
        }
        if outHeight != height {
            current = vertical(
                current, width: currentWidth, height: height, channels: channels, outHeight: outHeight)
        }
        return current
    }

    /// Resize an RGB image.
    package static func resize(_ image: RGB8Image, toWidth width: Int, toHeight height: Int) -> RGB8Image {
        if width == image.width && height == image.height { return image }
        return RGB8Image(
            width: width, height: height,
            pixels: resize(
                image.pixels, width: image.width, height: image.height, channels: 3, toWidth: width,
                toHeight: height))
    }

    @inline(__always) private static func clip8(_ acc: Int32, _ precision: Int) -> UInt8 {
        let v = acc >> Int32(precision)
        return UInt8(v < 0 ? 0 : (v > 255 ? 255 : v))
    }

    private static func horizontal(
        _ src: [UInt8], width: Int, height: Int, channels: Int, outWidth: Int
    ) -> [UInt8] {
        let t = taps(inputSize: width, outputSize: outWidth)
        let bias = Int32(1) << Int32(t.precision - 1)
        var out = [UInt8](repeating: 0, count: outWidth * height * channels)
        src.withUnsafeBufferPointer { s in
            t.weights.withUnsafeBufferPointer { w in
                out.withUnsafeMutableBufferPointer { d in
                    for y in 0..<height {
                        let srcRow = y * width * channels
                        let dstRow = y * outWidth * channels
                        for x in 0..<outWidth {
                            let first = srcRow + t.start[x] * channels
                            let wBase = x * t.stride
                            let n = t.count[x]
                            for c in 0..<channels {
                                var acc = bias
                                var p = first + c
                                for j in 0..<n {
                                    acc &+= Int32(s[p]) &* w[wBase + j]
                                    p += channels
                                }
                                d[dstRow + x * channels + c] = clip8(acc, t.precision)
                            }
                        }
                    }
                }
            }
        }
        return out
    }

    private static func vertical(
        _ src: [UInt8], width: Int, height: Int, channels: Int, outHeight: Int
    ) -> [UInt8] {
        let t = taps(inputSize: height, outputSize: outHeight)
        let bias = Int32(1) << Int32(t.precision - 1)
        let rowLength = width * channels
        var out = [UInt8](repeating: 0, count: rowLength * outHeight)
        var acc = [Int32](repeating: 0, count: rowLength)
        src.withUnsafeBufferPointer { s in
            out.withUnsafeMutableBufferPointer { d in
                acc.withUnsafeMutableBufferPointer { a in
                    for y in 0..<outHeight {
                        for k in 0..<rowLength { a[k] = bias }
                        let wBase = y * t.stride
                        for j in 0..<t.count[y] {
                            let weight = t.weights[wBase + j]
                            let row = (t.start[y] + j) * rowLength
                            for k in 0..<rowLength { a[k] &+= Int32(s[row + k]) &* weight }
                        }
                        let dstRow = y * rowLength
                        for k in 0..<rowLength { d[dstRow + k] = clip8(a[k], t.precision) }
                    }
                }
            }
        }
        return out
    }
}

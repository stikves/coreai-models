// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation

/// Aspect-preserving resize to multiples of k x patch size, then patchify and pad to the soft-token budget,
/// matching the reference image processor.
struct ImagePatcher: Sendable, Equatable {
    var patchSize: Int
    var maxSoftTokens: Int
    var poolingKernelSize: Int
    var rescaleFactor: Float
    var mean: [Float]
    var std: [Float]

    init(_ config: EmbeddingConfig.ImagePreprocessing) {
        let normalize = config.normalize ?? false
        patchSize = config.patchSize ?? 16
        maxSoftTokens = config.maxSoftTokens ?? 280
        poolingKernelSize = config.poolingKernelSize ?? 3
        rescaleFactor = config.rescaleFactor ?? 1.0 / 255.0
        mean = normalize ? (config.imageMean ?? [0, 0, 0]) : [0, 0, 0]
        std = normalize ? (config.imageStd ?? [1, 1, 1]) : [1, 1, 1]
    }

    /// Fixed patch budget `maxSoftTokens * k^2`.
    var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }

    /// Flattened patch width `3 * patchSize^2`.
    var patchDimension: Int { 3 * patchSize * patchSize }

    /// Patch tensors for one image.
    struct Patches: Sendable, Equatable {
        /// `[maxPatches, patchDimension]`, row-major, zero rows for padding.
        var pixelValues: [Float]
        /// `[maxPatches, 2]` `(x, y)` patch coordinates, `(-1, -1)` for padding.
        var positionIds: [Int32]
        /// Real soft tokens (`realPatches / k^2`).
        var numSoftTokens: Int
        var resizedHeight: Int
        var resizedWidth: Int
    }

    /// Target `(height, width)` for an input of the given size.
    func targetSize(height: Int, width: Int) throws -> (height: Int, width: Int) {
        guard height > 0, width > 0 else { throw EmbeddingError.invalidImage }
        let k = poolingKernelSize
        let targetPixels = Double(maxPatches * patchSize * patchSize)
        let factor = (targetPixels / Double(height * width)).squareRoot()
        let side = k * patchSize
        var th = Int((factor * Double(height) / Double(side)).rounded(.down)) * side
        var tw = Int((factor * Double(width) / Double(side)).rounded(.down)) * side
        if th == 0 && tw == 0 { throw EmbeddingError.invalidImage }
        let maxSide = (maxPatches / (k * k)) * side
        if th == 0 {
            th = side
            tw = min(Int((Double(width) / Double(height)).rounded(.down)) * side, maxSide)
        } else if tw == 0 {
            tw = side
            th = min(Int((Double(height) / Double(width)).rounded(.down)) * side, maxSide)
        }
        guard Double(th * tw) <= targetPixels else { throw EmbeddingError.invalidImage }
        return (th, tw)
    }

    /// Patches for `image`, resized with the reference-exact antialiased bicubic kernel.
    func patches(from image: CGImage) throws -> Patches {
        guard let rgb = RGB8Image(cgImage: image) else { throw EmbeddingError.invalidImage }
        let (th, tw) = try targetSize(height: rgb.height, width: rgb.width)
        let resized = BicubicResampler.resize(rgb, toWidth: tw, toHeight: th)
        return try patches(fromResized: resized)
    }

    private func patches(fromResized image: RGB8Image) throws -> Patches {
        let (h, w) = (image.height, image.width)
        var chw = [Float](repeating: 0, count: 3 * h * w)
        let plane = h * w
        image.pixels.withUnsafeBufferPointer { src in
            for i in 0..<plane {
                chw[i] = Float(src[i * 3])
                chw[plane + i] = Float(src[i * 3 + 1])
                chw[2 * plane + i] = Float(src[i * 3 + 2])
            }
        }
        return try patches(fromResizedCHW: chw, height: h, width: w)
    }

    /// Patchify a resized `[3, H, W]` image with values in `0...255`.
    private func patches(fromResizedCHW chw: [Float], height h: Int, width w: Int) throws -> Patches {
        let side = patchSize * poolingKernelSize
        guard h > 0, w > 0, h % side == 0, w % side == 0, chw.count == 3 * h * w else {
            throw EmbeddingError.invalidImage
        }
        let ph = h / patchSize
        let pw = w / patchSize
        let real = ph * pw
        guard real <= maxPatches else { throw EmbeddingError.invalidImage }
        let p = patchSize
        let pd = patchDimension
        var pixels = [Float](repeating: 0, count: maxPatches * pd)
        var positions = [Int32](repeating: -1, count: maxPatches * 2)
        let plane = h * w
        let scale = (0..<3).map { rescaleFactor / std[$0] }
        let offset = (0..<3).map { -mean[$0] / std[$0] }
        for py in 0..<ph {
            for px in 0..<pw {
                let patch = py * pw + px
                positions[patch * 2] = Int32(px)
                positions[patch * 2 + 1] = Int32(py)
                var dst = patch * pd
                for y in 0..<p {
                    let row = (py * p + y) * w + px * p
                    for x in 0..<p {
                        for c in 0..<3 {
                            pixels[dst] = chw[c * plane + row + x] * scale[c] + offset[c]
                            dst += 1
                        }
                    }
                }
            }
        }
        return Patches(
            pixelValues: pixels, positionIds: positions,
            numSoftTokens: real / (poolingKernelSize * poolingKernelSize),
            resizedHeight: h, resizedWidth: w)
    }
}

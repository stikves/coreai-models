// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation

/// An 8-bit RGB decode that matches the reference loader's RGB conversion, with alpha handled the same way.
/// Used before `BicubicResampler`.
package struct RGB8Image: Equatable, Sendable {
    package let width: Int
    package let height: Int
    /// `height * width * 3` bytes, rows top to bottom, interleaved `(r, g, b)`.
    package let pixels: [UInt8]

    package init(width: Int, height: Int, pixels: [UInt8]) {
        precondition(width > 0 && height > 0 && pixels.count == width * height * 3, "RGB8Image size mismatch")
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// Convert a decoded `CGImage` to RGB; returns nil when the image cannot be converted.
    package init?(cgImage image: CGImage) {
        let width = image.width
        let height = image.height
        guard width > 0, height > 0 else { return nil }
        if let direct = Self.directConversion(image) {
            self.init(width: width, height: height, pixels: direct)
            return
        }
        guard let drawn = Self.drawnConversion(image) else { return nil }
        self.init(width: width, height: height, pixels: drawn)
    }

    // MARK: - Direct (PIL-exact) conversion

    private struct Layout {
        var componentsPerPixel: Int
        /// Byte offset of r, g, b (or gray at index 0) inside a pixel, in component units.
        var colorIndex: [Int]
        var alphaIndex: Int?
        var premultiplied: Bool
    }

    private static func directConversion(_ image: CGImage) -> [UInt8]? {
        guard let space = image.colorSpace, !image.bitmapInfo.contains(.floatComponents) else { return nil }
        let bpc = image.bitsPerComponent
        guard bpc == 8 || bpc == 16 else { return nil }
        let model = space.model
        let colorComponents: Int
        switch model {
        case .monochrome: colorComponents = 1
        case .rgb: colorComponents = 3
        case .indexed:
            guard bpc == 8, image.bitsPerPixel == 8, space.baseColorSpace?.model == .rgb else { return nil }
            colorComponents = 1
        default: return nil
        }
        guard let layout = layout(image, colorComponents: colorComponents) else { return nil }
        guard image.bitsPerPixel == layout.componentsPerPixel * bpc else { return nil }
        if layout.premultiplied && bpc != 8 { return nil }
        guard let data = image.dataProvider?.data, let base = CFDataGetBytePtr(data) else { return nil }
        let length = CFDataGetLength(data)
        let bytesPerRow = image.bytesPerRow
        let width = image.width
        let height = image.height
        let bytesPerPixel = image.bitsPerPixel / 8
        guard bytesPerRow >= width * bytesPerPixel, length >= bytesPerRow * (height - 1) + width * bytesPerPixel
        else { return nil }

        var palette: [UInt8] = []
        if model == .indexed {
            guard let table = space.colorTable, table.count >= 3, table.count % 3 == 0 else { return nil }
            palette = table
        }

        let order = image.byteOrderInfo
        let little16 = bpc == 16 && order == .order16Little
        guard bpc == 8 || order == .order16Little || order == .order16Big || order == .orderDefault else {
            return nil
        }

        var out = [UInt8](repeating: 0, count: width * height * 3)
        out.withUnsafeMutableBufferPointer { dst in
            for y in 0..<height {
                let row = base + y * bytesPerRow
                for x in 0..<width {
                    let px = row + x * bytesPerPixel
                    @inline(__always) func sample(_ index: Int) -> Int {
                        if bpc == 8 { return Int(px[index]) }
                        let hi = little16 ? px[index * 2 + 1] : px[index * 2]
                        let lo = little16 ? px[index * 2] : px[index * 2 + 1]
                        return Int(hi) << 8 | Int(lo)
                    }
                    let o = (y * width + x) * 3
                    switch model {
                    case .indexed:
                        let entry = min(Int(px[0]), palette.count / 3 - 1) * 3
                        dst[o] = palette[entry]
                        dst[o + 1] = palette[entry + 1]
                        dst[o + 2] = palette[entry + 2]
                    case .monochrome:
                        var v = sample(layout.colorIndex[0])
                        if bpc == 16 { v = min(v, 255) }  // PIL I;16 -> L clips
                        if layout.premultiplied, let a = layout.alphaIndex { v = unpremultiply(v, Int(px[a])) }
                        let g = UInt8(v)
                        dst[o] = g
                        dst[o + 1] = g
                        dst[o + 2] = g
                    default:
                        for c in 0..<3 {
                            var v = sample(layout.colorIndex[c])
                            if bpc == 16 { v >>= 8 }  // PIL RGB;16B keeps the MSB
                            if layout.premultiplied, let a = layout.alphaIndex { v = unpremultiply(v, Int(px[a])) }
                            dst[o + c] = UInt8(v)
                        }
                    }
                }
            }
        }
        return out
    }

    private static func layout(_ image: CGImage, colorComponents n: Int) -> Layout? {
        let alpha = image.alphaInfo
        let hasAlphaSlot: Bool
        let alphaFirst: Bool
        let premultiplied: Bool
        switch alpha {
        case .none: (hasAlphaSlot, alphaFirst, premultiplied) = (false, false, false)
        case .last, .noneSkipLast: (hasAlphaSlot, alphaFirst, premultiplied) = (true, false, false)
        case .first, .noneSkipFirst: (hasAlphaSlot, alphaFirst, premultiplied) = (true, true, false)
        case .premultipliedLast: (hasAlphaSlot, alphaFirst, premultiplied) = (true, false, true)
        case .premultipliedFirst: (hasAlphaSlot, alphaFirst, premultiplied) = (true, true, true)
        case .alphaOnly: return nil
        @unknown default: return nil
        }
        let cpp = n + (hasAlphaSlot ? 1 : 0)
        // Logical component order, then the memory permutation for 8-bit packed little-endian orders.
        var logical = Array(0..<cpp)  // logical slot -> memory slot
        let order = image.byteOrderInfo
        if image.bitsPerComponent == 8 {
            switch order {
            case .orderDefault, .order32Big, .order16Big: break
            case .order32Little:
                guard cpp == 4 else { return nil }
                logical = logical.reversed()
            case .order16Little:
                guard cpp == 2 else { return nil }
                logical = logical.reversed()
            default: return nil
            }
        }
        let colorStart = alphaFirst && hasAlphaSlot ? 1 : 0
        let colorIndex = (0..<n).map { logical[colorStart + $0] }
        let isRealAlpha = alpha != .noneSkipLast && alpha != .noneSkipFirst && alpha != .none
        let alphaIndex = isRealAlpha ? logical[alphaFirst ? 0 : n] : nil
        return Layout(
            componentsPerPixel: cpp, colorIndex: colorIndex, alphaIndex: alphaIndex, premultiplied: premultiplied)
    }

    /// Best effort inverse of 8-bit premultiplication (lossy; decoders for PNG/JPEG deliver straight alpha).
    @inline(__always) private static func unpremultiply(_ c: Int, _ a: Int) -> Int {
        a == 0 ? 0 : min(255, (c * 255 + a / 2) / a)
    }

    // MARK: - Fallback

    private static func drawnConversion(_ image: CGImage) -> [UInt8]? {
        let width = image.width
        let height = image.height
        guard let space = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var rgbx = [UInt8](repeating: 0, count: width * height * 4)
        let drawn = rgbx.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                    bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { return nil }
        var out = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) {
            out[i * 3] = rgbx[i * 4]
            out[i * 3 + 1] = rgbx[i * 4 + 1]
            out[i * 3 + 2] = rgbx[i * 4 + 2]
        }
        return out
    }
}

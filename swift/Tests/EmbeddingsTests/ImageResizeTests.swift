// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import CoreAIEmbeddings

/// `ImageResizeFixtures/manifest.json`: each PNG with its PIL `convert("RGB")` decode and the
/// torchvision uint8 antialiased bicubic resize of that decode.
struct ResizeManifest: Decodable {
    struct Case: Decodable {
        let name: String
        let input: String
        let mode: String
        let width: Int
        let height: Int
        let targetHeight: Int
        let targetWidth: Int
        let decoded: String
        let resized: String

        enum CodingKeys: String, CodingKey {
            case name, input, mode, width, height, decoded, resized
            case targetHeight = "target_height"
            case targetWidth = "target_width"
        }
    }
    let cases: [Case]
}

@Suite struct ImageResizeTests {
    @Test(arguments: [
        "rgb_photo_down", "rgba_semi_alpha", "gray_down", "palette", "rgb16", "width_only", "checker_contrast",
    ])
    func matchesPIL(_ name: String) throws {
        let directory = try #require(Bundle.module.url(forResource: "ImageResizeFixtures", withExtension: nil))
        let manifest = try JSONDecoder().decode(
            ResizeManifest.self, from: Data(contentsOf: directory.appending(path: "manifest.json")))
        let c = try #require(manifest.cases.first { $0.name == name })
        let expectedDecoded = try [UInt8](Data(contentsOf: directory.appending(path: c.decoded)))
        let expectedResized = try [UInt8](Data(contentsOf: directory.appending(path: c.resized)))

        let source = try #require(CGImageSourceCreateWithURL(directory.appending(path: c.input) as CFURL, nil))
        let decoded = try RGB8Image(cgImage: #require(CGImageSourceCreateImageAtIndex(source, 0, nil)))
        #expect(decoded.width == c.width && decoded.height == c.height)
        #expect(decoded.pixels == expectedDecoded, "decode \(c.name) (\(c.mode))")

        let resized = AntialiasedBicubicResize.resize(
            RGB8Image(width: c.width, height: c.height, pixels: expectedDecoded),
            toWidth: c.targetWidth, toHeight: c.targetHeight)
        #expect(resized.pixels == expectedResized, "resize \(c.name)")
    }
}

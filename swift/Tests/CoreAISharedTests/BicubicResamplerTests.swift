// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import ImageIO
import Testing

@testable import CoreAIShared

/// `manifest.json` of the bundled reference set (see the fixtures README).
private struct ResizeReferenceManifest: Decodable {
    struct Case: Decodable {
        let name: String
        let mode: String
        let width: Int
        let height: Int
        let targetHeight: Int
        let targetWidth: Int
    }
    let cases: [Case]

    static func load(_ directory: URL) throws -> ResizeReferenceManifest {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(Self.self, from: Data(contentsOf: directory.appending(path: "manifest.json")))
    }
}

@Suite("BicubicResampler")
struct BicubicResamplerTests {
    private static let directory: URL? = Bundle.module.url(forResource: "BicubicResamplerFixtures", withExtension: nil)

    private static var caseNames: [String] {
        guard let directory, let manifest = try? ResizeReferenceManifest.load(directory) else { return [] }
        return manifest.cases.map(\.name)
    }

    private static func decode(_ url: URL) throws -> RGB8Image {
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
        return try #require(RGB8Image(cgImage: image))
    }

    @Test("bit-exact vs the reference RGB conversion and uint8 resize", arguments: caseNames)
    fileprivate func fixture(name: String) throws {
        let directory = try #require(Self.directory)
        let c = try #require(try ResizeReferenceManifest.load(directory).cases.first { $0.name == name })
        let expectedDecoded = try [UInt8](Data(contentsOf: directory.appending(path: "\(c.name).decoded.rgb")))
        let expectedResized = try [UInt8](Data(contentsOf: directory.appending(path: "\(c.name).resized.rgb")))

        let decoded = try Self.decode(directory.appending(path: "\(c.name).png"))
        #expect(decoded.width == c.width && decoded.height == c.height)
        #expect(decoded.pixels == expectedDecoded, "decode \(c.name) (\(c.mode))")

        // Resize alone, from the reference decode.
        let fromReference = BicubicResampler.resize(
            RGB8Image(width: c.width, height: c.height, pixels: expectedDecoded),
            toWidth: c.targetWidth, toHeight: c.targetHeight)
        #expect(fromReference.pixels == expectedResized, "resize \(c.name)")

        // Full host path.
        let full = BicubicResampler.resize(decoded, toWidth: c.targetWidth, toHeight: c.targetHeight)
        #expect(full.pixels == expectedResized)
    }

    @Test("quantized taps: per-output weights sum to 2^precision (±rounding), upscale support 2")
    func taps() {
        for (input, output) in [(37, 96), (200, 96), (2048, 768), (5, 48), (768, 768 + 48)] {
            let t = BicubicResampler.taps(inputSize: input, outputSize: output)
            for i in 0..<output {
                let sum = (0..<t.count[i]).reduce(Int32(0)) { $0 + t.weights[i * t.stride + $1] }
                #expect(abs(Int(sum) - (1 << t.precision)) <= t.count[i], "\(input)->\(output) i=\(i)")
            }
            if input < output { #expect(t.count.allSatisfy { $0 <= 4 }) }
        }
    }

    @Test("unchanged size returns the input")
    func identity() {
        let image = RGB8Image(width: 48, height: 48, pixels: (0..<(48 * 48 * 3)).map { UInt8($0 % 251) })
        #expect(BicubicResampler.resize(image, toWidth: 48, toHeight: 48) == image)
    }
}

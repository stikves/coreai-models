// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreGraphics
import Foundation
import TestUtilities
import Testing

@testable import CoreAIEmbeddings

enum ImageTestSupport {
    /// RGB image whose pixel (x, y) is (x % 256, y % 256, (x + y) % 256).
    static func gradientImage(width: Int, height: Int) -> CGImage? {
        image(width: width, height: height) { x, y in [UInt8(x % 256), UInt8(y % 256), UInt8((x + y) % 256)] }
    }

    static func solidImage(width: Int, height: Int) -> CGImage? {
        image(width: width, height: height) { _, _ in [128, 128, 128] }
    }

    /// RGB image from a per-pixel `(x, y) -> [r, g, b]` closure.
    static func image(width: Int, height: Int, pixel: (Int, Int) -> [UInt8]) -> CGImage? {
        var bytes = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let rgb = pixel(x, y)
                let i = (y * width + x) * 4
                bytes[i] = rgb[0]
                bytes[i + 1] = rgb[1]
                bytes[i + 2] = rgb[2]
            }
        }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let provider = CGDataProvider(data: Data(bytes) as CFData)
        else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
            space: space, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}

@Suite("Image preprocessing")
struct ImagePatcherTests {
    let patcher = ImagePatcher(.init())

    /// Target sizes recorded by the reference image processor.
    @Test(
        "aspect-preserving target size matches the reference processor",
        arguments: [
            (512, 512, 768, 768, 256), (172, 448, 480, 1296, 270), (660, 550, 864, 720, 270),
            (328, 400, 720, 864, 270), (300, 451, 624, 960, 260), (872, 1000, 720, 816, 255),
            (1411, 1411, 768, 768, 256), (303, 384, 672, 864, 252),
        ])
    func targetSizes(h: Int, w: Int, th: Int, tw: Int, softTokens: Int) throws {
        let size = try patcher.targetSize(height: h, width: w)
        #expect(size.height == th && size.width == tw)
        #expect((th / 16) * (tw / 16) / 9 == softTokens)
    }

    @Test("an image already at the target size is patchified in (py, px, c) order with (x, y) positions")
    func unchangedSize() throws {
        let small = ImagePatcher(.init(patchSize: 2, rescaleFactor: nil, maxSoftTokens: 4, poolingKernelSize: 1))
        // 4 x 2 image with channel c at (x, y) = c * 100 + y * 10 + x; the target size is also 2 x 4.
        let image = try #require(
            ImageTestSupport.image(width: 4, height: 2) { x, y in (0..<3).map { UInt8($0 * 100 + y * 10 + x) } })
        #expect(try small.targetSize(height: 2, width: 4) == (2, 4))
        let p = try small.patches(from: image)
        #expect(p.numSoftTokens == 2)
        #expect(p.resizedHeight == 2 && p.resizedWidth == 4)
        #expect(p.positionIds == [0, 0, 1, 0, -1, -1, -1, -1])
        let r: Float = 1.0 / 255.0
        // Patch 0: pixels (y0,x0) (y0,x1) (y1,x0) (y1,x1), each (r, g, b).
        let expected0: [Float] = [0, 100, 200, 1, 101, 201, 10, 110, 210, 11, 111, 211].map { $0 * r }
        #expect(zip(p.pixelValues.prefix(12), expected0).allSatisfy { abs($0 - $1) < 1e-6 })
        #expect(abs(p.pixelValues[12] - 2 * r) < 1e-6)  // patch 1 starts at x = 2
        #expect(p.pixelValues.count == 4 * 12)
        #expect(p.pixelValues.suffix(24).allSatisfy { $0 == 0 })
    }

    @Test("a larger image is resized, padded to the budget and kept in [0, 1]")
    func fromCGImage() throws {
        let image = try #require(ImageTestSupport.gradientImage(width: 300, height: 200))
        let p = try patcher.patches(from: image)
        let size = try patcher.targetSize(height: 200, width: 300)
        #expect(p.resizedHeight == size.height && p.resizedWidth == size.width)
        #expect(p.pixelValues.count == patcher.maxPatches * patcher.patchDimension)
        #expect(p.positionIds.count == patcher.maxPatches * 2)
        #expect(p.numSoftTokens == (size.height / 16) * (size.width / 16) / 9)
        #expect(p.pixelValues.allSatisfy { $0 >= 0 && $0 <= 1 })
    }
}

/// The text fixture extended with an image tower over fake components.
private func twoTowerMetadata() throws -> Data {
    try EmbeddingFixtures.editedMetadata { block in
        var towers = block["towers"] as! [[String: Any]]
        towers.append([
            "modality": "image", "pipeline": ["image_encoder", "token_table", "encoder"], "output": "embedding",
        ])
        block["towers"] = towers
        var components = block["components"] as! [String: Any]
        components["image_encoder"] = ["function": "main", "output": "soft_tokens"]
        components["token_table"] = ["function": "main", "inputs": ["input_ids"], "output": "inputs_embeds"]
        components["encoder"] = [
            "functions": [["function": "len512_b1", "batch": 1, "length": 512]], "output": "embedding",
        ]
        block["components"] = components
        block["merge"] = ["placeholder_ids": ["image": 258880], "token_ids": ["boi": 255999, "eoi": 258882]]
        block["preprocessing"] = ["image": ["patch_size": 16, "rescale_factor": 1.0 / 255.0, "max_soft_tokens": 280]]
    }
}

/// Text and image towers over fake components, keyed by asset name.
private func fakeTwoTowerModel(
    dimension: Int = 768, hidden: Int = 8
) throws -> (EmbeddingModel, [String: FakeEmbeddingComponent]) {
    var root = try #require(JSONSerialization.jsonObject(with: twoTowerMetadata()) as? [String: Any])
    root["assets"] = [
        "text_encoder": "text_encoder.aimodel", "image_encoder": "image_encoder.aimodel",
        "token_table": "token_table.aimodel", "encoder": "encoder.aimodel",
    ]
    let metadata = try JSONSerialization.data(withJSONObject: root)
    let components = [
        "text_encoder": FakeEmbeddingComponent(dimension: dimension),
        "image_encoder": FakeEmbeddingComponent.softTokens(hidden: hidden),
        "token_table": FakeEmbeddingComponent(dimension: hidden) { _, inputs in
            [Float](repeating: -1, count: (inputs["input_ids"]?.shape[1] ?? 0) * hidden)
        },
        "encoder": FakeEmbeddingComponent.constantEncoder(dimension: dimension),
    ]
    return (try EmbeddingFixtures.model(metadata: metadata, components: components), components)
}

@Suite("Per-tower lazy loading")
struct LazyLoadingTests {
    private static let imageComponents = ["image_encoder", "token_table", "encoder"]

    @Test("a text request loads only the text tower and its smallest function; an image request loads the rest")
    func textThenImage() async throws {
        let (model, components) = try fakeTwoTowerModel()
        _ = try await model.embed(.text("hello"))
        let text = try #require(components["text_encoder"])
        #expect(await text.loadCount == 1)
        #expect(await text.prepared == ["len128_b1"])
        #expect(await text.callCount(function: "len128_b1") == 1)
        for name in Self.imageComponents { #expect(await components[name]!.loadCount == 0) }

        // A second text request loads and prepares nothing new.
        _ = try await model.embed(.text("again"))
        #expect(await text.loadCount == 1)
        #expect(await text.prepared == ["len128_b1"])
        #expect(await text.callCount(function: "len128_b1") == 2)

        let image = try #require(ImageTestSupport.gradientImage(width: 64, height: 64))
        _ = try await model.embed(.image(image))
        for name in Self.imageComponents { #expect(await components[name]!.loadCount == 1) }
        #expect(await text.loadCount == 1)
        #expect(await components["encoder"]!.prepared == ["len512_b1"])
        #expect(await components["token_table"]!.prepared == ["main"])
        #expect(await components["encoder"]!.callCount(function: "len512_b1") == 1)
    }

    @Test("concurrent loads share one prepare per component")
    func concurrentLoads() async throws {
        let (model, components) = try fakeTwoTowerModel()
        async let a: Void = model.load(modalities: [.text, .image])
        async let b: Void = model.load(modalities: [.image])
        _ = try await (a, b)
        for component in components.values { #expect(await component.loadCount == 1) }
    }
}

@Suite("Image tower with fake components")
struct ImageTowerFakeTests {
    @Test("soft tokens are scattered over <image> placeholders inside <bos><boi>...<eoi><eos>")
    func imagePipeline() async throws {
        let (model, components) = try fakeTwoTowerModel()
        #expect(model.supportedModalities == [.text, .image])
        let image = try #require(ImageTestSupport.gradientImage(width: 512, height: 512))
        let e = try await model.embed(.image(image), dimensions: 256)
        #expect(e.dimension == 256)
        #expect(e.tokenCount == 256 + 4)  // 768x768 -> 256 soft tokens + bos/boi/eoi/eos

        let call = try #require(await components["encoder"]!.calls.first)
        #expect(call.function == "len512_b1")
        #expect(call.output == "embedding")
        #expect(call.shapes["inputs_embeds"] == [1, 512, 8])
        guard case .float32(let embeds) = call.inputs["inputs_embeds"]?.storage,
            case .int32(let mask) = call.inputs["attention_mask"]?.storage
        else {
            Issue.record("unexpected encoder inputs")
            return
        }
        #expect(mask.prefix(260).allSatisfy { $0 == 1 } && mask.dropFirst(260).allSatisfy { $0 == 0 })
        // Row 0 (bos) and row 1 (boi) keep the table value; row 2 is soft token 0.
        #expect(embeds[0..<16].allSatisfy { $0 == -1 })
        // Soft-token values come from `FakeEmbeddingComponent.softTokens` (`% 7`).
        #expect(Array(embeds[16..<24]) == (0..<8).map { Float($0 % 7) })
        #expect(embeds[(257 * 8)..<(258 * 8)].allSatisfy { $0 != -1 })  // last placeholder row (soft token 255)
        #expect(embeds[(258 * 8)..<(260 * 8)].allSatisfy { $0 == -1 })  // eoi, eos
    }

    @Test("an image tower without its assets is an invalid bundle")
    func missingImageAssets() async throws {
        let data = try EmbeddingFixtures.editedMetadata { block in
            var towers = block["towers"] as! [[String: Any]]
            towers.append(["modality": "image", "pipeline": ["image_encoder", "token_table", "encoder"]])
            block["towers"] = towers
            block["merge"] = ["placeholder_ids": ["image": 9], "token_ids": ["boi": 7, "eoi": 8]]
        }
        let model = try EmbeddingFixtures.model(
            metadata: data, components: ["text_encoder": FakeEmbeddingComponent(dimension: 768)])
        let image = try #require(ImageTestSupport.solidImage(width: 64, height: 64))
        await #expect(throws: EmbeddingError.invalidBundle("the bundle declares no asset 'image_encoder'")) {
            try await model.embed(.image(image))
        }
    }

    @Test("a declared asset whose file is missing reports the path")
    func missingAssetFile() async throws {
        let url = URL(fileURLWithPath: "bundle/image_encoder.aimodel")
        let component = try CoreAIEmbeddingComponent(name: "image_encoder", modelURL: url, computeUnits: ["gpu"])
        await #expect {
            try await component.load()
        } throws: { error in
            guard case .invalidBundle(let detail) = error as? EmbeddingError else { return false }
            return detail.contains(url.path)
        }
    }

    @Test("a Core AI failure names the component and compute unit")
    func componentFailureContext() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        let url = directory.appending(path: "broken.aimodel")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data("not a model".utf8).write(to: url.appending(path: "junk"))
        let component = try CoreAIEmbeddingComponent(name: "broken", modelURL: url, computeUnits: ["neural_engine"])
        await #expect {
            try await component.load()
        } throws: { error in
            guard case .componentFailed(let name, _, let unit, let cached, _) = error as? EmbeddingError else {
                return false
            }
            return name == "broken" && unit == "neural_engine" && !cached
                && (error as? EmbeddingError)?.errorDescription?.contains("Neural Engine") == true
        }
        let stale = EmbeddingError.componentFailed(
            component: "encoder", function: "len128_b1", computeUnit: "neural_engine", cached: true, reason: "x")
        #expect(stale.errorDescription?.contains("--clear-coreai-cache") == true)
        #expect(stale.errorDescription?.contains("PreparedModel.clearCache(at:)") == true)
        #expect(stale.errorDescription?.hasPrefix("encoder function 'len128_b1' failed on the Neural Engine: x.") == true)
    }
}

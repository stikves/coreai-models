// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIEmbeddings
import CoreAIVectorIndex
import CoreGraphics
import Foundation
import ImageIO
import TestUtilities
import UniformTypeIdentifiers

@testable import embedding_search

/// A real `EmbeddingModel` over deterministic fake components.
///
/// Text: components 0/1/2 count the words red/green/blue; every other word adds 0.1 to one of
/// components 3...7. Image: components 0/1/2 are the mean R/G/B, plus a small constant. Vectors are
/// L2-normalized; the model re-normalizes Matryoshka prefixes.
struct SearchTestModel {
    let model: EmbeddingModel
    let text: FakeEmbeddingComponent
    let imageEncoder: FakeEmbeddingComponent

    /// Strings of the attended rows of a text-encoder call (the fixture tokenizer maps text to UTF-8 bytes).
    static func rows(of call: FakeEmbeddingComponent.Call) -> [String] { rows(call.inputs) }

    static func rows(_ inputs: [String: EmbeddingTensor]) -> [String] {
        guard case .int32(let ids) = inputs["input_ids"]?.storage,
            case .int32(let mask) = inputs["attention_mask"]?.storage,
            let shape = inputs["input_ids"]?.shape
        else { return [] }
        return (0..<shape[0]).map { row in
            let bytes = (0..<shape[1]).compactMap { column -> UInt8? in
                let index = row * shape[1] + column
                return mask[index] == 1 ? UInt8(truncatingIfNeeded: ids[index]) : nil
            }
            return String(decoding: bytes, as: UTF8.self)
        }
    }

    static func make(
        space: (id: String, revision: String) = ("test/fake", "1"), image: Bool = true,
        prompts: [String: String] = ["document": "document: ", "query": "query: "]
    ) throws -> SearchTestModel {
        let hidden = 8
        var assets = ["text_encoder": "text_encoder.aimodel"]
        var components: [String: Any] = [
            "text_encoder": ["function": "main", "output": "embedding"]
        ]
        var towers: [[String: Any]] = [
            [
                "modality": "text", "pipeline": ["text_encoder"], "output": "embedding",
                "inputs": ["input_ids": "int32[B,L]", "attention_mask": "int32[B,L]"],
                "batch_sizes": [64], "length_buckets": [256],
            ]
        ]
        if image {
            for name in ["image_encoder", "token_table", "encoder"] { assets[name] = "\(name).aimodel" }
            components["image_encoder"] = ["function": "main", "output": "soft_tokens"]
            components["token_table"] = ["function": "main", "inputs": ["input_ids"], "output": "inputs_embeds"]
            components["encoder"] = [
                "functions": [["function": "len16_b1", "batch": 1, "length": 16]], "output": "embedding",
            ]
            towers.append([
                "modality": "image", "pipeline": ["image_encoder", "token_table", "encoder"], "output": "embedding",
            ])
        }
        var block: [String: Any] = [
            "space": ["id": space.id, "revision": space.revision], "dimension": hidden,
            "matryoshka_dimensions": [8, 4], "components": components, "towers": towers,
            "precision": ["compute": "float32", "forbidden": [String](), "compute_units": ["gpu"]],
            "text": [
                "special_tokens": ["bos": false, "eos": false, "pad_token_id": 0],
                "truncation": ["max_tokens": 256], "prompts": prompts,
                "document_template": "title: {title} | text: {text}",
            ] as [String: Any],
        ]
        if image {
            block["merge"] = ["placeholder_ids": ["image": 9], "token_ids": ["boi": 7, "eoi": 8]]
            block["preprocessing"] = ["image": ["patch_size": 16, "max_soft_tokens": 4, "pooling_kernel_size": 1]]
        }
        let metadata = try JSONSerialization.data(withJSONObject: [
            "metadata_version": "0.2", "kind": "embedding", "name": "search-fake", "assets": assets,
            "embedding": block,
        ] as [String: Any])

        let text = FakeEmbeddingComponent(dimension: hidden) { _, inputs in
            rows(inputs).flatMap { FakeEmbeddingComponent.normalized(textVector($0)) }
        }
        // Soft tokens carry the mean color of the real patches; the encoder reads the first soft-token row.
        let imageEncoder = FakeEmbeddingComponent(dimension: hidden) { _, inputs in
            guard case .float32(let pixels) = inputs["pixel_values"]?.storage,
                case .int32(let positions) = inputs["position_ids"]?.storage
            else { return [] }
            let patchWidth = pixels.count / (positions.count / 2)
            var sums = [Float](repeating: 0, count: 3)
            var count: Float = 0
            for patch in 0..<(positions.count / 2) where positions[patch * 2] >= 0 {
                for i in 0..<patchWidth { sums[i % 3] += pixels[patch * patchWidth + i] }
                count += Float(patchWidth / 3)
            }
            let row = sums.map { $0 / max(count, 1) } + [Float](repeating: 0.05, count: hidden - 3)
            return Array((0..<4).map { _ in row }.joined())
        }
        let table = FakeEmbeddingComponent(dimension: hidden) { _, inputs in
            [Float](repeating: 0, count: (inputs["input_ids"]?.shape[1] ?? 0) * hidden)
        }
        let encoder = FakeEmbeddingComponent(dimension: hidden) { _, inputs in
            guard case .float32(let embeds) = inputs["inputs_embeds"]?.storage else { return [] }
            return FakeEmbeddingComponent.normalized(Array(embeds[hidden..<(2 * hidden)]))  // row 1: first soft token
        }
        let model = try EmbeddingTestBundle.model(
            metadata: metadata,
            components: ["text_encoder": text, "image_encoder": imageEncoder, "token_table": table, "encoder": encoder],
            tokenizer: MockTokenizer())
        return SearchTestModel(model: model, text: text, imageEncoder: imageEncoder)
    }

    static func textVector(_ s: String) -> [Float] {
        var v = [Float](repeating: 0, count: 8)
        v[3] = 0.01
        for word in s.lowercased().split(whereSeparator: { !$0.isLetter }) {
            switch word {
            case "red": v[0] += 1
            case "green": v[1] += 1
            case "blue": v[2] += 1
            default: v[3 + Int(word.unicodeScalars.reduce(0) { $0 + $1.value } % 5)] += 0.1
            }
        }
        return v
    }
}

enum TestFiles {
    static func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appending(path: "embedding-search-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Writes a solid-color image (`png` or `jpeg`).
    static func writeImage(_ url: URL, red: CGFloat, green: CGFloat, blue: CGFloat, type: UTType = .png) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        let ctx = CGContext(
            data: nil, width: 16, height: 12, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(red: red, green: green, blue: blue, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: 16, height: 12))
        let image = ctx.makeImage()!
        let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil)!
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }

    static func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }
}

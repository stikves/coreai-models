// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation
import Testing

@testable import CoreAIEmbeddings

extension EmbeddingBlock {
    func tower(for modality: EmbeddingModality) -> Tower? { towers.first { $0.modality == modality } }
}

/// The `metadata.json` files of exported bundles, in `Resources/Bundles`.
enum Fixtures {
    static let names = [
        "clip-vit-b32", "clip-vit-b32-text", "clip-vit-b32-fused", "clap-htsat", "embeddinggemma-2",
        "embeddinggemma-2-ios",
    ]

    static func data(_ name: String) throws -> Data {
        try Data(contentsOf: #require(Bundle.module.url(forResource: "Bundles/\(name)", withExtension: "json")))
    }

    /// The fixture as a JSON object.
    static func json(_ name: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: data(name)) as? [String: Any])
    }

    static func block(_ name: String) throws -> EmbeddingBlock {
        try block(name) { _ in }
    }

    /// Decodes `name` after `edit` changes its `embedding` object.
    static func block(_ name: String, editing edit: (inout [String: Any]) -> Void) throws -> EmbeddingBlock {
        var root = try json(name)
        var embedding = try #require(root["embedding"] as? [String: Any])
        edit(&embedding)
        root["embedding"] = embedding
        let assets = try #require(root["assets"] as? [String: String])
        return try EmbeddingBlock(metadata: JSONSerialization.data(withJSONObject: root), assets: assets)
    }

    /// An RGB image with pixel `(x, y)` colored by `color`.
    static func image(width: Int, height: Int, color: (Int, Int) -> (UInt8, UInt8, UInt8)) throws -> CGImage {
        var pixels = [UInt8](repeating: 255, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let (r, g, b) = color(x, y)
                pixels.replaceSubrange((y * width + x) * 4..<(y * width + x) * 4 + 3, with: [r, g, b])
            }
        }
        let provider = try #require(CGDataProvider(data: Data(pixels) as CFData))
        return try #require(
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent))
    }
}

extension Dictionary where Key == String, Value == Any {
    /// Applies `edit` to the object at `path` (keys separated by dots).
    mutating func edit(_ path: String, _ edit: (inout [String: Any]) -> Void) {
        let keys = path.split(separator: ".").map(String.init)
        guard let first = keys.first else { return edit(&self) }
        var child = self[first] as? [String: Any] ?? [:]
        child.edit(keys.dropFirst().joined(separator: "."), edit)
        self[first] = child
    }
}

/// One recorded call of a fake program.
struct FakeCall: Sendable {
    let component: String
    let function: String
    let inputs: [String: HostTensor]
}

/// Records the calls of every fake program of a model.
actor CallLog {
    private(set) var calls: [FakeCall] = []
    func append(_ call: FakeCall) { calls.append(call) }
    func calls(of component: String) -> [FakeCall] { calls.filter { $0.component == component } }
}

/// A program of one asset that answers every function with outputs of the declared shapes. Each
/// output row `b` is filled by `value(component, output, b, inputs)`, or by its row index plus one.
struct FakeProgram: Program {
    let asset: String
    let block: EmbeddingBlock
    let log: CallLog
    let value: @Sendable (String, String, Int, [String: HostTensor]) -> [Float]?

    func run(function: String, inputs: [String: HostTensor]) async throws -> [String: HostTensor] {
        let (component, spec) = try #require(
            block.components.first {
                $0.value.asset == asset && $0.value.functions.contains { $0.function == function }
            })
        await log.append(FakeCall(component: component, function: function, inputs: inputs))
        let batch = inputs.values.first?.shape.first ?? 1
        let length = inputs.values.lazy.compactMap { tensor -> Int? in tensor.shape.count > 1 ? tensor.shape[1] : nil }
            .first
        var outputs: [String: HostTensor] = [:]
        for (name, out) in spec.outputs {
            let rowShape = out.shape.dropFirst().map { axis -> Int in
                switch axis {
                case .length: return length ?? 1
                case .fixed(let n): return n
                default: return 1
                }
            }
            let rowCount = rowShape.reduce(1, *)
            var values: [Float] = []
            for b in 0..<batch {
                let row = value(component, name, b, inputs) ?? [Float](repeating: Float(b + 1), count: rowCount)
                values +=
                    row.count == rowCount
                    ? row : Array(row.prefix(rowCount)) + [Float](repeating: 0, count: max(0, rowCount - row.count))
            }
            outputs[name] = HostTensor(float32: values, shape: [batch] + rowShape)
        }
        return outputs
    }
}

/// Fake models over the fixture metadata.
enum FakeModels {
    /// An `EmbeddingBundle` for fixture `name`, rooted in a temporary directory.
    static func bundle(_ name: String) throws -> EmbeddingBundle {
        let bundle = try ModelBundle(raw: Fixtures.data(name), bundlePath: URL(fileURLWithPath: NSTemporaryDirectory()))
        return try EmbeddingBundle(bundle: bundle)
    }

    /// Token ids: BOS (when the bundle has one), one id per word (its length + 10), EOS.
    static func text(_ config: TextConfig) -> TextPreparer {
        TextPreparer(config: config) { string in
            (config.bosTokenId.map { [Int($0)] } ?? []) + string.split(separator: " ").map { $0.count + 10 }
                + [Int(config.eosTokenId)]
        }
    }

    static func model(
        _ name: String, log: CallLog = CallLog(),
        value: @escaping @Sendable (String, String, Int, [String: HostTensor]) -> [Float]? = { _, _, _, _ in nil }
    ) throws -> EmbeddingModel {
        let bundle = try bundle(name)
        return try EmbeddingModel(
            bundle: bundle,
            programs: { _, component in FakeProgram(asset: component.asset, block: bundle.block, log: log, value: value)
            },
            text: { text(bundle.block.text) })
    }
}

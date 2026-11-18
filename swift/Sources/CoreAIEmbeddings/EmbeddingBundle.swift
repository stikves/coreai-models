// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

// The reader for `kind: "embedding"` bundles: `python/src/coreai_models/embedding/contract.py`
// R1-R10, plus the EmbeddingGemma 2 keys listed in `embedding/embedding_gemma2/__init__.py`.
// Decoding is strict (R5): an unknown key, tower or enum value throws `invalidMetadata`.

/// A `kind: "embedding"` bundle: the common `ModelBundle` plus the decoded `embedding` block.
struct EmbeddingBundle: Sendable {
    let bundle: ModelBundle
    let block: EmbeddingBlock

    /// Reads `metadata.json` from the bundle directory at `url`.
    ///
    /// - Throws: `ModelBundle.BundleError` for a missing or malformed `metadata.json` or another
    ///   kind, `EmbeddingError` when the `embedding` block breaks the contract.
    init(at url: URL) throws {
        try self.init(bundle: ModelBundle(at: url))
    }

    init(bundle: ModelBundle) throws {
        guard bundle.kind == .embedding else {
            throw ModelBundle.BundleError.kindMismatch(expected: .embedding, got: bundle.kind)
        }
        self.bundle = bundle
        self.block = try EmbeddingBlock(metadata: bundle.raw, assets: bundle.assets)
    }

    /// The bundled tokenizer folder.
    var tokenizerURL: URL { bundle.bundlePath.appending(path: "tokenizer") }
}

/// The `embedding` block (R2).
struct EmbeddingBlock: Sendable {
    var space: EmbeddingSpace
    var dimension: Int
    /// `logits = logitScale * cosine` (R9); `nil` for models without one.
    var logitScale: Double?
    var components: [String: Component]
    var towers: [Tower]
    var text: TextConfig
    var image: ImageConfig?
    /// Output widths, native first: `[dimension]` unless the bundle lists Matryoshka widths.
    var matryoshkaDimensions: [Int]
    /// Whether a vector cut to a smaller width is L2-normalized again.
    var renormalizeAfterTruncation: Bool

    /// Decodes and validates the `embedding` block of `metadata`.
    ///
    /// - Parameter assets: the bundle's `assets` map; every component must name one of its keys.
    init(metadata: Data, assets: [String: String]) throws {
        struct Envelope: Decodable { let embedding: EmbeddingBlock? }
        let envelope: Envelope
        do {
            envelope = try JSONDecoder().decode(Envelope.self, from: metadata)
        } catch let error as EmbeddingError {
            throw error
        } catch {
            throw EmbeddingError.invalidMetadata(Self.describe(error))
        }
        guard let block = envelope.embedding else {
            throw EmbeddingError.invalidMetadata("missing 'embedding'")
        }
        try block.validate(assets: assets)
        self = block
    }

    private static func describe(_ error: Error) -> String {
        switch error {
        case DecodingError.keyNotFound(let key, let context):
            return "missing '\(key.stringValue)' at \(path(context))"
        case DecodingError.typeMismatch(_, let context), DecodingError.valueNotFound(_, let context):
            return "wrong type at \(path(context))"
        case DecodingError.dataCorrupted(let context):
            return "\(context.debugDescription) at \(path(context))"
        default:
            return "\(error)"
        }
    }

    private static func path(_ context: DecodingError.Context) -> String {
        context.codingPath.map { $0.intValue.map(String.init) ?? $0.stringValue }.joined(separator: ".")
    }
}

/// The space a vector lives in (R2). Vectors are comparable only within one space.
public struct EmbeddingSpace: Codable, Sendable, Hashable {
    public let id: String
    public let revision: String

    enum CodingKeys: String, CodingKey, CaseIterable { case id, revision }

    public init(id: String, revision: String) {
        self.id = id
        self.revision = revision
    }
}

/// A tower's input kind.
public enum EmbeddingModality: String, Sendable { case text, image }

/// The components a modality runs, in order (R4). The last one's `embedding` output is the vector.
struct Tower: Sendable {
    var modality: EmbeddingModality
    var pipeline: [String]
}

// MARK: - Decoding

extension EmbeddingBlock: Decodable {
    enum CodingKeys: String, CodingKey, CaseIterable {
        case space, dimension, normalized, metric, components, towers, text, image, audio
        case logitScale = "logit_scale"
        case computePrecision = "compute_precision"
        case matryoshkaDimensions = "matryoshka_dimensions"
        case renormalizeAfterTruncation = "renormalize_after_truncation"
    }

    init(from decoder: Decoder) throws {
        try decoder.refuseUnknownKeys(CodingKeys.self, in: "embedding")
        let c = try decoder.container(keyedBy: CodingKeys.self)
        space = try c.decode(EmbeddingSpace.self, forKey: .space)
        try c.superDecoder(forKey: .space).refuseUnknownKeys(EmbeddingSpace.CodingKeys.self, in: "space")
        dimension = try c.decode(Int.self, forKey: .dimension)
        // Checked, not used: programs take their declared types, and the graph normalizes.
        _ = try c.decode(Bool.self, forKey: .normalized)
        _ = try c.decodeEnum(Metric.self, forKey: .metric)
        _ = try c.decodeEnum(ComputePrecision.self, forKey: .computePrecision)
        logitScale = try c.decodeIfPresent(Double.self, forKey: .logitScale)
        components = try c.decode([String: Component].self, forKey: .components)
        towers = try Tower.decodeRunnable(c.nestedUnkeyedContainer(forKey: .towers))
        text = try c.decode(TextConfig.self, forKey: .text)
        image = try c.decodeIfPresent(ImageConfig.self, forKey: .image)
        let widths = try c.decodeIfPresent([Int].self, forKey: .matryoshkaDimensions) ?? []
        matryoshkaDimensions = widths.isEmpty ? [dimension] : widths
        renormalizeAfterTruncation = try c.decodeIfPresent(Bool.self, forKey: .renormalizeAfterTruncation) ?? false
    }

    private enum Metric: String { case cosine }
    private enum ComputePrecision: String { case float32, bfloat16, float16 }

    /// The checks of `contract.validate_block`, plus the ones that need decoded values.
    func validate(assets: [String: String]) throws {
        var problems: [String] = []
        if dimension <= 0 { problems.append("dimension must be positive") }
        if matryoshkaDimensions.first != dimension || !matryoshkaDimensions.allSatisfy({ (1...dimension).contains($0) })
        {
            problems.append("matryoshka_dimensions must start with \(dimension) and stay in 1...\(dimension)")
        }
        var functions: Set<[String]> = []
        for (name, component) in components.sorted(by: { $0.key < $1.key }) {
            if assets[component.asset] == nil {
                problems.append("\(name): asset '\(component.asset)' not in assets")
            }
            if component.functions.isEmpty { problems.append("\(name): functions must not be empty") }
            for entry in component.functions where !functions.insert([component.asset, entry.function]).inserted {
                problems.append("\(name): duplicate function '\(entry.function)' in asset '\(component.asset)'")
            }
        }
        let modalities = towers.map(\.modality)
        if modalities.filter({ $0 == .text }).count != 1 || Set(modalities).count != modalities.count {
            problems.append("towers must hold one text tower and distinct modalities")
        }
        for tower in towers {
            let missing = tower.pipeline.filter { components[$0] == nil }
            if tower.pipeline.isEmpty || !missing.isEmpty {
                problems.append("the \(tower.modality.rawValue) pipeline names unknown components \(missing)")
            } else if let last = tower.pipeline.last, components[last]?.outputs["embedding"] == nil {
                problems.append(
                    "the \(tower.modality.rawValue) pipeline ends in '\(last)', which has no embedding output")
            }
        }
        if (image != nil) != modalities.contains(.image) {
            problems.append("the image block must be present exactly with the image tower")
        }
        if !problems.isEmpty { throw EmbeddingError.invalidMetadata(problems.joined(separator: "; ")) }
    }
}

extension Tower: Decodable {
    enum CodingKeys: String, CodingKey, CaseIterable { case modality, pipeline }

    init(from decoder: Decoder) throws {
        try decoder.refuseUnknownKeys(CodingKeys.self, in: "towers[]")
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let name = try c.decode(String.self, forKey: .modality)
        guard let modality = EmbeddingModality(rawValue: name) else {
            throw EmbeddingError.invalidMetadata("unknown modality '\(name)'")
        }
        self.modality = modality
        pipeline = try c.decode([String].self, forKey: .pipeline)
    }

    /// Contract modalities this runtime does not run yet. Their towers are skipped, so the rest of
    /// the bundle runs: a CLAP bundle runs text.
    static let notYetRun: Set<String> = ["audio"]

    /// The towers of `list`, without those of `notYetRun` modalities.
    static func decodeRunnable(_ list: UnkeyedDecodingContainer) throws -> [Tower] {
        var list = list
        var towers: [Tower] = []
        while !list.isAtEnd {
            let entry = try list.superDecoder()
            let modality = try entry.container(keyedBy: CodingKeys.self).decode(String.self, forKey: .modality)
            if !notYetRun.contains(modality) { towers.append(try Tower(from: entry)) }
        }
        return towers
    }
}

/// A coding key for any string, used to list the keys of a JSON object.
struct AnyCodingKey: CodingKey {
    var stringValue: String
    var intValue: Int? { nil }
    init(stringValue: String) { self.stringValue = stringValue }
    init?(intValue: Int) { nil }
}

extension Decoder {
    /// Throws `EmbeddingError.invalidMetadata` if the object here has a key `Keys` does not list.
    func refuseUnknownKeys<Keys: CodingKey & CaseIterable>(_ keys: Keys.Type, in path: String) throws {
        let known = Set(Keys.allCases.map(\.stringValue))
        let unknown = try container(keyedBy: AnyCodingKey.self).allKeys.map(\.stringValue)
            .filter { !known.contains($0) }.sorted()
        if !unknown.isEmpty {
            throw EmbeddingError.invalidMetadata("\(path): unknown keys \(unknown)")
        }
    }
}

extension KeyedDecodingContainer {
    /// Decodes a string enum, throwing `invalidMetadata` with the value when it is unknown.
    func decodeEnum<T: RawRepresentable>(_ type: T.Type, forKey key: Key) throws -> T where T.RawValue == String {
        let raw = try decode(String.self, forKey: key)
        guard let value = T(rawValue: raw) else {
            throw EmbeddingError.invalidMetadata("unknown \(key.stringValue) '\(raw)'")
        }
        return value
    }
}

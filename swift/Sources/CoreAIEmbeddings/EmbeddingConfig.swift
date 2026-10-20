// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreAIVectorIndex
import Foundation

/// The `embedding` block of an embedding bundle's metadata.json (mirrors contract.py).
struct EmbeddingConfig: Decodable, Sendable, Equatable {
    var space: VectorSpace
    var dimension: Int
    var matryoshkaDimensions: [Int]?
    var renormalizeAfterTruncation: Bool?
    var normalized: Bool?
    var metric: String?
    var precision: Precision?
    var components: [String: Component]?
    var towers: [Tower]
    var text: Text?
    var merge: Merge?
    var preprocessing: Preprocessing?

    struct Precision: Decodable, Sendable, Equatable {
        var compute: String?
        var forbidden: [String]?
        var computeUnits: [String]?
    }

    /// One static `(batch, length)` trace of a component.
    struct FunctionSpec: Decodable, Sendable, Equatable {
        var function: String
        var batch: Int
        var length: Int
    }

    /// How an encoder takes its attention mask.
    enum MaskEncoding: String, Decodable, Sendable, Equatable {
        case attentionMask = "attention_mask"
        case additiveBias = "additive_bias"
    }

    /// One asset of the bundle: a single function or one function per static `(batch, length)`.
    struct Component: Decodable, Sendable, Equatable {
        var function: String?
        var functions: [FunctionSpec]?
        var inputs: [String]?
        var output: String?
        var computeUnits: [String]?
        var maskEncoding: MaskEncoding?
    }

    /// A user-facing entry point for one modality.
    struct Tower: Decodable, Sendable, Equatable {
        var modality: String
        var pipeline: [String]
        var inputs: [String: String]?
        var output: String?
        var batchSizes: [Int]?
        var lengthBuckets: [Int]?
        var maxBatchTokens: Int?
    }

    struct SpecialTokens: Decodable, Sendable, Equatable {
        var bos: Bool?
        var eos: Bool?
        var bosTokenId: Int?
        var eosTokenId: Int?
        var padTokenId: Int?
    }

    struct Truncation: Decodable, Sendable, Equatable {
        var keepEos: Bool?
        var maxTokens: Int?
    }

    struct Text: Decodable, Sendable, Equatable {
        var tokenizer: String?
        var paddingSide: String?
        var specialTokens: SpecialTokens?
        var truncation: Truncation?
        var modelMaxTokens: Int?
        var prompts: [String: String]?
        var defaultPromptName: String?
        var documentTemplate: String?
    }

    struct Merge: Decodable, Sendable, Equatable {
        var placeholderIds: [String: Int]?
        var tokenIds: [String: Int]?
    }

    struct ImagePreprocessing: Decodable, Sendable, Equatable {
        var patchSize: Int?
        var rescaleFactor: Float?
        var normalize: Bool?
        var maxSoftTokens: Int?
        var poolingKernelSize: Int?
        var imageMean: [Float]?
        var imageStd: [Float]?
    }

    struct Preprocessing: Decodable, Sendable, Equatable {
        var image: ImagePreprocessing?
    }

    // MARK: - Accessors

    /// Allowed Matryoshka widths, native first.
    var widths: [Int] {
        guard let widths = matryoshkaDimensions, !widths.isEmpty else { return [dimension] }
        return widths
    }

    var renormalize: Bool { renormalizeAfterTruncation ?? true }

    func component(_ name: String) -> Component? { components?[name] }

    /// The component's own compute units, else the bundle's.
    func computeUnits(of component: String) -> [String]? {
        self.component(component)?.computeUnits ?? precision?.computeUnits
    }

    /// Towers whose modality this runtime knows; unknown modalities are skipped.
    var knownTowers: [(modality: EmbeddingModality, tower: Tower)] {
        towers.compactMap { tower in EmbeddingModality(rawValue: tower.modality).map { ($0, tower) } }
    }

    func tower(for modality: EmbeddingModality) -> Tower? {
        knownTowers.first { $0.modality == modality }?.tower
    }

    /// Structural checks that need no assets.
    func validate() throws {
        guard dimension > 0 else { throw EmbeddingError.invalidBundle("dimension must be > 0") }
        guard widths.first == dimension else {
            throw EmbeddingError.invalidBundle("matryoshka_dimensions must start with the native dimension")
        }
        guard widths.allSatisfy({ (1...dimension).contains($0) }) else {
            throw EmbeddingError.invalidBundle("matryoshka_dimensions must be in 1...\(dimension)")
        }
        guard !knownTowers.isEmpty else { throw EmbeddingError.invalidBundle("no supported towers declared") }
        if let metric, metric != "cosine" {
            throw EmbeddingError.invalidBundle("unsupported metric '\(metric)' (supported: cosine)")
        }
        if let compute = precision?.compute, (precision?.forbidden ?? []).contains(compute) {
            throw EmbeddingError.invalidBundle("compute precision '\(compute)' is listed as forbidden")
        }
        for name in Set(knownTowers.flatMap(\.tower.pipeline)).sorted() {
            _ = try ComputeUnit.require(computeUnits(of: name), component: name)
        }
    }
}

/// An embedding bundle: the common `ModelBundle` plus the decoded and validated `embedding` block.
struct EmbeddingBundle: Sendable {
    let bundle: ModelBundle
    let config: EmbeddingConfig

    init(at url: URL) throws {
        let bundle: ModelBundle
        do {
            bundle = try ModelBundle(at: url)
        } catch let error as ModelBundle.BundleError {
            throw EmbeddingError.invalidBundle(error.description)
        }
        try self.init(bundle: bundle)
    }

    init(bundle: ModelBundle) throws {
        guard bundle.kind == .embedding else {
            throw EmbeddingError.invalidBundle("expected bundle kind embedding, got \(bundle.kind)")
        }
        self.bundle = bundle
        self.config = try Self.decode(bundle.raw)
    }

    /// Decode and validate the `embedding` block of raw metadata.json bytes.
    static func decode(_ raw: Data) throws -> EmbeddingConfig {
        struct Envelope: Decodable { let embedding: EmbeddingConfig? }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let envelope: Envelope
        do {
            envelope = try decoder.decode(Envelope.self, from: raw)
        } catch {
            throw EmbeddingError.invalidBundle("malformed embedding metadata: \(error)")
        }
        guard let config = envelope.embedding else {
            throw EmbeddingError.invalidBundle("metadata has no embedding block")
        }
        try config.validate()
        return config
    }

    /// Tokenizer folder (`text.tokenizer`, default `tokenizer/`).
    var tokenizerURL: URL { bundle.bundlePath.appending(path: config.text?.tokenizer ?? "tokenizer") }
}

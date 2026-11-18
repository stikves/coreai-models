// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

/// The `text` block (R6): token ids, truncation and the default prompt.
struct TextConfig: Sendable {
    /// The tokenizer adds BOS; the host does not.
    var addBosToken: Bool
    /// The tokenizer adds EOS; the host does not.
    var addEosToken: Bool
    /// `nil` when the tokenizer adds no BOS (Qwen3-Embedding).
    var bosTokenId: Int32?
    var eosTokenId: Int32
    var padTokenId: Int32
    /// Longest token sequence, special tokens included.
    var maxTokens: Int
    /// Truncation keeps EOS as the last token.
    var keepEos: Bool
    /// Prefix of every text: `prompts[default_prompt_name]`, or empty.
    var prompt: String
}

/// The `image` block (R7).
struct ImageConfig: Sendable {
    enum Strategy: String, Sendable {
        /// Resize the shortest edge to `size`, crop the center square (CLIP).
        case centerCrop = "center_crop"
        /// The largest size with sides divisible by `patchSize * poolingKernelSize` and at most
        /// `maxSoftTokens * poolingKernelSize²` patches, cut into patches (EmbeddingGemma 2).
        case aspectRatioPreserving = "aspect_ratio_preserving"
    }

    /// Patch geometry and image token ids of `aspectRatioPreserving`.
    struct Patches: Sendable {
        var patchSize: Int
        var poolingKernelSize: Int
        var maxSoftTokens: Int
        var imageTokenId: Int32
        var boiTokenId: Int32
        var eoiTokenId: Int32

        /// Patches per image after padding: `maxSoftTokens * poolingKernelSize²`.
        var maxPatches: Int { maxSoftTokens * poolingKernelSize * poolingKernelSize }
        /// Values per patch: `3 * patchSize²`.
        var patchDimension: Int { 3 * patchSize * patchSize }
    }

    var strategy: Strategy
    var mean: [Float]
    var std: [Float]
    /// The crop side of `centerCrop`.
    var size: Int?
    /// The patch geometry of `aspectRatioPreserving`.
    var patches: Patches?
}

// MARK: - Decoding

extension TextConfig: Decodable {
    enum CodingKeys: String, CodingKey, CaseIterable {
        case truncation, prompts
        case embeddedTokenizer = "embedded_tokenizer"
        case addBosToken = "add_bos_token"
        case addEosToken = "add_eos_token"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case padTokenId = "pad_token_id"
        case paddingSide = "padding_side"
        case defaultPromptName = "default_prompt_name"
        // Accepted and not read.
        case modelMaxTokens = "model_max_tokens"
        case documentTemplate = "document_template"
    }

    struct Truncation: Decodable {
        enum CodingKeys: String, CodingKey, CaseIterable {
            case maxTokens = "max_tokens"
            case keepEos = "keep_eos"
        }

        let maxTokens: Int
        let keepEos: Bool

        init(from decoder: Decoder) throws {
            try decoder.refuseUnknownKeys(CodingKeys.self, in: "text.truncation")
            let c = try decoder.container(keyedBy: CodingKeys.self)
            maxTokens = try c.decode(Int.self, forKey: .maxTokens)
            keepEos = try c.decode(Bool.self, forKey: .keepEos)
        }
    }

    init(from decoder: Decoder) throws {
        try decoder.refuseUnknownKeys(CodingKeys.self, in: "text")
        let c = try decoder.container(keyedBy: CodingKeys.self)
        guard try c.decode(Bool.self, forKey: .embeddedTokenizer) else {
            throw EmbeddingError.invalidMetadata("text.embedded_tokenizer must be true")
        }
        guard try c.decode(String.self, forKey: .paddingSide) == "right" else {
            throw EmbeddingError.invalidMetadata("text.padding_side must be 'right'")
        }
        addBosToken = try c.decode(Bool.self, forKey: .addBosToken)
        addEosToken = try c.decode(Bool.self, forKey: .addEosToken)
        bosTokenId = try c.decodeIfPresent(Int32.self, forKey: .bosTokenId)
        eosTokenId = try c.decode(Int32.self, forKey: .eosTokenId)
        padTokenId = try c.decode(Int32.self, forKey: .padTokenId)
        let truncation = try c.decode(Truncation.self, forKey: .truncation)
        maxTokens = truncation.maxTokens
        keepEos = truncation.keepEos
        let prompts = try c.decodeIfPresent([String: String].self, forKey: .prompts) ?? [:]
        let defaultPromptName = try c.decodeIfPresent(String.self, forKey: .defaultPromptName)
        if addBosToken != (bosTokenId != nil) {
            throw EmbeddingError.invalidMetadata("text.add_bos_token must be true exactly when bos_token_id is set")
        }
        if maxTokens < 2 { throw EmbeddingError.invalidMetadata("text.truncation.max_tokens must be >= 2") }
        if let name = defaultPromptName, prompts[name] == nil {
            throw EmbeddingError.invalidMetadata("default_prompt_name '\(name)' is not in prompts")
        }
        prompt = defaultPromptName.flatMap { prompts[$0] } ?? ""
    }
}

extension ImageConfig: Decodable {
    enum CodingKeys: String, CodingKey, CaseIterable {
        case imageStrategy = "image_strategy"
        case imageSize = "image_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case patchSize = "patch_size"
        case poolingKernelSize = "pooling_kernel_size"
        case maxSoftTokens = "max_soft_tokens"
        case imageTokenId = "image_token_id"
        case boiTokenId = "boi_token_id"
        case eoiTokenId = "eoi_token_id"
    }

    init(from decoder: Decoder) throws {
        try decoder.refuseUnknownKeys(CodingKeys.self, in: "image")
        let c = try decoder.container(keyedBy: CodingKeys.self)
        strategy = try c.decodeEnum(Strategy.self, forKey: .imageStrategy)
        mean = try c.decode([Float].self, forKey: .imageMean)
        std = try c.decode([Float].self, forKey: .imageStd)
        guard mean.count == 3, std.count == 3, std.allSatisfy({ $0 != 0 }) else {
            throw EmbeddingError.invalidMetadata("image_mean and image_std must hold 3 values, std non-zero")
        }
        let patchKeys: [CodingKeys] = [
            .patchSize, .poolingKernelSize, .maxSoftTokens, .imageTokenId, .boiTokenId, .eoiTokenId,
        ]
        switch strategy {
        case .centerCrop:
            let extra = patchKeys.filter(c.contains)
            guard extra.isEmpty else {
                throw EmbeddingError.invalidMetadata("center_crop takes no \(extra.map(\.stringValue))")
            }
            let size = try c.decode(Int.self, forKey: .imageSize)
            guard size > 0 else { throw EmbeddingError.invalidMetadata("image_size must be positive") }
            self.size = size
            patches = nil
        case .aspectRatioPreserving:
            guard !c.contains(.imageSize) else {
                throw EmbeddingError.invalidMetadata("aspect_ratio_preserving takes no image_size")
            }
            size = nil
            patches = Patches(
                patchSize: try c.decode(Int.self, forKey: .patchSize),
                poolingKernelSize: try c.decode(Int.self, forKey: .poolingKernelSize),
                maxSoftTokens: try c.decode(Int.self, forKey: .maxSoftTokens),
                imageTokenId: try c.decode(Int32.self, forKey: .imageTokenId),
                boiTokenId: try c.decode(Int32.self, forKey: .boiTokenId),
                eoiTokenId: try c.decode(Int32.self, forKey: .eoiTokenId))
            guard let p = patches, p.patchSize > 0, p.poolingKernelSize > 0, p.maxSoftTokens > 0 else {
                throw EmbeddingError.invalidMetadata(
                    "patch_size, pooling_kernel_size and max_soft_tokens must be positive")
            }
        }
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIVectorIndex
import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAIEmbeddings

/// Shared fixtures for the CoreAIEmbeddings tests.
enum EmbeddingFixtures {
    static let space = VectorSpace(id: "google/embeddinggemma-2", revision: "914f7f89142e33e7")

    /// metadata.json of the text-only (GPU) EmbeddingGemma 2 bundle.
    static func textMetadata() throws -> Data { try resource("embedding_text_metadata") }

    /// metadata.json of the Neural Engine bundle: token table on the GPU, additive-bias encoder and
    /// image encoder on the Neural Engine.
    static func aneMetadata() throws -> Data { try resource("embedding_ane_metadata") }

    static func textConfig() throws -> EmbeddingConfig { try EmbeddingBundle.decode(textMetadata()) }
    static func aneConfig() throws -> EmbeddingConfig { try EmbeddingBundle.decode(aneMetadata()) }

    private static func resource(_ name: String) throws -> Data {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json"))
        return try Data(contentsOf: url)
    }

    /// Gemma-style BPE fixture: `<pad>`=0, `<unk>`=1, `<eos>`=2, `<bos>`=3.
    static var tokenizerFolder: URL {
        get throws { try #require(Bundle.module.url(forResource: "GemmaStyleTokenizer", withExtension: nil)) }
    }

    static func tokenizer() async throws -> any Tokenizer {
        try await AutoTokenizer.from(modelFolder: tokenizerFolder)
    }

    /// `base` (default: the text fixture) re-encoded after `edit` changes its `embedding` block.
    static func editedMetadata(base: Data? = nil, _ edit: (inout [String: Any]) -> Void) throws -> Data {
        var root = try #require(JSONSerialization.jsonObject(with: base ?? textMetadata()) as? [String: Any])
        var block = try #require(root["embedding"] as? [String: Any])
        edit(&block)
        root["embedding"] = block
        return try JSONSerialization.data(withJSONObject: root)
    }

    /// Model over `metadata` (default: the text fixture) with fake components and the fixture tokenizer.
    static func model(
        metadata: Data? = nil, components: [String: FakeEmbeddingComponent]
    ) throws -> EmbeddingModel {
        try EmbeddingTestBundle.model(
            metadata: metadata ?? textMetadata(), components: components, tokenizerFolder: tokenizerFolder)
    }

    /// Text model over a fake `text_encoder`.
    static func textModel(_ component: FakeEmbeddingComponent? = nil) throws -> EmbeddingModel {
        try model(components: ["text_encoder": component ?? FakeEmbeddingComponent(dimension: 768)])
    }
}

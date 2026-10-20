// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreAIVectorIndex
import Foundation
import TestUtilities
import Testing

@testable import CoreAIEmbeddings

@Suite("Embedding metadata")
struct EmbeddingMetadataTests {
    @Test("decodes the exporter's text-tower metadata")
    func decodesTextFixture() throws {
        let raw = try EmbeddingFixtures.textMetadata()
        let bundle = try EmbeddingBundle(bundle: ModelBundle(raw: raw, bundlePath: URL(fileURLWithPath: "bundle")))
        #expect(bundle.bundle.kind == .embedding)
        #expect(bundle.bundle.name == "embeddinggemma-2-text")
        #expect(bundle.bundle.assets["text_encoder"] == "text_encoder.aimodel")
        #expect(bundle.tokenizerURL.lastPathComponent == "tokenizer")

        let c = bundle.config
        #expect(c.space == EmbeddingFixtures.space)
        #expect(c.dimension == 768)
        #expect(c.widths == [768, 512, 256, 128])
        #expect(c.renormalize)
        #expect(c.metric == "cosine")
        #expect(c.precision?.compute == "float32")
        #expect(c.precision?.forbidden == ["float16"])
        #expect(c.precision?.computeUnits == ["gpu"])

        let functions = try #require(c.component("text_encoder")?.functions)
        #expect(functions.count == 8)
        #expect(functions.contains(.init(function: "len128_b32", batch: 32, length: 128)))
        #expect(!functions.contains { $0.length == 2048 && $0.batch == 32 })  // over max_batch_tokens
        #expect(c.computeUnits(of: "text_encoder") == ["gpu"])

        let tower = try #require(c.tower(for: .text))
        #expect(tower.pipeline == ["text_encoder"])
        #expect(tower.inputs == ["input_ids": "int32[B,L]", "attention_mask": "int32[B,L]"])
        #expect(tower.output == "embedding")
        #expect(tower.lengthBuckets == [128, 512, 2048])
        #expect(tower.batchSizes == [1, 8, 32])
        #expect(tower.maxBatchTokens == 16384)
        #expect(c.tower(for: .image) == nil)

        let text = try #require(c.text)
        #expect(text.tokenizer == "tokenizer")
        #expect(text.paddingSide == "right")
        #expect(text.specialTokens?.bos == true && text.specialTokens?.eos == true)
        #expect(text.specialTokens?.bosTokenId == 2 && text.specialTokens?.eosTokenId == 1)
        #expect(text.specialTokens?.padTokenId == 0)
        #expect(text.truncation?.keepEos == true)
        #expect(text.truncation?.maxTokens == 2048)
        #expect(text.modelMaxTokens == 8192)
        #expect(text.prompts?["query"] == "task: search result | query: ")
        #expect(text.defaultPromptName == nil)
        #expect(text.documentTemplate == "title: {title} | text: {text}")
    }

    @Test("ignores unknown keys, skips unknown towers, decodes image blocks")
    func tolerantDecoding() throws {
        let data = try EmbeddingFixtures.editedMetadata { block in
            block["future_field"] = ["anything": [1, 2, 3]]
            var towers = block["towers"] as! [[String: Any]]
            towers[0]["new_tower_key"] = "x"
            towers.append([
                "modality": "image", "pipeline": ["image_encoder", "token_table", "encoder"],
            ])
            towers.append(["modality": "audio", "pipeline": ["audio_encoder"]])
            block["towers"] = towers
            block["merge"] = [
                "placeholder_ids": ["image": 258880], "token_ids": ["boi": 255999, "eoi": 258882],
            ]
            block["preprocessing"] = [
                "image": ["patch_size": 16, "rescale_factor": 0.00392156, "normalize": false, "max_soft_tokens": 280]
            ]
        }
        let c = try EmbeddingBundle.decode(data)
        #expect(c.towers.count == 3)
        #expect(c.knownTowers.map(\.modality) == [.text, .image])
        #expect(c.merge?.placeholderIds?["image"] == 258880)
        #expect(c.merge?.tokenIds?["eoi"] == 258882)
        #expect(c.preprocessing?.image?.patchSize == 16)
        #expect(c.preprocessing?.image?.maxSoftTokens == 280)
    }

    @Test("an empty matryoshka list means the native width only")
    func emptyMatryoshka() throws {
        let data = try EmbeddingFixtures.editedMetadata { $0["matryoshka_dimensions"] = [Int]() }
        #expect(try EmbeddingBundle.decode(data).widths == [768])
    }

    @Test("validation fails closed")
    func invalidBlocks() throws {
        let edits: [(label: String, edit: (inout [String: Any]) -> Void)] = [
            ("a Matryoshka list that skips the native width", { $0["matryoshka_dimensions"] = [512, 256] }),
            ("a width above the native one", { $0["matryoshka_dimensions"] = [768, 1024] }),
            ("a non-cosine metric", { $0["metric"] = "dot" }),
            ("a zero dimension", { $0["dimension"] = 0 }),
            ("no towers", { $0["towers"] = [Any]() }),
        ]
        for (label, edit) in edits {
            let data = try EmbeddingFixtures.editedMetadata(edit)
            #expect(throws: EmbeddingError.self, "\(label)") { try EmbeddingBundle.decode(data) }
        }
    }

    @Test("forbidden compute precision and unsupported compute units are refused")
    func precisionAndUnits() throws {
        let forbidden = try EmbeddingFixtures.editedMetadata { block in
            var p = block["precision"] as! [String: Any]
            p["compute"] = "float16"
            block["precision"] = p
        }
        #expect(throws: EmbeddingError.self) { try EmbeddingBundle.decode(forbidden) }
        let cpu = try EmbeddingFixtures.editedMetadata { block in
            var components = block["components"] as! [String: Any]
            var encoder = components["text_encoder"] as! [String: Any]
            encoder["compute_units"] = ["cpu"]
            components["text_encoder"] = encoder
            block["components"] = components
        }
        #expect(throws: EmbeddingError.self) { try EmbeddingBundle.decode(cpu) }
    }

    @Test("a malformed or missing embedding block is an invalid bundle")
    func malformed() {
        #expect(throws: EmbeddingError.self) { try EmbeddingBundle.decode(Data(#"{"embedding": {"dimension": "x"}}"#.utf8)) }
        #expect(throws: EmbeddingError.self) { try EmbeddingBundle.decode(Data("{}".utf8)) }
    }

    @Test("a non-embedding bundle is refused")
    func kindMismatch() throws {
        let raw = Data(#"{"metadata_version":"0.2","kind":"llm","name":"x","assets":{}}"#.utf8)
        let bundle = try ModelBundle(raw: raw, bundlePath: URL(fileURLWithPath: "bundle"))
        #expect(throws: EmbeddingError.self) { try EmbeddingBundle(bundle: bundle) }
    }
}

@Suite("Embedding math and identity")
struct EmbeddingMathTests {
    let space = EmbeddingFixtures.space

    @Test("Matryoshka truncation keeps the prefix direction and re-normalizes")
    func truncateRenormalize() throws {
        let v = EmbeddingMath.l2Normalize((0..<768).map { Float(sin(Double($0) * 0.3)) })
        let e = Embedding(space: space, vector: v)
        for dims in [512, 256, 128] {
            let t = try e.truncated(to: dims)
            #expect(t.dimension == dims)
            #expect(abs(EmbeddingMath.l2Norm(t.vector) - 1) < 1e-5)
            #expect(EmbeddingMath.cosineSimilarity(t.vector, Array(v.prefix(dims))) > 0.99999)
            #expect(t.space == space)
        }
        #expect(throws: EmbeddingError.self) { try e.truncated(to: 0) }
        #expect(throws: EmbeddingError.self) { try e.truncated(to: 769) }
    }

    @Test("similarity throws on space or dimension mismatch")
    func mismatchErrors() throws {
        let a = Embedding(space: space, vector: [1, 0, 0, 0])
        let otherSpace = Embedding(space: VectorSpace(id: space.id, revision: "other"), vector: [1, 0, 0, 0])
        let shorter = Embedding(space: space, vector: [1, 0])
        #expect(throws: EmbeddingError.spaceMismatch(space, otherSpace.space)) { try a.cosineSimilarity(to: otherSpace) }
        #expect(throws: EmbeddingError.dimensionMismatch(4, 2)) { try a.cosineSimilarity(to: shorter) }
        // Truncating the longer one makes them comparable.
        #expect(try a.truncated(to: 2).cosineSimilarity(to: shorter) > 0.99999)
    }

    @Test("cosine similarity of orthogonal and opposite vectors")
    func cosine() throws {
        let a = Embedding(space: space, vector: [1, 0, 0])
        let b = Embedding(space: space, vector: [-1, 0, 0])
        let c = Embedding(space: space, vector: [0, 1, 0])
        #expect(try a.cosineSimilarity(to: a) > 0.99999)
        #expect(try a.cosineSimilarity(to: b) < -0.99999)
        #expect(abs(try a.cosineSimilarity(to: c)) < 1e-6)
    }

    @Test("vDSP helpers match scalar references")
    func helpers() {
        let a: [Float] = [3, 4]
        #expect(EmbeddingMath.l2Norm(a) == 5)
        #expect(EmbeddingMath.l2Normalize(a) == [0.6, 0.8])
        #expect(EmbeddingMath.l2Normalize([0, 0]) == [0, 0])
        #expect(!EmbeddingMath.isFinite([1, .nan]))
        #expect(EmbeddingMath.cosineSimilarity([1, 0], [1, 0, 0]) == 0)
    }
}

@Suite("Embedding and VectorIndex glue")
struct EmbeddingVectorIndexTests {
    @Test("descriptor from a model, append and search embeddings, refuse another space")
    func indexRoundTrip() async throws {
        let model = try EmbeddingFixtures.textModel()
        let descriptor = try VectorIndexDescriptor(model: model, dimensions: 256)
        #expect(descriptor.dimension == 256)
        #expect(descriptor.space == EmbeddingFixtures.space)
        #expect(throws: EmbeddingError.self) { try VectorIndexDescriptor(model: model, dimensions: 300) }

        var index = try VectorIndex(descriptor: descriptor)
        let docs = try await model.embed(
            [.text("Paris is the capital of France"), .text("the sky is blue")], promptName: "document",
            dimensions: 256)
        try index.append(id: "paris", embedding: docs[0], attributes: ["kind": "text"])
        try index.append(id: "sky", embedding: docs[1])
        #expect(try index.search(docs[1], k: 1).first?.id == "sky")
        #expect(try index.search(docs[1], k: 2, filter: ["kind": "text"]).map(\.id) == ["paris"])

        let foreign = Embedding(space: VectorSpace(id: "other", revision: "1"), vector: docs[0].vector)
        #expect(throws: VectorIndexError.self) { try index.append(id: "x", embedding: foreign) }
    }
}

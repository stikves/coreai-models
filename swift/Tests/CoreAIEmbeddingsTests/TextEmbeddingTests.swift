// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAIEmbeddings

/// The canonical prompt names every embedding bundle defines.
private let canonicalPrompts = [
    "classification", "clustering", "code_retrieval", "document", "fact_checking", "query", "question_answering",
    "sentence_similarity",
]

@Suite("Text preprocessing")
struct TextEmbeddingPreprocessorTests {
    func preprocessor(
        _ edit: ((inout [String: Any]) -> Void)? = nil
    ) throws -> (TextEmbeddingPreprocessor, EmbeddingConfig) {
        let data = try edit.map { try EmbeddingFixtures.editedMetadata($0) } ?? EmbeddingFixtures.textMetadata()
        let config = try EmbeddingBundle.decode(data)
        let tower = try #require(config.tower(for: .text))
        let pre = try TextEmbeddingPreprocessor(
            configuration: config, tower: tower, bosTokenId: 3, eosTokenId: 2, padTokenId: 5)
        return (pre, config)
    }

    @Test("prompt by name plus BOS/EOS from metadata around tokenizer ids")
    func promptAndSpecialTokens() async throws {
        let tok = try await EmbeddingFixtures.tokenizer()
        let (pre, _) = try preprocessor { block in
            var text = block["text"] as! [String: Any]
            text["prompts"] = ["query": "Question: ", "document": "Answer: "]
            block["text"] = text
        }
        let text = "What is the capital of France?"
        let plain = tok.encode(text: text, addSpecialTokens: false).map(Int32.init)
        let prompted = tok.encode(text: "Question: " + text, addSpecialTokens: false).map(Int32.init)
        #expect(prompted.count > plain.count)

        // Fixture ids: bos 2, eos 1, pad 0 win over the tokenizer's.
        let noPrompt = try pre.tokenize(text, promptName: nil, tokenizer: tok)
        #expect(noPrompt.ids == [2] + plain + [1])
        #expect(!noPrompt.truncated)
        #expect(pre.padTokenId == 0)

        let query = try pre.tokenize(text, promptName: "query", tokenizer: tok)
        #expect(query.ids == [2] + prompted + [1])

        #expect(throws: EmbeddingError.unknownPromptName("nope", available: ["document", "query"])) {
            try pre.tokenize(text, promptName: "nope", tokenizer: tok)
        }
    }

    @Test("default_prompt_name applies when no prompt is given")
    func defaultPrompt() async throws {
        let tok = try await EmbeddingFixtures.tokenizer()
        let (pre, _) = try preprocessor { block in
            var text = block["text"] as! [String: Any]
            text["prompts"] = ["query": "Question: "]
            text["default_prompt_name"] = "query"
            block["text"] = text
        }
        let ids = try pre.tokenize("Paris", promptName: nil, tokenizer: tok).ids
        let expected = tok.encode(text: "Question: Paris", addSpecialTokens: false).map(Int32.init)
        #expect(ids == [2] + expected + [1])
    }

    @Test("special tokens are optional per metadata")
    func noSpecialTokens() async throws {
        let tok = try await EmbeddingFixtures.tokenizer()
        let (pre, _) = try preprocessor { block in
            var text = block["text"] as! [String: Any]
            text["special_tokens"] = ["bos": false, "eos": false]
            block["text"] = text
        }
        let ids = try pre.tokenize("Paris", promptName: nil, tokenizer: tok).ids
        #expect(ids == tok.encode(text: "Paris", addSpecialTokens: false).map(Int32.init))
    }

    @Test("right truncation keeps EOS and reports it")
    func truncationKeepsEOS() throws {
        let (pre, _) = try preprocessor { block in
            var text = block["text"] as! [String: Any]
            var trunc = text["truncation"] as! [String: Any]
            trunc["max_tokens"] = 8
            text["truncation"] = trunc
            block["text"] = text
        }
        #expect(pre.maximumSequenceLength == 8)
        let t = try pre.assemble(content: Array(10..<30))
        #expect(t.ids == [2, 10, 11, 12, 13, 14, 15, 1])
        #expect(t.truncated)
        // Fully formed ids are never truncated.
        #expect(throws: EmbeddingError.inputTooLong(tokens: 9, maximum: 8)) {
            try pre.validateTokens(Array(repeating: 5, count: 9))
        }
        #expect(try pre.validateTokens([2, 5, 1]).ids == [2, 5, 1])
    }

    @Test("max length is capped by the largest bucket and model_max_tokens")
    func maximumLength() throws {
        let (pre, _) = try preprocessor()
        #expect(pre.maximumSequenceLength == 2048)
        #expect(pre.lengthBuckets == [128, 512, 2048])
    }

    @Test("smallest bucket >= length; too long throws")
    func bucketing() throws {
        let (pre, _) = try preprocessor()
        #expect(try pre.bucket(for: 2) == 128)
        #expect(try pre.bucket(for: 128) == 128)
        #expect(try pre.bucket(for: 129) == 512)
        #expect(try pre.bucket(for: 2048) == 2048)
        #expect(throws: EmbeddingError.self) { try pre.bucket(for: 2049) }
    }

    @Test("batch plan groups by bucket and picks static (batch, length) functions")
    func planning() throws {
        let (pre, _) = try preprocessor()
        let plans = try pre.plan(lengths: [5, 130, 7, 600])
        #expect(
            plans == [
                .init(indices: [0, 2], batch: 8, length: 128, function: "len128_b8"),
                .init(indices: [1], batch: 1, length: 512, function: "len512_b1"),
                .init(indices: [3], batch: 1, length: 2048, function: "len2048_b1"),
            ])

        let many = try pre.plan(lengths: Array(repeating: 10, count: 37))
        #expect(many.map(\.function) == ["len128_b32", "len128_b8"])
        #expect(many.map(\.indices.count) == [32, 5])
        #expect(many.flatMap(\.indices) == Array(0..<37))

        // 2048 has no b32 function (max_batch_tokens); 9 rows -> b8 + b1.
        let long = try pre.plan(lengths: Array(repeating: 1000, count: 9))
        #expect(long.map(\.function) == ["len2048_b8", "len2048_b1"])
    }

    @Test("dynamic graph plan respects max batch and max_batch_tokens")
    func dynamicPlanning() throws {
        let (pre, _) = try preprocessor { block in
            block["components"] = ["text_encoder": ["function": "main"]]
            var towers = block["towers"] as! [[String: Any]]
            towers[0]["batch_sizes"] = [1, 16]
            towers[0]["max_batch_tokens"] = 1024
            block["towers"] = towers
        }
        #expect(pre.functions.isEmpty)
        let plans = try pre.plan(lengths: Array(repeating: 200, count: 5) + Array(repeating: 10, count: 20))
        // 512-bucket: 1024/512 = 2 rows per call; 128-bucket: min(16, 8) = 8 rows per call.
        #expect(plans.map(\.batch) == [2, 2, 1, 8, 8, 4])
        #expect(plans.allSatisfy { $0.function == nil })
    }

    @Test("right padding with mask; dummy rows attend one token")
    func packing() throws {
        let (pre, _) = try preprocessor()
        let (ids, mask) = pre.pack([[2, 7, 1], [2, 1]], batch: 3, length: 4)
        #expect(ids == [2, 7, 1, 0, 2, 1, 0, 0, 0, 0, 0, 0])
        #expect(mask == [1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    }
}

@Suite("EmbeddingModel with a fake component")
struct EmbeddingModelFakeTests {
    @Test("embeds in order, pads batches, is padding invariant, honors Matryoshka widths")
    func embedText() async throws {
        let component = FakeEmbeddingComponent(dimension: 768)
        let model = try EmbeddingFixtures.textModel(component)
        #expect(model.space == EmbeddingFixtures.space)
        #expect(model.supportedModalities == [.text])
        #expect(model.matryoshkaDimensions == [768, 512, 256, 128])

        let texts = ["Paris", "What is the capital of France?", "London"]
        let batch = try await model.embed(texts.map { .text($0) }, promptName: "query")
        #expect(batch.count == 3)
        let calls = await component.calls
        #expect(calls.count == 1)
        #expect(calls[0].function == "len128_b8")
        #expect(calls[0].shapes["input_ids"] == [8, 128])
        #expect(calls[0].output == "embedding")
        #expect(calls[0].inputs["attention_mask"] != nil && calls[0].inputs["bias"] == nil)

        for (i, text) in texts.enumerated() {
            let alone = try await model.embed(.text(text), promptName: "query")
            #expect(try alone.cosineSimilarity(to: batch[i]) > 0.99999)
            #expect(alone.tokenCount == batch[i].tokenCount)
            #expect(abs(EmbeddingMath.l2Norm(alone.vector) - 1) < 1e-5)
        }
        #expect(try batch[0].cosineSimilarity(to: batch[1]) < 0.9999)

        let short = try await model.embed(.text("Paris"), promptName: "query", dimensions: 128)
        #expect(short.dimension == 128)
        #expect(abs(EmbeddingMath.l2Norm(short.vector) - 1) < 1e-5)
        #expect(try short.cosineSimilarity(to: batch[0].truncated(to: 128)) > 0.99999)
        await #expect(throws: EmbeddingError.unsupportedDimension(100, supported: [768, 512, 256, 128])) {
            try await model.embed(.text("Paris"), dimensions: 100)
        }
    }

    @Test("the eight canonical prompts embed; a prompt changes the vector; an unknown name throws")
    func prompts() async throws {
        let model = try EmbeddingFixtures.textModel()
        #expect(model.promptNames == canonicalPrompts)
        var vectors: [Embedding] = []
        for name in canonicalPrompts { vectors.append(try await model.embed(.text("Paris"), promptName: name)) }
        let plain = try await model.embed(.text("Paris"))
        for vector in vectors { #expect(try vector.cosineSimilarity(to: plain) < 0.9999) }
        await #expect(throws: EmbeddingError.unknownPromptName("STS", available: canonicalPrompts)) {
            try await model.embed(.text("Paris"), promptName: "STS")
        }
    }

    @Test("fully formed token ids bypass tokenization; document formatting fills the template")
    func tokensAndDocuments() async throws {
        let model = try EmbeddingFixtures.textModel()
        let q = try await model.embed(.text("Paris"), promptName: "query")
        let ids = try await model.tokenize("Paris", promptName: "query")
        let viaTokens = try await model.embed(tokens: [ids], dimensions: nil)[0]
        #expect(try viaTokens.cosineSimilarity(to: q) > 0.99999)
        #expect(model.formatDocument(title: nil, text: "body") == "title: none | text: body")
        #expect(model.formatDocument(title: "T", text: "body") == "title: T | text: body")
    }

    @Test("non-finite output is a component failure")
    func nonFinite() async throws {
        let nan = FakeEmbeddingComponent(dimension: 768) { _, inputs in
            [Float](repeating: .nan, count: (inputs["input_ids"]?.shape[0] ?? 1) * 768)
        }
        let model = try EmbeddingFixtures.textModel(nan)
        await #expect {
            try await model.embed(.text("Paris"))
        } throws: { error in
            guard case .componentFailed(let component, _, let unit, _, _) = error as? EmbeddingError else { return false }
            return component == "text_encoder" && unit == "gpu"
        }
    }

    @Test("an input over the largest bucket is refused; empty tokens too")
    func limits() async throws {
        let model = try EmbeddingFixtures.textModel()
        await #expect(throws: EmbeddingError.inputTooLong(tokens: 3000, maximum: 2048)) {
            try await model.embed(tokens: [[Int32](repeating: 10, count: 3000)], dimensions: nil)
        }
        await #expect(throws: EmbeddingError.emptyInput) { try await model.embed(tokens: [[]], dimensions: nil) }
    }

    @Test("image input on a text-only bundle throws unsupportedModality")
    func imageOnTextBundle() async throws {
        let model = try EmbeddingFixtures.textModel()
        let image = try #require(ImageTestSupport.solidImage(width: 64, height: 32))
        await #expect(throws: EmbeddingError.unsupportedModality(.image)) {
            try await model.embed(.image(image))
        }
    }
}

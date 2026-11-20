// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import CoreGraphics
import Foundation
import FoundationModels
import TestUtilities
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// MARK: - Streaming detokenization
//
// Each streaming site turns generated ids into text deltas. Joined, the deltas must equal the decode of
// all the ids at once. These tests script real token ids through every site and compare.
//
// The fixtures are real tokenizers cut down to the tokens these texts use, with their original ids:
// - `ByteLevelEmojiTokenizer`: Qwen3, byte-level BPE. `🐶` is three tokens, the last two continuation
//   bytes (`ĠðŁ`, `Ĳ`, `¶`); `🎉` likewise.
// - `MetaspaceByteFallbackTokenizer`: Mistral v0.3, Metaspace with byte fallback. `🍕` is four
//   `<0xHH>` tokens, a newline is `<0x0A>`, four spaces after it are `▁▁▁` + `▁return`, and `\r` is
//   its own token.
// - `GraphemeByteFallbackTokenizer`: Gemma 4, byte fallback. A combining mark is its own token after
//   the letter (`nai` + U+0308), each regional indicator of a flag is its own token, and `\r` is
//   `<0x0D>`.
// The ids and texts come from the Python `tokenizers` library, on these files and on the full ones.
//
// Failures they catch:
// - Held multi-byte character: a site that keeps only the last token as context, after a character
//   split over several tokens, keeps a lone continuation byte. Every later decode then contains
//   U+FFFD, and the text after that character is never emitted.
// - Character diffs: a combining mark or a second regional indicator extends the previous
//   `Character`, so a diff by `Character` count emits it twice, or emits the line again.
// - CRLF: `"\r\n"` is one `Character`, so after `"\r"` was emitted, a diff by `Character` drops the
//   `"\n"`.
// - Newline reset: decoding `▁▁▁` without the newline before it drops one of its spaces.

/// One scripted generation: token ids and the text Hugging Face decodes them to.
struct StreamingCase: CustomTestStringConvertible, Sendable {
    let name: String
    let fixture: StreamingFixture
    let ids: [Int32]
    let text: String

    var testDescription: String { name }

    /// The part of `text` a FoundationModels session returns as the response: what follows the
    /// reasoning block, if there is one.
    var responseText: String {
        guard let close = text.range(of: "</think>") else { return text }
        return String(text[close.upperBound...])
    }
}

enum StreamingFixture: String, Sendable {
    case byteLevel = "ByteLevelEmojiTokenizer"
    case metaspace = "MetaspaceByteFallbackTokenizer"
    case grapheme = "GraphemeByteFallbackTokenizer"

    /// Vocabulary size of the full tokenizer; the ids keep their original values.
    var vocabSize: Int {
        switch self {
        case .byteLevel: 151_669
        case .metaspace: 32_768
        case .grapheme: 262_144
        }
    }

    func load() async throws -> any Tokenizer {
        let url = try #require(TestResources.url(for: rawValue))
        return try await AutoTokenizer.from(modelFolder: url)
    }
}

enum StreamingCases {
    static let text: [StreamingCase] = [
        StreamingCase(
            name: "byte-level emoji then text", fixture: .byteLevel,
            ids: [37523, 11162, 238, 114, 146110, 835],
            text: "emoji \u{1F436}\u{1F431} end"),
        StreamingCase(
            name: "byte-level emoji inside reasoning", fixture: .byteLevel,
            ids: [
                151667, 198, 40, 1075, 11162, 238, 114, 264, 2696, 624, 151668, 271, 21396, 2618, 11162, 236, 231,
                2814,
            ],
            text: "<think>\nI like \u{1F436} a lot.\n</think>\n\nGreat job \u{1F389} done"),
        StreamingCase(
            name: "byte-fallback emoji then text", fixture: .metaspace,
            ids: [3655, 29473, 1011, 930, 912, 920, 30780, 1105, 1151],
            text: "food \u{1F355}\u{1F680} yum"),
        StreamingCase(
            name: "metaspace CRLF", fixture: .metaspace,
            ids: [9895, 1392, 29569, 781, 4778, 1757],
            text: "Line one\r\nLine two"),
        StreamingCase(
            name: "metaspace indentation after newline", fixture: .metaspace,
            ids: [1569, 1053, 29500, 29481, 2097, 781, 3055, 1372, 1036],
            text: "def f(s):\n    return s"),
        StreamingCase(
            name: "combining marks", fixture: .grapheme,
            ids: [41461, 241151, 560, 24364, 238288],
            text: "nai\u{0308}ve cafe\u{0301}"),
        StreamingCase(
            name: "regional indicators", fixture: .grapheme,
            ids: [27475, 236743, 243721, 243912, 247994, 245707],
            text: "flags \u{1F1FA}\u{1F1F8}\u{1F1EF}\u{1F1F5}"),
        StreamingCase(
            name: "byte-fallback CRLF", fixture: .grapheme,
            ids: [4539, 886, 236761, 251, 107, 4539, 1156, 236761],
            text: "Line one.\r\nLine two."),
    ]

    /// The same failures inside JSON, for the schema-guided strategies. Line breaks and indentation
    /// sit between JSON tokens, where the grammar allows whitespace; it does not allow `\r`, so there
    /// is no CRLF case.
    static let json: [StreamingCase] = [
        StreamingCase(
            name: "JSON, byte-level emoji", fixture: .byteLevel,
            ids: [4913, 64, 788, 330, 37523, 11162, 238, 114, 146110, 835, 9207],
            text: "{\"a\": \"emoji \u{1F436}\u{1F431} end\"}"),
        StreamingCase(
            name: "JSON, indentation and byte-fallback emoji", fixture: .metaspace,
            ids: [
                29519, 781, 3055, 1113, 29476, 2032, 1113, 19740, 29473, 1011, 930, 912, 920, 30780, 1105, 1151, 29507,
                781, 29520,
            ],
            text: "{\n    \"a\": \"food \u{1F355}\u{1F680} yum\"\n}"),
        StreamingCase(
            name: "JSON, combining marks and regional indicators", fixture: .grapheme,
            ids: [
                14937, 236746, 1083, 623, 41461, 241151, 560, 24364, 238288, 236743, 243721, 243912, 247994, 245707,
                25938,
            ],
            text: "{\"a\": \"nai\u{0308}ve cafe\u{0301} \u{1F1FA}\u{1F1F8}\u{1F1EF}\u{1F1F5}\"}"),
    ]

    static let schema = #"{"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}"#
}

/// Compares by Unicode scalar: `String ==` treats canonically equivalent text as equal.
private func expectSameScalars(
    _ streamed: String, _ expected: String, sourceLocation: SourceLocation = #_sourceLocation
) {
    #expect(
        Array(streamed.unicodeScalars) == Array(expected.unicodeScalars),
        "streamed \(streamed.debugDescription), expected \(expected.debugDescription)",
        sourceLocation: sourceLocation)
}

private func concatenated(_ stream: some AsyncSequence<GenerationResult, Error>) async throws -> String {
    var text = ""
    for try await result in stream {
        text += result.text
    }
    return text
}

@Suite("Streaming detokenization")
struct StreamingDetokenizationTests {
    @Test(
        "The fixture decodes the ids to the Hugging Face text",
        arguments: StreamingCases.text + StreamingCases.json)
    func fixtureDecode(_ testCase: StreamingCase) async throws {
        let tokenizer = try await testCase.fixture.load()
        expectSameScalars(tokenizer.decode(tokens: testCase.ids.map(Int.init)), testCase.text)
    }

    @Test("Vanilla decoding streams the full decode", arguments: StreamingCases.text)
    func vanilla(_ testCase: StreamingCase) async throws {
        let tokenizer = try await testCase.fixture.load()
        let eos = try #require(tokenizer.eosTokenId)
        let scripted = testCase.ids + [Int32(eos)]
        let stream = try await VanillaDecodingStrategy().decode(
            from: .tokens([eos]),
            tokenizer: tokenizer,
            inferenceEngine: MockEngine(tokens: scripted, vocabSize: nil),
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: scripted.count),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
        )
        expectSameScalars(try await concatenated(stream), testCase.text)
    }

    @Test("Constrained decoding streams the full decode", arguments: StreamingCases.json)
    func constrained(_ testCase: StreamingCase) async throws {
        let tokenizer = try await testCase.fixture.load()
        let vocabSize = testCase.fixture.vocabSize
        let stream = try await ConstrainedDecodingStrategy(jsonSchema: StreamingCases.schema, vocabSize: vocabSize)
            .decode(
                from: .tokens([1]),
                tokenizer: tokenizer,
                inferenceEngine: MockEngine(tokens: testCase.ids, vocabSize: vocabSize),
                samplingConfiguration: .greedy,
                options: InferenceOptions(maxTokens: testCase.ids.count),
                stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
            )
        expectSameScalars(try await concatenated(stream), testCase.text)
    }

    @Test("Pipelined constrained decoding streams the full decode", arguments: StreamingCases.json)
    func pipelinedConstrained(_ testCase: StreamingCase) async throws {
        let tokenizer = try await testCase.fixture.load()
        let stream = try await PipelinedConstrainedDecodingStrategy(
            jsonSchema: StreamingCases.schema, vocabSize: testCase.fixture.vocabSize
        )
        .decode(
            from: .tokens([1]),
            tokenizer: tokenizer,
            inferenceEngine: MockConstrainedEngine(scriptedTokens: testCase.ids),
            samplingConfiguration: .greedy,
            options: InferenceOptions(maxTokens: testCase.ids.count),
            stopSequences: StopSequences(for: tokenizer, additionalEosTokenIds: [])
        )
        expectSameScalars(try await concatenated(stream), testCase.text)
    }

    @Test("A CoreAILanguageModel session returns the full decode", arguments: StreamingCases.text)
    func languageModelSession(_ testCase: StreamingCase) async throws {
        let tokenizer = try await testCase.fixture.load()
        let eos = try #require(tokenizer.eosTokenId)
        let scripted = testCase.ids + [Int32(eos)]

        let directory = FileManager.default.temporaryDirectory.appending(
            path: "StreamingDetokenizationTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let metadata = """
            {"metadata_version": "0.2", "kind": "llm", "name": "scripted", "assets": {"main": "model.aimodel"},
             "language": {"tokenizer": "scripted", "vocab_size": \(testCase.fixture.vocabSize),
                          "max_context_length": 4096}}
            """
        try metadata.write(to: directory.appending(path: "metadata.json"), atomically: true, encoding: .utf8)
        let bundle = try LanguageModelBundle(at: directory)

        let configuration = CoreAILanguageModel.CoreAIExecutor.Configuration(
            url: directory, variant: nil, kvCacheStrategy: .auto, modelIdentifier: bundle.name,
            samplingConfig: .greedy, vocabSize: bundle.vocabSize, prefillChunkSize: nil,
            prefillChunkThreshold: nil)
        let resources = ModelResources { MockEngine(tokens: scripted, vocabSize: nil) }
        ModelResources.register(resources, for: configuration)
        let model = CoreAILanguageModel(
            configuration: configuration, bundle: bundle, tokenizer: tokenizer, resources: resources,
            maxWhitespaceCount: nil)

        let session = LanguageModelSession(model: model)
        let response = try await session.respond(
            to: "hi", options: GenerationOptions(maximumResponseTokens: scripted.count))
        expectSameScalars(response.content, testCase.responseText)
    }

    @Test("A CoreAIVisionLanguageModel session returns the full decode", arguments: StreamingCases.text)
    func visionLanguageModelSession(_ testCase: StreamingCase) async throws {
        let tokenizer = try await testCase.fixture.load()
        let eos = try #require(tokenizer.eosTokenId)
        let scripted = testCase.ids + [Int32(eos)]
        let model = CoreAIVisionLanguageModel(
            executorConfiguration: CoreAIVLMExecutor.Configuration(
                bundleURL: URL(fileURLWithPath: "/scripted-vlm-\(UUID().uuidString)"),
                engine: ScriptedImageEngine(tokens: scripted, vocabSize: nil),
                tokenizer: tokenizer,
                visionConfig: VisionConfig(imageSize: 1, patchSize: 1, imageTokenCount: 1, imageTokenId: 0),
                additionalStopTokenIds: []))
        let image = try #require(makeSolidCGImage(r: 0, g: 0, b: 255, side: 1))

        let session = LanguageModelSession(model: model)
        let response = try await session.respond(
            options: GenerationOptions(maximumResponseTokens: scripted.count)
        ) {
            Attachment(image)
            "Describe the image."
        }
        expectSameScalars(response.content, testCase.text)
    }
}

/// `MockEngine` that also takes an image: it encodes one placeholder and generates its scripted tokens.
final class ScriptedImageEngine: MockEngine, MultimodalInferenceEngine, @unchecked Sendable {
    var acceptsMultimodalInput: Bool { true }

    func encodeImage(at url: URL) async throws -> InputEmbeddings {
        throw InferenceRuntimeError.invalidState("not used")
    }

    func encodeImage(cgImage: CGImage) async throws -> InputEmbeddings {
        try InputEmbeddings(
            embeddings: NDArray(shape: [1, 1, 2], scalarType: .float16), embeddingPositions: 0..<1)
    }

    func encodeVideo(_ video: VideoInput) async throws -> InputEmbeddings {
        throw InferenceRuntimeError.invalidState("not used")
    }

    func generate(
        with input: InputEmbeddings, tokens: [TokenId], samplingConfiguration: SamplingConfiguration,
        inferenceOptions: InferenceOptions
    ) async throws -> GenerationSequence {
        try await generate(
            with: tokens, samplingConfiguration: samplingConfiguration, inferenceOptions: inferenceOptions)
    }
}

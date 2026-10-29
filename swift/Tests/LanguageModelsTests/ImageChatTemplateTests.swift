//
// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause
//

import Foundation
import Testing
import Tokenizers

@testable import CoreAILanguageModels

// The MinimalSmolVLMTokenizer fixture carries SmolVLM2-256M-Video-Instruct's chat template verbatim.
// That template only renders typed content ([{"type": "image"}, {"type": "text", ...}]); plain
// string content renders an empty user turn.
@Suite("Image chat template")
struct ImageChatTemplateTests {
    static let imageTokenId: Int32 = 126

    private func smolVLMTokenizer() async throws -> any Tokenizer {
        let url = try #require(TestResources.url(for: "MinimalSmolVLMTokenizer"))
        return try await AutoTokenizer.from(modelFolder: url)
    }

    @Test("String content renders no image placeholder with the SmolVLM template")
    func stringContentDropsTheTurn() async throws {
        let tokenizer = try await smolVLMTokenizer()
        let tokens = try tokenizer.applyChatTemplate(messages: [["role": "user", "content": "<image>\nhi"]])
        #expect(!tokens.contains(Int(Self.imageTokenId)))
    }

    @Test("Image chat template falls back to typed content and keeps the prompt")
    func typedContentFallback() async throws {
        let tokenizer = try await smolVLMTokenizer()
        let tokens = try #require(
            PromptUtils.applyImageChatTemplate(text: "hi", imageTokenId: Self.imageTokenId, tokenizer: tokenizer))
        #expect(tokens.filter { $0 == Self.imageTokenId }.count == 1)
        #expect(
            tokenizer.decode(tokens: tokens.map { Int($0) }, skipSpecialTokens: false)
                == "<|im_start|>User:<image>hi<end_of_utterance>\nAssistant:")
    }

    @Test("Expanded SmolVLM prompt wraps the image run like the HF processor")
    func expandedPromptMatchesProcessorLayout() async throws {
        let tokenizer = try await smolVLMTokenizer()
        let config = VisionConfig(
            imageSize: 512, patchSize: 16, imageTokenCount: 2, imageTokenId: Self.imageTokenId,
            imagePrefixTokenIds: [127, 128], imageSuffixTokenIds: [127])
        let templated = try #require(
            PromptUtils.applyImageChatTemplate(text: "hi", imageTokenId: Self.imageTokenId, tokenizer: tokenizer))
        let tokens = try #require(config.expandImagePlaceholder(in: templated, count: 2))
        #expect(
            tokenizer.decode(tokens: tokens.map { Int($0) }, skipSpecialTokens: false)
                == "<|im_start|>User:<fake_token_around_image><global-img><image><image>"
                + "<fake_token_around_image>hi<end_of_utterance>\nAssistant:")
    }

    @Test("CoreAIVLMExecutor (FoundationModels path) builds the HF SmolVLM layout, not the Qwen fallback")
    func libraryPromptBuilderUsesTemplate() async throws {
        let tokenizer = try await smolVLMTokenizer()
        let config = VisionConfig(
            imageSize: 512, patchSize: 16, imageTokenCount: 2, imageTokenId: Self.imageTokenId,
            imagePrefixTokenIds: [127, 128], imageSuffixTokenIds: [127])
        let tokens = CoreAIVLMExecutor.buildPromptTokens(
            userText: "hi", imageTokenCount: 2, visionConfig: config, tokenizer: tokenizer)
        #expect(
            tokenizer.decode(tokens: tokens.map { Int($0) }, skipSpecialTokens: false)
                == "<|im_start|>User:<fake_token_around_image><global-img><image><image>"
                + "<fake_token_around_image>hi<end_of_utterance>\nAssistant:")
    }
}

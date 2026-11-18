// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIEmbeddings

@Suite struct EmbeddingBundleTests {
    @Test func clip() throws {
        let block = try Fixtures.block("clip-vit-b32")
        #expect(block.space.id == "openai/clip-vit-base-patch32")
        #expect(block.dimension == 512)
        #expect(block.logitScale.map { abs($0 - 100) < 1e-3 } == true)
        #expect(block.matryoshkaDimensions == [512])
        #expect(block.text.maxTokens == 77 && block.text.padTokenId == 49407 && block.text.prompt.isEmpty)
        let image = try #require(block.image)
        #expect(image.strategy == .centerCrop && image.size == 224 && image.patches == nil)
        let text = try #require(block.components["text_encoder"])
        #expect(text.inputs["input_ids"] == (try TensorSpec(parsing: "int32[B,L]")))
        #expect(text.functions == [FunctionEntry(function: "text_encoder", batch: 1...64, length: 77...77)])
        #expect(block.tower(for: .image)?.pipeline == ["image_encoder"])
    }

    /// The audio tower is skipped until audio input lands; the text tower runs.
    @Test func clapRunsText() throws {
        let block = try Fixtures.block("clap-htsat")
        #expect(block.towers.map(\.modality) == [.text])
        #expect(block.components["audio_encoder"] != nil)
    }

    @Test func embeddingGemma2() throws {
        let block = try Fixtures.block("embeddinggemma-2")
        #expect(block.matryoshkaDimensions == [768, 512, 256, 128])
        #expect(block.renormalizeAfterTruncation)
        #expect(block.logitScale == nil)
        #expect(block.text.prompt.isEmpty)
        let patches = try #require(block.image?.patches)
        #expect(patches.maxPatches == 2520 && patches.patchDimension == 768)
        #expect(block.tower(for: .image)?.pipeline == ["image_encoder", "token_table", "encoder"])
        #expect(block.components["token_table"]?.functions.first?.length == 2...8192)
    }

    @Test func embeddingGemma2NeuralEngine() throws {
        let block = try Fixtures.block("embeddinggemma-2-ios")
        let encoder = try #require(block.components["encoder"])
        #expect(encoder.computeUnits == [.neuralEngine])
        #expect(encoder.bias == AdditiveBias(masked: -30000, unmasked: 0))
        #expect(encoder.inputs["bias"]?.dtype == .float16)
        #expect(block.tower(for: .text)?.pipeline == ["token_table", "encoder"])
    }

    @Test func fusedBundleKeepsSimilarity() throws {
        let block = try Fixtures.block("clip-vit-b32-fused")
        #expect(Set(block.components.values.map(\.asset)) == ["model"])
        #expect(block.components["similarity"]?.inputs["pixel_values"]?.shape.first == .secondBatch)
    }

    // MARK: - Refusals (R5)

    /// (label, message, edit) on `clip-vit-b32`.
    static let refusals: [(String, String, @Sendable (inout [String: Any]) -> Void)] = [
        ("unknown block key", "unknown keys", { $0["pooling"] = "mean" }),
        ("unknown space key", "space: unknown keys", { $0.edit("space") { $0["license"] = "mit" } }),
        ("missing space", "missing 'space'", { $0["space"] = nil }),
        (
            "unknown component key", "unknown keys",
            { $0.edit("components.text_encoder") { $0["dynamic_batch"] = true } }
        ),
        ("unknown text key", "unknown keys", { $0.edit("text") { $0["special_tokens"] = [:] } }),
        ("unknown image key", "unknown keys", { $0.edit("image") { $0["resize"] = "bicubic" } }),
        (
            "unknown function key", "unknown keys",
            {
                $0.edit("components.image_encoder") { $0["functions"] = [["function": "f", "batch": 1, "shape": 2]] }
            }
        ),
        ("unknown metric", "unknown metric", { $0["metric"] = "dot" }),
        ("unknown precision", "unknown compute_precision", { $0["compute_precision"] = "int8" }),
        ("unknown strategy", "unknown image_strategy", { $0.edit("image") { $0["image_strategy"] = "pad" } }),
        (
            "unknown compute unit", "unknown compute unit",
            {
                $0.edit("components.text_encoder") { $0["compute_units"] = ["tpu"] }
            }
        ),
        (
            "unknown modality", "unknown modality",
            {
                $0["towers"] = [
                    ["modality": "text", "pipeline": ["text_encoder"]],
                    ["modality": "video", "pipeline": ["image_encoder"]],
                ]
            }
        ),
        (
            "bad tensor spec", "bad tensor spec",
            {
                $0.edit("components.text_encoder") {
                    $0["inputs"] = ["input_ids": "int64[B,L]", "attention_mask": "int32[B,L]"]
                }
            }
        ),
        (
            "bad axis", "bad axis",
            {
                $0.edit("components.text_encoder") {
                    $0["inputs"] = ["input_ids": "int32[B,T]", "attention_mask": "int32[B,L]"]
                }
            }
        ),
        ("missing asset", "not in assets", { $0.edit("components.text_encoder") { $0["asset"] = "nope" } }),
        (
            "missing component", "unknown components",
            {
                $0["towers"] = [
                    ["modality": "text", "pipeline": ["text_encoder"]], ["modality": "image", "pipeline": ["vision"]],
                ]
            }
        ),
        ("image tower without block", "image block", { $0["image"] = nil }),
        (
            "two text towers", "one text tower",
            {
                $0["towers"] = [
                    ["modality": "text", "pipeline": ["text_encoder"]],
                    ["modality": "text", "pipeline": ["text_encoder"]],
                ]
            }
        ),
        ("empty functions", "must not be empty", { $0.edit("components.text_encoder") { $0["functions"] = [] } }),
        (
            "duplicate function", "duplicate function",
            {
                $0.edit("components.image_encoder") {
                    $0["asset"] = "text_encoder"
                    $0["functions"] = [["function": "text_encoder", "batch": 1]]
                }
            }
        ),
        (
            "bad range", "bad range",
            {
                $0.edit("components.image_encoder") {
                    $0["functions"] = [["function": "f", "batch": ["min": 4, "max": 2]]]
                }
            }
        ),
        (
            "pipeline without embedding", "no embedding output",
            {
                $0.edit("components.image_encoder") { $0["outputs"] = ["features": "float32[B,512]"] }
            }
        ),
        ("center_crop without size", "image_size", { $0.edit("image") { $0["image_size"] = nil } }),
        ("left padding", "padding_side", { $0.edit("text") { $0["padding_side"] = "left" } }),
        (
            "default prompt not in prompts", "default_prompt_name",
            { $0.edit("text") { $0["default_prompt_name"] = "query" } }
        ),
        ("matryoshka not native first", "matryoshka_dimensions", { $0["matryoshka_dimensions"] = [256, 512] }),
        ("patch keys on center_crop", "center_crop takes no", { $0.edit("image") { $0["patch_size"] = 16 } }),
    ]

    @Test(arguments: refusals.indices)
    func refuses(_ index: Int) throws {
        let (label, message, edit) = Self.refusals[index]
        let error = #expect(throws: EmbeddingError.self, "\(label)") {
            try Fixtures.block("clip-vit-b32", editing: edit)
        }
        #expect(error?.description.contains(message) == true, "\(label): \(error?.description ?? "no error")")
    }

    @Test func refusesBiasWithoutInput() throws {
        #expect(throws: EmbeddingError.self) {
            try Fixtures.block("embeddinggemma-2-ios") {
                $0.edit("components.encoder") {
                    $0["inputs"] = ["inputs_embeds": "float16[B,L,512]", "attention_mask": "int32[B,L]"]
                }
            }
        }
    }

    @Test func textWithoutBos() throws {
        let block = try Fixtures.block("clip-vit-b32-text") {
            $0.edit("text") {
                $0["bos_token_id"] = NSNull()
                $0["add_bos_token"] = false
            }
        }
        #expect(block.text.bosTokenId == nil && !block.text.addBosToken)
        #expect(throws: EmbeddingError.self) {
            try Fixtures.block("clip-vit-b32-text") { $0.edit("text") { $0["bos_token_id"] = NSNull() } }
        }
        #expect(throws: EmbeddingError.self) {
            try Fixtures.block("clip-vit-b32-text") { $0.edit("text") { $0["add_bos_token"] = false } }
        }
    }
}

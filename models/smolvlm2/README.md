# SmolVLM2

HuggingFace's SmolVLM2 vision-language models for on-device inference via Core AI.

## Supported Models

| Model                        | Parameters | Vision encoder        | Image tokens | macOS | iOS |
| ---------------------------- | ---------- | --------------------- | ------------ | ----- | --- |
| SmolVLM2-256M-Video-Instruct | 256M       | SigLIP base, 512 px   | 64           | Yes   | No  |
| SmolVLM2-500M-Video-Instruct | 500M       | SigLIP base, 512 px   | 64           | Yes   | No  |
| SmolVLM2-2.2B-Instruct       | 2.2B       | SigLIP so400m, 384 px | 81           | Yes   | No  |

Each image is encoded as SmolVLM2's single global view and wrapped in
`<fake_token_around_image><global-img>` … `<fake_token_around_image>`, as the HuggingFace processor does
with image splitting off. The bundles take single images; the 256M and 500M checkpoints are the
"Video-Instruct" releases.

## Export models

```bash
uv run coreai.vlm.export smolvlm2-256m
uv run coreai.vlm.export smolvlm2-500m
uv run coreai.vlm.export smolvlm2-2.2b
```

Each export writes one bundle (`kind=vlm`) holding the FP16 text decoder, the token-embedding lookup, the
vision encoder and connector, and the tokenizer. See the [VLM README](../vlm/README.md#bundle-layout) for the
layout.

**Options:**

```bash
# Text decoder and embedding only, for text-only use
uv run coreai.vlm.export smolvlm2-256m --skip-vision

# Compress the text decoder (the vision encoder and embedding stay full precision)
uv run coreai.vlm.export smolvlm2-256m --compression 4bit

# KV cache context length (default: 4096)
uv run coreai.vlm.export smolvlm2-256m --max-context-length 8192

# Custom output directory
uv run coreai.vlm.export smolvlm2-256m --output-dir path/to/exports
```

## Run a Core AI Vision-Language Model

### In your macOS applications via Foundation Models

```swift
import FoundationModels
import CoreAILanguageModels

let model = try await CoreAIVisionLanguageModel(resourcesAt: modelURL)

let session = LanguageModelSession(model: model)

let response = try await session.respond {
    Prompt {
        Attachment(image)
        "What is in this image?"
    }
}

print(response)
```

`CoreAIVisionLanguageModel` takes one image attachment per prompt and decodes greedily.

### On your Mac using built-in Command Line Tool

```bash
swift run -c release llm-runner --model path/to/exported_model_folder \
  --image path/to/photo.jpg --prompt "Describe this image."
```

## Evaluation

The exported model definitions match HuggingFace on FP32 weights: the text decoder's next-token NLL is the
same with 100% top-1 agreement, and the vision encoder and connector outputs are bit-identical.

Text-only perplexity score on the [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level)
dataset computed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md)
with the Core AI PyTorch models.

| Model | Compression      | BPW   | Platform | Perplexity |
| ----- | ---------------- | ----- | -------- | ---------- |
| 256M  | none (`float16`) | 16.00 | macOS    | 28.29      |
| 500M  | none (`float16`) | 16.00 | macOS    | 19.74      |
| 2.2B  | none (`float16`) | 16.00 | macOS    | 13.32      |

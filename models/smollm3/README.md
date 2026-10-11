# SmolLM3

HuggingFace's SmolLM3 for on-device inference via Core AI.

SmolLM3 is a Qwen2-style decoder (GQA with 4 KV heads, tied embeddings, 128k vocabulary)
that skips rotary position embeddings on every 4th layer (NoPE). The instruct model supports
switchable reasoning: thinking is on by default; disable it with `/no_think` in the system
prompt or `enable_thinking=False` when applying the chat template.

SmolLM3 is the successor to [SmolLM2](../smollm2), not a drop-in replacement: it comes in one size,
uses a different tokenizer and chat template, and is macOS-only here. Both families stay supported.

|                | SmolLM2                       | SmolLM3                        |
| -------------- | ----------------------------- | ------------------------------ |
| Sizes          | 135M, 360M, 1.7B              | 3B                             |
| Context        | 8K                            | 64K                            |
| Vocabulary     | 49K                           | 128K                           |
| Reasoning mode | No                            | Yes (`/no_think` to disable)   |
| Platforms      | macOS, iOS                    | macOS                          |

## Supported Models

| Model           | Parameters | macOS | iOS |
| --------------- | ---------- | ----- | --- |
| SmolLM3-3B      | 3B         | Yes   | No  |
| SmolLM3-3B-Base | 3B         | Yes   | No  |

## Export models

```bash
# Instruct (INT4 with INT8 on blocks 0-1 and lm_head; FP16 embedding)
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B

# Base
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B-Base
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B --compression none

# Smaller KV cache
uv run coreai.llm.export HuggingFaceTB/SmolLM3-3B --max-context-length 8192
```

The KV cache is about 72 KB per token in FP16, so the full 64k context needs about 4.8 GB.

## Run a Core AI Language Model

### In your macOS applications via Foundation Models

```swift
import FoundationModels
import CoreAILanguageModels

let model = try await CoreAILanguageModel(resourcesAt: modelURL)

let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "What is quantum computing?")

print(response)
```

### On your Mac using built-in Command Line Tool

```bash
swift run -c release llm-runner --model path/to/exported_model_folder --prompt "Hello"
```

## Benchmark a Core AI Language Model

```bash
swift run -c release llm-benchmark --model path/to/exported_model_folder
```

Defaults: 512 prompt tokens, 1024 generation tokens, 5 trials. Override with `-p`, `-g`, and `-n`.

## Evaluation

Perplexity on [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) and
tinyMMLU accuracy, computed with the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
on the Core AI PyTorch models (context 16384, `float16`). BPW is the exported bundle size over the parameter
count; the mixed recipe stores a separate INT8 `lm_head` next to the FP16 embedding.

| Model     | Compression                                  | BPW   | Platform | Perplexity | tinyMMLU |
| --------- | -------------------------------------------- | ----- | -------- | ---------- | -------- |
| 3B        | none (`float16`)                             | 16.07 | macOS    | 10.67      | 0.599    |
| 3B        | [INT4, INT8 blocks 0-1 + lm_head][mixed]     | 6.46  | macOS    | 11.30      | 0.588    |
| 3B-Base   | none (`float16`)                             | 16.07 | macOS    | 10.14      | 0.615    |
| 3B-Base   | [INT4, INT8 blocks 0-1 + lm_head][mixed]     | 6.46  | macOS    | 10.80      | 0.599    |

[mixed]: smollm3_mixed_4bit_8bit.yaml

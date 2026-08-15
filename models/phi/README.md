# Phi

Microsoft's Phi-3, Phi-3.5, and Phi-4 mini models for on-device inference via Core AI.

## Supported Models

| Model                  | Parameters | Context | macOS | iOS |
| ---------------------- | ---------- | ------- | ----- | --- |
| Phi-4-mini-instruct    | 3.8B       | 131072  | Yes   | No  |
| Phi-3.5-mini-instruct  | 3.8B       | 131072  | Yes   | No  |
| Phi-3-mini-4k-instruct | 3.8B       | 4096    | Yes   | Yes |

## Setup to export models

If you haven't installed `uv`, install it by
```bash
brew install uv
```

## Export models

```bash
# Phi-4-mini (recommended)
uv run coreai.llm.export microsoft/Phi-4-mini-instruct

# Phi-3.5-mini
uv run coreai.llm.export microsoft/Phi-3.5-mini-instruct

# Phi-3-mini (4K context)
uv run coreai.llm.export microsoft/Phi-3-mini-4k-instruct

# Phi-3-mini iOS
uv run coreai.llm.export microsoft/Phi-3-mini-4k-instruct --platform iOS
```

**Options:**

```bash
# Full precision
uv run coreai.llm.export microsoft/Phi-4-mini-instruct --compression none

# Custom output directory
uv run coreai.llm.export microsoft/Phi-4-mini-instruct --output-dir ./my-models/

# Preview resolved config without exporting
uv run coreai.llm.export microsoft/Phi-4-mini-instruct --dry-run
```

## Run a Core AI Language Model

### In your iOS and macOS applications via Foundation Models

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

Perplexity score on the [`WikiText-2`](https://huggingface.co/datasets/EleutherAI/wikitext_document_level) dataset computed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/wikitext/README.md) with the Core AI PyTorch models.

| Model        | Compression                               | Platform | Perplexity Score |
| ------------ | ----------------------------------------- | -------- | ---------------- |
| Phi-3-mini   | none (`float16`)                          | macOS    | 9.47             |
| Phi-3-mini   | [INT4 with FP16 embedding][phi-4bit-yaml] | macOS    | 11.24            |
| Phi-3.5-mini | none (`float16`)                          | macOS    | 9.98             |
| Phi-3.5-mini | [INT4 with FP16 embedding][phi-4bit-yaml] | macOS    | 12.04            |
| Phi-4-mini   | none (`float16`)                          | macOS    | 11.12            |
| Phi-4-mini   | [INT4 with FP16 embedding][phi-4bit-yaml] | macOS    | 12.80            |

The embedding is kept at FP16 because the embedding tensor is large relative to the
model — excluding it from INT4 improves generation quality.

[phi-4bit-yaml]: phi_4bit_embedding_excluded.yaml

## Architecture Notes

- All three models share the `phi3` architecture class
- Fused gate+up MLP (`gate_up_proj` chunked into gate and up)
- **Phi-4**: GQA (32 Q / 8 KV heads) with separate Q/K/V projections, LongRoPE,
  `partial_rotary_factor=0.75`
- **Phi-3.5**: MHA (32/32 heads) with fused QKV, LongRoPE
- **Phi-3**: MHA with fused QKV, sliding window attention (window=2047)

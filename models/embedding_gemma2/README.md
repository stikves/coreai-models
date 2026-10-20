# EmbeddingGemma 2

EmbeddingGemma 2 is a multimodal retrieval embedding model that maps text and images into one shared space: every input becomes a single 768-dimensional, L2-normalized vector, compared with cosine similarity. The leading 512, 256, or 128 dimensions form valid Matryoshka embeddings (truncated and re-normalized) for smaller indexes.[^1]

`coreai.embedding.export` writes a `kind: embedding` bundle that the `CoreAIEmbeddings` Swift library and the `embedding-search` tool run.

## Supported models

| Short name         | HuggingFace ID            | Parameters                    | Platforms  |
| ------------------ | ------------------------- | ----------------------------- | ---------- |
| `embeddinggemma-2` | `google/embeddinggemma-2` | 744M (text 271M + image 168M) | macOS, iOS |

The checkpoint also contains an audio encoder, which the exporter does not load.

## Setup

If you haven't installed `uv`, install it by

```bash
brew install uv
```

## Export

```sh
uv run coreai.embedding.export embeddinggemma-2                 # macOS (GPU): text + image
uv run coreai.embedding.export embeddinggemma-2 --platform iOS  # Neural Engine
uv run coreai.embedding.export embeddinggemma-2 --skip-vision   # text only
uv run coreai.embedding.export embeddinggemma-2 --dry-run       # print the plan
```

The macOS bundle is a GPU bundle in `float32`; `bfloat16` is also accepted. `float16` is refused because the model overflows in float16. The iOS bundle is a Neural Engine bundle in `float16`, using scale-invariant norms, and it also runs on a Mac.

Bundles go to `exports/` by default: `exports/embeddinggemma-2` (text and image), `exports/embeddinggemma-2-text` (`--skip-vision`), `exports/embeddinggemma-2-ios`, and `exports/embeddinggemma-2-ios-text`. Pass `--output-dir` to write elsewhere.

**Options**

| Flag                                             | Description                                                   | Default    |
| ------------------------------------------------ | ------------------------------------------------------------- | ---------- |
| `--platform {macOS,iOS}`                         | Target platform: GPU bundle or Neural Engine bundle           | `macOS`    |
| `--skip-vision`                                  | Export the text tower only                                    | off        |
| `--compute-precision {float32,bfloat16,float16}` | Compute precision for the macOS bundle (`float16` is refused) | `float32`  |
| `--output-dir PATH`                              | Directory for the bundle                                      | `exports/` |
| `--overwrite`                                    | Replace an existing bundle                                    | off        |
| `--dry-run`                                      | Print the export plan without writing                         | off        |
| `--list-models`                                  | List the registry models and exit                             | off        |
| `--include-debug-info`                           | Embed full debug information in the `.aimodel` files          | off        |
| `-v`, `--verbose`                                | Show more console output                                      | off        |

### Bundle layout

| File                    | macOS       | iOS | Role                                                                                       |
| ----------------------- | ----------- | --- | ------------------------------------------------------------------------------------------ |
| `metadata.json`         | yes         | yes | Bundle contract                                                                            |
| `tokenizer/`            | yes         | yes | Tokenizer files                                                                            |
| `text_encoder.aimodel`  | yes         | -   | Text tower                                                                                 |
| `token_table.aimodel`   | with images | yes | Token ids to embeddings (GPU)                                                              |
| `encoder.aimodel`       | with images | yes | Transformer and pooling: the image tower on macOS; both towers on the Neural Engine on iOS |
| `image_encoder.aimodel` | yes         | yes | Pixels to soft tokens (omitted with `--skip-vision`)                                       |

Both bundles stamp the same embedding space, so their vectors can share an index; the first Neural Engine load specializes the graphs and later loads reuse the cache. Neural Engine text inputs run one at a time.

### Task prompts

Text inputs are prefixed with a task prompt. Select one by name with `promptName` in Swift or `--prompt-name` on the command line. Images take no prompt.

| Prompt                | Use for                                                 | Prefix                                |
| --------------------- | ------------------------------------------------------- | ------------------------------------- |
| `query`               | Search queries: retrieval, reranking, bitext mining     | `task: search result \| query:`       |
| `document`            | Documents to search over                                | `title: none \| text:`                |
| `sentence_similarity` | Semantic similarity, pair classification, summarization | `task: sentence similarity \| query:` |
| `classification`      | Classification, multi-label classification              | `task: classification \| query:`      |
| `clustering`          | Clustering                                              | `task: clustering \| query:`          |
| `question_answering`  | Question answering                                      | `task: question answering \| query:`  |
| `fact_checking`       | Fact checking                                           | `task: fact checking \| query:`       |
| `code_retrieval`      | Code and instruction retrieval                          | `task: code retrieval \| query:`      |

For retrieval, embed queries with `query` and documents with `document`. Documents with a title can use the bundle's document template: format the input as `title: {title} | text: {text}` and embed it with the `document` prompt.

## Running

### In your iOS and macOS applications

```swift
import CoreAIEmbeddings
import CoreAIVectorIndex

let model = try EmbeddingModel(bundleAt: URL(fileURLWithPath: "exports/embeddinggemma-2"))
try await model.load()

// Build an index sized for the model's native width (pass `dimensions: 256` for a Matryoshka prefix).
var index = try VectorIndex(descriptor: VectorIndexDescriptor(model: model))

let documents = ["The Eiffel Tower is in Paris.", "Photosynthesis converts light into chemical energy."]
let documentEmbeddings = try await model.embed(documents.map { .text($0) }, promptName: "document")
for (i, embedding) in documentEmbeddings.enumerated() {
    try index.append(id: "doc-\(i)", embedding: embedding, attributes: ["kind": "text"])
}

let photo = try await model.embed([.image(cgImage)])[0]
try index.append(id: "photo", embedding: photo, attributes: ["kind": "image"])

let query = try await model.embed(.text("Where is the Eiffel Tower?"), promptName: "query")
for result in try index.search(query, k: 3) {
    print(result.id, result.score)
}

try index.save(to: URL(fileURLWithPath: "corpus.cvi"))
let reopened = try VectorIndex.open(at: URL(fileURLWithPath: "corpus.cvi"), expecting: index.descriptor)
```

The index records the embedding space (model ID and revision) and width, and rejects vectors from a different space.

### On your Mac using built-in Command Line Tool

```bash
# Embed a folder of images and a text corpus (.jsonl with {"text", "id"?, "title"?} or one document per line)
swift run -c release embedding-search index --model exports/embeddinggemma-2 \
    --images path/to/photos --texts path/to/docs.jsonl --out corpus.cvi

# Search with a text or image query
swift run -c release embedding-search query --model exports/embeddinggemma-2 \
    --index corpus.cvi --text "a dog on a beach" --k 5
swift run -c release embedding-search query --model exports/embeddinggemma-2 \
    --index corpus.cvi --image path/to/query.jpg --kind image

# Many queries in one process from a JSON array
swift run -c release embedding-search query --model exports/embeddinggemma-2 \
    --index corpus.cvi --queries-json queries.json --output-json results.json

# Print one embedding
swift run -c release embedding-search embed --model exports/embeddinggemma-2 \
    --text "hello world" --prompt-name query
```

`index` and `query` default to the `document` and `query` prompts; `--dimensions` selects a Matryoshka width, and `index --storage float16` halves the index size. Run `embedding-search <command> --help` for every option and the JSON input and output formats.

PNG inputs embed identically to the reference preprocessing, while JPEG inputs can differ slightly because the image decoders differ.

## License

The export recipe in this repository is licensed under the [BSD-3-Clause license](../../LICENSE). The EmbeddingGemma 2 model weights are released by Google DeepMind under the Apache-2.0 license, and deployments must follow the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).

[^1]: [Documentation](https://ai.google.dev/gemma/docs/embeddinggemma) · [HuggingFace](https://huggingface.co/google/embeddinggemma-2)

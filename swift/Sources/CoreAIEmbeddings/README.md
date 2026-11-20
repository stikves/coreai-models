# CoreAIEmbeddings

Runs `kind: "embedding"` bundles, which `coreai.embedding.export` writes: texts and images in, one vector per input out. Vectors from one bundle's towers share a space and compare by cosine. The bundle format is [`embedding/contract.py`](../../../python/src/coreai_models/embedding/contract.py).

```swift
import CoreAIEmbeddings

var model = try await CoreAIEmbeddingModel(resourcesAt: URL(filePath: "exports/embeddinggemma-2"))
let query = try await model.embed(["which photo shows a cat?"])[0]
let photos = try await model.embed(images: images)
let best = try photos.max { try query.cosine($0) < query.cosine($1) }
```

The runtime has no per-model code. Each tower's `pipeline` runs in order over named tensors: the first inputs come from the modality (token ids, a cropped image, or image patches with their token ids), later ones are earlier outputs or are built from them (soft tokens merged into `inputs_embeds`, an additive attention `bias`, `pool_weights`). Each input takes the shortest function length that holds it, and inputs of one length share calls of the largest batch that fits. A bundle with an input nothing provides is refused at `init`.

## Models

| Bundle                           | Input       | Text tokens | Output                              |
|----------------------------------|-------------|-------------|-------------------------------------|
| `clip-vit-b32`, `-fused`, `-ios` | text, image | 77          | 512, `logit_scale` 100              |
| `clip-vit-b32-text`              | text        | 77          | 512, `logit_scale` 100              |
| `clap-htsat`, `-text`            | text        | 77          | 512, `logit_scale` 18.66            |
| `embeddinggemma-2`               | text, image | 2,048       | 768; Matryoshka 512/256/128         |
| `embeddinggemma-2-ios`           | text, image | 1,024       | as above, same space                |
| `qwen3-embedding-0.6b`           | text        | 2,048       | 1024; Matryoshka 512/256/128/64/32  |

**Text tokens** is the bundle's `text.truncation.max_tokens`, special tokens included: a longer text is cut to it with EOS kept last, as the Hugging Face tokenizer cuts it, and its embedding's `status` says so. The macOS and iOS EmbeddingGemma 2 bundles share a space but cut at different lengths, so a text over 1,024 tokens embeds differently on the two. `init` refuses a bundle whose limit is longer than its longest text function.

A full CLAP bundle runs its text tower; its audio tower is skipped until audio input lands. The first load of a Neural Engine program specializes it, which takes minutes for `embeddinggemma-2-ios`; later loads use the Core AI cache.

## API

```swift
public protocol EmbeddingModel: ~Copyable {
    var space: EmbeddingSpace { get }
    var dimension: Int { get }
    var matryoshkaDimensions: [Int] { get }
    var modalities: Set<EmbeddingModality> { get }
    var logitScale: Double? { get }
    mutating func embed(_ inputs: [EmbeddingInput], dimensions: Int?) async throws -> [Embedding]
    func truncate(_ embedding: Embedding, to width: Int) throws -> Embedding
}

extension EmbeddingModel where Self: ~Copyable {
    mutating func embed(_ inputs: [EmbeddingInput]) async throws -> [Embedding]
    mutating func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding]
    mutating func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding]
}

public struct CoreAIEmbeddingModel: EmbeddingModel, ~Copyable, Sendable {
    init(resourcesAt url: URL) async throws
    mutating func embed(_ inputs: [EmbeddingInput], dimensions: Int? = nil) async throws -> [Embedding]
    mutating func load(_ modalities: Set<EmbeddingModality>? = nil) async throws
    mutating func warmup(_ modalities: Set<EmbeddingModality>? = nil) async throws
    func truncate(_ embedding: Embedding, to width: Int) throws -> Embedding
}

public final class SharedEmbeddingModel: EmbeddingModel, Sendable {
    init(_ model: consuming CoreAIEmbeddingModel, maxQueueDepth: Int = .max)
    func embed(_ inputs: [EmbeddingInput], dimensions: Int? = nil) async throws -> [Embedding]
    func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding]
    func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding]
    func load(_ modalities: Set<EmbeddingModality>? = nil) async throws
    func warmup(_ modalities: Set<EmbeddingModality>? = nil) async throws
    func truncate(_ embedding: Embedding, to width: Int) throws -> Embedding
}
```

- **`CoreAIEmbeddingModel`** has one owner and runs one call at a time: its calls are `mutating`, so two at once do not compile. It cannot be copied, and dropping it releases its programs. An actor cannot call it on a stored property; give the actor a `SharedEmbeddingModel` instead.
- **`SharedEmbeddingModel`** takes the model over and serves any number of callers, one call at a time in the order they arrive. A call waiting behind `maxQueueDepth` others throws `queueFull`; a call cancelled while it waits throws `CancellationError`.
- **`embed`** returns one vector per input, in input order; `EmbeddingInput` is `.text(_:)`, `.image(_:)` or a string literal, so texts and images can mix. Texts go in after the bundle's `default_prompt_name` prompt when it has one; no exported bundle sets one, so texts go in as they are.
- **`dimensions`** is one of `matryoshkaDimensions`; `nil` is the native width. `truncate(_:to:)` cuts a stored vector to such a width, renormalized when the bundle says so, without running the model.
- **Loading:** each tower loads on its first call, and each function on the first call that needs it. `load(_:)` loads the tokenizer and programs up front; `warmup(_:)` also runs every function once on placeholder inputs. `embed` and `warmup` throw `CancellationError` before the next program call.
- **`Embedding`** holds `vector`, `space` and `status`, is `Codable`, and `cosine(_:)` throws `incompatibleEmbeddings` across spaces or widths. `status` is `.complete`, or `.truncated(tokens:limit:)` for a text of `tokens` tokens that the model saw only the first `limit` of (the **Text tokens** column above); images are always `.complete`. `truncate(_:to:)` keeps the status and `cosine` ignores it. Zero-shot labels (CLIP): `Float(model.logitScale ?? 1) * image.cosine(label)`, softmax over the labels.
- **Create ML:** `CoreAIEmbeddingTextExtractor(model:dimensions:)` (`String`) and `CoreAIEmbeddingImageExtractor(model:dimensions:context:)` (`CIImage`, an `ImageFeatureExtractor`) take a `SharedEmbeddingModel`, so both can share one loaded model. They go where `ImageFeaturePrint` does and return `MLShapedArray<Float>` of shape `[dimensions]`. Init throws `unsupportedDimension`, or `modalityNotInBundle` for an image extractor over a text-only bundle.

## Errors

`EmbeddingError` names the failing input, width or program. `init` also throws `ModelBundle.BundleError` for a missing or invalid asset.

| Case                                         | When                                                                               |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `invalidMetadata(_:)`                        | The `embedding` block breaks the contract, or `max_tokens` exceeds its functions   |
| `modalityNotInBundle(_:)`                    | The call needs a tower the bundle does not have                                    |
| `unsupportedPipeline(_:_:)`                  | A tower's pipeline needs an input no earlier stage or rule provides                |
| `unsupportedDimension(_:supported:)`         | A width not in `matryoshkaDimensions`                                              |
| `incompatibleEmbeddings(_:)`                 | `cosine` across spaces or widths, `truncate` across spaces                         |
| `imageConversionFailed`                      | The image cannot be converted to RGB                                               |
| `unexpectedOutput(component:_:)`             | A program returned something other than its declared outputs                       |
| `programFailed(asset:function:unit:reason:)` | Core AI failed to load or run a program; `unit` is `gpu`, `neural_engine` or `cpu` |
| `queueFull(depth:)`                          | A `SharedEmbeddingModel` call finds `maxQueueDepth` calls already waiting          |

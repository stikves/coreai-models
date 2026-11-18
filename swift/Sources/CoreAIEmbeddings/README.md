# CoreAIEmbeddings

Runs `kind: "embedding"` bundles, which `coreai.embedding.export` writes: texts and images in, one vector per input out. Vectors from one bundle's towers share a space and compare by cosine. The bundle format is [`embedding/contract.py`](../../../python/src/coreai_models/embedding/contract.py).

```swift
import CoreAIEmbeddings

let model = try await EmbeddingModel(resourcesAt: URL(filePath: "exports/embeddinggemma-2"))
let query = try await model.embed(["which photo shows a cat?"])[0]
let photos = try await model.embed(images: images)
let best = try photos.max { try query.cosine($0) < query.cosine($1) }
```

The runtime has no per-model code. Each tower's `pipeline` runs in order over named tensors: the first inputs come from the modality (token ids, a cropped image, or image patches with their token ids), later ones are earlier outputs or are built from them (soft tokens merged into `inputs_embeds`, an additive attention `bias`, `pool_weights`). Each input takes the shortest function length that holds it, and inputs of one length share calls of the largest batch that fits. A bundle with an input nothing provides is refused at `init`.

## Models

| Bundle                           | Input       | Output                      |
|----------------------------------|-------------|-----------------------------|
| `clip-vit-b32`, `-fused`, `-ios` | text, image | 512, `logit_scale` 100      |
| `clip-vit-b32-text`              | text        | 512, `logit_scale` 100      |
| `clap-htsat`, `-text`            | text        | 512, `logit_scale` 18.66    |
| `embeddinggemma-2`               | text, image | 768; Matryoshka 512/256/128 |
| `embeddinggemma-2-ios`           | text, image | as above, same space        |

A full CLAP bundle runs its text tower; its audio tower is skipped until audio input lands. The first load of a Neural Engine program specializes it, which takes minutes for `embeddinggemma-2-ios`; later loads use the Core AI cache.

## API

```swift
public actor EmbeddingModel {
    init(resourcesAt url: URL) async throws

    func embed(_ inputs: [EmbeddingInput], dimensions: Int? = nil) async throws -> [Embedding]
    func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding]
    func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding]
    nonisolated func truncate(_ embedding: Embedding, to width: Int) throws -> Embedding

    func load(_ modalities: Set<EmbeddingModality>? = nil) async throws
    func warmup(_ modalities: Set<EmbeddingModality>? = nil) async throws
    func unload() async

    nonisolated var space: EmbeddingSpace { get }
    nonisolated var dimension: Int { get }
    nonisolated var matryoshkaDimensions: [Int] { get }
    nonisolated var modalities: Set<EmbeddingModality> { get }
    nonisolated var logitScale: Double? { get }
}
```

- **`embed`** returns one vector per input, in input order; `EmbeddingInput` is `.text(_:)`, `.image(_:)` or a string literal, so texts and images can mix. Texts go in after the bundle's `default_prompt_name` prompt when it has one; no exported bundle sets one, so texts go in as they are.
- **`dimensions`** is one of `matryoshkaDimensions`; `nil` is the native width. `truncate(_:to:)` cuts a stored vector to such a width, renormalized when the bundle says so, without running the model.
- **Loading:** each tower loads on its first call, and each function on the first call that needs it. `load(_:)` loads the tokenizer and programs up front; `warmup(_:)` also runs every function once on placeholder inputs. `embed` and `warmup` throw `CancellationError` before the next program call.
- **`Embedding`** holds `vector` and `space`, is `Codable`, and `cosine(_:)` throws `incompatibleEmbeddings` across spaces or widths. Zero-shot labels (CLIP): `Float(model.logitScale ?? 1) * image.cosine(label)`, softmax over the labels.
- **Create ML:** `CoreAIEmbeddingTextExtractor(model:dimensions:)` (`String`) and `CoreAIEmbeddingImageExtractor(model:dimensions:context:)` (`CIImage`, an `ImageFeatureExtractor`) go where `ImageFeaturePrint` does and return `MLShapedArray<Float>` of shape `[dimensions]`. Init throws `unsupportedDimension`, or `modalityNotInBundle` for an image extractor over a text-only bundle.

## Errors

`EmbeddingError` names the failing input, width or program. `init` also throws `ModelBundle.BundleError` for a missing or invalid asset.

| Case                                         | When                                                                               |
| -------------------------------------------- | ---------------------------------------------------------------------------------- |
| `invalidMetadata(_:)`                        | The `embedding` block breaks the contract                                          |
| `modalityNotInBundle(_:)`                    | The call needs a tower the bundle does not have                                    |
| `unsupportedPipeline(_:_:)`                  | A tower's pipeline needs an input no earlier stage or rule provides                |
| `unsupportedDimension(_:supported:)`         | A width not in `matryoshkaDimensions`                                              |
| `incompatibleEmbeddings(_:)`                 | `cosine` across spaces or widths, `truncate` across spaces                         |
| `imageConversionFailed`                      | The image cannot be converted to RGB                                               |
| `unexpectedOutput(component:_:)`             | A program returned something other than its declared outputs                       |
| `programFailed(asset:function:unit:reason:)` | Core AI failed to load or run a program; `unit` is `gpu`, `neural_engine` or `cpu` |

# VLM Export Plan

Export plan for adding Vision-Language Model support to coreai-models.
Order: SmolVLM2 → Gemma3 → InternVL3.

## 1. SmolVLM2-256M (Apache 2.0)

### Model IDs
- `HuggingFaceTB/SmolVLM2-256M-Video-Instruct` (instruction-tuned)
- `HuggingFaceTB/SmolVLM2-2.2B-Instruct` (larger variant)

### Architecture Overview

```
Image → SigLIP ViT → pixel_shuffle(2x) → Linear → scatter into text embeddings → LlamaDecoder → logits
```

### Vision Encoder (SigLIP-based)
- **Type**: SmolVLMVisionTransformer (SigLIP variant)
- **hidden_size**: 1152
- **intermediate_size**: 3072 (for 256M; larger for 2.2B)
- **num_hidden_layers**: 12
- **num_attention_heads**: 16
- **image_size**: 224 (for 256M, 384 for 2.2B)
- **patch_size**: 32 (for 256M; 14 for 2.2B)
- **Activation**: gelu_pytorch_tanh
- **No CLS token** — all patch tokens used
- **Output**: `[1, num_patches, 1152]` where num_patches = (224/32)^2 = 49

### Connector (Projection)
- **pixel_shuffle**: scale_factor=2, reduces sequence length by 4x
  - Input: `[1, 49, 1152]`
  - After pixel_shuffle: `[1, 49/4, 1152*4]` = `[1, 12, 4608]`
  - (Note: 49 is not divisible by 4 cleanly — actual num_patches may differ for 256M)
- **SimpleMLP**: Single `nn.Linear(vision_hidden * scale_factor^2, text_hidden, bias=False)`
  - Input: `[1, N, 4608]`
  - Output: `[1, N, text_hidden]`

For 256M with patch_size=32, image_size=224:
- num_patches = (224/32)^2 = 49 → pixel_shuffle produces 49/4 ≈ 12 tokens
- image_seq_len = (224/32)^2 / (2^2) = 49/4 ≈ 12

### Text Decoder (SmolLM2 = Llama architecture)
- **model_type**: "llama" (confirmed from config class)
- **Architecture**: Identical to Llama — GQA, RMSNorm, SiLU MLP, RoPE
- **For 256M variant** (estimated from model name):
  - hidden_size: ~576
  - num_layers: ~12-16
  - num_heads: varies
  - vocab_size: 128256+
  - rope_theta: TBD (need to download config)
- **Weights load into existing `LlamaForCausalLM`**: YES with config changes only

### Image Token Injection
- **image_token_id**: 128257 (default from SmolVLMConfig)
- **Method**: `torch.where(image_mask, image_embeds, text_embeds)` — same scatter-merge pattern as our VLM engine
- **Position IDs**: Contiguous across text+vision tokens

### Image Preprocessing
- **Normalization**: IMAGENET_STANDARD (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
- **rescale_factor**: 1/255
- **Resize**: longest_edge=364 (max_image_size), aspect ratio preserving
- **do_image_splitting**: True (can split large images into sub-images — for v1 we skip this)
- **Resampling**: LANCZOS

### Weight Prefixes (from modeling code)
- Vision: `model.vision_model.*`
- Connector: `model.connector.*`
- Text: `model.text_model.*`
- LM head: `lm_head.*`

### Export Strategy

1. **Text decoder**: Use existing `LlamaForCausalLM` (or minimal subclass) with
   `inputs_embeds` variant. Register as `"smolvlm"` with `hf_config_attr="text_config"`,
   `hf_state_dict_prefix="model.text_model."`.

2. **Vision encoder**: Export `SmolVLMVisionTransformer` + `SmolVLMConnector` as a
   fused `vision.aimodel`. Static shapes for fixed 224x224 input (v1).
   Output: `[1, image_seq_len, text_hidden]` in float16.

3. **Embed model**: `embed.aimodel` — token embedding lookup (same pattern as Qwen3-VL).

4. **Bundle metadata**:
```json
{
  "kind": "vlm",
  "vision": {
    "image_size": 224,
    "patch_size": 32,
    "image_token_count": 12,
    "image_token_id": 128257,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "rescale_factor": 1.0
  }
}
```

### Key Differences from Qwen3-VL Export
- Simpler projection (1 linear vs 2-layer MLP)
- Standard Llama decoder (no custom RoPE like M-RoPE)
- IMAGENET normalization, not CLIP
- Smaller patch count (12 tokens vs 196)
- No custom KV cache needed (standard cache.py should work — no Metal prefill crash expected with only 12 vision tokens)

### Confirmed Config (from downloaded model)

```
Vision:
  hidden_size: 768
  num_hidden_layers: (SigLIP default, ~12)
  num_attention_heads: 12
  image_size: 512
  patch_size: 16
  position_embeddings: 1024 (32x32 grid)
  activation: gelu_pytorch_tanh

Connector:
  scale_factor: 4
  pixel_shuffle: 1024 patches → 64 tokens (÷16)
  projection: Linear(768*16=12288, 576, bias=False)

Text (SmolLM2 = Llama):
  model_type: "llama"
  hidden_size: 576
  num_hidden_layers: 30
  num_attention_heads: 9
  num_key_value_heads: 3
  intermediate_size: 1536
  head_dim: 64
  rope_theta: 100000
  rms_norm_eps: 1e-5
  vocab_size: 49280
  max_position_embeddings: 8192
  tie_word_embeddings: false
  attention_bias: false
  mlp_bias: false

Top-level:
  image_token_id: 49190
  scale_factor: 4
  image_seq_len: (512/16)^2 / (4^2) = 1024/16 = 64
```

### Weight Prefixes (confirmed from safetensors)
- `model.vision_model.embeddings.*`
- `model.vision_model.encoder.layers.N.*`
- `model.vision_model.post_layernorm.*`
- `model.connector.modality_projection.proj.weight` — shape [576, 12288]
- `model.text_model.embed_tokens.weight` — shape [49280, 576]
- `model.text_model.layers.N.{self_attn,mlp,input_layernorm,post_attention_layernorm}.*`
- `model.text_model.norm.weight`
- `lm_head.weight` — shape [49280, 576]

### Implementation Plan

Files to create:
1. `python/src/coreai_models/models/gpu/smolvlm.py` — text decoder (inputs_embeds)
2. `python/export_smolvlm.py` — export script
3. Registry entry in `python/src/coreai_models/models/registry.py`

Text decoder approach: Copy Mistral model structure (same primitives),
remove sliding window, accept inputs_embeds instead of input_ids.

### Risks
- pixel_shuffle with scale_factor=4 on a 32x32 grid — need to verify reshape logic
- do_image_splitting disabled for v1 — may affect quality on large images
- 512px input is larger than typical SigLIP (usually 224/384) — verify export works

---

## 2. Gemma3 4B Multimodal (Gemma TOS — custom license)

### Model ID
- `google/gemma-3-4b-it` (already cached locally)

### Architecture Overview

```
Image → SigLIP ViT (896x896, 27 layers) → AvgPool(4x4) → RMSNorm → Linear → scatter → Gemma3Decoder → logits
```

### Confirmed Config (from cached model)

```
Vision (SigLIP):
  hidden_size: 1152
  num_hidden_layers: 27
  num_attention_heads: 16
  image_size: 896
  patch_size: 14
  intermediate_size: 4304
  model_type: siglip_vision_model
  patches_per_image: 64 (896/14)
  total_patches: 4096 (64x64)

Projector (Gemma3MultiModalProjector):
  - AvgPool2d(kernel_size=4, stride=4): 64x64 → 16x16 = 256 tokens
  - RMSNorm(1152)
  - matmul with mm_input_projection_weight [1152, 2560] (no bias)
  Output: [1, 256, 2560]

Text (Gemma3TextConfig):
  hidden_size: 2560
  num_hidden_layers: 34
  num_attention_heads: 8 (default)
  num_key_value_heads: 4 (default)
  head_dim: 256
  intermediate_size: 10240
  vocab_size: 262208
  rope_theta: 1000000.0
  max_position_embeddings: 131072
  sliding_window: 1024
  rope_scaling: {factor: 8.0, rope_type: "linear"}

Top-level:
  image_token_index: 262144
  mm_tokens_per_image: 256
```

### Weight Prefixes (from safetensors index)
- `vision_tower.vision_model.embeddings.*`
- `vision_tower.vision_model.encoder.layers.N.*`
- `multi_modal_projector.mm_input_projection_weight` — shape [1152, 2560]
- `multi_modal_projector.mm_soft_emb_norm.weight` — shape [1152]
- `language_model.model.embed_tokens.weight`
- `language_model.model.layers.N.*`
- `language_model.model.norm.weight`
- (no separate lm_head — tie_word_embeddings=True implied)

### Image Preprocessing
- **Normalization**: IMAGENET_STANDARD (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
- **rescale_factor**: 1/255
- **Resize**: 896x896 fixed (with Pan-and-Scan for large images — skip for v1)
- **Resampling**: BICUBIC

### Key Differences from SmolVLM2
- **Much larger vision encoder**: 27 layers vs 12, 4096 patches vs 1024
- **Different spatial reduction**: AvgPool2d(4x4) vs pixel_shuffle(4x)
- **Larger text decoder**: 2560 hidden, 34 layers, head_dim=256
- **Sliding window attention**: alternating global/local layers in text decoder
- **Text decoder already exists**: `gemma3_text` in registry (but needs inputs_embeds variant)
- **Same normalization**: IMAGENET_STANDARD [0.5, 0.5, 0.5]

### Implementation Plan

1. **Text decoder (inputs_embeds)**: Subclass existing `Gemma3ForCausalLM` or create
   `Gemma3VLMForCausalLM` that takes inputs_embeds. The existing model handles
   sliding window + RoPE scaling already.

2. **Vision encoder**: Export SigLIP ViT (27 layers, 896x896 fixed input).
   Large model (~400M vision params) — export may take a while.

3. **Projector**: Fuse AvgPool + RMSNorm + matmul into vision export or keep separate.
   Keeping it fused with vision is simpler (one less function to manage).

4. **Bundle metadata**:
```json
{
  "kind": "vlm",
  "vision": {
    "image_size": 896,
    "patch_size": 14,
    "image_token_count": 256,
    "image_token_id": 262144,
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "rescale_factor": 1.0
  }
}
```

### Risks
- 4B model is large — export + compilation will be slow
- Sliding window attention in text decoder needs testing with VLM prefill
- 896x896 vision input is 4x the pixels of SmolVLM (512x512) — memory concerns
- rope_scaling with linear factor=8 — verify our existing gemma3 impl handles it
- License is Gemma TOS (not standard OSS) — may limit public release

---

## 3. InternVL3-1B (MIT)

*Research pending — third in queue.*

Key facts known:
- InternViT-300M vision encoder (448x448)
- Pixel shuffle + 2-layer MLP projection
- Qwen2.5-0.5B text decoder (existing qwen2 in registry)
- Multi-crop tiling (up to 12 tiles) — complex preprocessing
- CLIP normalization

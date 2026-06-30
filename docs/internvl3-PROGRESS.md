# InternVL3 Export -- PROGRESS

**Last updated**: 2026-06-30 by python-worker
**Last gate result**: `HF_HUB_OFFLINE=1 uv run python python/export_internvl3.py --num-layers 2 --overwrite` -- PASS

## STATUS
| Phase | Status | Gate Result |
|-------|--------|-------------|
| 1. Research & architecture | Done | Config confirmed from cached model |
| 2. Text decoder (inputs_embeds) | Done | internvl3.py created, registry entry added |
| 3. Vision encoder + projector export | Done | InternViT + pixel_shuffle + MLP projector in export script |
| 4. Export script + bundle | Done | All 3 aimodels + tokenizer + metadata.json exported |
| 5. E2E test with VLM engine | Pending | |

**Current blocker**: None
**Next action**: E2E test with VLM engine (Swift side)

## CONFIG (from downloaded model)
```
Vision (InternViT-300M):
  hidden_size: 1024
  num_hidden_layers: 24
  num_attention_heads: 16
  intermediate_size: 4096
  image_size: 448
  patch_size: 14
  num_patches: 1024
  norm_type: layer_norm
  activation: gelu
  has_class_embedding: yes (prepended, removed before pixel_shuffle)
  position_embedding: [1, 1025, 1024] (cls + patches)
  layer_scale: ls1, ls2 per layer
  fused_qkv: attn.qkv (already fused in safetensors)

Projector (mlp1):
  pixel_shuffle(downsample_ratio=0.5, ps_version=v2): 1024 patches → 256 tokens, hidden 1024→4096
  LayerNorm(4096) with bias
  Linear(4096, 896, bias=True)
  GELU
  Linear(896, 896, bias=True)

Text (Qwen2.5-0.5B):
  hidden_size: 896
  num_hidden_layers: 24
  num_attention_heads: 14
  num_key_value_heads: 2
  intermediate_size: 4864
  vocab_size: 151674
  max_position_embeddings: 32768
  rope_theta: 1000000.0
  rms_norm_eps: 1e-6
  tie_word_embeddings: False
  model_type: qwen2
  No qk_norm (confirmed: no q_norm/k_norm keys in safetensors)

Top-level:
  model_type: internvl_chat (NOT in transformers — manual loading needed)
  downsample_ratio: 0.5
  ps_version: v2
  image_token_id: 151667 (<IMG_CONTEXT>)
```

## Weight Prefixes
- `vision_model.embeddings.*`
- `vision_model.encoder.layers.N.*` (norm1, norm2, ls1, ls2, attn.qkv, attn.proj, mlp.fc1, mlp.fc2)
- `mlp1.0.{weight,bias}` — LayerNorm(4096)
- `mlp1.1.{weight,bias}` — Linear(4096→896)
- `mlp1.3.{weight,bias}` — Linear(896→896)
- `language_model.model.embed_tokens.weight`
- `language_model.model.layers.N.*`
- `language_model.model.norm.weight`
- `language_model.lm_head.weight`

## DECISIONS
- D1: Use existing Qwen2 model as base for text decoder (same arch, in registry)
- D2: Single-crop v1 (maxTiles=1 in config, tiling logic deferred)
- D3: Manual safetensors loading (model_type internvl_chat not in transformers)
- D4: CLIP normalization (confirmed from InternVL image processor)
- D5: Fuse vision encoder + pixel_shuffle + MLP projector into one vision.aimodel
- D6: Vision encoder runs in float32 (weights loaded as bf16, cast to f32 before export)

## LEARNINGS
- InternVL3-1B uses model_type "internvl_chat" which is NOT in HF transformers
- The transformers/models/internvl/ module has "internvl" type — different model class
- Must load weights manually from safetensors (same pattern as SmolVLM/Qwen3-VL)
- Qwen2 text decoder: 14 heads with 2 kv_heads = GQA ratio 7:1
- Projector has bias on all layers (unlike SmolVLM's bias-free connector)
- pixel_shuffle with ratio 0.5 = spatial dims halved, channel dim 4x'd
- InternViT uses fused QKV (already fused in safetensors, unlike SmolVLM's separate q/k/v)
- InternViT has layer_scale (ls1, ls2) multipliers per layer
- InternViT has class_embedding prepended and removed before pixel_shuffle
- Vision weights are bfloat16 in safetensors; must cast to float32 for export since input is float32
- No q_norm/k_norm in the language model layers (confirmed by checking safetensors keys)
- Registry uses hf_config_attr=None since InternVL uses "llm_config" not "text_config"

## FILES CREATED/MODIFIED
- `python/src/coreai_models/models/gpu/internvl3.py` (NEW) — text decoder
- `python/src/coreai_models/models/registry.py` (MODIFIED) — added internvl3 entry
- `python/export_internvl3.py` (NEW) — full VLM export script

---
## DETAILED LOG (append-only, newest first)

### 2026-06-30 -- Phase 2-4: Text decoder + export script
- Created InternVL3ForCausalLM in gpu/internvl3.py (inputs_embeds variant of Qwen2)
- Added registry entry for "internvl3" with hf_config_attr=None
- Created export_internvl3.py with InternViT vision encoder, embed, and text decoder
- First attempt failed: conv2d input float32 vs bias bfloat16 mismatch in vision encoder
- Fix: cast vision model to float32 after loading weights (`.float().eval()`)
- Second attempt: PASS — all 3 aimodels + tokenizer + metadata exported successfully
- **Result**: Gate PASS — `HF_HUB_OFFLINE=1 uv run python python/export_internvl3.py --num-layers 2 --overwrite` completes without error

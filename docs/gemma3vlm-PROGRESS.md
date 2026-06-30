# Gemma3 VLM Export -- PROGRESS

**Last updated**: 2026-06-30 by python-worker
**Last gate result**: `--num-layers 2` export PASS (all 3 aimodels + tokenizer + metadata)

## STATUS
| Phase | Status | Gate Result |
|-------|--------|-------------|
| 1. Research & architecture | Done | Config confirmed, plan in docs/vlm-export-plan.md |
| 2. Text decoder (inputs_embeds) | Done | gemma3_vlm.py created, exports successfully |
| 3. Vision encoder + projector export | Done | vision.aimodel exports (SigLIP + AvgPool + RMSNorm + Linear) |
| 4. Export script + bundle | Done | Full bundle exports with --num-layers 2 |
| 5. E2E test with VLM engine | Pending | |

**Current blocker**: None
**Next action**: E2E test with VLM engine (full model export + inference quality check)

## DECISIONS
- D1: Reuse existing gemma3_text.py model architecture (sliding window, RoPE scaling all handled)
- D2: Fuse vision encoder + projector (AvgPool + RMSNorm + Linear) into single vision.aimodel
- D3: Fixed 896x896 input for v1 (no Pan-and-Scan)
- D4: Same branch sukru/smolvlm (multi-VLM export PR)
- D5: Vision encoder runs in float32 (input is float32 pixels), casts to fp16 at output
- D6: lm_head.weight = clone of embed_tokens.weight (tie_word_embeddings workaround for inputs_embeds variant)

## LEARNINGS
- Text decoder already in registry as gemma3_text with sliding window + rope_scaling
- Projector is: AvgPool2d(4,4) + RMSNorm(1152) + matmul(1152->2560)
- Weight prefixes: vision_tower.*, multi_modal_projector.*, language_model.model.*
- 4096 patches (64x64) -> 256 tokens after AvgPool
- image_token_index: 262144, mm_tokens_per_image: 256
- SigLIP vision encoder: 27 layers, 1152 hidden, same arch family as SmolVLM
- Vision weights load as bfloat16 from safetensors -- must cast to float32 before export (conv2d input is float32)
- Gemma3 tie_word_embeddings=True means no separate lm_head in HF checkpoint; must clone embed_tokens -> lm_head

## FILES CREATED/MODIFIED
- `python/src/coreai_models/models/gpu/gemma3_vlm.py` — NEW: inputs_embeds variant text decoder
- `python/export_gemma3vlm.py` — NEW: full VLM bundle export script
- `python/src/coreai_models/models/registry.py` — MODIFIED: added gemma3_vlm entry

---
## DETAILED LOG (append-only, newest first)

### 2026-06-30 -- Phase 2-4: Implement text decoder + export script
- Created `gemma3_vlm.py` based on gemma3_text.py but with inputs_embeds variant (no embed_tokens)
- Created `export_gemma3vlm.py` following SmolVLM pattern but with Gemma3 specifics:
  - SigLIP ViT 27-layer + AvgPool(4,4) + RMSNorm + Linear projector
  - Embed model with baked-in embed_scale (sqrt(2560))
  - Text decoder with fused qkv + fused qk_norm
- First attempt failed: conv2d input float32 vs weight bfloat16 mismatch
  - Fix: cast vision model to float32 before export
- Second attempt: PASS -- all components export cleanly
- Verified bundle: vision.aimodel, embed.aimodel, gemma3_4b_vlm.aimodel, tokenizer/, metadata.json
- Registry entry added with hf_config_attr="text_config", hf_state_dict_prefix="language_model.model."

# SmolVLM2 Export -- PROGRESS

**Last updated**: 2026-06-30 by python-worker
**Last gate result**: Both 256M and 2.2B export with `--num-layers 2` -- PASS

## STATUS
| Phase | Status | Gate Result |
|-------|--------|-------------|
| 1. Research & architecture | Done | Config confirmed, plan in docs/vlm-export-plan.md |
| 2. Text decoder (inputs_embeds) | Done | Model loads, fuses qkv, exports successfully |
| 3. Vision encoder + connector export | Done | 12-layer SigLIP + pixel_shuffle + Linear exported |
| 4. Export script + bundle | Done | Full bundle: text + embed + vision + tokenizer + metadata |
| 5. Multi-variant support | Done | Refactored: --model-id flag, config-driven, sharded safetensors |
| 6. E2E test with VLM engine | Pending | |

**Current blocker**: None
**Next action**: E2E integration test with VLM Swift runner

## DECISIONS
- D1: Use Mistral model as template (same Llama arch, same primitives)
- D2: Fuse vision encoder + connector into single vision.aimodel
- D3: Fixed 512x512 input for v1 (no image splitting)
- D4: Target coreai-models-mine repo, sukru/smolvlm branch
- D5: Created `models/gpu/` directory for VLM-specific models (inputs_embeds variant)
- D6: Manual safetensors loading for text decoder (lm_head.weight not under model.text_model. prefix)
- D7: ShardedSafetensors class abstracts single-file vs index.json model loading

## LEARNINGS
- SmolLM2 text decoder is model_type="llama" -- identical structure to Mistral sans sliding window
- scale_factor=4 for 256M (1024->64 tokens), scale_factor=3 for 2.2B (729->81 tokens)
- Connector is single Linear(hidden*scale^2 -> text_hidden) -- trivially fused with vision encoder
- `lm_head.weight` in SmolVLM lives at top-level (not under `model.text_model.`) so
  `from_hf_memory_efficient` with `hf_state_dict_prefix` cannot load it directly.
  Manual safetensors loading is needed for the text decoder.
- HF_HUB_OFFLINE=1 env var enables offline mode for snapshot_download (proxy blocks network)
- Vision encoder uses standard SigLIP attention (with bias) -- no RoPE, no causal mask
- 2.2B uses image_size=384, patch_size=14 -> floor(384/14)=27, 27^2=729 patches (non-integer division is fine, Conv2d with stride=14 floors automatically)
- 2.2B safetensors are sharded (model-00001-of-00002.safetensors + index.json)
- AutoConfig resolves all needed fields (hidden_size, num_attention_heads, intermediate_size) even when not in raw JSON

---
## DETAILED LOG (append-only, newest first)

### 2026-06-30 -- Phase 5: Multi-variant refactoring
- Refactored `python/export_smolvlm.py` to support multiple model sizes
- Added `--model-id` argument (default: 256M)
- Removed all hardcoded constants (IMAGE_SIZE, PATCH_SIZE, VISION_HIDDEN_SIZE, etc.)
- SmolVLMVisionEncoder now accepts config params in __init__
- Added ShardedSafetensors class for 2.2B (multiple .safetensors files + index.json)
- Bundle name derived from model_id: "smolvlm_256m", "smolvlm_2_2b"
- Verified text decoder model class (`models/gpu/smolvlm.py`) works for 2.2B without changes
- **Result**: Both variants export successfully with `--num-layers 2`
  - 2.2B: vision(384px, 27x27 grid, 1152 hidden, 16 heads, 27 layers), text(2048 hidden, 32 heads, 24 layers)
  - 256M: vision(512px, 32x32 grid, 768 hidden, 12 heads, 12 layers), text(576 hidden, 9 heads, 30 layers)

### 2026-06-29 -- Phase 2-4: Text decoder + export script implementation
- Created `python/src/coreai_models/models/gpu/smolvlm.py` -- inputs_embeds text decoder
- Created `python/export_smolvlm.py` -- full VLM bundle export script
- Added registry entry for "smolvlm" in `models/registry.py`
- First attempt: `from_hf_memory_efficient` failed because `lm_head.weight` not prefixed
- Fix: manual safetensors loading in export script with explicit key mapping
- **Result**: `uv run python python/export_smolvlm.py --num-layers 2` PASS
  - vision.aimodel: exported (12 SigLIP layers + connector)
  - embed.aimodel: exported (embedding lookup)
  - smolvlm_256m.aimodel: exported (2 text decoder layers)
  - metadata.json: correct VLM schema with vision config
  - tokenizer/: saved from HF

# SmolVLM2 Export -- PROGRESS

**Last updated**: 2026-06-29 by python-worker
**Last gate result**: `uv run python python/export_smolvlm.py --num-layers 2` -- PASS

## STATUS
| Phase | Status | Gate Result |
|-------|--------|-------------|
| 1. Research & architecture | Done | Config confirmed, plan in docs/vlm-export-plan.md |
| 2. Text decoder (inputs_embeds) | Done | Model loads, fuses qkv, exports successfully |
| 3. Vision encoder + connector export | Done | 12-layer SigLIP + pixel_shuffle + Linear exported |
| 4. Export script + bundle | Done | Full bundle: text + embed + vision + tokenizer + metadata |
| 5. E2E test with VLM engine | Pending | |

**Current blocker**: None
**Next action**: E2E integration test with VLM Swift runner

## DECISIONS
- D1: Use Mistral model as template (same Llama arch, same primitives)
- D2: Fuse vision encoder + connector into single vision.aimodel
- D3: Fixed 512x512 input for v1 (no image splitting)
- D4: Target coreai-models-mine repo, sukru/smolvlm branch
- D5: Created `models/gpu/` directory for VLM-specific models (inputs_embeds variant)
- D6: Manual safetensors loading for text decoder (lm_head.weight not under model.text_model. prefix)

## LEARNINGS
- SmolLM2 text decoder is model_type="llama" -- identical structure to Mistral sans sliding window
- scale_factor=4 (not 2) -- pixel_shuffle reduces 1024->64 tokens
- Connector is single Linear(12288->576) -- trivially fused with vision encoder
- `lm_head.weight` in SmolVLM lives at top-level (not under `model.text_model.`) so
  `from_hf_memory_efficient` with `hf_state_dict_prefix` cannot load it directly.
  Manual safetensors loading is needed for the text decoder.
- HF_HUB_OFFLINE=1 env var enables offline mode for snapshot_download (proxy blocks network)
- Vision encoder uses standard SigLIP attention (with bias) -- no RoPE, no causal mask

---
## DETAILED LOG (append-only, newest first)

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

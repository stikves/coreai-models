# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""VLM export pipeline orchestration.

Exports a HuggingFace VLM (LLaVA-1.5 today) to a model bundle with
three sub-models declared in metadata's `assets` map:

    {
      "models": {
        "main":      "model.aimodel",   # LLM with embedding input (stateful)
        "vision":    "vision.aimodel",  # CLIP encoder + projector
        "embedding": "embed.aimodel"    # text embed_tokens lookup
      }
    }

The model component reuses the same auto-aliased KV cache pattern as
text-only LLM exports (`coreai_models.export.macos.export_to_coreai`),
yielding a 2-input + 2-state main graph at runtime.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from coreai_models.export.bundle import METADATA_VERSION
from coreai_models.export.compression import get_c4, quantize_pytorch_model
from coreai_models.export.macos import export_to_coreai
from coreai_models.export.metadata import build_aimodel_metadata
from coreai_models.export.presets import get_preset
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.vlm.components import LLAVA_COMPONENTS, ComponentSpec
from coreai_models.vlm.llava import LLaVATextEmbedder, load_llava_components
from coreai_models.vlm.models import get_family_key

logger = logging.getLogger(__name__)


@dataclass
class VLMExportConfig:
    """Configuration for a VLM export."""

    hf_model_id: str
    output_dir: str = "outputs"
    compute_precision: str = "float16"
    compression: str = "none"
    """Quantization preset name (`'none'`, `'4bit'`, etc.). Applied only to
    the language_model component (the 12 GB one for LLaVA-1.5 7B); vision
    encoder and embed_tokens stay at full precision for now."""

    max_context_length: int | None = None
    overwrite: bool = False


def export_vlm(config: VLMExportConfig) -> str:
    """Export a VLM to a multi-component model bundle.

    Returns the path to the bundle directory.
    """
    return asyncio.run(_async_export_vlm(config))


async def _async_export_vlm(config: VLMExportConfig) -> str:
    family_key = get_family_key(config.hf_model_id)
    if family_key != "llava":
        # Today's component registry only has LLaVA. Other families plug in
        # by adding their own ComponentSpec tuple + branch here.
        raise ValueError(
            f"VLM family '{family_key}' is not supported by this exporter yet. "
            f"Currently supported: 'llava'."
        )

    target_dtype = _resolve_precision(config.compute_precision)

    # 1. Load + split the HF VLM into our three components.
    loaded = load_llava_components(
        config.hf_model_id,
        target_dtype=target_dtype,
        max_context_length=config.max_context_length,
    )

    # 1a. Apply pre-export torch quantization to the language_model.
    if config.compression and config.compression != "none":
        _apply_language_model_quantization(loaded, config, target_dtype)

    # 2. Prepare output bundle directory.
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_name = config.hf_model_id.split("/")[-1]
    if config.compression and config.compression != "none":
        suffix = get_preset(config.compression).get("suffix") or config.compression
        bundle_name = f"{bundle_name}_{suffix}"
    bundle_path = output_dir / bundle_name
    bundle_path.mkdir(parents=True, exist_ok=True)

    # 3. Export each component.
    for spec in LLAVA_COMPONENTS:
        asset_path = bundle_path / spec.asset_name
        if asset_path.exists():
            if config.overwrite:
                import shutil

                shutil.rmtree(asset_path)
            else:
                raise FileExistsError(
                    f"{asset_path} already exists. Use --overwrite to replace it."
                )
        await _export_component(spec, loaded, asset_path, config.hf_model_id)

    # 4. Sidecars: tokenizer + metadata.
    _save_tokenizer(bundle_path / "tokenizer", config.hf_model_id)
    _write_metadata(
        bundle_path,
        config.hf_model_id,
        loaded["text_config"],
        loaded["vision_config"],
        config.compression,
    )

    logger.info(f"Export complete: {bundle_path}")
    return str(bundle_path)


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------


def _apply_language_model_quantization(
    loaded: dict[str, Any], config: VLMExportConfig, target_dtype: torch.dtype
) -> None:
    """Apply pre-export torch quantization to the LLM backbone.

    Quantizes against the regular `forward(input_ids, ...)` signature — the
    resulting quantized weights are the same Linear layers reused by
    `forward_from_embeddings`, so the model component's export still runs
    through the embedding-input path.

    Updates `loaded["language_model"]` in place; rebuilds
    `loaded["text_embedder"]` so it points at the new (quantized) model's
    embed_tokens.
    """
    preset = get_preset(config.compression)
    torch_quantization_config = preset.get("torch_quantization_config")
    if torch_quantization_config is None:
        return  # Preset has no torch quant config (e.g. palettization-only or 'none')

    logger.info(
        f"Applying pre-export torch quantization (preset={config.compression}) to language_model..."
    )

    language_model = loaded["language_model"]
    text_config = loaded["text_config"]

    max_context_length = getattr(text_config, "max_position_embeddings", 2048)
    vocab_size = text_config.vocab_size
    query_len = max(8, max_context_length // 4)

    input_ids = torch.randint(1, vocab_size, (1, query_len), dtype=torch.int32)
    position_ids = (
        torch.arange(query_len + 1024, dtype=torch.int32).unsqueeze(0).expand(1, query_len + 1024)
    )
    k_cache, v_cache = KVCache.create_cache_tensors(text_config, dtype=target_dtype)

    quantization_inputs = (input_ids, position_ids, k_cache, v_cache)
    quantization_dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_ids", min=2, max=max_context_length - 2)},
        "position_ids": {1: torch.export.Dim("seq_pos", min=query_len, max=max_context_length - 1)},
        "k_cache": None,
        "v_cache": None,
    }

    def get_calibration_data():  # type: ignore[no-untyped-def]
        tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)
        return get_c4(tokenizer)

    # Augment exclusions to keep `embed_tokens` at full precision. The shared
    # `4bit` preset only excludes SDPA / RoPE / RMSNorm — `nn.Embedding` is
    # quantized by default. For VLM that's bad: at runtime the LLM consumes a
    # mixed sequence of int4-decoded text rows and fp16 image-projector rows
    # (scatter-merged); keeping embed_tokens at fp16 avoids that precision
    # asymmetry. Cost is one tensor (~250 MB for LLaVA-1.5 7B's
    # vocab × hidden × 2B).
    quant_cfg = dict(torch_quantization_config)
    module_configs = dict(quant_cfg.get("module_type_configs") or {})
    module_configs.setdefault("torch.nn.modules.sparse.Embedding", None)
    quant_cfg["module_type_configs"] = module_configs

    # Use plain `symmetric` instead of `symmetric_with_clipping` for VLM.
    # The shared `4bit` preset uses SYMMETRIC_WITH_CLIPPING which restricts
    # int4 to (-7, 7) for "equal bins" symmetry, throwing away one negative
    # level (15 effective levels vs 16). For text-only LLMs the loss is
    # negligible. For VLM at the same block-size=32 the dropped bin matters
    # more on outlier blocks — using the full (-8, 7) range produces
    # noticeably better captions in side-by-side comparisons.
    quant_cfg = _override_weight_qscheme(quant_cfg, "symmetric")

    quantized_model = quantize_pytorch_model(
        language_model,
        quantization_inputs,
        quantization_dynamic_shapes,
        quant_cfg,
        calibration_data_fn=get_calibration_data,
    )

    loaded["language_model"] = quantized_model
    # text_embedder pointed at the *old* model's embed_tokens — repoint it.
    loaded["text_embedder"] = LLaVATextEmbedder(quantized_model.model.embed_tokens)
    loaded["text_embedder"].eval()


# ---------------------------------------------------------------------------
# Per-component export
# ---------------------------------------------------------------------------


async def _export_component(
    spec: ComponentSpec, loaded: dict[str, Any], asset_path: Path, hf_model_id: str
) -> None:
    """Export one component, writing a `.aimodel` directory at `asset_path`.

    All components route through `export_to_coreai`. The KV-cache
    auto-aliasing kicks in for components whose input/output names match
    on the same key (e.g. `keyCache` in + `keyCache` out for the model
    component). Vision and embedding don't have matching names, so they
    end up as plain stateless graphs.
    """
    logger.info(f"Exporting component '{spec.component_key}' → {asset_path.name}...")
    wrapper = spec.wrapper_fn(loaded)
    wrapper.eval()

    reference_inputs, dynamic_shapes = spec.dummy_fn(loaded)
    program = export_to_coreai(
        wrapper,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=spec.input_names,
        output_names=spec.output_names,
        state_names=spec.state_names if spec.state_names else None,
    )

    program.optimize()
    metadata = build_aimodel_metadata(hf_model_id, component=spec.component_key)
    program.save_asset(asset_path, metadata)
    del program


# ---------------------------------------------------------------------------
# Sidecar files
# ---------------------------------------------------------------------------


def _save_tokenizer(dest: Path, hf_model_id: str) -> None:
    logger.info(f"Saving tokenizer from {hf_model_id} → {dest}...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokenizer.save_pretrained(str(dest))


def _write_metadata(
    bundle_path: Path,
    hf_model_id: str,
    text_config: Any,
    vision_config: Any,
    compression: str,
) -> None:
    """Write metadata.json (0.2 schema) with VLM multi-component assets."""
    image_size = getattr(vision_config, "image_size", 336)
    patch_size = getattr(vision_config, "patch_size", 14)
    num_patches = (image_size // patch_size) ** 2

    metadata: dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "kind": "vlm",
        "name": bundle_path.name,
        "assets": {spec.component_key: spec.asset_name for spec in LLAVA_COMPONENTS},
        "language": {
            "tokenizer": hf_model_id,
            "vocab_size": getattr(text_config, "vocab_size", None),
            "max_context_length": getattr(text_config, "max_position_embeddings", None),
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "image_size": image_size,
            "patch_size": patch_size,
            "image_token_count": num_patches,
            "image_token_id": 32000,
        },
        "source": {
            "model_definition": "torch",
            "hf_model_id": hf_model_id,
        },
        "compression": compression if compression != "none" else None,
        "compilation": {
            "date": datetime.now().astimezone().isoformat(),
            "targets": [],
        },
    }
    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_precision(precision_str: str) -> torch.dtype:
    precision_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = precision_map.get(precision_str)
    if dtype is None:
        raise ValueError(
            f"Unsupported compute_precision '{precision_str}'. "
            f"Supported: {', '.join(precision_map.keys())}"
        )
    return dtype


def _override_weight_qscheme(quant_cfg: dict[str, Any], qscheme: str) -> dict[str, Any]:
    """Return a copy of `quant_cfg` with weight qscheme overridden.

    Drills into `global_config.op_state_spec.weight.qscheme`. If any layer
    in the path is missing, the original config is returned unchanged
    (keeps this resilient to preset-shape changes upstream).
    """
    out = dict(quant_cfg)
    global_cfg = out.get("global_config")
    if not isinstance(global_cfg, dict):
        return out
    global_cfg = dict(global_cfg)
    op_state = global_cfg.get("op_state_spec")
    if not isinstance(op_state, dict):
        return out
    op_state = dict(op_state)
    weight = op_state.get("weight")
    if not isinstance(weight, dict):
        return out
    weight = dict(weight)
    weight["qscheme"] = qscheme
    op_state["weight"] = weight
    global_cfg["op_state_spec"] = op_state
    out["global_config"] = global_cfg
    return out

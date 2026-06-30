#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export Gemma3 4B as a complete VLM bundle.

Produces exports/gemma3_4b_vlm.llmasset/ with:
  - gemma3_4b_vlm.aimodel     (text decoder, inputs_embeds variant)
  - embed.aimodel              (token embedding lookup with embed_scale baked in)
  - vision.aimodel             (SigLIP ViT 27-layer + AvgPool + RMSNorm + Linear projector)
  - tokenizer/                 (embedded HF tokenizer)
  - metadata.json              (bundle manifest, kind=vlm)
"""

import argparse
import json
import logging
import math
import os
import re
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HF_MODEL_ID = "google/gemma-3-4b-it"

# Vision constants (SigLIP ViT)
IMAGE_SIZE = 896
PATCH_SIZE = 14
GRID_SIZE = IMAGE_SIZE // PATCH_SIZE  # 64
NUM_PATCHES = GRID_SIZE ** 2  # 4096
VISION_HIDDEN_SIZE = 1152
VISION_NUM_HEADS = 16
VISION_HEAD_DIM = VISION_HIDDEN_SIZE // VISION_NUM_HEADS  # 72
VISION_INTERMEDIATE_SIZE = 4304
VISION_NUM_LAYERS = 27
VISION_LAYER_NORM_EPS = 1e-6

# Projector constants
AVGPOOL_KERNEL = 4
AVGPOOL_STRIDE = 4
PROJECTED_GRID = GRID_SIZE // AVGPOOL_KERNEL  # 16
IMAGE_SEQ_LEN = PROJECTED_GRID ** 2  # 256
TEXT_HIDDEN_SIZE = 2560

# Text decoder constants
IMAGE_TOKEN_ID = 262144
VOCAB_SIZE = 262208
EMBED_SCALE = TEXT_HIDDEN_SIZE ** 0.5  # sqrt(2560) ~ 50.6


def _find_repo_root() -> Path:
    """Walk up from this file to find the workspace root."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Vision Encoder + Projector (fused into a single exportable module)
# ---------------------------------------------------------------------------


class Gemma3VisionEncoder(nn.Module):
    """SigLIP ViT + AvgPool + RMSNorm + Linear projector for Gemma3.

    Takes pixel_values [1, 3, 896, 896] and outputs [1, 256, 2560] float16.
    """

    def __init__(self, num_layers: int = VISION_NUM_LAYERS) -> None:
        super().__init__()
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, VISION_HIDDEN_SIZE, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=True
        )
        # Learnable position embeddings for 64x64 grid = 4096 patches
        self.position_embedding = nn.Embedding(NUM_PATCHES, VISION_HIDDEN_SIZE)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList(
            [VisionEncoderLayer() for _ in range(num_layers)]
        )

        # Post-layernorm (LayerNorm with bias)
        self.post_layernorm = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

        # Projector: AvgPool2d(4,4) + RMSNorm + Linear
        self.proj_norm_weight = nn.Parameter(torch.ones(VISION_HIDDEN_SIZE))
        self.proj_linear_weight = nn.Parameter(
            torch.empty(VISION_HIDDEN_SIZE, TEXT_HIDDEN_SIZE)
        )

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm: weight * x * rsqrt(mean(x^2) + eps)."""
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + VISION_LAYER_NORM_EPS)
        return (weight.float() * x).to(x.dtype)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: [1, 3, 896, 896] float32

        Returns:
            [1, 256, 2560] float16 - vision embeddings ready for text decoder
        """
        # Patch embedding: [1, 3, 896, 896] -> [1, 1152, 64, 64]
        x = self.patch_embedding(pixel_values)
        # Flatten spatial: [1, 1152, 64, 64] -> [1, 4096, 1152]
        x = x.flatten(2).transpose(1, 2)

        # Add position embeddings
        position_ids = torch.arange(NUM_PATCHES, device=x.device)
        x = x + self.position_embedding(position_ids)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Post-layernorm
        x = self.post_layernorm(x)

        # AvgPool2d(4,4): reshape to spatial grid, pool, flatten back
        # [1, 4096, 1152] -> [1, 64, 64, 1152] -> permute -> [1, 1152, 64, 64]
        batch_size = x.shape[0]
        x = x.reshape(batch_size, GRID_SIZE, GRID_SIZE, VISION_HIDDEN_SIZE)
        x = x.permute(0, 3, 1, 2)  # [1, 1152, 64, 64]
        x = F.avg_pool2d(x, kernel_size=AVGPOOL_KERNEL, stride=AVGPOOL_STRIDE)  # [1, 1152, 16, 16]
        x = x.flatten(2).transpose(1, 2)  # [1, 256, 1152]

        # RMSNorm (projector norm)
        x = self._rms_norm(x, self.proj_norm_weight)

        # Linear projection: [1, 256, 1152] @ [1152, 2560] -> [1, 256, 2560]
        x = torch.matmul(x, self.proj_linear_weight)

        return x.to(torch.float16)


class VisionEncoderLayer(nn.Module):
    """Single SigLIP encoder layer with pre-norm."""

    def __init__(self) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)
        self.layer_norm2 = nn.LayerNorm(VISION_HIDDEN_SIZE, eps=VISION_LAYER_NORM_EPS)

        self.self_attn = VisionAttention()
        self.mlp = VisionMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.layer_norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x


class VisionAttention(nn.Module):
    """Multi-head attention for vision encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.num_heads = VISION_NUM_HEADS
        self.head_dim = VISION_HEAD_DIM

        self.q_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.k_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.v_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)
        self.out_proj = nn.Linear(VISION_HIDDEN_SIZE, VISION_HIDDEN_SIZE, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(attn)


class VisionMLP(nn.Module):
    """MLP for vision encoder (GELU approximate tanh activation)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(VISION_HIDDEN_SIZE, VISION_INTERMEDIATE_SIZE, bias=True)
        self.fc2 = nn.Linear(VISION_INTERMEDIATE_SIZE, VISION_HIDDEN_SIZE, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Embedding model (with embed_scale baked in)
# ---------------------------------------------------------------------------


class EmbedModel(nn.Module):
    """Token embedding lookup for Gemma3 VLM text pipeline.

    Bakes in the embed_scale (sqrt(hidden_size)) so the runtime gets
    correctly-scaled embeddings without needing to know about the scaling.
    """

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.embed_scale = hidden_size ** 0.5

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings and apply embed_scale.

        Args:
            input_ids: [1, seq_len] int32

        Returns:
            [1, seq_len, hidden_size] float16
        """
        embeddings = self.embed_tokens(input_ids)
        return (embeddings * torch.tensor(
            self.embed_scale, dtype=embeddings.dtype, device=embeddings.device
        )).to(torch.float16)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _resolve_safetensors_files(model_dir: str) -> list[str]:
    """Resolve safetensors file paths (single or sharded)."""
    single_path = os.path.join(model_dir, "model.safetensors")
    index_path = os.path.join(model_dir, "model.safetensors.index.json")

    if os.path.isfile(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_filenames = sorted(set(index["weight_map"].values()))
        return [os.path.join(model_dir, fn) for fn in shard_filenames]
    elif os.path.isfile(single_path):
        return [single_path]
    else:
        raise FileNotFoundError(
            f"No safetensors files found in {model_dir}. "
            "Expected model.safetensors or model.safetensors.index.json."
        )


def _open_safetensors(safetensors_files: list[str]):
    """Build a key-to-file mapping from multiple safetensors files."""
    key_to_file: dict[str, str] = {}
    for path in safetensors_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118
                key_to_file[key] = path
    return key_to_file


def _get_tensor(key_to_file: dict[str, str], key: str) -> torch.Tensor:
    """Load a single tensor by key from the appropriate shard."""
    path = key_to_file[key]
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(key)


def _load_vision_weights(
    vision_model: Gemma3VisionEncoder,
    key_to_file: dict[str, str],
    num_layers: int | None = None,
) -> None:
    """Load vision encoder + projector weights from the full model safetensors."""
    # Patch embedding
    vision_model.patch_embedding.weight.data = _get_tensor(
        key_to_file, "vision_tower.vision_model.embeddings.patch_embedding.weight"
    )
    vision_model.patch_embedding.bias.data = _get_tensor(
        key_to_file, "vision_tower.vision_model.embeddings.patch_embedding.bias"
    )

    # Position embedding
    vision_model.position_embedding.weight.data = _get_tensor(
        key_to_file, "vision_tower.vision_model.embeddings.position_embedding.weight"
    )

    # Post-layernorm
    vision_model.post_layernorm.weight.data = _get_tensor(
        key_to_file, "vision_tower.vision_model.post_layernorm.weight"
    )
    vision_model.post_layernorm.bias.data = _get_tensor(
        key_to_file, "vision_tower.vision_model.post_layernorm.bias"
    )

    # Projector: RMSNorm weight + linear projection weight
    vision_model.proj_norm_weight.data = _get_tensor(
        key_to_file, "multi_modal_projector.mm_soft_emb_norm.weight"
    )
    vision_model.proj_linear_weight.data = _get_tensor(
        key_to_file, "multi_modal_projector.mm_input_projection_weight"
    )

    # Determine how many layers to load
    actual_layers = len(vision_model.encoder_layers)
    load_layers = min(actual_layers, num_layers) if num_layers is not None else actual_layers

    logger.info(f"Loading {load_layers} vision encoder layers...")

    # Load each encoder layer
    for i in range(load_layers):
        layer = vision_model.encoder_layers[i]
        prefix = f"vision_tower.vision_model.encoder.layers.{i}"

        # Layer norms
        layer.layer_norm1.weight.data = _get_tensor(key_to_file, f"{prefix}.layer_norm1.weight")
        layer.layer_norm1.bias.data = _get_tensor(key_to_file, f"{prefix}.layer_norm1.bias")
        layer.layer_norm2.weight.data = _get_tensor(key_to_file, f"{prefix}.layer_norm2.weight")
        layer.layer_norm2.bias.data = _get_tensor(key_to_file, f"{prefix}.layer_norm2.bias")

        # Self-attention
        layer.self_attn.q_proj.weight.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.q_proj.weight"
        )
        layer.self_attn.q_proj.bias.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.q_proj.bias"
        )
        layer.self_attn.k_proj.weight.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.k_proj.weight"
        )
        layer.self_attn.k_proj.bias.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.k_proj.bias"
        )
        layer.self_attn.v_proj.weight.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.v_proj.weight"
        )
        layer.self_attn.v_proj.bias.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.v_proj.bias"
        )
        layer.self_attn.out_proj.weight.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.out_proj.weight"
        )
        layer.self_attn.out_proj.bias.data = _get_tensor(
            key_to_file, f"{prefix}.self_attn.out_proj.bias"
        )

        # MLP
        layer.mlp.fc1.weight.data = _get_tensor(key_to_file, f"{prefix}.mlp.fc1.weight")
        layer.mlp.fc1.bias.data = _get_tensor(key_to_file, f"{prefix}.mlp.fc1.bias")
        layer.mlp.fc2.weight.data = _get_tensor(key_to_file, f"{prefix}.mlp.fc2.weight")
        layer.mlp.fc2.bias.data = _get_tensor(key_to_file, f"{prefix}.mlp.fc2.bias")


def _load_embed_weights(embed_model: EmbedModel, key_to_file: dict[str, str]) -> None:
    """Load embedding weights from the full model safetensors."""
    embed_model.embed_tokens.weight.data = _get_tensor(
        key_to_file, "language_model.model.embed_tokens.weight"
    )


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_vision(
    key_to_file: dict[str, str],
    output_path: Path,
    num_vision_layers: int | None = None,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the vision encoder + projector as vision.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    actual_layers = num_vision_layers if num_vision_layers is not None else VISION_NUM_LAYERS
    logger.info(f"Building vision encoder ({actual_layers} layers)...")
    vision_model = Gemma3VisionEncoder(num_layers=actual_layers)
    _load_vision_weights(vision_model, key_to_file, num_layers=num_vision_layers)
    # Vision encoder runs in float32 (input is float32 pixel values, output is cast to fp16)
    vision_model = vision_model.float().eval()

    # Vision accepts float32 input (pixel values), outputs float16 (cast at end of forward)
    pixel_values = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)

    reference_inputs = {"pixel_values": pixel_values}

    logger.info("Exporting vision encoder to Core AI...")
    coreai_program = export_to_coreai(
        vision_model,
        reference_inputs,
        dynamic_shapes=None,
        input_names=("pixel_values",),
        output_names=("image_embeds",),
        state_names=None,
    )
    coreai_program.optimize()

    metadata = build_aimodel_metadata(HF_MODEL_ID, component="vision")
    coreai_program.save_asset(output_path, metadata)
    logger.info(f"Saved vision.aimodel to {output_path}")


def export_embed(
    key_to_file: dict[str, str],
    output_path: Path,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the token embedding lookup (with embed_scale) as embed.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    logger.info("Building embedding model...")
    embed_model = EmbedModel(VOCAB_SIZE, TEXT_HIDDEN_SIZE)
    _load_embed_weights(embed_model, key_to_file)
    embed_model = embed_model.to(target_dtype).eval()

    # Dynamic seq_len for embedding lookup
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 16), dtype=torch.int32)

    reference_inputs = {"input_ids": input_ids}
    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_len", min=1, max=131072)},
    }

    logger.info("Exporting embedding model to Core AI...")
    coreai_program = export_to_coreai(
        embed_model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("input_ids",),
        output_names=("embeddings",),
        state_names=None,
    )
    coreai_program.optimize()

    metadata = build_aimodel_metadata(HF_MODEL_ID, component="embedding")
    coreai_program.save_asset(output_path, metadata)
    logger.info(f"Saved embed.aimodel to {output_path}")


def export_text_decoder(
    model_dir: str,
    key_to_file: dict[str, str],
    output_path: Path,
    text_config,
    target_dtype: torch.dtype = torch.float16,
    num_layers: int | None = None,
    max_context_length: int = 131072,
) -> None:
    """Export the text decoder (inputs_embeds variant) as the main .aimodel."""
    from coreai_models.export._constants import (
        QUANT_TRACE_OFFSET,
        QUANT_TRACE_QUERY_LEN,
        TRACE_KV_CACHE_SEQ_LEN,
    )
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata
    from coreai_models.models.gpu.gemma3_vlm import Gemma3VLMForCausalLM
    from coreai_models.primitives.macos.cache import KVCache

    logger.info("Loading text decoder...")

    # Build config with optional layer truncation
    config = Gemma3VLMForCausalLM._get_reauthored_config(
        text_config, max_context_length, num_layers=num_layers
    )

    # Create model on meta device
    model = Gemma3VLMForCausalLM(config, model_device="meta")
    model.to(dtype=target_dtype)

    # Load weights from safetensors manually
    # Source prefix: language_model.model.* -> stripped to model.*
    state_dict: dict[str, torch.Tensor] = {}
    prefix = "language_model.model."

    for key, file_path in key_to_file.items():
        if key.startswith(prefix):
            stripped = key.removeprefix(prefix)
            # Skip layers beyond num_layers
            if num_layers is not None:
                match = re.search(r"layers\.(\d+)\.", stripped)
                if match and int(match.group(1)) >= num_layers:
                    continue
            with safe_open(file_path, framework="pt", device="cpu") as f:
                tensor = f.get_tensor(key)
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            state_dict[f"model.{stripped}"] = tensor

    # For lm_head: Gemma3 uses tie_word_embeddings=True, so lm_head.weight = embed_tokens.weight
    # We need lm_head in the decoder but embed_tokens goes to embed.aimodel
    # Copy embed_tokens as lm_head
    embed_key = "model.embed_tokens.weight"
    if embed_key in state_dict:
        state_dict["lm_head.weight"] = state_dict[embed_key].clone()

    # Apply weight mutations (qkv fusion, qk_norm fusion, drop embed_tokens)
    model._mutate_state_dict(state_dict)

    # Load into model
    strict = num_layers is None
    model.load_state_dict(state_dict, assign=True, strict=strict)
    del state_dict

    model = model.eval()

    config = model.config
    batch_size = 1
    query_len = QUANT_TRACE_QUERY_LEN

    # inputs_embeds instead of input_ids
    inputs_embeds = torch.randn(
        batch_size, query_len, config.hidden_size, dtype=target_dtype
    )
    position_ids = (
        torch.arange(query_len + QUANT_TRACE_OFFSET, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, query_len + QUANT_TRACE_OFFSET)
    )

    # Create KV cache
    saved_max_pos = config.max_position_embeddings
    config.max_position_embeddings = TRACE_KV_CACHE_SEQ_LEN
    k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=target_dtype)
    config.max_position_embeddings = saved_max_pos

    reference_inputs = {
        "inputs_embeds": inputs_embeds,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }

    dynamic_shapes = {
        "inputs_embeds": {1: torch.export.Dim("query_len", max=max_context_length - 2)},
        "position_ids": {
            1: torch.export.Dim(
                "seq_pos", min=QUANT_TRACE_QUERY_LEN, max=max_context_length - 1
            )
        },
        "k_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_context_length
            )
        },
        "v_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq_len", min=TRACE_KV_CACHE_SEQ_LEN, max=max_context_length
            )
        },
    }

    logger.info("Exporting text decoder to Core AI...")
    coreai_program = export_to_coreai(
        model,
        reference_inputs,
        dynamic_shapes=dynamic_shapes,
        input_names=("inputs_embeds", "position_ids"),
        output_names=("logits",),
        state_names=("k_cache", "v_cache"),
    )
    coreai_program.optimize()

    metadata = build_aimodel_metadata(HF_MODEL_ID, component="text_decoder")
    coreai_program.save_asset(output_path, metadata)
    logger.info(f"Saved text decoder to {output_path}")


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


def write_vlm_metadata(bundle_path: Path, hf_config, name: str) -> None:
    """Write metadata.json for the VLM bundle."""
    text_config = hf_config.text_config

    metadata = {
        "metadata_version": "0.2",
        "kind": "vlm",
        "name": name,
        "assets": {
            "main": f"{name}.aimodel",
            "embedding": "embed.aimodel",
            "vision": "vision.aimodel",
        },
        "language": {
            "tokenizer": HF_MODEL_ID,
            "vocab_size": text_config.vocab_size,
            "max_context_length": text_config.max_position_embeddings,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "image_size": IMAGE_SIZE,
            "patch_size": PATCH_SIZE,
            "image_token_count": IMAGE_SEQ_LEN,
            "image_token_id": IMAGE_TOKEN_ID,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1.0,
        },
        "source": {
            "model_definition": "torch",
            "hf_model_id": HF_MODEL_ID,
        },
    }

    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")


def write_tokenizer(bundle_path: Path, model_dir: str) -> None:
    """Save the HF tokenizer into the bundle from local model dir."""
    tokenizer_dir = bundle_path / "tokenizer"
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.save_pretrained(str(tokenizer_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Gemma3 4B VLM bundle")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: <repo>/exports/)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Truncate text decoder to N layers (for smoke testing)",
    )
    parser.add_argument(
        "--num-vision-layers",
        type=int,
        default=None,
        help="Truncate vision encoder to N layers (for smoke testing)",
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip vision encoder export",
    )
    parser.add_argument(
        "--skip-embed",
        action="store_true",
        help="Skip embedding model export",
    )
    parser.add_argument(
        "--skip-text",
        action="store_true",
        help="Skip text decoder export",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=True,
        help="Overwrite existing output (default: True)",
    )
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        root = _find_repo_root()
        output_dir = root / "exports"

    bundle_name = "gemma3_4b_vlm"
    bundle_path = output_dir / f"{bundle_name}.llmasset"

    if bundle_path.exists() and args.overwrite:
        shutil.rmtree(bundle_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting Gemma3 4B VLM to {bundle_path}")

    # Download/resolve model files (use local cache if network unavailable)
    try:
        model_dir = snapshot_download(
            HF_MODEL_ID,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )
    except (OSError, Exception):
        logger.info("Network unavailable, attempting offline resolution...")
        model_dir = snapshot_download(
            HF_MODEL_ID,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            local_files_only=True,
        )

    logger.info(f"Using model files from: {model_dir}")

    # Build key-to-file mapping for all safetensors
    safetensors_files = _resolve_safetensors_files(model_dir)
    key_to_file = _open_safetensors(safetensors_files)

    # Load HF config for metadata
    hf_config = AutoConfig.from_pretrained(model_dir)
    text_config = hf_config.text_config

    target_dtype = torch.float16

    # When --num-layers is set for smoke testing, also limit vision layers
    # unless --num-vision-layers is explicitly provided
    num_vision_layers = args.num_vision_layers
    if num_vision_layers is None and args.num_layers is not None:
        num_vision_layers = args.num_layers

    # 1. Export vision encoder + projector
    if not args.skip_vision:
        vision_output = bundle_path / "vision.aimodel"
        export_vision(key_to_file, vision_output, num_vision_layers, target_dtype)

    # 2. Export embedding model
    if not args.skip_embed:
        embed_output = bundle_path / "embed.aimodel"
        export_embed(key_to_file, embed_output, target_dtype)

    # 3. Export text decoder
    if not args.skip_text:
        text_output = bundle_path / f"{bundle_name}.aimodel"
        export_text_decoder(
            model_dir,
            key_to_file,
            text_output,
            text_config=text_config,
            target_dtype=target_dtype,
            num_layers=args.num_layers,
        )

    # 4. Write tokenizer
    write_tokenizer(bundle_path, model_dir)

    # 5. Write VLM metadata
    write_vlm_metadata(bundle_path, hf_config, bundle_name)

    logger.info(f"Gemma3 4B VLM export complete: {bundle_path}")


if __name__ == "__main__":
    main()

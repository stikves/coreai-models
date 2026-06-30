#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export SmolVLM2 variants as complete VLM bundles.

Supports both SmolVLM2-256M and SmolVLM2-2.2B (and future sizes).
All model constants are read from the HF config at runtime.

Produces exports/<name>.llmasset/ with:
  - <name>.aimodel             (text decoder, inputs_embeds variant)
  - embed.aimodel              (token embedding lookup)
  - vision.aimodel             (SigLIP ViT + pixel_shuffle connector)
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
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


def _find_repo_root() -> Path:
    """Walk up from this file to find the workspace root."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return Path(__file__).resolve().parent


def _derive_bundle_name(model_id: str) -> str:
    """Derive a short bundle name from the HF model ID.

    Examples:
        HuggingFaceTB/SmolVLM2-256M-Video-Instruct -> smolvlm_256m
        HuggingFaceTB/SmolVLM2-2.2B-Instruct -> smolvlm_2_2b
    """
    # Extract the model name part after the slash
    name = model_id.split("/")[-1]  # e.g. SmolVLM2-256M-Video-Instruct

    # Find the size indicator (e.g. 256M, 2.2B)
    match = re.search(r"SmolVLM2?-([0-9.]+[BMbm])", name)
    if match:
        size = match.group(1).lower()  # e.g. "256m" or "2.2b"
        # Replace dots with underscores for filesystem safety
        size = size.replace(".", "_")
        return f"smolvlm_{size}"

    # Fallback: sanitize the full name
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


# ---------------------------------------------------------------------------
# Sharded safetensors loader
# ---------------------------------------------------------------------------


class ShardedSafetensors:
    """Handles both single-file and sharded (index.json) safetensors."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")

        if os.path.exists(index_path):
            # Sharded: load the index
            with open(index_path) as f:
                index = json.load(f)
            self.weight_map = index["weight_map"]
            self.sharded = True
        else:
            # Single file
            self.single_path = os.path.join(model_dir, "model.safetensors")
            self.weight_map = None
            self.sharded = False

        # Cache of open file handles
        self._handles: dict[str, object] = {}

    def _get_handle(self, shard_file: str):
        """Get or open a safetensors file handle."""
        if shard_file not in self._handles:
            path = os.path.join(self.model_dir, shard_file)
            self._handles[shard_file] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_file]

    def get_tensor(self, key: str) -> torch.Tensor:
        """Load a tensor by key, resolving shards as needed."""
        if self.sharded:
            shard_file = self.weight_map[key]
            handle = self._get_handle(shard_file)
            return handle.get_tensor(key)
        else:
            handle = self._get_handle("model.safetensors")
            return handle.get_tensor(key)

    def keys(self) -> list[str]:
        """List all available tensor keys."""
        if self.sharded:
            return list(self.weight_map.keys())
        else:
            handle = self._get_handle("model.safetensors")
            return list(handle.keys())

    def close(self) -> None:
        """Close all handles."""
        self._handles.clear()


# ---------------------------------------------------------------------------
# Vision Encoder + Connector (fused into a single exportable module)
# ---------------------------------------------------------------------------


class SmolVLMVisionEncoder(nn.Module):
    """SigLIP ViT + pixel_shuffle connector for SmolVLM2.

    All dimensions are derived from the HF config at construction time.
    Supports non-square grids (e.g., 27x27 for image_size=384, patch_size=14).
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        vision_hidden_size: int,
        vision_num_heads: int,
        vision_intermediate_size: int,
        text_hidden_size: int,
        scale_factor: int,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        # Grid size: floor(image_size / patch_size) per side
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # Derived connector dimensions
        self.connector_in_features = vision_hidden_size * (scale_factor ** 2)

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, vision_hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )
        # Learnable position embeddings
        self.position_embedding = nn.Embedding(self.num_patches, vision_hidden_size)

        # Transformer encoder layers (populated from weights)
        self.encoder_layers = nn.ModuleList()

        # Post-layernorm
        self.post_layernorm = nn.LayerNorm(vision_hidden_size)

        # Connector: pixel_shuffle + linear projection
        self.connector_proj = nn.Linear(self.connector_in_features, text_hidden_size, bias=False)

        # Store for building encoder layers
        self._vision_hidden_size = vision_hidden_size
        self._vision_num_heads = vision_num_heads
        self._vision_intermediate_size = vision_intermediate_size

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel shuffle: [B, seq, hidden] -> [B, seq/scale^2, hidden*scale^2].

        Rearranges spatial patches into channel dimension.
        """
        batch_size, seq_len, hidden = x.shape
        h = w = int(math.sqrt(seq_len))
        # Reshape to spatial grid
        x = x.reshape(batch_size, h, w, hidden)
        # Reshape into scale_factor blocks
        sf = self.scale_factor
        new_h = h // sf
        new_w = w // sf
        x = x.reshape(batch_size, new_h, sf, new_w, sf, hidden)
        # Merge scale dimensions into hidden
        x = x.permute(0, 1, 3, 2, 4, 5)  # [B, new_h, new_w, sf, sf, hidden]
        x = x.reshape(batch_size, new_h * new_w, sf * sf * hidden)
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: [1, 3, image_size, image_size] float32

        Returns:
            [1, image_seq_len, text_hidden_size] float16
        """
        # Patch embedding
        x = self.patch_embedding(pixel_values)
        # Flatten spatial: [1, hidden, grid, grid] -> [1, num_patches, hidden]
        x = x.flatten(2).transpose(1, 2)

        # Add position embeddings
        position_ids = torch.arange(self.num_patches, device=x.device)
        x = x + self.position_embedding(position_ids)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Post-layernorm
        x = self.post_layernorm(x)

        # Pixel shuffle
        x = self._pixel_shuffle(x)

        # Linear projection
        x = self.connector_proj(x)

        return x.to(torch.float16)


class VisionEncoderLayer(nn.Module):
    """Single SigLIP encoder layer with pre-norm."""

    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.self_attn = VisionAttention(hidden_size, num_heads)
        self.mlp = VisionMLP(hidden_size, intermediate_size)

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

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=True)

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

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Embedding model (simple nn.Embedding wrapper)
# ---------------------------------------------------------------------------


class EmbedModel(nn.Module):
    """Token embedding lookup for the VLM text pipeline."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Look up token embeddings.

        Args:
            input_ids: [1, seq_len] int32

        Returns:
            [1, seq_len, hidden_size] float16
        """
        return self.embed_tokens(input_ids).to(torch.float16)


# ---------------------------------------------------------------------------
# Weight loading helpers
# ---------------------------------------------------------------------------


def _load_vision_weights(
    vision_model: SmolVLMVisionEncoder,
    tensors: ShardedSafetensors,
) -> None:
    """Load vision encoder + connector weights from the model safetensors."""
    all_keys = tensors.keys()

    # Patch embedding
    vision_model.patch_embedding.weight.data = tensors.get_tensor(
        "model.vision_model.embeddings.patch_embedding.weight"
    )
    vision_model.patch_embedding.bias.data = tensors.get_tensor(
        "model.vision_model.embeddings.patch_embedding.bias"
    )

    # Position embedding
    vision_model.position_embedding.weight.data = tensors.get_tensor(
        "model.vision_model.embeddings.position_embedding.weight"
    )

    # Post-layernorm
    vision_model.post_layernorm.weight.data = tensors.get_tensor(
        "model.vision_model.post_layernorm.weight"
    )
    vision_model.post_layernorm.bias.data = tensors.get_tensor(
        "model.vision_model.post_layernorm.bias"
    )

    # Connector projection
    vision_model.connector_proj.weight.data = tensors.get_tensor(
        "model.connector.modality_projection.proj.weight"
    )

    # Count encoder layers
    layer_keys = [k for k in all_keys if k.startswith("model.vision_model.encoder.layers.")]
    layer_indices = set()
    for k in layer_keys:
        parts = k.split(".")
        layer_indices.add(int(parts[4]))
    num_layers = max(layer_indices) + 1

    logger.info(f"Loading {num_layers} vision encoder layers...")

    # Build encoder layers with proper dimensions
    hidden_size = vision_model._vision_hidden_size
    num_heads = vision_model._vision_num_heads
    intermediate_size = vision_model._vision_intermediate_size

    vision_model.encoder_layers = nn.ModuleList(
        [VisionEncoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
    )

    # Load each encoder layer
    for i in range(num_layers):
        layer = vision_model.encoder_layers[i]
        prefix = f"model.vision_model.encoder.layers.{i}"

        # Layer norms
        layer.layer_norm1.weight.data = tensors.get_tensor(f"{prefix}.layer_norm1.weight")
        layer.layer_norm1.bias.data = tensors.get_tensor(f"{prefix}.layer_norm1.bias")
        layer.layer_norm2.weight.data = tensors.get_tensor(f"{prefix}.layer_norm2.weight")
        layer.layer_norm2.bias.data = tensors.get_tensor(f"{prefix}.layer_norm2.bias")

        # Self-attention
        layer.self_attn.q_proj.weight.data = tensors.get_tensor(
            f"{prefix}.self_attn.q_proj.weight"
        )
        layer.self_attn.q_proj.bias.data = tensors.get_tensor(f"{prefix}.self_attn.q_proj.bias")
        layer.self_attn.k_proj.weight.data = tensors.get_tensor(
            f"{prefix}.self_attn.k_proj.weight"
        )
        layer.self_attn.k_proj.bias.data = tensors.get_tensor(f"{prefix}.self_attn.k_proj.bias")
        layer.self_attn.v_proj.weight.data = tensors.get_tensor(
            f"{prefix}.self_attn.v_proj.weight"
        )
        layer.self_attn.v_proj.bias.data = tensors.get_tensor(f"{prefix}.self_attn.v_proj.bias")
        layer.self_attn.out_proj.weight.data = tensors.get_tensor(
            f"{prefix}.self_attn.out_proj.weight"
        )
        layer.self_attn.out_proj.bias.data = tensors.get_tensor(
            f"{prefix}.self_attn.out_proj.bias"
        )

        # MLP
        layer.mlp.fc1.weight.data = tensors.get_tensor(f"{prefix}.mlp.fc1.weight")
        layer.mlp.fc1.bias.data = tensors.get_tensor(f"{prefix}.mlp.fc1.bias")
        layer.mlp.fc2.weight.data = tensors.get_tensor(f"{prefix}.mlp.fc2.weight")
        layer.mlp.fc2.bias.data = tensors.get_tensor(f"{prefix}.mlp.fc2.bias")


def _load_embed_weights(embed_model: EmbedModel, tensors: ShardedSafetensors) -> None:
    """Load embedding weights from the model safetensors."""
    embed_model.embed_tokens.weight.data = tensors.get_tensor(
        "model.text_model.embed_tokens.weight"
    )


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_vision(
    tensors: ShardedSafetensors,
    output_path: Path,
    hf_config,
    model_id: str,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the vision encoder + connector as vision.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    vision_config = hf_config.vision_config
    text_config = hf_config.text_config

    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    vision_hidden_size = vision_config.hidden_size
    vision_num_heads = vision_config.num_attention_heads
    vision_intermediate_size = vision_config.intermediate_size
    text_hidden_size = text_config.hidden_size
    scale_factor = hf_config.scale_factor

    logger.info(
        f"Building vision encoder: image={image_size}, patch={patch_size}, "
        f"hidden={vision_hidden_size}, heads={vision_num_heads}, scale={scale_factor}"
    )

    vision_model = SmolVLMVisionEncoder(
        image_size=image_size,
        patch_size=patch_size,
        vision_hidden_size=vision_hidden_size,
        vision_num_heads=vision_num_heads,
        vision_intermediate_size=vision_intermediate_size,
        text_hidden_size=text_hidden_size,
        scale_factor=scale_factor,
    )
    _load_vision_weights(vision_model, tensors)
    vision_model = vision_model.eval()

    # Vision accepts float32 input (pixel values), outputs float16 (cast at end of forward)
    pixel_values = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)

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

    metadata = build_aimodel_metadata(model_id, component="vision")
    coreai_program.save_asset(output_path, metadata)
    logger.info(f"Saved vision.aimodel to {output_path}")


def export_embed(
    tensors: ShardedSafetensors,
    output_path: Path,
    vocab_size: int,
    hidden_size: int,
    model_id: str,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the token embedding lookup as embed.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    logger.info("Building embedding model...")
    embed_model = EmbedModel(vocab_size, hidden_size)
    _load_embed_weights(embed_model, tensors)
    embed_model = embed_model.to(target_dtype).eval()

    # Dynamic seq_len for embedding lookup
    input_ids = torch.randint(0, vocab_size, (1, 16), dtype=torch.int32)

    reference_inputs = {"input_ids": input_ids}
    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_len", min=1, max=8192)},
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

    metadata = build_aimodel_metadata(model_id, component="embedding")
    coreai_program.save_asset(output_path, metadata)
    logger.info(f"Saved embed.aimodel to {output_path}")


def export_text_decoder(
    model_dir: str,
    tensors: ShardedSafetensors,
    output_path: Path,
    text_config,
    model_id: str,
    target_dtype: torch.dtype = torch.float16,
    num_layers: Optional[int] = None,
    max_context_length: int = 8192,
) -> None:
    """Export the text decoder (inputs_embeds variant) as the main .aimodel."""
    from coreai_models.export._constants import (
        QUANT_TRACE_OFFSET,
        QUANT_TRACE_QUERY_LEN,
        TRACE_KV_CACHE_SEQ_LEN,
    )
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata
    from coreai_models.models.gpu.smolvlm import SmolVLMForCausalLMEmbeddings
    from coreai_models.primitives.macos.cache import KVCache

    logger.info("Loading text decoder...")

    # Build config with optional layer truncation
    config = SmolVLMForCausalLMEmbeddings._get_reauthored_config(
        text_config, max_context_length, num_layers=num_layers
    )

    # Create model on meta device
    model = SmolVLMForCausalLMEmbeddings(config, model_device="meta")
    model.to(dtype=target_dtype)

    # Load weights from safetensors manually
    # We need: model.text_model.* (stripped to model.*) and lm_head.*
    state_dict: dict[str, torch.Tensor] = {}
    prefix = "model.text_model."

    all_keys = tensors.keys()
    for key in all_keys:
        if key.startswith(prefix):
            stripped = key.removeprefix(prefix)
            # Skip layers beyond num_layers
            if num_layers is not None:
                match = re.search(r"layers\.(\d+)\.", stripped)
                if match and int(match.group(1)) >= num_layers:
                    continue
            tensor = tensors.get_tensor(key)
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            state_dict[f"model.{stripped}"] = tensor
        elif key == "lm_head.weight":
            tensor = tensors.get_tensor(key)
            if tensor.dtype != target_dtype:
                tensor = tensor.to(target_dtype)
            state_dict["lm_head.weight"] = tensor

    # Apply weight mutations (qkv fusion, drop embed_tokens)
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

    metadata = build_aimodel_metadata(model_id, component="text_decoder")
    coreai_program.save_asset(output_path, metadata)
    logger.info(f"Saved text decoder to {output_path}")


# ---------------------------------------------------------------------------
# Bundle assembly
# ---------------------------------------------------------------------------


def write_vlm_metadata(bundle_path: Path, hf_config, model_id: str, name: str) -> None:
    """Write metadata.json for the VLM bundle."""
    text_config = hf_config.text_config
    vision_config = hf_config.vision_config

    image_size = vision_config.image_size
    patch_size = vision_config.patch_size
    scale_factor = hf_config.scale_factor
    grid_size = image_size // patch_size
    num_patches = grid_size * grid_size
    image_seq_len = num_patches // (scale_factor ** 2)
    image_token_id = hf_config.image_token_id

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
            "tokenizer": model_id,
            "vocab_size": text_config.vocab_size,
            "max_context_length": text_config.max_position_embeddings,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "image_size": image_size,
            "patch_size": patch_size,
            "image_token_count": image_seq_len,
            "image_token_id": image_token_id,
            "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5],
            "rescale_factor": 1.0,
        },
        "source": {
            "model_definition": "torch",
            "hf_model_id": model_id,
        },
    }

    metadata_path = bundle_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Wrote metadata to {metadata_path}")


def write_tokenizer(bundle_path: Path, model_id: str) -> None:
    """Save the HF tokenizer into the bundle."""
    tokenizer_dir = bundle_path / "tokenizer"
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(str(tokenizer_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SmolVLM2 VLM bundle")
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HuggingFace model ID (default: {DEFAULT_MODEL_ID})",
    )
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

    model_id = args.model_id

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        root = _find_repo_root()
        output_dir = root / "exports"

    bundle_name = _derive_bundle_name(model_id)
    bundle_path = output_dir / f"{bundle_name}.llmasset"

    if bundle_path.exists() and args.overwrite:
        shutil.rmtree(bundle_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {model_id} as '{bundle_name}' to {bundle_path}")

    # Download/resolve model files (use local cache if network unavailable)
    try:
        model_dir = snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
        )
    except (OSError, Exception):
        logger.info("Network unavailable, attempting offline resolution...")
        model_dir = snapshot_download(
            model_id,
            allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            local_files_only=True,
        )
    logger.info(f"Using model files from: {model_dir}")

    # Load HF config for metadata and dimensions
    hf_config = AutoConfig.from_pretrained(model_dir)
    text_config = hf_config.text_config

    # Open sharded safetensors
    tensors = ShardedSafetensors(model_dir)

    target_dtype = torch.float16

    # 1. Export vision encoder + connector
    if not args.skip_vision:
        vision_output = bundle_path / "vision.aimodel"
        export_vision(tensors, vision_output, hf_config, model_id, target_dtype)

    # 2. Export embedding model
    if not args.skip_embed:
        embed_output = bundle_path / "embed.aimodel"
        export_embed(
            tensors,
            embed_output,
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            model_id=model_id,
            target_dtype=target_dtype,
        )

    # 3. Export text decoder
    if not args.skip_text:
        text_output = bundle_path / f"{bundle_name}.aimodel"
        export_text_decoder(
            model_dir,
            tensors,
            text_output,
            text_config=text_config,
            model_id=model_id,
            target_dtype=target_dtype,
            num_layers=args.num_layers,
        )

    # 4. Write tokenizer
    write_tokenizer(bundle_path, model_id)

    # 5. Write VLM metadata
    write_vlm_metadata(bundle_path, hf_config, model_id, bundle_name)

    # Clean up
    tensors.close()

    logger.info(f"SmolVLM2 export complete: {bundle_path}")


if __name__ == "__main__":
    main()

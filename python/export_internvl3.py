#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export InternVL3-1B as a complete VLM bundle.

Produces exports/internvl3_1b.llmasset/ with:
  - internvl3_1b.aimodel         (text decoder, inputs_embeds variant)
  - embed.aimodel                (token embedding lookup)
  - vision.aimodel               (InternViT + pixel_shuffle + MLP projector)
  - tokenizer/                   (embedded HF tokenizer)
  - metadata.json                (bundle manifest, kind=vlm)
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

DEFAULT_MODEL_ID = "OpenGVLab/InternVL3-1B"


def _find_repo_root() -> Path:
    """Walk up from this file to find the workspace root."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Sharded safetensors loader
# ---------------------------------------------------------------------------


class ShardedSafetensors:
    """Handles both single-file and sharded (index.json) safetensors."""

    def __init__(self, model_dir: str) -> None:
        self.model_dir = model_dir
        index_path = os.path.join(model_dir, "model.safetensors.index.json")

        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            self.weight_map = index["weight_map"]
            self.sharded = True
        else:
            self.single_path = os.path.join(model_dir, "model.safetensors")
            self.weight_map = None
            self.sharded = False

        self._handles: dict[str, object] = {}

    def _get_handle(self, shard_file: str):
        if shard_file not in self._handles:
            path = os.path.join(self.model_dir, shard_file)
            self._handles[shard_file] = safe_open(path, framework="pt", device="cpu")
        return self._handles[shard_file]

    def get_tensor(self, key: str) -> torch.Tensor:
        if self.sharded:
            shard_file = self.weight_map[key]
            handle = self._get_handle(shard_file)
            return handle.get_tensor(key)
        else:
            handle = self._get_handle("model.safetensors")
            return handle.get_tensor(key)

    def keys(self) -> list[str]:
        if self.sharded:
            return list(self.weight_map.keys())
        else:
            handle = self._get_handle("model.safetensors")
            return list(handle.keys())

    def close(self) -> None:
        self._handles.clear()


# ---------------------------------------------------------------------------
# Vision Encoder (InternViT + pixel_shuffle + MLP projector)
# ---------------------------------------------------------------------------


class InternVL3VisionEncoder(nn.Module):
    """InternViT-300M + pixel_shuffle + MLP projector for InternVL3.

    Architecture:
    - Patch embedding (Conv2d) + learnable position embedding
    - Class embedding (prepended to patch sequence, then removed after encoder)
    - 24 transformer layers with pre-norm, layer scale, GELU MLP
    - pixel_shuffle (spatial downsample by 0.5) folds spatial patches into channels
    - MLP projector: LayerNorm → Linear → GELU → Linear
    """

    def __init__(
        self,
        image_size: int = 448,
        patch_size: int = 14,
        hidden_size: int = 1024,
        num_heads: int = 16,
        intermediate_size: int = 4096,
        text_hidden_size: int = 896,
        downsample_ratio: float = 0.5,
        num_layers: int = 24,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_size = intermediate_size
        self.downsample_ratio = downsample_ratio

        # Grid: 448/14 = 32 -> 1024 patches
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size  # 1024

        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )

        # Class embedding [1, 1, hidden_size]
        self.class_embedding = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Position embedding for class_token + patches: [1, 1025, hidden_size]
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, hidden_size))

        # Transformer encoder
        self.encoder_layers = nn.ModuleList(
            [InternViTLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)]
        )

        # MLP projector (mlp1): LayerNorm → Linear(4096, 896) → GELU → Linear(896, 896)
        # After pixel_shuffle with ratio 0.5: channels become hidden_size / (ratio^2) = 1024 / 0.25 = 4096
        connector_in = int(hidden_size / (downsample_ratio**2))  # 4096
        self.proj_layernorm = nn.LayerNorm(connector_in)
        self.proj_linear1 = nn.Linear(connector_in, text_hidden_size, bias=True)
        self.proj_linear2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """InternVL's pixel_shuffle (ps_version=v2).

        Input: [B, num_patches, hidden] (class token already removed)
        After reshape to spatial: [B, H, W, C]
        After pixel_shuffle(0.5): spatial dims halved, channels 4x'd
        Output: [B, (H/2)*(W/2), C*4]
        """
        batch_size, seq_len, hidden = x.shape
        h = w = int(math.sqrt(seq_len))

        # Reshape to spatial [B, W, H, C] (InternVL uses W, H ordering)
        x = x.reshape(batch_size, w, h, hidden)

        # ps_version v2: standard pixel_shuffle downsample
        scale_factor = self.downsample_ratio
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)
        new_c = int(hidden / (scale_factor * scale_factor))  # This gives hidden / 0.25 = 4*hidden... wait

        # Actually the InternVL pixel_shuffle works differently:
        # It reshapes the spatial dims DOWN by scale_factor, packing into channels
        # Input: [B, W=32, H=32, C=1024]
        # Step 1: view as [B, W, H*scale, C/scale] = [B, 32, 16, 2048]
        # Step 2: permute to [B, H*scale, W, C/scale] = [B, 16, 32, 2048]
        # Step 3: view as [B, H*scale, W*scale, C/(scale^2)] = [B, 16, 16, 4096]
        # Step 4: permute to [B, W*scale, H*scale, C/(scale^2)] = [B, 16, 16, 4096]
        # Final flatten: [B, 256, 4096]
        B, W, H, C = x.size()
        x = x.view(B, W, int(H * scale_factor), int(C / scale_factor))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, int(H * scale_factor), int(W * scale_factor), int(C / (scale_factor * scale_factor)))
        x = x.permute(0, 2, 1, 3).contiguous()

        # Flatten spatial dims: [B, new_w, new_h, new_c] -> [B, new_w*new_h, new_c]
        x = x.reshape(batch_size, -1, x.shape[-1])
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: [1, 3, 448, 448] float32

        Returns:
            [1, 256, 896] float16
        """
        batch_size = pixel_values.shape[0]

        # Patch embedding: [B, 3, 448, 448] -> [B, 1024, 32, 32] -> [B, 1024, 1024]
        x = self.patch_embedding(pixel_values)
        x = x.flatten(2).transpose(1, 2)  # [B, 1024, 1024]

        # Prepend class embedding
        cls_tokens = self.class_embedding.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 1025, 1024]

        # Add position embedding
        x = x + self.position_embedding

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Remove class token (take only patch tokens)
        x = x[:, 1:, :]  # [B, 1024, 1024]

        # Pixel shuffle downsample
        x = self._pixel_shuffle(x)  # [B, 256, 4096]

        # MLP projector
        x = self.proj_layernorm(x)
        x = self.proj_linear1(x)
        x = F.gelu(x)
        x = self.proj_linear2(x)

        return x.to(torch.float16)


class InternViTLayer(nn.Module):
    """Single InternViT encoder layer with pre-norm and layer scale."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

        self.attn = InternViTAttention(hidden_size, num_heads)
        self.mlp = InternViTMLP(hidden_size, intermediate_size)

        # Layer scale parameters
        self.ls1 = nn.Parameter(torch.ones(hidden_size))
        self.ls2 = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1 * self.attn(self.norm1(x))
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class InternViTAttention(nn.Module):
    """Multi-head attention for InternViT with fused QKV."""

    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Fused QKV projection
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, seq, head_dim]
        q, k, v = qkv.unbind(0)

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.proj(attn)


class InternViTMLP(nn.Module):
    """MLP for InternViT (GELU activation)."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Embedding model
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
    vision_model: InternVL3VisionEncoder,
    tensors: ShardedSafetensors,
    num_layers: Optional[int] = None,
) -> None:
    """Load vision encoder + projector weights from safetensors."""
    # Patch embedding
    vision_model.patch_embedding.weight.data = tensors.get_tensor(
        "vision_model.embeddings.patch_embedding.weight"
    )
    vision_model.patch_embedding.bias.data = tensors.get_tensor(
        "vision_model.embeddings.patch_embedding.bias"
    )

    # Class embedding [1, 1, 1024]
    vision_model.class_embedding.data = tensors.get_tensor(
        "vision_model.embeddings.class_embedding"
    )

    # Position embedding [1, 1025, 1024]
    vision_model.position_embedding.data = tensors.get_tensor(
        "vision_model.embeddings.position_embedding"
    )

    # Projector (mlp1)
    vision_model.proj_layernorm.weight.data = tensors.get_tensor("mlp1.0.weight")
    vision_model.proj_layernorm.bias.data = tensors.get_tensor("mlp1.0.bias")
    vision_model.proj_linear1.weight.data = tensors.get_tensor("mlp1.1.weight")
    vision_model.proj_linear1.bias.data = tensors.get_tensor("mlp1.1.bias")
    vision_model.proj_linear2.weight.data = tensors.get_tensor("mlp1.3.weight")
    vision_model.proj_linear2.bias.data = tensors.get_tensor("mlp1.3.bias")

    # Determine how many encoder layers to load
    actual_layers = len(vision_model.encoder_layers)
    if num_layers is not None:
        actual_layers = min(actual_layers, num_layers)

    logger.info(f"Loading {actual_layers} vision encoder layers...")

    for i in range(actual_layers):
        layer = vision_model.encoder_layers[i]
        prefix = f"vision_model.encoder.layers.{i}"

        # Layer norms
        layer.norm1.weight.data = tensors.get_tensor(f"{prefix}.norm1.weight")
        layer.norm1.bias.data = tensors.get_tensor(f"{prefix}.norm1.bias")
        layer.norm2.weight.data = tensors.get_tensor(f"{prefix}.norm2.weight")
        layer.norm2.bias.data = tensors.get_tensor(f"{prefix}.norm2.bias")

        # Layer scale
        layer.ls1.data = tensors.get_tensor(f"{prefix}.ls1")
        layer.ls2.data = tensors.get_tensor(f"{prefix}.ls2")

        # Attention (fused qkv)
        layer.attn.qkv.weight.data = tensors.get_tensor(f"{prefix}.attn.qkv.weight")
        layer.attn.qkv.bias.data = tensors.get_tensor(f"{prefix}.attn.qkv.bias")
        layer.attn.proj.weight.data = tensors.get_tensor(f"{prefix}.attn.proj.weight")
        layer.attn.proj.bias.data = tensors.get_tensor(f"{prefix}.attn.proj.bias")

        # MLP
        layer.mlp.fc1.weight.data = tensors.get_tensor(f"{prefix}.mlp.fc1.weight")
        layer.mlp.fc1.bias.data = tensors.get_tensor(f"{prefix}.mlp.fc1.bias")
        layer.mlp.fc2.weight.data = tensors.get_tensor(f"{prefix}.mlp.fc2.weight")
        layer.mlp.fc2.bias.data = tensors.get_tensor(f"{prefix}.mlp.fc2.bias")


def _load_embed_weights(embed_model: EmbedModel, tensors: ShardedSafetensors) -> None:
    """Load embedding weights from safetensors."""
    embed_model.embed_tokens.weight.data = tensors.get_tensor(
        "language_model.model.embed_tokens.weight"
    )


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_vision(
    tensors: ShardedSafetensors,
    output_path: Path,
    model_id: str,
    num_vision_layers: Optional[int] = None,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the vision encoder + projector as vision.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    n_layers = num_vision_layers if num_vision_layers is not None else 24

    logger.info(
        f"Building vision encoder: image=448, patch=14, hidden=1024, "
        f"heads=16, layers={n_layers}"
    )

    vision_model = InternVL3VisionEncoder(
        image_size=448,
        patch_size=14,
        hidden_size=1024,
        num_heads=16,
        intermediate_size=4096,
        text_hidden_size=896,
        downsample_ratio=0.5,
        num_layers=n_layers,
    )
    _load_vision_weights(vision_model, tensors, num_layers=num_vision_layers)
    # Vision encoder runs in float32 (input is float32, output cast to float16 at end)
    vision_model = vision_model.float().eval()

    # Vision accepts float32 input, outputs float16
    pixel_values = torch.randn(1, 3, 448, 448, dtype=torch.float32)

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
    model_id: str,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the token embedding lookup as embed.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    vocab_size = 151674
    hidden_size = 896

    logger.info("Building embedding model...")
    embed_model = EmbedModel(vocab_size, hidden_size)
    _load_embed_weights(embed_model, tensors)
    embed_model = embed_model.to(target_dtype).eval()

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
    tensors: ShardedSafetensors,
    output_path: Path,
    model_id: str,
    target_dtype: torch.dtype = torch.float16,
    num_layers: Optional[int] = None,
    max_context_length: int = 8192,
) -> None:
    """Export the text decoder (inputs_embeds variant) as the main .aimodel."""
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Config

    from coreai_models.export._constants import (
        QUANT_TRACE_OFFSET,
        QUANT_TRACE_QUERY_LEN,
        TRACE_KV_CACHE_SEQ_LEN,
    )
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata
    from coreai_models.models.gpu.internvl3 import InternVL3ForCausalLM
    from coreai_models.primitives.macos.cache import KVCache

    logger.info("Loading text decoder...")

    # Build Qwen2Config from the InternVL3 llm_config
    qwen_config = Qwen2Config(
        hidden_size=896,
        num_hidden_layers=num_layers if num_layers is not None else 24,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        vocab_size=151674,
        max_position_embeddings=max_context_length,
        rope_theta=1000000.0,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )

    config = InternVL3ForCausalLM._get_reauthored_config(
        qwen_config, max_context_length, num_layers=num_layers
    )

    # Create model on meta device
    model = InternVL3ForCausalLM(config, model_device="meta")
    model.to(dtype=target_dtype)

    # Load weights from safetensors manually
    state_dict: dict[str, torch.Tensor] = {}
    prefix = "language_model.model."

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
        elif key == "language_model.lm_head.weight":
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


def write_vlm_metadata(bundle_path: Path, model_id: str, name: str) -> None:
    """Write metadata.json for the VLM bundle."""
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
            "vocab_size": 151674,
            "max_context_length": 32768,
            "embedded_tokenizer": True,
            "function_map": {"main": ["main"]},
        },
        "vision": {
            "image_size": 448,
            "patch_size": 14,
            "image_token_count": 256,
            "image_token_id": 151667,
            "image_mean": [0.48145466, 0.4578275, 0.40821073],
            "image_std": [0.26862954, 0.26130258, 0.27577711],
            "rescale_factor": 1.0,
            "max_tiles": 1,
            "include_thumbnail": False,
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


def write_tokenizer(bundle_path: Path, model_dir: str) -> None:
    """Save the HF tokenizer into the bundle from local model directory."""
    tokenizer_dir = bundle_path / "tokenizer"
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.save_pretrained(str(tokenizer_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export InternVL3-1B VLM bundle")
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

    model_id = args.model_id
    bundle_name = "internvl3_1b"

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        root = _find_repo_root()
        output_dir = root / "exports"

    bundle_path = output_dir / f"{bundle_name}.llmasset"

    if bundle_path.exists() and args.overwrite:
        shutil.rmtree(bundle_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting {model_id} as '{bundle_name}' to {bundle_path}")

    # Resolve model directory (use local cache — HF_HUB_OFFLINE=1)
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

    # Open safetensors
    tensors = ShardedSafetensors(model_dir)

    target_dtype = torch.float16

    # Use --num-layers for vision layers too if --num-vision-layers not specified
    num_vision_layers = args.num_vision_layers if args.num_vision_layers is not None else args.num_layers

    # 1. Export vision encoder + projector
    if not args.skip_vision:
        vision_output = bundle_path / "vision.aimodel"
        export_vision(tensors, vision_output, model_id, num_vision_layers, target_dtype)

    # 2. Export embedding model
    if not args.skip_embed:
        embed_output = bundle_path / "embed.aimodel"
        export_embed(tensors, embed_output, model_id, target_dtype)

    # 3. Export text decoder
    if not args.skip_text:
        text_output = bundle_path / f"{bundle_name}.aimodel"
        export_text_decoder(
            tensors,
            text_output,
            model_id=model_id,
            target_dtype=target_dtype,
            num_layers=args.num_layers,
        )

    # 4. Write tokenizer
    write_tokenizer(bundle_path, model_dir)

    # 5. Write VLM metadata
    write_vlm_metadata(bundle_path, model_id, bundle_name)

    # Clean up
    tensors.close()

    logger.info(f"InternVL3 export complete: {bundle_path}")


if __name__ == "__main__":
    main()

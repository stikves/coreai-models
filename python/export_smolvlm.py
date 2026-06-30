#!/usr/bin/env python3
# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Export SmolVLM2-256M as a complete VLM bundle.

Produces exports/smolvlm_256m.llmasset/ with:
  - smolvlm_256m.aimodel    (text decoder, inputs_embeds variant)
  - embed.aimodel            (token embedding lookup)
  - vision.aimodel           (SigLIP ViT + pixel_shuffle connector)
  - tokenizer/               (embedded HF tokenizer)
  - metadata.json            (bundle manifest, kind=vlm)
"""

import argparse
import json
import logging
import math
import os
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

HF_MODEL_ID = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"

# Vision constants
IMAGE_SIZE = 512
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 1024
VISION_HIDDEN_SIZE = 768
VISION_NUM_HEADS = 12
VISION_HEAD_DIM = VISION_HIDDEN_SIZE // VISION_NUM_HEADS  # 64

# Connector constants
SCALE_FACTOR = 4
IMAGE_SEQ_LEN = NUM_PATCHES // (SCALE_FACTOR ** 2)  # 64
CONNECTOR_IN_FEATURES = VISION_HIDDEN_SIZE * (SCALE_FACTOR ** 2)  # 12288
TEXT_HIDDEN_SIZE = 576

# Text decoder constants
IMAGE_TOKEN_ID = 49190


def _find_repo_root() -> Path:
    """Walk up from this file to find the workspace root."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return d
        d = d.parent
    return Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Vision Encoder + Connector (fused into a single exportable module)
# ---------------------------------------------------------------------------


class SmolVLMVisionEncoder(nn.Module):
    """SigLIP ViT + pixel_shuffle connector for SmolVLM2.

    Takes pixel_values [1, 3, 512, 512] and outputs [1, 64, 576] float16.
    """

    def __init__(self) -> None:
        super().__init__()
        # Patch embedding
        self.patch_embedding = nn.Conv2d(
            3, VISION_HIDDEN_SIZE, kernel_size=PATCH_SIZE, stride=PATCH_SIZE, bias=True
        )
        # Learnable position embeddings for 32x32 grid = 1024 patches
        self.position_embedding = nn.Embedding(NUM_PATCHES, VISION_HIDDEN_SIZE)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList()  # will be populated from weights

        # Post-layernorm
        self.post_layernorm = nn.LayerNorm(VISION_HIDDEN_SIZE)

        # Connector: pixel_shuffle + linear projection
        self.connector_proj = nn.Linear(CONNECTOR_IN_FEATURES, TEXT_HIDDEN_SIZE, bias=False)

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel shuffle: [B, seq, hidden] -> [B, seq/scale^2, hidden*scale^2].

        Rearranges spatial patches into channel dimension.
        """
        batch_size, seq_len, hidden = x.shape
        h = w = int(math.sqrt(seq_len))
        # Reshape to spatial grid
        x = x.reshape(batch_size, h, w, hidden)
        # Reshape into scale_factor blocks
        new_h = h // SCALE_FACTOR
        new_w = w // SCALE_FACTOR
        x = x.reshape(batch_size, new_h, SCALE_FACTOR, new_w, SCALE_FACTOR, hidden)
        # Merge scale dimensions into hidden
        x = x.permute(0, 1, 3, 2, 4, 5)  # [B, new_h, new_w, sf, sf, hidden]
        x = x.reshape(batch_size, new_h * new_w, SCALE_FACTOR * SCALE_FACTOR * hidden)
        return x

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            pixel_values: [1, 3, 512, 512] float32

        Returns:
            [1, 64, 576] float16 - vision embeddings ready for text decoder
        """
        # Patch embedding: [1, 3, 512, 512] -> [1, 768, 32, 32]
        x = self.patch_embedding(pixel_values)
        # Flatten spatial: [1, 768, 32, 32] -> [1, 1024, 768]
        x = x.flatten(2).transpose(1, 2)

        # Add position embeddings
        position_ids = torch.arange(NUM_PATCHES, device=x.device)
        x = x + self.position_embedding(position_ids)

        # Transformer encoder
        for layer in self.encoder_layers:
            x = layer(x)

        # Post-layernorm
        x = self.post_layernorm(x)

        # Pixel shuffle: [1, 1024, 768] -> [1, 64, 12288]
        x = self._pixel_shuffle(x)

        # Linear projection: [1, 64, 12288] -> [1, 64, 576]
        x = self.connector_proj(x)

        return x.to(torch.float16)


class VisionEncoderLayer(nn.Module):
    """Single SigLIP encoder layer with pre-norm."""

    def __init__(self, hidden_size: int = VISION_HIDDEN_SIZE) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.self_attn = VisionAttention(hidden_size)
        self.mlp = VisionMLP(hidden_size)

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

    def __init__(self, hidden_size: int = VISION_HIDDEN_SIZE) -> None:
        super().__init__()
        self.num_heads = VISION_NUM_HEADS
        self.head_dim = hidden_size // self.num_heads

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

    def __init__(self, hidden_size: int = VISION_HIDDEN_SIZE) -> None:
        super().__init__()
        intermediate_size = hidden_size * 4  # SigLIP default
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
    safetensors_path: str,
) -> None:
    """Load vision encoder + connector weights from the full model safetensors."""
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        all_keys = list(f.keys())

        # Patch embedding
        vision_model.patch_embedding.weight.data = f.get_tensor(
            "model.vision_model.embeddings.patch_embedding.weight"
        )
        vision_model.patch_embedding.bias.data = f.get_tensor(
            "model.vision_model.embeddings.patch_embedding.bias"
        )

        # Position embedding
        vision_model.position_embedding.weight.data = f.get_tensor(
            "model.vision_model.embeddings.position_embedding.weight"
        )

        # Post-layernorm
        vision_model.post_layernorm.weight.data = f.get_tensor(
            "model.vision_model.post_layernorm.weight"
        )
        vision_model.post_layernorm.bias.data = f.get_tensor(
            "model.vision_model.post_layernorm.bias"
        )

        # Connector projection
        vision_model.connector_proj.weight.data = f.get_tensor(
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

        # Build encoder layers
        vision_model.encoder_layers = nn.ModuleList(
            [VisionEncoderLayer() for _ in range(num_layers)]
        )

        # Load each encoder layer
        for i in range(num_layers):
            layer = vision_model.encoder_layers[i]
            prefix = f"model.vision_model.encoder.layers.{i}"

            # Layer norms
            layer.layer_norm1.weight.data = f.get_tensor(f"{prefix}.layer_norm1.weight")
            layer.layer_norm1.bias.data = f.get_tensor(f"{prefix}.layer_norm1.bias")
            layer.layer_norm2.weight.data = f.get_tensor(f"{prefix}.layer_norm2.weight")
            layer.layer_norm2.bias.data = f.get_tensor(f"{prefix}.layer_norm2.bias")

            # Self-attention
            layer.self_attn.q_proj.weight.data = f.get_tensor(
                f"{prefix}.self_attn.q_proj.weight"
            )
            layer.self_attn.q_proj.bias.data = f.get_tensor(f"{prefix}.self_attn.q_proj.bias")
            layer.self_attn.k_proj.weight.data = f.get_tensor(
                f"{prefix}.self_attn.k_proj.weight"
            )
            layer.self_attn.k_proj.bias.data = f.get_tensor(f"{prefix}.self_attn.k_proj.bias")
            layer.self_attn.v_proj.weight.data = f.get_tensor(
                f"{prefix}.self_attn.v_proj.weight"
            )
            layer.self_attn.v_proj.bias.data = f.get_tensor(f"{prefix}.self_attn.v_proj.bias")
            layer.self_attn.out_proj.weight.data = f.get_tensor(
                f"{prefix}.self_attn.out_proj.weight"
            )
            layer.self_attn.out_proj.bias.data = f.get_tensor(
                f"{prefix}.self_attn.out_proj.bias"
            )

            # MLP
            layer.mlp.fc1.weight.data = f.get_tensor(f"{prefix}.mlp.fc1.weight")
            layer.mlp.fc1.bias.data = f.get_tensor(f"{prefix}.mlp.fc1.bias")
            layer.mlp.fc2.weight.data = f.get_tensor(f"{prefix}.mlp.fc2.weight")
            layer.mlp.fc2.bias.data = f.get_tensor(f"{prefix}.mlp.fc2.bias")


def _load_embed_weights(embed_model: EmbedModel, safetensors_path: str) -> None:
    """Load embedding weights from the full model safetensors."""
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        embed_model.embed_tokens.weight.data = f.get_tensor(
            "model.text_model.embed_tokens.weight"
        )


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------


def export_vision(
    safetensors_path: str,
    output_path: Path,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the vision encoder + connector as vision.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    logger.info("Building vision encoder...")
    vision_model = SmolVLMVisionEncoder()
    _load_vision_weights(vision_model, safetensors_path)
    vision_model = vision_model.eval()

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
    safetensors_path: str,
    output_path: Path,
    vocab_size: int,
    hidden_size: int,
    target_dtype: torch.dtype = torch.float16,
) -> None:
    """Export the token embedding lookup as embed.aimodel."""
    from coreai_models.export.macos import export_to_coreai
    from coreai_models.export.metadata import build_aimodel_metadata

    logger.info("Building embedding model...")
    embed_model = EmbedModel(vocab_size, hidden_size)
    _load_embed_weights(embed_model, safetensors_path)
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

    metadata = build_aimodel_metadata(HF_MODEL_ID, component="embedding")
    coreai_program.save_asset(output_path, metadata)
    logger.info(f"Saved embed.aimodel to {output_path}")


def export_text_decoder(
    model_dir: str,
    safetensors_path: str,
    output_path: Path,
    text_config,
    target_dtype: torch.dtype = torch.float16,
    num_layers: int | None = None,
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

    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():  # noqa: SIM118
            if key.startswith(prefix):
                stripped = key.removeprefix(prefix)
                # Skip layers beyond num_layers
                if num_layers is not None:
                    import re

                    match = re.search(r"layers\.(\d+)\.", stripped)
                    if match and int(match.group(1)) >= num_layers:
                        continue
                tensor = f.get_tensor(key)
                if tensor.dtype != target_dtype:
                    tensor = tensor.to(target_dtype)
                state_dict[f"model.{stripped}"] = tensor
            elif key == "lm_head.weight":
                tensor = f.get_tensor(key)
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


def write_tokenizer(bundle_path: Path) -> None:
    """Save the HF tokenizer into the bundle."""
    tokenizer_dir = bundle_path / "tokenizer"
    logger.info("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    tokenizer.save_pretrained(str(tokenizer_dir))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SmolVLM2-256M VLM bundle")
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

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        root = _find_repo_root()
        output_dir = root / "exports"

    bundle_name = "smolvlm_256m"
    bundle_path = output_dir / f"{bundle_name}.llmasset"

    if bundle_path.exists() and args.overwrite:
        shutil.rmtree(bundle_path)
    bundle_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting SmolVLM2-256M to {bundle_path}")

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
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    logger.info(f"Using model files from: {model_dir}")

    # Load HF config for metadata
    hf_config = AutoConfig.from_pretrained(model_dir)
    text_config = hf_config.text_config

    target_dtype = torch.float16

    # 1. Export vision encoder + connector
    if not args.skip_vision:
        vision_output = bundle_path / "vision.aimodel"
        export_vision(safetensors_path, vision_output, target_dtype)

    # 2. Export embedding model
    if not args.skip_embed:
        embed_output = bundle_path / "embed.aimodel"
        export_embed(
            safetensors_path,
            embed_output,
            vocab_size=text_config.vocab_size,
            hidden_size=text_config.hidden_size,
            target_dtype=target_dtype,
        )

    # 3. Export text decoder
    if not args.skip_text:
        text_output = bundle_path / f"{bundle_name}.aimodel"
        export_text_decoder(
            model_dir,
            safetensors_path,
            text_output,
            text_config=text_config,
            target_dtype=target_dtype,
            num_layers=args.num_layers,
        )

    # 4. Write tokenizer
    write_tokenizer(bundle_path)

    # 5. Write VLM metadata
    write_vlm_metadata(bundle_path, hf_config, bundle_name)

    logger.info(f"SmolVLM2-256M export complete: {bundle_path}")


if __name__ == "__main__":
    main()

# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SmolVLM2 vision encoder + connector for Core AI export.

Standalone re-implementation of the SmolVLM2 (Idefics3-style) vision tower
that loads directly from the HF safetensors checkpoint. The text decoder
lives in ``smolvlm.py``; this file covers the vision pipeline:

  pixel_values [1, 3, H, W]
    -> Conv2d patch embedding (stride == patch_size, "valid" padding)
    -> learned absolute position embeddings (direct lookup at native res)
    -> N SigLIP transformer layers (pre-norm attention + GELU-tanh MLP, biased)
    -> post-layernorm
    -> pixel shuffle (scale_factor x scale_factor spatial merge)
    -> connector projection (no bias)
  -> image_features [1, N, text_hidden_size]

Unlike Qwen3-VL/Qwen2.5-VL, SmolVLM2's SigLIP tower has no rotary position
embeddings and no windowed attention: it is a bidirectional ViT with learned
absolute position embeddings over a fixed square grid, which is why this
lives as its own module rather than reusing ``StaticVisionEncoder`` (that
class is specific to the Qwen patchify + 3D-RoPE geometry).

Weight key mapping from HF checkpoint::

  model.vision_model.embeddings.*   -> vision_model.embeddings.*   (patch + position embeddings)
  model.vision_model.encoder.*      -> vision_model.encoder.*      (transformer layers)
  model.vision_model.post_layernorm.* -> vision_model.post_layernorm.*
  model.connector.modality_projection.proj.weight -> connector.proj.weight

Architecture notes (``HuggingFaceTB/SmolVLM2-256M-Video-Instruct``):

  All geometry (image_size, patch_size, hidden_size, num_attention_heads,
  intermediate_size, num_hidden_layers, scale_factor) is read from the HF
  ``vision_config`` / top-level config at construction time, so this same
  module covers other SmolVLM2 sizes (e.g. 2.2B) without code changes.
"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from coreai_models.models.base import load_tensors_for_keys, resolve_safetensors_files

# ---------------------------------------------------------------------------
# Transformer building blocks (SigLIP-style: biased projections, GELU-tanh)
# ---------------------------------------------------------------------------


class SmolVLMVisionEmbeddings(nn.Module):
    """Conv2d patch embedding + learned absolute position embeddings."""

    def __init__(self, hidden_size: int, image_size: int, patch_size: int) -> None:
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=True
        )
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.position_embedding = nn.Embedding(self.num_patches, hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # [1, 3, H, W] -> [1, hidden, grid, grid] -> [1, num_patches, hidden]
        x = self.patch_embedding(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        position_ids = torch.arange(self.num_patches, device=x.device)
        return x + self.position_embedding(position_ids)


class SmolVLMVisionAttention(nn.Module):
    """Bidirectional multi-head self-attention (all projections biased)."""

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

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        attn = attn.transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.out_proj(attn)


class SmolVLMVisionMLP(nn.Module):
    """``fc1 -> GELU(tanh approx) -> fc2``, both projections biased."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=True)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class SmolVLMVisionEncoderLayer(nn.Module):
    """Pre-norm SigLIP transformer layer."""

    def __init__(
        self, hidden_size: int, num_heads: int, intermediate_size: int, layer_norm_eps: float
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.self_attn = SmolVLMVisionAttention(hidden_size, num_heads)
        self.mlp = SmolVLMVisionMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x


class SmolVLMVisionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_hidden_layers: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SmolVLMVisionEncoderLayer(hidden_size, num_heads, intermediate_size, layer_norm_eps)
                for _ in range(num_hidden_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SmolVLMVisionTransformer(nn.Module):
    """SigLIP ViT: embeddings -> encoder -> post-layernorm.

    Module names/nesting mirror the HF checkpoint's ``model.vision_model.*``
    hierarchy 1:1, so weight loading needs only a prefix strip.
    """

    def __init__(
        self,
        hidden_size: int,
        image_size: int,
        patch_size: int,
        num_heads: int,
        intermediate_size: int,
        num_hidden_layers: int,
        layer_norm_eps: float,
    ) -> None:
        super().__init__()
        self.embeddings = SmolVLMVisionEmbeddings(hidden_size, image_size, patch_size)
        self.encoder = SmolVLMVisionEncoder(
            hidden_size, num_heads, intermediate_size, num_hidden_layers, layer_norm_eps
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.embeddings(pixel_values)
        x = self.encoder(x)
        return self.post_layernorm(x)


# ---------------------------------------------------------------------------
# Connector: pixel shuffle (spatial merge) + linear projection
# ---------------------------------------------------------------------------


class SmolVLMConnector(nn.Module):
    """Pixel-shuffle spatial merge + projection to the text hidden size.

    ``[B, seq, vision_hidden]`` -> ``[B, seq/scale^2, vision_hidden*scale^2]``
    -> ``[B, seq/scale^2, text_hidden]``
    """

    def __init__(self, vision_hidden_size: int, text_hidden_size: int, scale_factor: int) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.proj = nn.Linear(vision_hidden_size * (scale_factor**2), text_hidden_size, bias=False)

    def _pixel_shuffle(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden = x.shape
        h = w = int(math.sqrt(seq_len))
        sf = self.scale_factor
        x = x.reshape(batch_size, h, w, hidden)
        x = x.reshape(batch_size, h // sf, sf, w // sf, sf, hidden)
        x = x.permute(0, 1, 3, 2, 4, 5)  # [B, h/sf, w/sf, sf, sf, hidden]
        return x.reshape(batch_size, (h // sf) * (w // sf), sf * sf * hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self._pixel_shuffle(x))


# ---------------------------------------------------------------------------
# Combined vision model (encoder + connector, from_pretrained loader)
# ---------------------------------------------------------------------------


class SmolVLMVisionModel(nn.Module):
    """End-to-end SmolVLM2 vision pipeline.

    Takes raw pixel values and produces features aligned to the text hidden
    space, ready to be scattered into the token-embedding sequence.

    Input:  ``pixel_values [1, 3, H, W]``  (single image, RGB, pre-normalized)
    Output: ``image_features [1, N, text_hidden_size]``  (float32)

    where ``N = (H / patch_size / scale_factor) * (W / patch_size / scale_factor)``.
    """

    def __init__(
        self,
        *,
        vision_hidden_size: int,
        vision_num_heads: int,
        vision_intermediate_size: int,
        vision_num_hidden_layers: int,
        image_size: int,
        patch_size: int,
        text_hidden_size: int,
        scale_factor: int,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.scale_factor = scale_factor

        self.vision_model = SmolVLMVisionTransformer(
            hidden_size=vision_hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            num_heads=vision_num_heads,
            intermediate_size=vision_intermediate_size,
            num_hidden_layers=vision_num_hidden_layers,
            layer_norm_eps=layer_norm_eps,
        )
        self.connector = SmolVLMConnector(vision_hidden_size, text_hidden_size, scale_factor)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[1, 3, H, W]`` -> ``[1, N, text_hidden_size]`` (float32)."""
        hidden_states = self.vision_model(pixel_values)
        return self.connector(hidden_states)

    # ---- pretrained loader ------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id_or_path: str,
        dtype: torch.dtype = torch.float32,
    ) -> "SmolVLMVisionModel":
        """Load the vision model from a SmolVLM2 HF checkpoint.

        Downloads the checkpoint (or uses cached), reads the vision and
        top-level configs, creates the model on meta device, then loads
        weights from safetensors.

        Args:
            model_id_or_path: HuggingFace model ID
                (e.g. ``"HuggingFaceTB/SmolVLM2-256M-Video-Instruct"``) or
                local path.
            dtype: target dtype for parameters (default: float32, suitable
                for vision encoder accuracy — matches the Muse Glimmer and
                Qwen vision encoders, which are also exported at fp32).
        """
        if os.path.isdir(model_id_or_path):
            model_dir = model_id_or_path
        else:
            model_dir = snapshot_download(
                model_id_or_path,
                allow_patterns=[
                    "*.safetensors",
                    "*.safetensors.index.json",
                    "config.json",
                ],
            )

        # Use AutoConfig (not a raw json.load) so field defaults/aliases the
        # HF config class fills in (e.g. fields omitted from config.json
        # because they match the class default) are resolved correctly.
        hf_config = AutoConfig.from_pretrained(model_dir)
        vision_cfg = hf_config.vision_config
        text_cfg = hf_config.text_config

        model = cls(
            vision_hidden_size=vision_cfg.hidden_size,
            vision_num_heads=vision_cfg.num_attention_heads,
            vision_intermediate_size=vision_cfg.intermediate_size,
            vision_num_hidden_layers=vision_cfg.num_hidden_layers,
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            text_hidden_size=text_cfg.hidden_size,
            scale_factor=hf_config.scale_factor,
            layer_norm_eps=vision_cfg.layer_norm_eps,
        )
        model = model.to(dtype=dtype)

        safetensors_files = resolve_safetensors_files(model_dir)

        # Build key->file index for vision + connector keys only.
        vision_key_to_file: dict[str, str] = {}
        from safetensors import safe_open

        for path in safetensors_files:
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    if key.startswith("model.vision_model.") or key.startswith("model.connector."):
                        vision_key_to_file[key] = path

        raw_sd = load_tensors_for_keys(vision_key_to_file, dtype)

        model.load_state_dict(map_hf_vision_state_dict(raw_sd), strict=True, assign=True)

        meta_params = [n for n, p in model.named_parameters() if p.is_meta]
        if meta_params:
            raise RuntimeError(
                f"Vision model has unloaded parameters on meta device: {meta_params}"
            )

        return model.eval()


def map_hf_vision_state_dict(hf_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map HF ``model.vision_model.*`` / ``model.connector.*`` keys to SmolVLMVisionModel keys."""
    mapped: dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_state_dict.items():
        if hf_key == "model.connector.modality_projection.proj.weight":
            mapped["connector.proj.weight"] = tensor
        elif hf_key.startswith("model.vision_model."):
            mapped[hf_key.removeprefix("model.")] = tensor
    return mapped

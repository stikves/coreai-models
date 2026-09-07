# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Gemma 3n VLM components for CoreAI model export.

Vision tower: MobileNetV5 (224x224 → 256 visual tokens).
Text decoder: AltUp variant accepting inputs_embeds + input_ids (for per-layer inputs).
"""

import math

import torch
import torch.nn as nn
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nForCausalLM as HFGemma3nForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.models.macos.gemma3n import (
    Gemma3nForCausalLM,
    Gemma3nModel,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm


class Gemma3nVisionEncoder(nn.Module):
    """MobileNetV5 vision tower + multimodal embedder.

    Input:  pixel_values  float16 [1, 3, 224, 224]
    Output: image_embeds  float16 [1, 256, hidden_size]
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        embedder: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.soft_embedding_norm = embedder.soft_embedding_norm
        self.embedding_projection = embedder.embedding_projection
        self.post_projection_norm = embedder.post_projection_norm
        self.scale = math.sqrt(hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # MobileNetV5: [B, 3, 224, 224] → [B, 2048, 16, 16]
        features = self.vision_tower(pixel_values, do_pooling=False)
        # Reshape to sequence: [B, 2048, 16, 16] → [B, 256, 2048]
        b, c, h, w = features.shape
        features = features.reshape(b, c, h * w).transpose(1, 2)
        # Scale + embedder: norm → project → norm
        features = features * self.scale
        features = self.soft_embedding_norm(features)
        features = self.embedding_projection(features)
        features = self.post_projection_norm(features)
        return features


class Gemma3nModelEmbeddings(Gemma3nModel):
    """Gemma3n text decoder that accepts inputs_embeds instead of input_ids.

    For VLM: the runner merges vision embeddings into inputs_embeds at image
    token positions before calling this. input_ids is still needed for the
    per-layer embedding table (AltUp).
    """

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        per_layer_inputs = self._get_per_layer_inputs(input_ids, inputs_embeds)

        hidden_states = self._altup_expand(inputs_embeds)

        for layer in self.layers:
            idx = layer.layer_idx
            per_layer_input = per_layer_inputs[:, :, idx, :]
            hidden_states = layer(hidden_states, per_layer_input, position_ids, cache)

        hidden_states = self._altup_collapse(hidden_states)
        return self.norm(hidden_states)


class Gemma3nForCausalLMEmbeddings(Gemma3nForCausalLM):
    """Gemma3n VLM text decoder: takes (input_ids, inputs_embeds, position_ids).

    The embed_tokens table is removed from this graph (it lives in embed.aimodel).
    But embed_tokens_per_layer stays (needed for per-layer AltUp inputs).
    """

    def _init_model(self, config: Gemma3nTextConfig) -> None:
        self.model = Gemma3nModelEmbeddings(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, inputs_embeds, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        super()._mutate_state_dict(state_dict)
        # Remove embed_tokens (lives in embed.aimodel), keep embed_tokens_per_layer
        for k in list(state_dict.keys()):
            if "embed_tokens.weight" in k and "per_layer" not in k:
                del state_dict[k]

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

from coreai_models.models.macos.gemma3n import (
    Gemma3nForCausalLM,
    Gemma3nModel,
)
from coreai_models.primitives.macos.cache import KVCache


class Gemma3nVisionEncoder(nn.Module):
    """MobileNetV5 vision tower + multimodal embedder.

    Input:  pixel_values  float16 [1, 3, 224, 224]
    Output: image_embeds  float16 [1, num_soft_tokens, text_hidden_size]

    Mirrors ``Gemma3nModel.get_image_features``: reshape the conv feature map to a
    token sequence, scale by ``sqrt(vision_hidden_size)``, then run the embedder
    (soft_embedding_norm → embedding_projection → embedding_post_projection_norm).
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        embedder: nn.Module,
        vision_hidden_size: int,
        num_soft_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.soft_embedding_norm = embedder.soft_embedding_norm
        self.embedding_projection = embedder.embedding_projection
        self.embedding_post_projection_norm = embedder.embedding_post_projection_norm
        self.vision_hidden_size = vision_hidden_size
        self.num_soft_tokens = num_soft_tokens
        # HF scales by the *vision* hidden size (not the text hidden size).
        self.scale = math.sqrt(vision_hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # MobileNetV5: [B, 3, 224, 224] → [B, vision_hidden, H, W]
        vision_outputs = self.vision_tower(
            pixel_values=pixel_values, do_pooling=False, return_dict=True
        )
        features = vision_outputs.last_hidden_state
        # [B, C, H, W] → [B, H*W, C] (== [B, num_soft_tokens, vision_hidden])
        b = features.shape[0]
        features = features.reshape(
            b, self.vision_hidden_size, self.num_soft_tokens
        ).permute(0, 2, 1)
        # Scale + embedder: norm → project → norm
        features = features * self.scale
        features = self.soft_embedding_norm(features)
        features = self.embedding_projection(features)
        features = self.embedding_post_projection_norm(features)
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

    The decoder's forward never looks up the main embedding table — the runner
    computes inputs_embeds (via the separate embed.aimodel) and merges vision
    embeddings before calling this graph. embed_tokens therefore does not appear
    in the traced graph except through the tied lm_head, exactly as in the
    text-only decoder. embed_tokens_per_layer stays (per-layer AltUp inputs).

    State-dict handling is inherited unchanged from Gemma3nForCausalLM: the main
    embedding weight is loaded normally and tied to lm_head, so loading matches
    the proven text-only path (removing it would orphan the still-declared
    embed_tokens parameter on the meta device).
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

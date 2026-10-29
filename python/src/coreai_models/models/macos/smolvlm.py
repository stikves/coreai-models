# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SmolVLM2 text decoder (inputs_embeds variant) for VLM export.

SmolVLM2's text backbone is a standard Llama decoder (SmolLM2 under the hood).
Structurally identical to Mistral/Llama but accepts pre-computed embeddings
instead of token IDs, matching the ``inputs_embeds`` contract shared by every
VLM text decoder in ``coreai_models.vlm.export`` (Qwen3-VL, Muse Glimmer). The
embedding lookup is handled by a separate ``embed.aimodel``; the vision-token
scatter-merge happens upstream in the Swift runner.
"""

import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.smolvlm.modeling_smolvlm import (
    SmolVLMForConditionalGeneration as HFSmolVLMForConditionalGeneration,
)
from typing_extensions import Self, override

from coreai_models._hf import is_default_rope_scaling
from coreai_models.models.base import BaseForCausalLM
from coreai_models.models.macos.mistral import TransformerBlock
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm


class SmolVLMTextModel(nn.Module):
    """SmolVLM2 text backbone without token embeddings.

    Accepts pre-computed embeddings (from vision encoder or separate embed.aimodel).
    The decoder layers are Mistral's (same Llama-style attention, MLP and norms).
    """

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        assert is_default_rope_scaling(config), f"unsupported rope_scaling: {config.rope_scaling}"
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = inputs_embeds
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class SmolVLMForCausalLMEmbeddings(BaseForCausalLM):
    """Engine-compatible SmolVLM2 text decoder (inputs_embeds variant).

    Used as the ``main`` text decoder in a VLM bundle. Embeddings are provided
    externally (from ``embed.aimodel`` for text tokens or ``vision.aimodel``
    for image tokens, scatter-merged by the Swift runner before this graph
    runs).

    forward(inputs_embeds, position_ids, k_cache, v_cache) -> logits
    """

    # We don't use _HF_MODEL_CLASS because coreai_models.vlm.export loads
    # weights directly from safetensors (see load_model_from_safetensors),
    # bypassing from_hf_memory_efficient.
    _HF_MODEL_CLASS = None

    @classmethod
    def _get_reauthored_config(
        cls,
        hf_config,
        max_context_length: int | None = None,
        num_layers: int | None = None,
    ):
        # `hf_config` here is the top-level SmolVLM config; the text backbone's
        # LlamaConfig lives under `.text_config` (mirrors Qwen3VLForCausalLMEmbeddings
        # and MuseGlimmerForCausalLMEmbeddings).
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config
        if max_context_length is not None:
            text_config.max_position_embeddings = max_context_length
        if num_layers is not None:
            text_config.num_hidden_layers = num_layers
        return text_config

    @override
    def _init_model(self, config: LlamaConfig) -> None:
        self.model = SmolVLMTextModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(inputs_embeds, position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # Find the maximum layer index present in this state dict slice
        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) < 4:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        # Fuse q_proj + k_proj + v_proj into qkv_proj for each layer
        if max_layer >= 0:
            for i in range(max_layer + 1):
                combined_weight = []
                need_to_fuse = True
                for proj in ["q_proj", "k_proj", "v_proj"]:
                    weight_key = f"model.layers.{i}.self_attn.{proj}.weight"
                    if weight_key not in state_dict:
                        need_to_fuse = False
                        continue
                    combined_weight.append(state_dict[weight_key])
                    del state_dict[weight_key]
                if need_to_fuse:
                    state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.concat(
                        combined_weight, axis=0
                    )

        # Drop embed_tokens — it goes to the separate embed.aimodel
        embed_key = "model.embed_tokens.weight"
        if embed_key in state_dict:
            del state_dict[embed_key]


class SmolVLMForCausalLM(SmolVLMForCausalLMEmbeddings):
    """SmolVLM2 text decoder with its own token embedding (input_ids variant).

    Loads from the full VLM checkpoint and ignores the vision tower, so text-only
    evals (`--hf-id HuggingFaceTB/SmolVLM2-...`) run the same decoder modules the
    VLM bundle exports.

    forward(input_ids, position_ids, k_cache, v_cache) -> logits
    """

    _HF_MODEL_CLASS = HFSmolVLMForConditionalGeneration

    @override
    def _init_model(self, config: LlamaConfig) -> None:
        super()._init_model(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(self.embed_tokens(input_ids), position_ids, cache)
        return self.lm_head(out)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        for key in list(state_dict):
            if key.startswith(("model.vision_model.", "model.connector.")):
                del state_dict[key]
            elif key == "model.text_model.embed_tokens.weight":
                state_dict["embed_tokens.weight"] = state_dict.pop(key)
            elif key.startswith("model.text_model."):
                state_dict["model." + key.removeprefix("model.text_model.")] = state_dict.pop(key)
        super()._mutate_state_dict(state_dict)

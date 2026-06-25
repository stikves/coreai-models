# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as HFLlamaForCausalLM,
)
from typing_extensions import Self, override

from coreai_models.models.base import BaseForCausalLM
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.mlp import MLP
from coreai_models.primitives.macos.rms_norm import RMSNorm
from coreai_models.primitives.macos.rope import initialize_rope
from coreai_models.primitives.macos.sdpa import SDPA

USE_FUSED_KV = True


class Attention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # dimension for weights
        dim = config.hidden_size
        self.n_heads = n_heads = config.num_attention_heads
        self.n_kv_heads = n_kv_heads = config.num_key_value_heads
        self.head_dim = head_dim = getattr(config, "head_dim", dim // n_heads)

        # combined qkv projection into a single linear layer
        self.qkv_proj = nn.Linear(
            dim,
            n_heads * head_dim  # q
            + n_kv_heads * head_dim  # k
            + n_kv_heads * head_dim,  # v
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=config.attention_bias)

        # sdpa
        self.sdpa = SDPA(is_causal=True)

        # rope
        rope_scaling = getattr(config, "rope_scaling", None)
        rope_theta = getattr(config, "rope_theta", None)
        if rope_theta is None and rope_scaling is not None:
            rope_theta = rope_scaling.get("rope_theta", 10000.0)
        rope_theta = rope_theta or 10000.0
        assert rope_scaling is None or rope_scaling["rope_type"] == "llama3"
        self.rope = initialize_rope(
            dims=config.head_dim,
            base=rope_theta,
            scaling_config=rope_scaling,
        )

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        batch_size, query_len, _ = x.shape
        n_heads, n_kv_heads = self.n_heads, self.n_kv_heads

        # linear projection and slice for q, k, v
        qkv = (
            self.qkv_proj(x)
            .reshape(batch_size, query_len, n_heads + 2 * n_kv_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        # rope
        seq_len = position_ids.shape[-1]
        torch._check_is_size(query_len)
        torch._check_is_size(seq_len)
        offset = seq_len - query_len
        torch._check_is_size(offset)
        rope_positions = position_ids.narrow(-1, offset, query_len)

        if USE_FUSED_KV:
            query_key = qkv.narrow(1, 0, n_heads + n_kv_heads)
            query_key = self.rope(query_key, position_ids=rope_positions)
            query = query_key.narrow(1, 0, n_heads)
            key = query_key.narrow(1, n_heads, n_kv_heads)
        else:
            query = qkv.narrow(1, 0, n_heads)
            key = qkv.narrow(1, n_heads, n_kv_heads)
            query = self.rope(query, position_ids=rope_positions)
            key = self.rope(key, position_ids=rope_positions)

        value = qkv.narrow(1, n_heads + n_kv_heads, n_kv_heads)

        # update the cache
        if cache is not None:
            key, value = cache.update_and_fetch(
                self.layer_idx, offset, key, value, seq_len=seq_len, query_len=query_len
            )

        # sdpa
        output = (
            self.sdpa(query, key, value)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, query_len, self.n_heads * self.head_dim)
        )
        return self.o_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.self_attn = Attention(config, layer_idx=layer_idx)
        self.mlp = MLP(hidden_size, config.intermediate_size, bias=config.mlp_bias)

        # rms norm layer
        self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.IntTensor,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), position_ids, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LlamaModel(nn.Module):
    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor = None,
        cache: KVCache | None = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, position_ids, cache)
        return self.norm(h)


class LlamaForCausalLM(BaseForCausalLM):
    _HF_MODEL_CLASS = HFLlamaForCausalLM

    @override
    def _init_model(self, config: LlamaConfig) -> None:
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        cache = KVCache(k_cache, v_cache)
        out = self.model(input_ids, position_ids, cache)
        return self.lm_head(out)

    @BaseForCausalLM.cast_logits_bfloat16_to_float16
    def forward_from_embeddings(
        self,
        in_embeddings: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Same as `forward` but skips `embed_tokens` — the caller passes
        already-embedded inputs directly. Used by VLM exports where text and
        vision embeddings are pre-merged before the LLM main pass.

        Same in-place `KVCache` pattern as `forward`, so the export still
        gets auto-aliased KV cache → 2 inputs + 2 states + 1 output at runtime.
        """
        cache = KVCache(k_cache, v_cache)
        h = in_embeddings
        for layer in self.model.layers:
            h = layer(h, position_ids, cache)
        h = self.model.norm(h)
        return self.lm_head(h)

    @override
    def _mutate_state_dict(self: Self, state_dict: dict[str, torch.Tensor]) -> None:
        # first we get how many attention layers we got
        max_layer = -1
        for k in state_dict:
            name_split = k.split(".")
            if len(name_split) != 6:
                continue
            if not k.startswith("model.layers."):
                continue
            max_layer = max(max_layer, int(name_split[2]))

        if max_layer < 0:
            err = "invalid state_dict"
            raise ValueError(err)

        # construct the combine weight and bias
        for i in range(max_layer + 1):
            # Verify all keys exist before fusing
            weight_keys = [
                f"model.layers.{i}.self_attn.{proj}.weight"
                for proj in ["q_proj", "k_proj", "v_proj"]
            ]
            if all(k in state_dict for k in weight_keys):
                combined_weight = [state_dict.pop(k) for k in weight_keys]
                state_dict[f"model.layers.{i}.self_attn.qkv_proj.weight"] = torch.concat(
                    combined_weight, axis=0
                )

            # Handle biases if they exist
            bias_keys = [
                f"model.layers.{i}.self_attn.{proj}.bias" for proj in ["q_proj", "k_proj", "v_proj"]
            ]
            if all(k in state_dict for k in bias_keys):
                combined_bias = [state_dict.pop(k) for k in bias_keys]
                state_dict[f"model.layers.{i}.self_attn.qkv_proj.bias"] = torch.concat(
                    combined_bias, axis=0
                )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        super().load_state_dict(state_dict, strict=strict, assign=assign)
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

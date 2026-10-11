# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS SmolLM3 model parity with HuggingFace."""

import pytest
import torch
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Config
from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3ForCausalLM as HFSmolLM3ForCausalLM,
)

from coreai_models.models.macos.smollm3 import SmolLM3ForCausalLM
from coreai_models.models.registry import get_model_entry
from coreai_models.primitives.macos.cache import KVCache

VOCAB_SIZE = 100


def _make_config(**overrides) -> SmolLM3Config:
    # Interval 2 gives NoPE layers at 1 and 3, so most tests cover both kinds.
    kwargs = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=4,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=64,
        no_rope_layer_interval=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    kwargs.update(overrides)
    return SmolLM3Config(**kwargs)


def _make_models(config: SmolLM3Config, dtype: torch.dtype = torch.float32):
    torch.manual_seed(0)
    hf_model = HFSmolLM3ForCausalLM(config).to(dtype).eval()
    our_model = SmolLM3ForCausalLM(config, model_device="cpu")
    our_model.to(dtype).eval()
    sd = dict(hf_model.state_dict())
    our_model._mutate_state_dict(sd)
    our_model.load_state_dict(sd, assign=True, strict=True)
    return hf_model, our_model


class TestmacOSSmolLM3ForCausalLM:
    """Test macOS SmolLM3ForCausalLM against HuggingFace reference."""

    def test_no_rope_layers_wired(self):
        """Layers flagged 0 in ``no_rope_layers`` skip RoPE."""
        config = _make_config()
        assert config.no_rope_layers == [1, 0, 1, 0]
        model = SmolLM3ForCausalLM(config, model_device="cpu")
        assert [layer.self_attn.use_rope for layer in model.model.layers] == [
            True,
            False,
            True,
            False,
        ]

    @pytest.mark.parametrize("seq_len", [1, 8])
    def test_forward_parity(self, seq_len):
        """Prefill logits match HF for single- and multi-token inputs."""
        config = _make_config()
        hf_model, our_model = _make_models(config)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-5, rtol=1e-5)

    def test_decode_parity_after_prefill(self):
        """Prefill then single-token decode through the KV cache matches HF."""
        prompt_len, total_len = 6, 10
        config = _make_config()
        hf_model, our_model = _make_models(config)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, total_len))
        with torch.no_grad():
            hf_logits = hf_model(input_ids=input_ids).logits

        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)
        with torch.no_grad():
            position_ids = torch.arange(prompt_len, dtype=torch.int32).unsqueeze(0)
            out = our_model(input_ids[:, :prompt_len], position_ids, k_cache, v_cache)
            torch.testing.assert_close(out, hf_logits[:, :prompt_len], atol=1e-5, rtol=1e-5)
            for pos in range(prompt_len, total_len):
                position_ids = torch.arange(pos + 1, dtype=torch.int32).unsqueeze(0)
                out = our_model(input_ids[:, pos : pos + 1], position_ids, k_cache, v_cache)
                torch.testing.assert_close(out, hf_logits[:, pos : pos + 1], atol=1e-5, rtol=1e-5)

    def test_forward_parity_float16(self):
        """Parity holds in float16."""
        seq_len = 8
        config = _make_config()
        hf_model, our_model = _make_models(config, dtype=torch.float16)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float16)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-2, rtol=1e-2)

    def test_tied_embeddings(self):
        """lm_head shares the embedding weight after loading."""
        config = _make_config()
        _, our_model = _make_models(config)
        assert our_model.lm_head.weight is our_model.model.embed_tokens.weight

    def test_sliding_window_rejected(self):
        """Sliding-window NoPE layers are not supported and must fail loudly."""
        config = _make_config(use_sliding_window=True, sliding_window=16)
        with pytest.raises(ValueError, match="sliding"):
            SmolLM3ForCausalLM(config, model_device="cpu")

    def test_registry(self):
        """``smollm3`` model_type resolves to SmolLM3ForCausalLM."""
        assert get_model_entry("smollm3").macos_class is SmolLM3ForCausalLM

    @pytest.mark.parametrize("max_ctx, expect_freqs", [(32768, False), (65536, True)])
    def test_long_context_uses_f32_rope_freqs(self, max_ctx, expect_freqs):
        """Contexts past 32K pass f32 inverse frequencies so runtime angles stay in f32."""
        config = _make_config(max_position_embeddings=max_ctx)
        model = SmolLM3ForCausalLM(config, model_device="cpu").to(torch.float16)
        freqs = model.model.layers[0].self_attn._rope_freqs
        if expect_freqs:
            assert freqs is not None and freqs.dtype == torch.float32
        else:
            assert freqs is None

    def test_forward_parity_long_prefill(self):
        """2K-token prefill matches HF with a 64K-context config (f32 freqs path)."""
        seq_len = 2048
        config = _make_config(max_position_embeddings=65536)
        hf_model, our_model = _make_models(config)

        input_ids = torch.randint(0, VOCAB_SIZE, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_out = hf_model(input_ids=input_ids, position_ids=position_ids.long())

        torch.testing.assert_close(our_out, hf_out.logits, atol=1e-4, rtol=1e-4)

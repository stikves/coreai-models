# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for macOS Qwen2.5-VL text decoder parity with HuggingFace."""

import torch
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as HFQwen2_5VLForConditionalGeneration,
)

from coreai_models.models.macos.qwen2_5_vl import Qwen2_5VLForCausalLM
from coreai_models.primitives.macos.cache import KVCache


def _make_config(**overrides) -> Qwen2_5_VLTextConfig:
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=100,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        attention_bias=True,
        rope_parameters={
            "rope_type": "default",
            "mrope_section": [2, 3, 3],
        },
    )
    defaults.update(overrides)
    return Qwen2_5_VLTextConfig(**defaults)


def _load_hf_text_decoder(config):
    """Load only the text decoder from the HF conditional generation model."""
    from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLVisionConfig,
    )

    vision_config = Qwen2_5_VLVisionConfig(
        hidden_size=64, depth=1, num_heads=4, patch_size=14, spatial_merge_size=2
    )
    full_config = Qwen2_5_VLConfig(
        text_config=config.to_dict(), vision_config=vision_config.to_dict()
    )
    hf_model = HFQwen2_5VLForConditionalGeneration(full_config).to(torch.float32).eval()
    return hf_model


class TestQwen2_5VLForCausalLM:
    """Test macOS Qwen2_5VLForCausalLM against HuggingFace reference."""

    def test_forward_parity_single_token(self):
        config = _make_config()
        hf_model = _load_hf_text_decoder(config)
        our_model = Qwen2_5VLForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=False)

        input_ids = torch.randint(0, 100, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_text_out = hf_model.model.language_model(
                input_ids=input_ids,
                position_ids=position_ids.long().unsqueeze(0).expand(3, -1, -1),
            )
            hf_logits = hf_model.lm_head(hf_text_out.last_hidden_state)

        torch.testing.assert_close(our_out, hf_logits, atol=1e-5, rtol=1e-5)

    def test_forward_parity_multi_token(self):
        seq_len = 8
        config = _make_config()
        hf_model = _load_hf_text_decoder(config)
        our_model = Qwen2_5VLForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=False)

        input_ids = torch.randint(0, 100, (1, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            our_out = our_model(input_ids, position_ids, k_cache, v_cache)
            hf_text_out = hf_model.model.language_model(
                input_ids=input_ids,
                position_ids=position_ids.long().unsqueeze(0).expand(3, -1, -1),
            )
            hf_logits = hf_model.lm_head(hf_text_out.last_hidden_state)

        torch.testing.assert_close(our_out, hf_logits, atol=1e-5, rtol=1e-5)

    def test_output_shape(self):
        config = _make_config()
        our_model = Qwen2_5VLForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        batch, seq_len = 1, 6
        input_ids = torch.randint(0, 100, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = our_model(input_ids, position_ids, k_cache, v_cache)

        assert out.shape == (batch, seq_len, config.vocab_size)

    def test_has_attention_bias_no_qk_norm(self):
        """Qwen2.5-VL uses attention bias but no QK-norm (unlike Qwen3-VL)."""
        config = _make_config(num_hidden_layers=1)
        model = Qwen2_5VLForCausalLM(config, model_device="cpu")
        attn = model.model.layers[0].self_attn

        assert attn.qkv_proj.bias is not None
        assert not hasattr(attn, "qk_norm")

    def test_mutate_state_dict_fuses_qkv_with_bias(self):
        config = _make_config(num_hidden_layers=1)
        our_model = Qwen2_5VLForCausalLM(config, model_device="cpu")
        hf_model = _load_hf_text_decoder(config)

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)

        assert "model.layers.0.self_attn.qkv_proj.weight" in sd
        assert "model.layers.0.self_attn.qkv_proj.bias" in sd
        assert "model.layers.0.self_attn.q_proj.weight" not in sd
        assert "model.layers.0.self_attn.q_proj.bias" not in sd

    def test_mutate_state_dict_strips_vision_keys(self):
        config = _make_config(num_hidden_layers=1)
        our_model = Qwen2_5VLForCausalLM(config, model_device="cpu")
        hf_model = _load_hf_text_decoder(config)

        sd = dict(hf_model.state_dict())
        vision_keys_before = [k for k in sd if "visual" in k]
        assert len(vision_keys_before) > 0

        our_model._mutate_state_dict(sd)

        vision_keys_after = [k for k in sd if "visual" in k]
        assert len(vision_keys_after) == 0

    def test_incremental_decode(self):
        config = _make_config()
        hf_model = _load_hf_text_decoder(config)
        our_model = Qwen2_5VLForCausalLM(config, model_device="cpu")
        our_model.to(torch.float32).eval()

        sd = dict(hf_model.state_dict())
        our_model._mutate_state_dict(sd)
        our_model.load_state_dict(sd, assign=True, strict=False)

        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        input_ids = torch.randint(0, 100, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            our_model(input_ids, position_ids, k_cache, v_cache)

        next_token = torch.randint(0, 100, (1, 1))
        pos_ids_step2 = torch.arange(5, dtype=torch.int32).unsqueeze(0)
        with torch.no_grad():
            out2 = our_model(next_token, pos_ids_step2, k_cache, v_cache)

        assert out2.shape == (1, 1, config.vocab_size)
        with torch.no_grad():
            out2b = our_model(next_token, pos_ids_step2, k_cache, v_cache)
        torch.testing.assert_close(out2, out2b)

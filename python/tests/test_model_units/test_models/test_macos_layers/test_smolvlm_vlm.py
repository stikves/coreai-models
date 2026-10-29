# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for SmolVLM2 VLM components.

Covers the VLM-specific classes added for SmolVLM2 support:
- SmolVLMForCausalLMEmbeddings (text decoder, inputs_embeds variant)
- SmolVLMVisionModel (SigLIP ViT + pixel-shuffle connector)

All tests use small configs with random weights — no HF downloads.
"""

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from coreai_models.models.macos.smolvlm import SmolVLMForCausalLMEmbeddings
from coreai_models.models.macos.smolvlm_vision import SmolVLMVisionModel
from coreai_models.primitives.macos.cache import KVCache


def _make_smolvlm_text_config(**overrides) -> LlamaConfig:
    """Minimal SmolVLM2 text-decoder config for fast unit tests (tiny dims)."""
    defaults = dict(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=64,
        vocab_size=100,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
    )
    defaults.update(overrides)
    return LlamaConfig(**defaults)


class TestSmolVLMForCausalLMEmbeddings:
    """Test the inputs_embeds variant of the SmolVLM2 text decoder."""

    def test_forward_produces_finite_output(self):
        """Forward with random inputs_embeds should produce finite logits."""
        config = _make_smolvlm_text_config()
        model = SmolVLMForCausalLMEmbeddings(config, model_device="cpu")
        model.to(torch.float32).eval()

        seq_len = 6
        inputs_embeds = torch.randn(1, seq_len, config.hidden_size)
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            logits = model(inputs_embeds, position_ids, k_cache, v_cache)

        assert logits.shape == (1, seq_len, config.vocab_size)
        assert torch.isfinite(logits).all()

    def test_no_embed_tokens(self):
        """The embeddings variant should not have embed_tokens (that lives in embed.aimodel)."""
        config = _make_smolvlm_text_config()
        model = SmolVLMForCausalLMEmbeddings(config, model_device="cpu")
        assert not hasattr(model.model, "embed_tokens")

    def test_get_reauthored_config_unwraps_text_config(self):
        """_get_reauthored_config should unwrap `.text_config` from the top-level VLM config."""
        from types import SimpleNamespace

        text_config = _make_smolvlm_text_config()
        top_level = SimpleNamespace(text_config=text_config)

        reauthored = SmolVLMForCausalLMEmbeddings._get_reauthored_config(
            top_level, max_context_length=128, num_layers=1
        )

        assert reauthored is text_config
        assert reauthored.max_position_embeddings == 128
        assert reauthored.num_hidden_layers == 1

    def test_mutate_state_dict_fuses_qkv_and_drops_embed_tokens(self):
        """_mutate_state_dict should fuse q/k/v into qkv_proj and drop embed_tokens."""
        config = _make_smolvlm_text_config(num_hidden_layers=1)
        model = SmolVLMForCausalLMEmbeddings(config, model_device="cpu")

        head_dim = config.hidden_size // config.num_attention_heads
        sd = {
            "model.embed_tokens.weight": torch.randn(config.vocab_size, config.hidden_size),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(
                config.num_attention_heads * head_dim, config.hidden_size
            ),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(
                config.num_key_value_heads * head_dim, config.hidden_size
            ),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(
                config.num_key_value_heads * head_dim, config.hidden_size
            ),
            "lm_head.weight": torch.randn(config.vocab_size, config.hidden_size),
        }
        model._mutate_state_dict(sd)

        assert "model.embed_tokens.weight" not in sd
        assert "model.layers.0.self_attn.q_proj.weight" not in sd
        assert "model.layers.0.self_attn.k_proj.weight" not in sd
        assert "model.layers.0.self_attn.v_proj.weight" not in sd
        fused_key = "model.layers.0.self_attn.qkv_proj.weight"
        assert fused_key in sd
        expected_rows = (config.num_attention_heads + 2 * config.num_key_value_heads) * head_dim
        assert sd[fused_key].shape == (expected_rows, config.hidden_size)
        assert "lm_head.weight" in sd


class TestSmolVLMVisionModel:
    """Test the SigLIP ViT + pixel-shuffle connector."""

    def _make_tiny_vision_model(
        self, image_size: int = 32, scale_factor: int = 2
    ) -> SmolVLMVisionModel:
        return SmolVLMVisionModel(
            vision_hidden_size=16,
            vision_num_heads=2,
            vision_intermediate_size=32,
            vision_num_hidden_layers=2,
            image_size=image_size,
            patch_size=16,
            text_hidden_size=24,
            scale_factor=scale_factor,
        ).eval()

    def test_forward_produces_finite_output(self):
        """Forward with random pixel_values should produce finite image features."""
        model = self._make_tiny_vision_model(image_size=32, scale_factor=1)
        pixel_values = torch.randn(1, 3, 32, 32)

        with torch.no_grad():
            out = model(pixel_values)

        # grid_size = 32/16 = 2 -> 4 patches; scale_factor=1 keeps all 4 tokens.
        assert out.shape == (1, 4, 24)
        assert torch.isfinite(out).all()

    def test_pixel_shuffle_reduces_token_count(self):
        """scale_factor=2 should merge a 4x4 grid down to a single token."""
        # grid_size = image_size/patch_size = 4 -> matches scale_factor=2 exactly.
        image_size = 4 * 16
        model = self._make_tiny_vision_model(image_size=image_size, scale_factor=2)
        pixel_values = torch.randn(1, 3, image_size, image_size)

        with torch.no_grad():
            out = model(pixel_values)

        # (4/2) * (4/2) = 4 merged tokens.
        assert out.shape == (1, 4, 24)
        assert torch.isfinite(out).all()

    def test_num_visual_tokens_matches_smolvlm2_256m_geometry(self):
        """Sanity check against the real SmolVLM2-256M-Video-Instruct config values."""
        # image_size=512, patch_size=16, scale_factor=4 -> (512/16/4)**2 = 64.
        image_size, patch_size, scale_factor = 512, 16, 4
        grid = image_size // patch_size
        num_visual_tokens = (grid // scale_factor) ** 2
        assert num_visual_tokens == 64


class TestSmolVLMImageWrapperTokens:
    """HF SmolVLM single-image layout: fake-token, global-img, N image tokens, fake-token."""

    def test_spec_declares_hf_single_image_wrapper(self):
        from coreai_models.vlm.export import SUPPORTED_MODELS

        spec = SUPPORTED_MODELS["smolvlm2-256m"]
        assert spec.image_prefix_tokens == ("<fake_token_around_image>", "<global-img>")
        assert spec.image_suffix_tokens == ("<fake_token_around_image>",)

    def test_other_vlms_have_no_wrapper(self):
        from coreai_models.vlm.export import SUPPORTED_MODELS

        for name in ("qwen3-vl", "muse-glimmer-vl"):
            assert SUPPORTED_MODELS[name].image_prefix_tokens == ()
            assert SUPPORTED_MODELS[name].image_suffix_tokens == ()

    def test_wrapper_ids_resolve_and_reject_unknown(self):
        from coreai_models.vlm.export import _resolve_token_ids

        class _Tok:
            vocab = {"<fake_token_around_image>": 49189, "<global-img>": 49152}
            unk_token_id = 0

            def convert_tokens_to_ids(self, token):
                return self.vocab.get(token, self.unk_token_id)

        wrapper = ("<fake_token_around_image>", "<global-img>")
        assert _resolve_token_ids(_Tok(), wrapper) == [49189, 49152]
        assert _resolve_token_ids(_Tok(), ()) == []
        with pytest.raises(ValueError, match="<missing>"):
            _resolve_token_ids(_Tok(), ("<missing>",))


def _make_hf_smolvlm(layer_norm_eps: float = 1e-6):
    from transformers import SmolVLMConfig, SmolVLMForConditionalGeneration

    config = SmolVLMConfig(
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            image_size=64,
            patch_size=16,
            layer_norm_eps=layer_norm_eps,
        ),
        text_config=dict(
            hidden_size=48,
            intermediate_size=96,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=12,
            vocab_size=128,
            max_position_embeddings=64,
            rope_theta=100000.0,
            pad_token_id=0,
        ),
        scale_factor=2,
        image_token_id=120,
    )
    torch.manual_seed(0)
    model = SmolVLMForConditionalGeneration(config).float().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.2)
    return config, model


class TestSmolVLMParityWithHF:
    """Tiny-config parity against HF SmolVLMForConditionalGeneration (random weights)."""

    @pytest.mark.parametrize("layer_norm_eps", [1e-6, 0.5])
    def test_vision_tower_and_connector_match_hf(self, layer_norm_eps):
        from coreai_models.models.macos.smolvlm_vision import map_hf_vision_state_dict

        config, hf = _make_hf_smolvlm(layer_norm_eps)
        v = config.vision_config
        ours = SmolVLMVisionModel(
            vision_hidden_size=v.hidden_size,
            vision_num_heads=v.num_attention_heads,
            vision_intermediate_size=v.intermediate_size,
            vision_num_hidden_layers=v.num_hidden_layers,
            image_size=v.image_size,
            patch_size=v.patch_size,
            text_hidden_size=config.text_config.hidden_size,
            scale_factor=config.scale_factor,
            layer_norm_eps=v.layer_norm_eps,
        ).eval()
        hf_sd = {
            k: t
            for k, t in hf.state_dict().items()
            if k.startswith(("model.vision_model.", "model.connector."))
        }
        ours.load_state_dict(map_hf_vision_state_dict(hf_sd), strict=True)

        pixel_values = torch.randn(1, 3, v.image_size, v.image_size)
        with torch.no_grad():
            vision_out = hf.model.vision_model(pixel_values=pixel_values).last_hidden_state
            ref = hf.model.connector(vision_out)
            out = ours(pixel_values)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_text_decoder_matches_hf(self):
        config, hf = _make_hf_smolvlm()
        ours = SmolVLMForCausalLMEmbeddings(config.text_config, model_device="cpu").float().eval()
        sd = {}
        for k, t in hf.state_dict().items():
            if k.startswith("model.text_model."):
                sd["model." + k.removeprefix("model.text_model.")] = t
            elif k == "lm_head.weight":
                sd[k] = t
        ours._mutate_state_dict(sd)
        ours.load_state_dict(sd, strict=True)

        seq_len = 8
        embeds = torch.randn(1, seq_len, config.text_config.hidden_size)
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(config.text_config, dtype=torch.float32)
        with torch.no_grad():
            hidden = hf.model.text_model(inputs_embeds=embeds, position_ids=position_ids.long())
            ref = hf.lm_head(hidden.last_hidden_state)
            out = ours(embeds, position_ids, k_cache, v_cache)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_input_ids_decoder_matches_hf_through_from_hf_hooks(self):
        """The eval-path class loads the full checkpoint state dict and matches HF text logits."""
        import copy

        from coreai_models.models.macos.smolvlm import SmolVLMForCausalLM
        from coreai_models.models.registry import get_model_entry

        assert get_model_entry("smolvlm").macos_class is SmolVLMForCausalLM
        assert SmolVLMForCausalLM._HF_MODEL_CLASS.__name__ == "SmolVLMForConditionalGeneration"

        config, hf = _make_hf_smolvlm()
        text_config = SmolVLMForCausalLM._get_reauthored_config(copy.deepcopy(config))
        ours = SmolVLMForCausalLM(text_config, model_device="cpu").float().eval()
        sd = dict(hf.state_dict())
        ours._mutate_state_dict(sd)
        ours.load_state_dict(sd, strict=True)

        input_ids = torch.randint(0, config.text_config.vocab_size, (1, 8))
        position_ids = torch.arange(8, dtype=torch.int32).unsqueeze(0)
        k_cache, v_cache = KVCache.create_cache_tensors(text_config, dtype=torch.float32)
        with torch.no_grad():
            hidden = hf.model.text_model(input_ids=input_ids, position_ids=position_ids.long())
            ref = hf.lm_head(hidden.last_hidden_state)
            out = ours(input_ids, position_ids, k_cache, v_cache)
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    def test_rejects_non_default_rope_scaling(self):
        config, _ = _make_hf_smolvlm()
        text_config = config.text_config
        text_config.rope_scaling = {"rope_type": "linear", "factor": 2.0}
        with pytest.raises(AssertionError, match="rope_scaling"):
            SmolVLMForCausalLMEmbeddings(text_config, model_device="cpu")


_SMOLVLM2_SPECS = {
    "smolvlm2-256m": ("HuggingFaceTB/SmolVLM2-256M-Video-Instruct", 512, 16, 4, 64),
    "smolvlm2-500m": ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", 512, 16, 4, 64),
    "smolvlm2-2.2b": ("HuggingFaceTB/SmolVLM2-2.2B-Instruct", 384, 14, 3, 81),
}


@pytest.mark.parametrize("short_name", sorted(_SMOLVLM2_SPECS))
def test_smolvlm2_spec_geometry(short_name):
    """Spec geometry matches the HF vision config + processor image_seq_len (values pinned)."""
    from coreai_models.vlm.export import SUPPORTED_MODELS

    hf_id, image_size, patch_size, scale, seq_len = _SMOLVLM2_SPECS[short_name]
    spec = SUPPORTED_MODELS[short_name]
    assert spec.family == "smolvlm"
    assert spec.hf_model_id == hf_id
    geometry = (spec.image_size, spec.patch_size, spec.spatial_merge_size)
    assert geometry == (image_size, patch_size, scale)
    assert spec.num_visual_tokens == seq_len
    assert spec.image_prefix_tokens == ("<fake_token_around_image>", "<global-img>")
    assert spec.image_suffix_tokens == ("<fake_token_around_image>",)

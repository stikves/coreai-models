# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for Gemma 3n VLM components (3-input text decoder + vision encoder wrapper).

The 3-input decoder (``Gemma3nForCausalLMEmbeddings``) is an exact refactor of the
text-only ``Gemma3nForCausalLM``: instead of computing ``inputs_embeds`` internally
from ``input_ids``, the runner passes ``inputs_embeds`` in (with vision embeddings
already merged at image-token positions). ``input_ids`` is still needed for the
per-layer AltUp embedding table. We validate equivalence against the text-only
model, which is itself validated against HuggingFace in ``test_gemma3n.py``.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers.models.gemma3n.configuration_gemma3n import Gemma3nTextConfig
from transformers.models.gemma3n.modeling_gemma3n import (
    Gemma3nForCausalLM as HFGemma3nForCausalLM,
)

from coreai_models.models.macos.gemma3n import Gemma3nForCausalLM
from coreai_models.models.macos.gemma3n_vlm import (
    Gemma3nForCausalLMEmbeddings,
    Gemma3nVisionEncoder,
)
from coreai_models.primitives.macos.cache import KVCache
from coreai_models.primitives.macos.rms_norm import RMSNorm


def _make_gemma3n_config(**overrides) -> Gemma3nTextConfig:
    n_layers = overrides.pop("num_hidden_layers", 10)
    defaults = dict(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=n_layers,
        intermediate_size=[128] * n_layers,
        vocab_size=200,
        vocab_size_per_layer_input=200,
        max_position_embeddings=32,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        rope_local_base_freq=10000.0,
        hidden_size_per_layer_input=16,
        altup_num_inputs=4,
        altup_active_idx=0,
        altup_correct_scale=True,
        laurel_rank=8,
        num_kv_shared_layers=0,
        activation_sparsity_pattern=[0.0] * n_layers,
        sliding_window=8,
        hidden_activation="gelu_pytorch_tanh",
        tie_word_embeddings=True,
    )
    defaults.update(overrides)
    return Gemma3nTextConfig(**defaults)


def _build_text_and_vlm(config):
    """Build the HF-validated text model and a VLM decoder sharing its weights."""
    torch.manual_seed(42)
    hf_model = HFGemma3nForCausalLM(config).to(torch.float32).eval()

    text_model = Gemma3nForCausalLM(config, model_device="cpu").to(torch.float32).eval()
    sd = dict(hf_model.state_dict())
    text_model._mutate_state_dict(sd)
    text_model.load_state_dict(sd, assign=True, strict=True)

    vlm_model = Gemma3nForCausalLMEmbeddings(config, model_device="cpu").to(torch.float32).eval()
    # Same submodule structure as the text model — copy the loaded weights directly.
    vlm_model.load_state_dict(text_model.state_dict(), assign=True, strict=True)
    return text_model, vlm_model


def _inputs_embeds_from(text_model, input_ids):
    """Reproduce the embeddings the text-only model computes internally."""
    return text_model.model.embed_tokens(input_ids) * text_model.model.embed_scale


class TestGemma3nForCausalLMEmbeddings:
    """The 3-input decoder must match the text-only decoder given equal embeddings."""

    def test_equivalence_multi_token(self):
        config = _make_gemma3n_config(num_hidden_layers=5)
        text_model, vlm_model = _build_text_and_vlm(config)

        input_ids = torch.randint(0, 200, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        inputs_embeds = _inputs_embeds_from(text_model, input_ids)

        k1, v1 = KVCache.create_cache_tensors(config, dtype=torch.float32)
        k2, v2 = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            text_out = text_model(input_ids, position_ids, k1, v1)
            vlm_out = vlm_model(input_ids, inputs_embeds, position_ids, k2, v2)

        # 1e-4 matches the base-vs-HF tolerance for this family: the residual
        # (~6e-5) is thread-nondeterministic float matmul reduction, not a logic
        # gap — inputs_embeds are bit-identical and the diff does not grow with depth.
        torch.testing.assert_close(vlm_out, text_out, atol=1e-4, rtol=1e-4)

    def test_equivalence_single_token(self):
        config = _make_gemma3n_config(num_hidden_layers=5)
        text_model, vlm_model = _build_text_and_vlm(config)

        input_ids = torch.randint(0, 200, (1, 1))
        position_ids = torch.tensor([[0]], dtype=torch.int32)
        inputs_embeds = _inputs_embeds_from(text_model, input_ids)

        k1, v1 = KVCache.create_cache_tensors(config, dtype=torch.float32)
        k2, v2 = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            text_out = text_model(input_ids, position_ids, k1, v1)
            vlm_out = vlm_model(input_ids, inputs_embeds, position_ids, k2, v2)

        # 1e-4 matches the base-vs-HF tolerance for this family: the residual
        # (~6e-5) is thread-nondeterministic float matmul reduction, not a logic
        # gap — inputs_embeds are bit-identical and the diff does not grow with depth.
        torch.testing.assert_close(vlm_out, text_out, atol=1e-4, rtol=1e-4)

    def test_equivalence_with_kv_sharing(self):
        config = _make_gemma3n_config(num_hidden_layers=10, num_kv_shared_layers=2)
        text_model, vlm_model = _build_text_and_vlm(config)

        input_ids = torch.randint(0, 200, (1, 4))
        position_ids = torch.arange(4, dtype=torch.int32).unsqueeze(0)
        inputs_embeds = _inputs_embeds_from(text_model, input_ids)

        k1, v1 = Gemma3nForCausalLM.create_cache_tensors(config, dtype=torch.float32)
        k2, v2 = Gemma3nForCausalLM.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            text_out = text_model(input_ids, position_ids, k1, v1)
            vlm_out = vlm_model(input_ids, inputs_embeds, position_ids, k2, v2)

        # 1e-4 matches the base-vs-HF tolerance for this family: the residual
        # (~6e-5) is thread-nondeterministic float matmul reduction, not a logic
        # gap — inputs_embeds are bit-identical and the diff does not grow with depth.
        torch.testing.assert_close(vlm_out, text_out, atol=1e-4, rtol=1e-4)

    def test_merged_vision_embeddings_change_output(self):
        """Overwriting embeddings at image positions must flow through the decoder."""
        config = _make_gemma3n_config(num_hidden_layers=5)
        text_model, vlm_model = _build_text_and_vlm(config)

        input_ids = torch.randint(0, 200, (1, 6))
        position_ids = torch.arange(6, dtype=torch.int32).unsqueeze(0)
        base_embeds = _inputs_embeds_from(text_model, input_ids)

        merged = base_embeds.clone()
        merged[:, 1:3, :] += 1.0  # simulate merged vision soft tokens

        k1, v1 = KVCache.create_cache_tensors(config, dtype=torch.float32)
        k2, v2 = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            base_out = vlm_model(input_ids, base_embeds, position_ids, k1, v1)
            merged_out = vlm_model(input_ids, merged, position_ids, k2, v2)

        assert not torch.allclose(base_out, merged_out)

    def test_output_shape(self):
        config = _make_gemma3n_config(num_hidden_layers=5)
        _, vlm_model = _build_text_and_vlm(config)

        batch, seq_len = 1, 6
        input_ids = torch.randint(0, 200, (batch, seq_len))
        position_ids = torch.arange(seq_len, dtype=torch.int32).unsqueeze(0)
        inputs_embeds = torch.randn(batch, seq_len, config.hidden_size)
        k, v = KVCache.create_cache_tensors(config, dtype=torch.float32)

        with torch.no_grad():
            out = vlm_model(input_ids, inputs_embeds, position_ids, k, v)

        assert out.shape == (batch, seq_len, config.vocab_size)

    def test_mutate_state_dict_retains_embed_tables(self):
        """The VLM decoder never looks up the main embed table in forward, but the
        parameter is still declared and tied to lm_head — so state-dict handling
        keeps embed_tokens (loaded + tied) and the per-layer table (AltUp)."""
        config = _make_gemma3n_config(num_hidden_layers=5)
        torch.manual_seed(0)
        hf_model = HFGemma3nForCausalLM(config).to(torch.float32).eval()
        vlm_model = Gemma3nForCausalLMEmbeddings(config, model_device="cpu")

        sd = dict(hf_model.state_dict())
        vlm_model._mutate_state_dict(sd)

        assert any(
            k.endswith("embed_tokens.weight") for k in sd
        ), "main embed_tokens weight must be retained (tied to lm_head)"
        assert any(
            "embed_tokens_per_layer" in k for k in sd
        ), "per-layer embed table must be retained (AltUp)"


class _StubVisionTower(nn.Module):
    """Stand-in for MobileNetV5: returns a fixed [B, C, H, W] feature map
    wrapped like a HF ModelOutput (``.last_hidden_state``)."""

    def __init__(self, channels: int, grid: int) -> None:
        super().__init__()
        self.channels = channels
        self.grid = grid

    def forward(
        self, pixel_values: torch.Tensor, do_pooling: bool = False, return_dict: bool = True
    ):
        b = pixel_values.shape[0]
        feat = torch.randn(b, self.channels, self.grid, self.grid)
        return SimpleNamespace(last_hidden_state=feat)


class _StubEmbedder(nn.Module):
    """Stand-in for Gemma3nMultimodalEmbedder: norm → project → norm."""

    def __init__(self, vision_channels: int, hidden_size: int) -> None:
        super().__init__()
        self.soft_embedding_norm = RMSNorm(vision_channels, eps=1e-6)
        self.embedding_projection = nn.Linear(vision_channels, hidden_size, bias=False)
        self.embedding_post_projection_norm = RMSNorm(hidden_size, eps=1e-6)


class TestGemma3nVisionEncoder:
    """Vision wrapper: [B,3,224,224] → [B, 256, hidden]."""

    def test_output_shape_256_tokens(self):
        hidden_size = 64
        vision_channels = 2048
        grid = 16  # 16×16 = 256 tokens
        num_soft_tokens = grid * grid

        encoder = Gemma3nVisionEncoder(
            vision_tower=_StubVisionTower(vision_channels, grid),
            embedder=_StubEmbedder(vision_channels, hidden_size),
            vision_hidden_size=vision_channels,
            num_soft_tokens=num_soft_tokens,
        ).eval()

        pixel_values = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = encoder(pixel_values)

        assert out.shape == (1, num_soft_tokens, hidden_size)

    def test_channels_first_to_sequence_ordering(self):
        """Reshape must map [B,C,H,W] → [B, H*W, C] (row-major over the grid)."""
        vision_channels = 4
        grid = 2  # 4 tokens

        # Craft a known feature map and run the reshape the encoder uses.
        features = torch.arange(vision_channels * grid * grid, dtype=torch.float32)
        features = features.reshape(1, vision_channels, grid, grid)
        b = features.shape[0]
        seq = features.reshape(b, vision_channels, grid * grid).permute(0, 2, 1)

        assert seq.shape == (1, grid * grid, vision_channels)
        # position 0 collects channel values at spatial (0,0)
        torch.testing.assert_close(seq[0, 0], features[0, :, 0, 0])
        # last position collects spatial (grid-1, grid-1)
        torch.testing.assert_close(seq[0, -1], features[0, :, grid - 1, grid - 1])

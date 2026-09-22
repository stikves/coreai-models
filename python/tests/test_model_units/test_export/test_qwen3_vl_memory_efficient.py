# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Regression tests for ``from_hf_memory_efficient`` on the Qwen3-VL layout.

Qwen3-VL keeps its text weights under ``model.language_model.`` and its tie flag on
the top-level config. Exercises the shipped memory-efficient path against a tiny
synthetic checkpoint on disk: every parameter must leave the meta device, layer
truncation must be honoured, vision weights must be ignored, and both tied and
untied checkpoints must land the LM head. No HuggingFace download; the model is
built from local safetensors.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig, Qwen3VLTextConfig

from coreai_models.models.macos.qwen3_vl import Qwen3VLForCausalLM

_PREFIX = "model.language_model."
_HID = 32
_N_HEADS = 4
_N_KV_HEADS = 2
_HEAD_DIM = 8
_INTERMEDIATE = 64
_VOCAB = 48
_CKPT_LAYERS = 3


def _write_checkpoint(model_dir: Path, tied: bool) -> dict[str, torch.Tensor]:
    """Write a tiny Qwen3-VL checkpoint (config.json + model.safetensors) to disk.

    Returns the tensors written under their raw checkpoint keys so the caller can
    compare them against the loaded model.
    """
    torch.manual_seed(0)
    text_config = Qwen3VLTextConfig(
        hidden_size=_HID,
        num_hidden_layers=_CKPT_LAYERS,
        num_attention_heads=_N_HEADS,
        num_key_value_heads=_N_KV_HEADS,
        head_dim=_HEAD_DIM,
        intermediate_size=_INTERMEDIATE,
        vocab_size=_VOCAB,
        rms_norm_eps=1e-6,
    )
    Qwen3VLConfig(text_config=text_config.to_dict(), tie_word_embeddings=tied).save_pretrained(
        model_dir
    )

    q_dim = _N_HEADS * _HEAD_DIM
    kv_dim = _N_KV_HEADS * _HEAD_DIM

    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.float16)

    tensors: dict[str, torch.Tensor] = {
        f"{_PREFIX}embed_tokens.weight": rand(_VOCAB, _HID),
        f"{_PREFIX}norm.weight": rand(_HID),
        # Vision tower weights the text loader must ignore.
        "model.visual.patch_embed.proj.weight": rand(_HID, 16),
        "model.visual.blocks.0.norm1.weight": rand(_HID),
    }
    if not tied:
        tensors["lm_head.weight"] = rand(_VOCAB, _HID)

    for i in range(_CKPT_LAYERS):
        base = f"{_PREFIX}layers.{i}."
        tensors.update(
            {
                f"{base}self_attn.q_proj.weight": rand(q_dim, _HID),
                f"{base}self_attn.k_proj.weight": rand(kv_dim, _HID),
                f"{base}self_attn.v_proj.weight": rand(kv_dim, _HID),
                f"{base}self_attn.o_proj.weight": rand(_HID, q_dim),
                f"{base}self_attn.q_norm.weight": rand(_HEAD_DIM),
                f"{base}self_attn.k_norm.weight": rand(_HEAD_DIM),
                f"{base}mlp.gate_proj.weight": rand(_INTERMEDIATE, _HID),
                f"{base}mlp.up_proj.weight": rand(_INTERMEDIATE, _HID),
                f"{base}mlp.down_proj.weight": rand(_HID, _INTERMEDIATE),
                f"{base}input_layernorm.weight": rand(_HID),
                f"{base}post_attention_layernorm.weight": rand(_HID),
            }
        )

    save_file(tensors, str(model_dir / "model.safetensors"))
    return tensors


@pytest.mark.parametrize("tied", [True, False])
def test_from_hf_memory_efficient_loads_qwen3_vl(monkeypatch, tied):
    num_layers = 2
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp)
        written = _write_checkpoint(model_dir, tied=tied)

        monkeypatch.setattr(
            "coreai_models.models.base.snapshot_download",
            lambda *args, **kwargs: str(model_dir),
        )

        model = Qwen3VLForCausalLM.from_hf_memory_efficient(
            "dummy/qwen3-vl",
            max_context_length=128,
            num_layers=num_layers,
            hf_config_attr="text_config",
            hf_state_dict_prefix=_PREFIX,
        )

        meta = [name for name, p in model.named_parameters() if p.is_meta]
        assert meta == [], f"parameters left on meta device: {meta}"

        assert len(model.model.layers) == num_layers
        assert not any("visual" in name for name, _ in model.named_parameters())

        torch.testing.assert_close(
            model.model.embed_tokens.weight, written[f"{_PREFIX}embed_tokens.weight"]
        )
        torch.testing.assert_close(model.model.norm.weight, written[f"{_PREFIX}norm.weight"])

        attn0 = model.model.layers[0].self_attn
        torch.testing.assert_close(
            attn0.o_proj.weight, written[f"{_PREFIX}layers.0.self_attn.o_proj.weight"]
        )
        expected_qkv = torch.cat(
            [
                written[f"{_PREFIX}layers.0.self_attn.q_proj.weight"],
                written[f"{_PREFIX}layers.0.self_attn.k_proj.weight"],
                written[f"{_PREFIX}layers.0.self_attn.v_proj.weight"],
            ],
            dim=0,
        )
        torch.testing.assert_close(attn0.qkv_proj.weight, expected_qkv)

        if tied:
            assert model.lm_head.weight is model.model.embed_tokens.weight
        else:
            torch.testing.assert_close(model.lm_head.weight, written["lm_head.weight"])

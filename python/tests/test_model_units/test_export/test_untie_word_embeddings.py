# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Tests for the `coreai_models.untie_word_embeddings` compression-recipe option."""

from pathlib import Path

import pytest
import torch
from transformers.models.smollm3.modeling_smollm3 import SmolLM3Config

from coreai_models.export.compression import quantize_pytorch_model, untie_word_embeddings
from coreai_models.llm.export import _load_compression_config_object
from coreai_models.models.macos.smollm3 import SmolLM3ForCausalLM
from coreai_models.primitives.macos.cache import KVCache

pytest.importorskip("coreai_opt")
from coreai_opt import ExportBackend  # noqa: E402

VOCAB_SIZE = 512

_INT8_BLOCK32 = """
      op_state_spec:
        weight:
          dtype: int8
          qscheme: symmetric_with_clipping
          granularity:
            type: per_block
            block_size: 32
            axis: 1
      op_input_spec: null
      op_output_spec: null
"""


def _write_recipe(tmp_path: Path, untie: object | None) -> Path:
    """INT4 decoder, embedding excluded, lm_head named at INT8."""
    text = ""
    if untie is not None:
        text += f"coreai_models:\n  untie_word_embeddings: {untie}\n"
    text += (
        "quantization_config:\n"
        "  execution_mode: eager\n"
        "  global_config:\n"
        "    op_state_spec:\n"
        "      weight:\n"
        "        dtype: int4\n"
        "        qscheme: symmetric_with_clipping\n"
        "        granularity:\n"
        "          type: per_block\n"
        "          block_size: 32\n"
        "          axis: 1\n"
        "    op_input_spec: null\n"
        "    op_output_spec: null\n"
        "  module_type_configs:\n"
        "    coreai_models.primitives.macos.sdpa.SDPA: null\n"
        "    coreai_models.primitives.macos.rope.RoPE: null\n"
        "    coreai_models.primitives.macos.rms_norm.RMSNorm: null\n"
        "    torch.nn.modules.sparse.Embedding: null\n"
        "  module_name_configs:\n"
        "    lm_head:" + _INT8_BLOCK32
    )
    path = tmp_path / "recipe.yaml"
    path.write_text(text)
    return path


def _tiny_tied_model() -> tuple[SmolLM3ForCausalLM, SmolLM3Config]:
    torch.manual_seed(0)
    config = SmolLM3Config(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        intermediate_size=128,
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=64,
        no_rope_layer_interval=2,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SmolLM3ForCausalLM(config, model_device="cpu").float().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.5)
    model.lm_head.weight = model.model.embed_tokens.weight
    return model, config


def _quantize(model: SmolLM3ForCausalLM, config: SmolLM3Config, recipe: dict) -> torch.nn.Module:
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    position_ids = torch.arange(8, dtype=torch.int32).unsqueeze(0)
    k_cache, v_cache = KVCache.create_cache_tensors(config, dtype=torch.float32, seq_len=32)
    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_ids", max=30)},
        "position_ids": {1: torch.export.Dim("seq_pos", min=2, max=31)},
        "k_cache": None,
        "v_cache": None,
    }
    return quantize_pytorch_model(
        model,
        (input_ids, position_ids, k_cache, v_cache),
        dynamic_shapes,
        recipe,
        cache_seq_len=32,
        state_indices=(2, 3),
        export_backend=ExportBackend._TORCH,
    )


def test_loader_merges_untie_option(tmp_path):
    recipe = _load_compression_config_object(_write_recipe(tmp_path, "true"), "macOS")
    assert recipe["untie_word_embeddings"] is True


def test_loader_rejects_non_bool_untie_option(tmp_path):
    with pytest.raises(SystemExit, match="untie_word_embeddings"):
        _load_compression_config_object(_write_recipe(tmp_path, "'yes'"), "macOS")


def test_untied_recipe_quantizes_lm_head_only(tmp_path):
    model, config = _tiny_tied_model()
    ref_embed = model.model.embed_tokens.weight.detach().clone()
    recipe = _load_compression_config_object(_write_recipe(tmp_path, "true"), "macOS")

    quantized = _quantize(model, config, recipe)

    hidden = torch.randn(1, 8, 64)
    ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    with torch.no_grad():
        head_delta = (quantized.lm_head(hidden) - hidden @ ref_embed.T).abs().max().item()
        embed_out = quantized.model.embed_tokens(ids)
    assert type(quantized.model.embed_tokens) is torch.nn.Embedding
    torch.testing.assert_close(embed_out, ref_embed[ids], atol=0, rtol=0)
    assert head_delta > 0


def test_tied_recipe_quantizes_embedding_too(tmp_path):
    # coreai-opt resolves one config per shared weight: naming the tied lm_head
    # also quantizes the input embedding.
    model, config = _tiny_tied_model()
    ref_embed = model.model.embed_tokens.weight.detach().clone()
    recipe = _load_compression_config_object(_write_recipe(tmp_path, None), "macOS")

    quantized = _quantize(model, config, recipe)

    ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    with torch.no_grad():
        embed_delta = (quantized.model.embed_tokens(ids) - ref_embed[ids]).abs().max().item()
    assert embed_delta > 0


def test_untie_requires_tied_weights():
    model, _ = _tiny_tied_model()
    untie_word_embeddings(model)
    assert model.lm_head.weight is not model.model.embed_tokens.weight
    torch.testing.assert_close(model.lm_head.weight, model.model.embed_tokens.weight)
    with pytest.raises(ValueError, match="tied"):
        untie_word_embeddings(model)


@pytest.mark.parametrize("short_name", ["smollm3-3b", "smollm3-3b-base"])
def test_smollm3_presets_ship_untied_mixed_recipe(short_name):
    from coreai_models.model_registry import lookup_preset

    preset = lookup_preset(short_name, model_type="llm", variant="macOS")
    repo_root = Path(__file__).resolve().parents[4]
    recipe = _load_compression_config_object(repo_root / preset.compression_config, "macOS")
    assert recipe["untie_word_embeddings"] is True
    assert "lm_head" in recipe["module_name_configs"]

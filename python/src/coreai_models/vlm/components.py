# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""VLM component specifications.

Three components per VLM bundle, each exported as its own .aimodel:
- `vision`: CLIP encoder + projector (stateless, 1 input → 1 output).
- `embedding`: text `embed_tokens` lookup (stateless, 1 input → 1 output).
- `model`: LLM main with embedding input (stateful, 4-input/3-output
  declared, auto-aliased to 2-input + 2-state at runtime).

Mirrors the shape of `coreai_models.diffusion.components.ComponentSpec` but
with the LLM stateful path baked into the model component.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from coreai_models.primitives.macos.cache import KVCache
from coreai_models.vlm.llava import LLaVALanguageModelWrapper


@dataclass(frozen=True)
class ComponentSpec:
    """Everything needed to export one VLM component."""

    asset_name: str
    """Filename inside the bundle, e.g. 'vision.aimodel'."""

    component_key: str
    """Role key written into metadata's `models` map ('main', 'vision', 'embedding')."""

    input_names: tuple[str, ...]
    """Names assigned to the exported function's inputs (excludes states)."""

    output_names: tuple[str, ...]
    """Names assigned to the exported function's outputs (excludes states)."""

    wrapper_fn: Callable[[dict[str, Any]], torch.nn.Module]
    """Build the torch.nn.Module to export from the loaded LLaVA components."""

    dummy_fn: Callable[[dict[str, Any]], tuple[dict[str, Any], dict | None]]
    """Build (reference_inputs, dynamic_shapes_or_None) for `export_to_coreai`."""

    state_names: tuple[str, ...] = ()
    """Names of inputs that are mutated in place (KV cache)."""


# ---------------------------------------------------------------------------
# Wrapper builders — pull the right module out of the loaded LLaVA dict.
# ---------------------------------------------------------------------------


def _vision_wrapper(loaded: dict[str, Any]) -> torch.nn.Module:
    return loaded["vision"]


def _embedding_wrapper(loaded: dict[str, Any]) -> torch.nn.Module:
    return loaded["text_embedder"]


def _model_wrapper(loaded: dict[str, Any]) -> torch.nn.Module:
    # Wrap LlamaForCausalLM so its forward calls forward_from_embeddings
    # — keeps the model component's exported entrypoint named "main" with
    # an `in_embeddings` input rather than `input_ids`.
    return LLaVALanguageModelWrapper(loaded["language_model"])


# ---------------------------------------------------------------------------
# Dummy-input builders.
#
# - Stateless components return a plain tuple of positional tensors.
# - Stateful (model) component returns (reference_inputs_dict, dynamic_shapes)
#   matching `export_to_coreai`'s signature.
# ---------------------------------------------------------------------------


def _vision_dummy(loaded: dict[str, Any]) -> tuple[Any, ...]:
    vision_config = loaded["vision_config"]
    # Square spatial extent — CLIP-style vision encoders expect a fixed
    # H == W. Surfaced by the HF config; no defensive fallback because a
    # missing `image_size` would silently produce a different exported
    # graph shape than the runtime preprocessor uses.
    image_size: int = vision_config.image_size
    # CLIP vision encoder takes Float32 normalized RGB pixels NCHW.
    pixel_values = torch.zeros(1, 3, image_size, image_size, dtype=torch.float32)
    reference_inputs = {"pixel_values": pixel_values}
    dynamic_shapes = None
    return (reference_inputs, dynamic_shapes)


def _embedding_dummy(loaded: dict[str, Any]) -> tuple[Any, ...]:
    text_config = loaded["text_config"]
    vocab_size = getattr(text_config, "vocab_size", 32064)
    max_context_length = getattr(text_config, "max_position_embeddings", 2048)
    # Runtime seq_len varies (576 image placeholders + prompt tokens etc.),
    # so the seq dim must be dynamic.
    query_len = max(8, max_context_length // 4)
    input_ids = torch.randint(1, vocab_size, (1, query_len), dtype=torch.int32)
    reference_inputs = {"input_ids": input_ids}
    dynamic_shapes = {
        "input_ids": {1: torch.export.Dim("seq_ids", min=1, max=max_context_length - 1)}
    }
    return (reference_inputs, dynamic_shapes)


def _model_dummy(loaded: dict[str, Any]) -> tuple[Any, ...]:
    """Build (reference_inputs_dict, dynamic_shapes) for the LLM main export.

    Mirrors `coreai_models.export.macos._build_reference_inputs` but with
    `in_embeddings` instead of `input_ids`. Sizes scaled to fit LLaVA-1.5's
    2048-token context (the existing text-only path assumes much larger
    contexts and would produce a `Dim(min=2048, max=2047)` here).
    """
    language_model = loaded["language_model"]
    text_config = loaded["text_config"]
    target_dtype = next(language_model.parameters()).dtype

    max_context_length = getattr(text_config, "max_position_embeddings", 2048)
    hidden_size = getattr(text_config, "hidden_size", 4096)

    # Reference query window. ~1/4 of max keeps room for KV-cache padding
    # in the dummy and gives torch.export a meaningful symbolic-range hint.
    batch_size = 1
    query_len = max(8, max_context_length // 4)
    pad_len = max(8, max_context_length - query_len - 1)

    in_embeddings = torch.zeros(batch_size, query_len, hidden_size, dtype=target_dtype)
    position_ids = (
        torch.arange(query_len + pad_len, dtype=torch.int32)
        .unsqueeze(0)
        .expand(batch_size, query_len + pad_len)
    )
    k_cache, v_cache = KVCache.create_cache_tensors(text_config, dtype=target_dtype)

    reference_inputs = {
        "in_embeddings": in_embeddings,
        "position_ids": position_ids,
        "k_cache": k_cache,
        "v_cache": v_cache,
    }

    dynamic_shapes = {
        "in_embeddings": {1: torch.export.Dim("seq_ids", min=2, max=max_context_length - 2)},
        "position_ids": {1: torch.export.Dim("seq_pos", min=query_len, max=max_context_length - 1)},
        "k_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "k_seq_len", min=query_len, max=max_context_length
            )
        },
        "v_cache": {
            KVCache.seq_len_dim(): torch.export.Dim(
                "v_seq_len", min=query_len, max=max_context_length
            )
        },
    }

    return (reference_inputs, dynamic_shapes)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


# KV cache is declared as state_names so the Core AI converter creates
# proper mutable state handles.
LLAVA_COMPONENTS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        asset_name="vision.aimodel",
        component_key="vision",
        input_names=("pixel_values",),
        output_names=("image_embeds",),
        wrapper_fn=_vision_wrapper,
        dummy_fn=_vision_dummy,
    ),
    ComponentSpec(
        asset_name="embed.aimodel",
        component_key="embedding",
        input_names=("input_ids",),
        output_names=("text_embeddings",),
        wrapper_fn=_embedding_wrapper,
        dummy_fn=_embedding_dummy,
    ),
    ComponentSpec(
        asset_name="model.aimodel",
        component_key="main",
        input_names=("in_embeddings", "position_ids"),
        output_names=("logits",),
        state_names=("keyCache", "valueCache"),
        wrapper_fn=_model_wrapper,
        dummy_fn=_model_dummy,
    ),
)

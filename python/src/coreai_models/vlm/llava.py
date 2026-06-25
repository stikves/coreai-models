# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""LLaVA-1.5 components for VLM export.

Defines the model-family-specific bits (projector module, vision/text
backbones loader). The vision encoder is HF's `CLIPVisionModel` (used as
a black box); the projector is a 2-layer MLP we re-author to keep
weights inspectable; the LLM backbone reuses
`coreai_models.models.macos.llama.LlamaForCausalLM` (which already exposes
`forward_from_embeddings`).
"""

import logging
from typing import Any

import torch
import torch.nn as nn
from transformers import LlavaForConditionalGeneration

from coreai_models.models.macos.llama import LlamaForCausalLM

logger = logging.getLogger(__name__)


class LLaVAMultiModalProjector(nn.Module):
    """LLaVA-1.5 projector: Linear → GELU → Linear (both with bias).

    Maps CLIP vision features `[B, num_patches, vision_hidden]` into LLM
    embedding space `[B, num_patches, llm_hidden]`. No spatial pooling —
    all 576 patch tokens (24×24 at 336×336/patch=14) pass through directly.
    """

    def __init__(self, vision_hidden_size: int, text_hidden_size: int) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(vision_hidden_size, text_hidden_size, bias=True)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(text_hidden_size, text_hidden_size, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        h = self.linear_1(image_features)
        h = self.act(h)
        h = self.linear_2(h)
        return h


class LLaVAVisionTower(nn.Module):
    """Vision encoder + projector wrapped as a single forward.

    Input: pixel_values `[1, 3, H, W]` Float32.
    Output: image_embeds `[1, num_patches, llm_hidden]` Float32.

    `num_patches = (H/patch_size)**2` — 576 for LLaVA-1.5 at 336×336/14.
    The CLIP vision model emits a CLS token + patches; the LLaVA convention
    drops the CLS and keeps only the patch tokens.
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        projector: LLaVAMultiModalProjector,
        select_layer: int = -2,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector
        # LLaVA-1.5 uses the second-to-last hidden state as image features
        # (matches HF reference) — `output_hidden_states=True` + index -2.
        self.select_layer = select_layer

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_tower(
            pixel_values=pixel_values,
            output_hidden_states=True,
        )
        # outputs.hidden_states is a tuple of [B, 1+num_patches, vision_hidden].
        # LLaVA drops the CLS token (index 0) and uses index 1: onwards.
        image_features = outputs.hidden_states[self.select_layer][:, 1:, :]
        return self.projector(image_features)


class LLaVATextEmbedder(nn.Module):
    """Wraps the LLM's `embed_tokens` lookup as a standalone module.

    Input: input_ids `[1, S]` Int32.
    Output: text_embeds `[1, S, llm_hidden]` Float16/Float32 (matching the
    underlying weight dtype).
    """

    def __init__(self, embed_tokens: nn.Module) -> None:
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class LLaVALanguageModelWrapper(nn.Module):
    """Adapts `LlamaForCausalLM.forward_from_embeddings` to the same 4-input
    forward signature `export_macos_model` expects, but with an embedding input
    instead of token IDs. Auto-aliasing on `keyCache`/`valueCache` works the
    same way — output is 2-input + 2-state + 1-output at runtime.
    """

    def __init__(self, llama: LlamaForCausalLM) -> None:
        super().__init__()
        self.llama = llama

    def forward(
        self,
        in_embeddings: torch.Tensor,
        position_ids: torch.IntTensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        return self.llama.forward_from_embeddings(in_embeddings, position_ids, k_cache, v_cache)


# ---------------------------------------------------------------------------
# Loaders — extract LLaVA pieces from a HuggingFace `LlavaForConditionalGeneration`
# ---------------------------------------------------------------------------


def load_llava_components(
    hf_model_id: str,
    target_dtype: torch.dtype = torch.float16,
    max_context_length: int | None = None,
) -> dict[str, Any]:
    """Load a LLaVA-1.5 HF checkpoint and split it into the three components
    we want to export: vision (encoder + projector), embed (text embedding
    lookup), and model (LLM main with embedding input).

    Returns a dict with keys: `vision_tower`, `projector`, `text_embedder`,
    `language_model` (the latter is a `LlamaForCausalLM`, our re-authored
    version, whose `forward_from_embeddings` is what the model component
    exports).
    """
    logger.info(f"Loading LLaVA model {hf_model_id} (dtype={target_dtype})...")
    hf_model = LlavaForConditionalGeneration.from_pretrained(
        hf_model_id,
        dtype=target_dtype,
    )

    vision_config = hf_model.config.vision_config
    text_config = hf_model.config.text_config
    if max_context_length is not None:
        text_config.max_position_embeddings = max_context_length

    # LLaVA-1.5's text config carries `rope_scaling = {'rope_type': 'default',
    # 'rope_theta': 10000.0}`. Coreai-models's LlamaForCausalLM only accepts
    # `rope_scaling=None` or `rope_type=='llama3'`, so normalize the
    # "default" case to None and lift rope_theta to a top-level attribute
    # (which is what the rope-init expects).
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling is not None and rope_scaling.get("rope_type") == "default":
        text_config.rope_theta = rope_scaling.get("rope_theta", 10000.0)
        text_config.rope_scaling = None

    # --- Vision tower: HF's CLIPVisionModel, used as-is (black box). ---
    vision_tower_hf = hf_model.model.vision_tower
    vision_tower_hf.eval()

    # --- Projector: re-author with our weights so the export sees a
    # plain torch.nn.Module with named submodules (linear_1, linear_2).
    projector = LLaVAMultiModalProjector(
        vision_hidden_size=vision_config.hidden_size,
        text_hidden_size=text_config.hidden_size,
    ).to(dtype=target_dtype)
    projector.load_state_dict(
        hf_model.model.multi_modal_projector.state_dict(), strict=True, assign=True
    )
    projector.eval()

    # --- Language model: re-author as our LlamaForCausalLM so we get
    # `forward_from_embeddings` and the in-place KVCache pattern that
    # auto-aliases on export.
    #
    # NB: HF stores LLaVA's text backbone as `model.language_model.*` —
    # `LlamaForCausalLM.from_hf(hf_model_id)` doesn't know that mapping
    # and would attempt to load from a plain-Llama checkpoint structure
    # (which doesn't exist for a LLaVA model ID). Instead, build the
    # model from `text_config` and remap the state dict ourselves.
    language_model = LlamaForCausalLM(text_config, model_device="meta")
    language_model.to(dtype=target_dtype)

    text_state_dict: dict[str, torch.Tensor] = {}
    lm_prefix = "model.language_model."
    for key, value in hf_model.state_dict().items():
        if key.startswith(lm_prefix):
            text_state_dict["model." + key[len(lm_prefix) :]] = value
        elif key == "lm_head.weight":
            text_state_dict[key] = value

    # `_mutate_state_dict` fuses q/k/v projections etc. — must be called
    # before `load_state_dict` for our keys to match.
    language_model._mutate_state_dict(text_state_dict)
    language_model.load_state_dict(text_state_dict, strict=True, assign=True)
    language_model.eval()

    # --- Text embedder: the embed_tokens layer from our re-authored LLM.
    text_embedder = LLaVATextEmbedder(language_model.model.embed_tokens)
    text_embedder.eval()

    # We're done with the HF wrapper. Hold a direct ref to `vision_tower_hf`
    # (its `vision_tower` submodule) so we can drop the rest and free memory.
    del hf_model

    vision = LLaVAVisionTower(vision_tower_hf, projector)
    vision.eval()

    return {
        "vision": vision,
        "text_embedder": text_embedder,
        "language_model": language_model,
        "vision_config": vision_config,
        "text_config": text_config,
    }

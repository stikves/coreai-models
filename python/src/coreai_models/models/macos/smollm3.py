# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""SmolLM3: Qwen2-style decoder with NoPE on every `no_rope_layer_interval`-th layer.

The per-layer RoPE skip is handled by `qwen2.Attention` via `config.no_rope_layers`.
"""

from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3Config,
)
from transformers.models.smollm3.modeling_smollm3 import (
    SmolLM3ForCausalLM as HFSmolLM3ForCausalLM,
)
from typing_extensions import override

from coreai_models.models.macos.qwen2 import Qwen2ForCausalLM


class SmolLM3ForCausalLM(Qwen2ForCausalLM):
    _HF_MODEL_CLASS = HFSmolLM3ForCausalLM

    @override
    def _init_model(self, config: SmolLM3Config) -> None:
        if getattr(config, "use_sliding_window", False):
            raise ValueError("SmolLM3 sliding-window attention is not supported")
        super()._init_model(config)

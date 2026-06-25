# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""VLM (Vision-Language Model) export pipeline for coreai-models.

Multi-component exporter producing model bundles with three sub-models:
vision (CLIP encoder + projector), embedding (text embed_tokens), and main
(LLM decoder with embedding input). Mirrors the pattern of
`coreai_models.diffusion`.
"""

from coreai_models.vlm.models import SUPPORTED_MODELS, get_family_key, list_models
from coreai_models.vlm.pipeline import VLMExportConfig, export_vlm

__all__ = [
    "SUPPORTED_MODELS",
    "VLMExportConfig",
    "export_vlm",
    "get_family_key",
    "list_models",
]

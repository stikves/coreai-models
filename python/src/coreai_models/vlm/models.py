# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""Supported VLM model families.

Listed by `coreai.vlm.export --list-models`. Each entry is
(family-name, example HF model ID, family-key) where family-key drives
component-spec selection.
"""

# Each entry: (family name, example HF model ID, family key)
SUPPORTED_MODELS: list[tuple[str, str, str]] = [
    ("llava-1.5", "llava-hf/llava-1.5-7b-hf", "llava"),
]


def list_models() -> list[str]:
    """Return the names of supported VLM model families."""
    return [name for name, _, _ in SUPPORTED_MODELS]


def get_family_key(model_id: str) -> str:
    """Determine the family key for a given HF model ID.

    Returns one of the family keys (e.g. 'llava'). Raises ValueError for
    unknown models.
    """
    for _, known_id, key in SUPPORTED_MODELS:
        if model_id == known_id:
            return key
    raise ValueError(
        f"Unknown VLM model: '{model_id}'. Supported: {[mid for _, mid, _ in SUPPORTED_MODELS]}"
    )

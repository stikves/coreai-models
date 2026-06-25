# Copyright 2026 Apple Inc.
#
# Use of this source code is governed by a BSD-3-clause license that can
# be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

"""CLI entry point for `coreai.vlm.export`."""

import argparse
import logging
import sys
from pathlib import Path

from coreai_models.export.presets import (
    DEFAULT_MACOS_COMPRESSION_PRESET,
    list_macos_presets,
)
from coreai_models.model_registry import (
    presets_for_type,
    try_lookup_preset,
    try_lookup_preset_by_hf_id,
)
from coreai_models.vlm.models import SUPPORTED_MODELS, list_models
from coreai_models.vlm.pipeline import VLMExportConfig, export_vlm


def _default_output_dir() -> str:
    """Resolve exports/ relative to the workspace root."""
    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists() and (d / "python").exists():
            return str(d / "exports")
        d = d.parent
    return "exports"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreai.vlm.export",
        description="Export HuggingFace VLM models to a multi-component model bundle",
    )
    parser.add_argument(
        "model",
        nargs="?",
        help="HuggingFace model ID (e.g. llava-hf/llava-1.5-7b-hf)",
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output_dir",
        default=None,
        help="Output directory for the bundle (default: <repo-root>/exports/)",
    )
    parser.add_argument(
        "--compute-precision",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Compute precision for the exported model (default: float16)",
    )
    parser.add_argument(
        "--compression",
        default="none",
        help=(
            "Quantization preset for the language_model component. "
            f"Available: {', '.join(list_macos_presets())}. Default: 'none'. "
            f"Suggested: '{DEFAULT_MACOS_COMPRESSION_PRESET}' for ~4x model-size reduction. "
            "Vision encoder and embed_tokens stay full precision."
        ),
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=None,
        help="Override max context length (default: from HF config).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing component asset directories.",
    )
    parser.add_argument(
        "--experimental",
        action="store_true",
        help="Allow exporting models without a registry preset.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List supported VLM model families and exit.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    parser = build_parser()
    args = parser.parse_args()

    if args.list_models:
        print("Supported VLM models:")
        for p in presets_for_type("vlm"):
            print(f"  {p.short_name:20s}  {p.hf_id}")
        return

    if not args.model:
        parser.print_help()
        print(f"\nSupported families: {list_models()}")
        sys.exit(1)

    # Resolve model: short-name → preset, or HF ID → preset for defaults
    preset = None
    hf_model_id = args.model
    if "/" not in args.model:
        preset = try_lookup_preset(args.model, model_type="vlm")
        if preset is None and not args.experimental:
            available = [p.short_name for p in presets_for_type("vlm")]
            print(
                f"Error: '{args.model}' is not a registered VLM short-name and doesn't "
                "look like a HuggingFace ID (expected 'org/model').\n"
                f"Available: {available}",
                file=sys.stderr,
            )
            sys.exit(1)
        if preset is not None:
            hf_model_id = preset.hf_id
    else:
        preset = try_lookup_preset_by_hf_id(args.model, model_type="vlm")

    if preset is None and not args.experimental:
        known_ids = [hf_id for _, hf_id, _ in SUPPORTED_MODELS]
        print(
            f"Error: '{args.model}' is not a supported VLM model.\n"
            f"Supported: {known_ids}\n"
            "Pass --experimental to try exporting it anyway.",
            file=sys.stderr,
        )
        sys.exit(1)

    compute_precision = args.compute_precision
    compression = args.compression
    if preset is not None:
        if compute_precision == "float16" and preset.compute_precision:
            compute_precision = preset.compute_precision
        if compression == "none" and preset.compression and preset.compression != "none":
            compression = preset.compression

    config = VLMExportConfig(
        hf_model_id=hf_model_id,
        output_dir=args.output_dir or _default_output_dir(),
        compute_precision=compute_precision,
        compression=compression,
        max_context_length=args.max_context_length,
        overwrite=args.overwrite,
    )
    bundle_path = export_vlm(config)
    print(f"\nExported VLM bundle: {bundle_path}")


if __name__ == "__main__":
    main()

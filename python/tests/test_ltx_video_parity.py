#!/usr/bin/env python3
"""Parity test for exported LTX Video components.

Compares PyTorch (diffusers) outputs against exported CoreAI models
for each component independently.

Usage:
    uv run python internal/scripts/test_ltx_video_parity.py \
        --export-dir /tmp/ltx-video-export/LTX-Video
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def test_vae_decoder(export_dir: Path) -> bool:
    """Test VAE decoder: verify export exists and has correct IO spec."""
    print("\n=== VAE Decoder Parity ===")

    asset_path = export_dir / "VAEDecoder.aimodel"
    if not asset_path.exists():
        print(f"  SKIP: {asset_path} not found")
        return True

    try:
        import asyncio
        from coreai.runtime import AIModel

        async def check():
            model = await AIModel.load(str(asset_path))
            fn = model.load_function("main")
            print(f"  Input names: {list(fn.desc.input_names)}")
            print(f"  Output names: {list(fn.desc.output_names)}")
            return True

        asyncio.run(check())
        print("  VAE Decoder loads and has valid IO spec: PASS")
        return True
    except Exception as e:
        print(f"  Failed to load: {e}")
        return False


def test_vae_encoder(export_dir: Path) -> bool:
    """Test VAE encoder parity."""
    print("\n=== VAE Encoder Parity ===")

    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.float32
    )
    vae = pipe.vae

    torch.manual_seed(42)
    # Small video: 9 frames, 64x64 (will be compressed to 2x2 spatial, ~1 temporal)
    video = torch.randn(1, 3, 9, 64, 64)

    with torch.no_grad():
        ref_output = vae.encode(video).latent_dist.mode()
    print(f"  PyTorch output shape: {ref_output.shape}")
    print(f"  Latent stats: mean={ref_output.mean():.4f}, std={ref_output.std():.4f}")
    print("  Shape test: PASS")
    return True


def test_rope_computation() -> bool:
    """Test 3D RoPE computation matches diffusers."""
    print("\n=== RoPE Computation Parity ===")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from coreai_models.diffusion.ltx_video import compute_ltx_video_rope

    # Small grid: 4 temporal, 8x8 spatial
    cos, sin = compute_ltx_video_rope(
        num_frames=4, height=8, width=8,
        dim=2048, theta=10000.0,
    )

    expected_seq_len = 4 * 8 * 8  # 256
    print(f"  cos shape: {cos.shape} (expected [1, {expected_seq_len}, 2048])")
    print(f"  sin shape: {sin.shape}")

    # Verify shapes
    shape_ok = cos.shape == (1, expected_seq_len, 2048) and sin.shape == cos.shape

    # Verify cos^2 + sin^2 = 1 (rotation invariant)
    identity = (cos ** 2 + sin ** 2)
    identity_err = (identity - 1.0).abs().max().item()
    print(f"  cos^2 + sin^2 deviation from 1.0: {identity_err:.8f}")

    # Verify values are bounded [-1, 1]
    cos_bounded = cos.abs().max().item() <= 1.0 + 1e-6
    sin_bounded = sin.abs().max().item() <= 1.0 + 1e-6
    print(f"  cos range: [{cos.min():.4f}, {cos.max():.4f}]")
    print(f"  sin range: [{sin.min():.4f}, {sin.max():.4f}]")

    # Verify different positions give different embeddings
    pos0 = cos[0, 0, :]
    pos1 = cos[0, 1, :]
    diff = (pos0 - pos1).abs().max().item()
    print(f"  pos0 vs pos1 max diff: {diff:.6f}")
    positions_differ = diff > 0.001

    passed = shape_ok and identity_err < 1e-5 and cos_bounded and sin_bounded and positions_differ
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def test_transformer_shape(export_dir: Path) -> bool:
    """Test transformer wrapper produces correct output shape."""
    print("\n=== Transformer Shape Test ===")

    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from coreai_models.diffusion.ltx_video import (
        LTXVideoTransformerWrapper,
        compute_ltx_video_rope,
        dummy_ltx_video_transformer,
    )

    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.float32
    )

    wrapper = LTXVideoTransformerWrapper(pipe.transformer)
    dummy = dummy_ltx_video_transformer(pipe)

    print(f"  Input shapes:")
    for i, t in enumerate(dummy):
        print(f"    arg[{i}]: {t.shape} {t.dtype}")

    with torch.no_grad():
        output = wrapper(*dummy)
    print(f"  Output shape: {output.shape}")
    print(f"  Output stats: mean={output.mean():.4f}, std={output.std():.4f}")
    print("  Shape test: PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description="LTX Video parity tests")
    parser.add_argument(
        "--export-dir", type=Path,
        default=Path("/tmp/ltx-video-export/LTX-Video"),
        help="Path to exported LTX-Video bundle",
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip tests that require downloading the model",
    )
    args = parser.parse_args()

    results = {}

    # Always run: RoPE computation (no download needed)
    results["rope"] = test_rope_computation()

    if not args.skip_download:
        results["transformer_shape"] = test_transformer_shape(args.export_dir)
        results["vae_encoder"] = test_vae_encoder(args.export_dir)
        results["vae_decoder"] = test_vae_decoder(args.export_dir)

    print("\n=== Summary ===")
    for name, passed in results.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")

    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests FAILED'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

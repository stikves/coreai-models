#!/usr/bin/env python3
"""LTX Video step-by-step parity test.

Three-column comparison:
  A   = Full Python (diffusers) pipeline — ground truth
  A+B = Swift pipeline fed with Python's intermediate outputs at each step
  B   = Full Swift pipeline end-to-end

Dumps intermediate tensors at each stage and reports per-step divergence.

Usage:
    # Generate reference traces (Column A):
    uv run python python/tests/test_ltx_video_step_parity.py --generate-reference \
        --output-dir /tmp/ltx-parity --num-frames 25 --steps 5

    # Then run Swift with --parity-trace to dump B column intermediates,
    # and also with --feed-reference for A+B column.
    # (Swift integration TBD — for now this script does A and prepares inputs for A+B)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Stages in order
STAGES = [
    "01_tokenize",
    "02_text_encode",
    "03_latent_dims",
    "04_noise",
    "05_rope",
    "06_scheduler_sigmas",
    "07_denoise_step",  # repeated per step
    "08_unpack_latents",
    "09_vae_decode",
    "10_pixels",
]


def generate_reference(args):
    """Column A: run full Python pipeline, dump every intermediate."""
    from diffusers import LTXPipeline

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading LTX Video pipeline...")

    pipe = LTXPipeline.from_pretrained(
        "Lightricks/LTX-Video", torch_dtype=torch.float32
    )
    pipe.to(device)

    num_frames = args.num_frames
    height = args.height
    width = args.width
    steps = args.steps
    seed = args.seed
    prompt = args.prompt
    guidance_scale = args.guidance_scale

    print(f"Config: {num_frames} frames, {width}x{height}, {steps} steps, seed={seed}")
    print(f"Prompt: {prompt}")

    # === Stage 1: Tokenize ===
    print("\n[01] Tokenizing...")
    tokenizer = pipe.tokenizer
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=128,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    save_tensor(output_dir / "01_input_ids.npy", input_ids.cpu())
    save_tensor(output_dir / "01_attention_mask.npy", attention_mask.cpu())
    print(f"  input_ids shape: {input_ids.shape}, first 10: {input_ids[0,:10].tolist()}")

    # === Stage 2: Text Encode ===
    print("\n[02] Encoding text...")
    with torch.no_grad():
        text_output = pipe.text_encoder(input_ids, attention_mask=attention_mask)
        text_embeddings = text_output[0]  # last_hidden_state

    save_tensor(output_dir / "02_text_embeddings.npy", text_embeddings.cpu())
    print(f"  text_embeddings shape: {text_embeddings.shape}")
    print(f"  stats: mean={text_embeddings.mean():.6f}, std={text_embeddings.std():.6f}")

    # === Stage 3: Latent Dimensions ===
    print("\n[03] Computing latent dimensions...")
    vae_temporal_compression = 8
    vae_spatial_compression = 32
    latent_frames = (num_frames - 1) // vae_temporal_compression + 1
    latent_h = height // vae_spatial_compression
    latent_w = width // vae_spatial_compression
    seq_len = latent_frames * latent_h * latent_w
    latent_channels = 128

    dims = {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "latent_frames": latent_frames,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "seq_len": seq_len,
        "latent_channels": latent_channels,
    }
    with open(output_dir / "03_latent_dims.json", "w") as f:
        json.dump(dims, f, indent=2)
    print(f"  latent: {latent_frames}f x {latent_h}h x {latent_w}w = {seq_len} seq_len")

    # === Stage 4: Noise Generation ===
    print("\n[04] Generating noise...")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # LTX uses packed format [1, seq_len, channels]
    noise = torch.randn(1, seq_len, latent_channels, generator=generator, device="cpu").to(device)

    save_tensor(output_dir / "04_noise.npy", noise.cpu())
    print(f"  noise shape: {noise.shape}")
    print(f"  stats: mean={noise.mean():.6f}, std={noise.std():.6f}")

    # === Stage 5: RoPE ===
    print("\n[05] Computing RoPE...")
    rope_module = pipe.transformer.rope
    # The actual pipeline passes rope_interpolation_scale = (temporal/fps, spatial, spatial)
    vae_temporal = 8
    vae_spatial = 32
    frame_rate = fps if 'fps' in dir() else 24
    rope_interpolation_scale = (
        vae_temporal / frame_rate,
        vae_spatial,
        vae_spatial,
    )
    with torch.no_grad():
        rope_cos, rope_sin = rope_module(
            noise, num_frames=latent_frames, height=latent_h, width=latent_w,
            rope_interpolation_scale=rope_interpolation_scale
        )

    save_tensor(output_dir / "05_rope_cos.npy", rope_cos.cpu())
    save_tensor(output_dir / "05_rope_sin.npy", rope_sin.cpu())
    print(f"  rope_cos shape: {rope_cos.shape}")
    print(f"  rope_cos[0,:5]: {rope_cos[0,:5].tolist()[:5]}")

    # === Stage 6: Scheduler Setup ===
    print("\n[06] Setting up scheduler...")
    # LTX uses dynamic shifting — compute mu from resolution
    num_latent_pixels = seq_len * latent_channels
    mu = pipe.scheduler.config.get("base_shift", 0.5) + (
        pipe.scheduler.config.get("max_shift", 1.15) - pipe.scheduler.config.get("base_shift", 0.5)
    ) * (num_latent_pixels / (512 * 512))  # normalized by default resolution
    mu = min(mu, pipe.scheduler.config.get("max_shift", 1.15))
    pipe.scheduler.set_timesteps(steps, device=device, mu=mu)
    timesteps = pipe.scheduler.timesteps
    sigmas = pipe.scheduler.sigmas

    save_tensor(output_dir / "06_timesteps.npy", timesteps.cpu())
    save_tensor(output_dir / "06_sigmas.npy", sigmas.cpu())
    print(f"  timesteps: {timesteps.tolist()}")
    print(f"  sigmas: {sigmas.tolist()}")

    # === Stage 7: Denoising Loop ===
    print("\n[07] Denoising...")
    latents = noise * sigmas[0]

    for i, t in enumerate(timesteps):
        print(f"  Step {i+1}/{steps}: t={t.item():.4f}")

        # Save pre-step latents
        save_tensor(output_dir / f"07_step{i:02d}_input_latents.npy", latents.cpu())

        with torch.no_grad():
            model_output = pipe.transformer(
                hidden_states=latents,
                encoder_hidden_states=text_embeddings,
                timestep=t.unsqueeze(0),
                encoder_attention_mask=attention_mask,
                num_frames=latent_frames,
                height=latent_h,
                width=latent_w,
                rope_interpolation_scale=rope_interpolation_scale,
                return_dict=False,
            )[0]

        save_tensor(output_dir / f"07_step{i:02d}_model_output.npy", model_output.cpu())
        print(f"    output stats: mean={model_output.mean():.6f}, std={model_output.std():.6f}")

        # Scheduler step
        latents = pipe.scheduler.step(model_output, t, latents, return_dict=False)[0]

        save_tensor(output_dir / f"07_step{i:02d}_output_latents.npy", latents.cpu())

    # === Stage 8: Unpack Latents ===
    print("\n[08] Unpacking latents...")
    # [1, seq_len, 128] -> [1, 128, T, H, W]
    unpacked = latents.reshape(1, latent_frames, latent_h, latent_w, latent_channels)
    unpacked = unpacked.permute(0, 4, 1, 2, 3).contiguous()

    save_tensor(output_dir / "08_unpacked_latents.npy", unpacked.cpu())
    print(f"  unpacked shape: {unpacked.shape}")
    print(f"  stats: mean={unpacked.mean():.6f}, std={unpacked.std():.6f}")

    # === Stage 9: VAE Decode ===
    print("\n[09] VAE decoding...")
    with torch.no_grad():
        decoded = pipe.vae.decode(unpacked, return_dict=False)[0]

    save_tensor(output_dir / "09_decoded_video.npy", decoded.cpu())
    print(f"  decoded shape: {decoded.shape}")
    print(f"  pixel stats: mean={decoded.mean():.6f}, std={decoded.std():.6f}")

    # === Stage 10: Post-processing ===
    print("\n[10] Post-processing...")
    # Clamp to [0, 1] and convert to uint8
    pixels = decoded.clamp(0, 1)
    pixels_uint8 = (pixels * 255).to(torch.uint8)

    save_tensor(output_dir / "10_pixels_float.npy", pixels.cpu())
    save_tensor(output_dir / "10_pixels_uint8.npy", pixels_uint8.cpu())
    print(f"  pixel range: [{pixels.min():.4f}, {pixels.max():.4f}]")
    print(f"  output video: {pixels_uint8.shape}")

    # Save config for Swift comparison
    config = {
        "prompt": prompt,
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "steps": steps,
        "seed": seed,
        "guidance_scale": guidance_scale,
        "latent_frames": latent_frames,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "seq_len": seq_len,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n=== Reference generated: {output_dir} ===")
    print(f"Files: {sorted(p.name for p in output_dir.glob('*.npy'))}")


def compare_references(args):
    """Compare Column A vs Column B (or A+B) dumps."""
    ref_dir = Path(args.reference_dir)
    test_dir = Path(args.test_dir)

    print(f"Comparing: {ref_dir} (reference) vs {test_dir} (test)")
    print()

    results = []
    for ref_file in sorted(ref_dir.glob("*.npy")):
        test_file = test_dir / ref_file.name
        if not test_file.exists():
            print(f"  SKIP {ref_file.name}: not found in test dir")
            continue

        ref = np.load(ref_file)
        test = np.load(test_file)

        if ref.shape != test.shape:
            print(f"  FAIL {ref_file.name}: shape mismatch {ref.shape} vs {test.shape}")
            results.append((ref_file.name, False, "shape mismatch"))
            continue

        max_diff = np.abs(ref.astype(np.float64) - test.astype(np.float64)).max()
        cos_sim = cosine_similarity(ref.flatten(), test.flatten())
        mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)

        passed = cos_sim > 0.99 or max_diff < 0.01
        status = "PASS" if passed else "FAIL"
        print(f"  {status} {ref_file.name}: max_diff={max_diff:.6f}, cos_sim={cos_sim:.6f}, mse={mse:.8f}")
        results.append((ref_file.name, passed, f"max={max_diff:.6f} cos={cos_sim:.6f}"))

    print(f"\n=== {sum(1 for _,p,_ in results if p)}/{len(results)} passed ===")


def cosine_similarity(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def save_tensor(path, tensor):
    if isinstance(tensor, torch.Tensor):
        np.save(path, tensor.detach().numpy())
    else:
        np.save(path, tensor)


def main():
    parser = argparse.ArgumentParser(description="LTX Video step-by-step parity test")
    sub = parser.add_subparsers(dest="command")

    # Generate reference (Column A)
    gen = sub.add_parser("generate", help="Generate Python reference traces")
    gen.add_argument("--output-dir", type=str, default="/tmp/ltx-parity")
    gen.add_argument("--num-frames", type=int, default=25)
    gen.add_argument("--height", type=int, default=320)
    gen.add_argument("--width", type=int, default=512)
    gen.add_argument("--steps", type=int, default=5)
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--prompt", type=str, default="A cat walking in a garden")
    gen.add_argument("--guidance-scale", type=float, default=3.0)

    # Compare (A vs B or A vs A+B)
    cmp = sub.add_parser("compare", help="Compare reference vs test dumps")
    cmp.add_argument("--reference-dir", type=str, required=True)
    cmp.add_argument("--test-dir", type=str, required=True)

    args = parser.parse_args()

    if args.command == "generate":
        generate_reference(args)
    elif args.command == "compare":
        compare_references(args)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

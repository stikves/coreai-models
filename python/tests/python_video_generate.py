#!/usr/bin/env python3
"""Generate a reference video using the Python diffusers LTX Video pipeline.

Usage:
    uv run python python/tests/python_video_generate.py
    uv run python python/tests/python_video_generate.py --prompt "A dog running" --steps 30
    uv run python python/tests/python_video_generate.py --device cpu --output /tmp/ref.mp4
"""

import argparse
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="Generate video with LTX Video (Python/diffusers)")
    parser.add_argument("--prompt", type=str, default="A cat playing piano")
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cpu", "cuda"])
    parser.add_argument("--output", type=str, default="/tmp/ltx_python_reference.mp4")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float16", "bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    model_dtype = dtype_map[args.dtype]

    print(f"Loading LTX Video pipeline (dtype={args.dtype})...")
    from diffusers import LTXPipeline

    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=model_dtype)
    pipe.to(args.device)

    print(f"Config:")
    print(f"  Prompt:     {args.prompt}")
    print(f"  Frames:     {args.num_frames}")
    print(f"  Resolution: {args.width}x{args.height}")
    print(f"  Steps:      {args.steps}")
    print(f"  Seed:       {args.seed}")
    print(f"  Guidance:   {args.guidance_scale}")
    print(f"  Device:     {args.device}")
    print(f"  Output:     {args.output}")
    print()

    gen = torch.Generator("cpu").manual_seed(args.seed)

    print("Generating...")
    start = time.time()
    result = pipe(
        args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        generator=gen,
    )
    elapsed = time.time() - start

    video = result.frames[0]

    print(f"Generated {len(video)} frames in {elapsed:.1f}s")

    # Save as MP4 (needs opencv) or fallback to PNG frames
    if args.output.endswith(".mp4"):
        try:
            from diffusers.utils import export_to_video
            export_to_video(video, args.output, fps=24)
            print(f"Saved MP4 to {args.output}")
        except ImportError:
            from pathlib import Path
            out_dir = Path(args.output).with_suffix("")
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, frame in enumerate(video):
                frame.save(out_dir / f"frame_{i:04d}.png")
            print(f"opencv not found — saved {len(video)} PNGs to {out_dir}/")
    else:
        from pathlib import Path
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(video):
            frame.save(out_dir / f"frame_{i:04d}.png")
        print(f"Saved {len(video)} PNGs to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())

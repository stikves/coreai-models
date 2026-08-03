#!/usr/bin/env python3
"""Exhaustive trace of LTX Video pipeline internals.

Traces EVERY intermediate computation in the LTX Video pipeline at the finest
granularity, dumping exact numerical values needed to replicate in Swift.

Key questions answered:
  - Does the scheduler scale latents before transformer? (NO)
  - Does the scheduler scale model output? (NO)
  - What is the EXACT Euler step formula? (prev_sample = sample + dt * output, dt = sigma_next - sigma)
  - Is there shift_terminal clamping? (YES: 0.1)
  - Are sigmas from set_timesteps or computed differently? (Pipeline passes explicit sigmas)
  - Is there latent denormalization before VAE? (YES: latents * std / scaling_factor + mean)
  - What is mu and how is it computed? (Linear interpolation from base_shift/max_shift by seq_len)

Usage:
    cd /Users/sukru/dev/coreai-models-apple
    uv run python python/tests/test_ltx_exhaustive_trace.py

Output: /tmp/ltx-trace.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch


def trace_scheduler_internals(pipe, num_inference_steps, video_sequence_length, device):
    """Trace every scheduler computation step-by-step."""
    trace = {}

    scheduler = pipe.scheduler
    config = scheduler.config

    # 1. Scheduler config
    trace["scheduler_config"] = {
        "num_train_timesteps": config.num_train_timesteps,
        "shift": config.shift,
        "use_dynamic_shifting": config.use_dynamic_shifting,
        "base_image_seq_len": config.get("base_image_seq_len", 256),
        "max_image_seq_len": config.get("max_image_seq_len", 4096),
        "base_shift": config.get("base_shift", 0.5),
        "max_shift": config.get("max_shift", 1.15),
        "shift_terminal": config.get("shift_terminal", None),
        "invert_sigmas": config.invert_sigmas,
        "time_shift_type": config.get("time_shift_type", "exponential"),
    }

    # 2. mu computation (as done in pipeline_ltx.py line 722-728)
    base_seq_len = config.get("base_image_seq_len", 256)
    max_seq_len = config.get("max_image_seq_len", 4096)
    base_shift = config.get("base_shift", 0.5)
    max_shift = config.get("max_shift", 1.15)

    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = video_sequence_length * m + b

    trace["mu_computation"] = {
        "formula": "mu = video_seq_len * m + b, where m = (max_shift - base_shift) / (max_seq_len - base_seq_len), b = base_shift - m * base_seq_len",
        "video_sequence_length": int(video_sequence_length),
        "base_seq_len": int(base_seq_len),
        "max_seq_len": int(max_seq_len),
        "base_shift": float(base_shift),
        "max_shift": float(max_shift),
        "m": float(m),
        "b": float(b),
        "mu": float(mu),
    }

    # 3. Pre-shift sigmas (this is what the pipeline passes to set_timesteps)
    pre_shift_sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    trace["pre_shift_sigmas"] = {
        "formula": "np.linspace(1.0, 1/num_inference_steps, num_inference_steps)",
        "values": pre_shift_sigmas.tolist(),
    }

    # 4. After dynamic shift (exponential time shift)
    # Formula: sigma' = exp(mu) / (exp(mu) + (1/sigma - 1)^1.0)
    exp_mu = np.exp(mu)
    post_shift_sigmas = exp_mu / (exp_mu + (1.0 / pre_shift_sigmas - 1.0))
    trace["post_dynamic_shift_sigmas"] = {
        "formula": "exp(mu) / (exp(mu) + (1/sigma - 1))",
        "exp_mu": float(exp_mu),
        "values": post_shift_sigmas.tolist(),
    }

    # 5. After stretch_shift_to_terminal
    shift_terminal = config.get("shift_terminal", None)
    if shift_terminal:
        one_minus_z = 1.0 - post_shift_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - shift_terminal)
        stretched = 1.0 - (one_minus_z / scale_factor)

        trace["stretch_shift_to_terminal"] = {
            "formula": "stretched = 1 - ((1 - sigma) / ((1 - sigma[-1]) / (1 - shift_terminal)))",
            "shift_terminal": float(shift_terminal),
            "one_minus_z": one_minus_z.tolist(),
            "scale_factor": float(scale_factor),
            "stretched_sigmas": stretched.tolist(),
        }
        final_sigmas_before_append = stretched
    else:
        trace["stretch_shift_to_terminal"] = {"applied": False}
        final_sigmas_before_append = post_shift_sigmas

    # 6. Final sigmas (with terminal 0 appended)
    final_sigmas = np.append(final_sigmas_before_append, 0.0)
    trace["final_sigmas_with_terminal_zero"] = final_sigmas.tolist()

    # 7. Timesteps (sigmas * num_train_timesteps)
    timesteps = final_sigmas_before_append * config.num_train_timesteps
    trace["timesteps"] = timesteps.tolist()

    # 8. Now actually call set_timesteps to get the real scheduler values
    sigmas_input = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps)
    scheduler.set_timesteps(num_inference_steps, device=device, sigmas=sigmas_input.tolist(), mu=mu)

    actual_sigmas = scheduler.sigmas.cpu().numpy()
    actual_timesteps = scheduler.timesteps.cpu().numpy()

    trace["actual_scheduler_sigmas"] = actual_sigmas.tolist()
    trace["actual_scheduler_timesteps"] = actual_timesteps.tolist()

    # Verify our manual computation matches
    trace["manual_vs_actual_max_diff"] = float(np.max(np.abs(final_sigmas - actual_sigmas)))

    # 9. Per-step dt values
    dt_values = []
    for i in range(num_inference_steps):
        sigma_i = actual_sigmas[i]
        sigma_next = actual_sigmas[i + 1]
        dt = sigma_next - sigma_i
        dt_values.append({
            "step": i,
            "sigma": float(sigma_i),
            "sigma_next": float(sigma_next),
            "dt": float(dt),
            "dt_is_negative": dt < 0,
        })
    trace["per_step_dt"] = dt_values

    return trace, mu


def trace_full_pipeline(args):
    """Run the full pipeline with exhaustive tracing."""
    from diffusers import LTXPipeline

    trace = {}
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    trace["device"] = device

    print(f"Loading LTX Video pipeline on {device}...")
    pipe = LTXPipeline.from_pretrained("Lightricks/LTX-Video", torch_dtype=torch.float32)
    pipe.to(device)

    # Configuration
    num_frames = 25
    height = 320
    width = 512
    steps = 5
    seed = 42
    prompt = "A cat walking in a garden"

    trace["config"] = {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "steps": steps,
        "seed": seed,
        "prompt": prompt,
    }

    # Latent dimensions
    vae_temporal = 8
    vae_spatial = 32
    latent_frames = (num_frames - 1) // vae_temporal + 1
    latent_h = height // vae_spatial
    latent_w = width // vae_spatial
    video_seq_len = latent_frames * latent_h * latent_w
    latent_channels = 128

    trace["latent_dims"] = {
        "vae_temporal_compression": vae_temporal,
        "vae_spatial_compression": vae_spatial,
        "latent_frames": latent_frames,
        "latent_h": latent_h,
        "latent_w": latent_w,
        "video_seq_len": video_seq_len,
        "latent_channels": latent_channels,
    }

    # ==== SCHEDULER TRACE ====
    print("\n=== Tracing scheduler internals ===")
    scheduler_trace, mu = trace_scheduler_internals(pipe, steps, video_seq_len, device)
    trace["scheduler"] = scheduler_trace

    # ==== NOISE GENERATION ====
    print("\n=== Tracing noise generation ===")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    # Pipeline generates [B, C, F, H, W] then packs; since patch_size=1, packing is just reshape
    noise_5d = torch.randn(1, latent_channels, latent_frames, latent_h, latent_w,
                           generator=generator, device="cpu")
    # Pack: [1, C, F, H, W] -> [1, F*H*W, C] (for patch_size=1, just reshape)
    noise_packed = noise_5d.permute(0, 2, 3, 4, 1).reshape(1, video_seq_len, latent_channels)
    noise_packed = noise_packed.to(device)

    trace["noise"] = {
        "generation": "torch.randn([1, 128, 4, 10, 16], seed=42) then pack to [1, 640, 128]",
        "initial_latent_scaling": "NONE - raw noise used as initial latents (no sigma multiplication)",
        "note": "In flow-matching at sigma=1.0, x_1 = pure noise. No scaling needed.",
        "packed_shape": list(noise_packed.shape),
        "first_5_values": noise_packed[0, 0, :5].cpu().tolist(),
        "last_5_values": noise_packed[0, -1, -5:].cpu().tolist(),
        "mean": float(noise_packed.mean()),
        "std": float(noise_packed.std()),
    }

    # ==== TEXT ENCODING ====
    print("\n=== Tracing text encoding ===")
    tokenizer = pipe.tokenizer
    text_inputs = tokenizer(prompt, padding="max_length", max_length=128, truncation=True, return_tensors="pt")
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    with torch.no_grad():
        text_output = pipe.text_encoder(input_ids, attention_mask=attention_mask)
        text_embeddings = text_output[0]

    trace["text_encoding"] = {
        "input_ids_first_10": input_ids[0, :10].cpu().tolist(),
        "attention_mask_sum": int(attention_mask.sum()),
        "text_embeddings_shape": list(text_embeddings.shape),
        "text_embeddings_mean": float(text_embeddings.mean()),
        "text_embeddings_std": float(text_embeddings.std()),
    }

    # ==== DENOISING LOOP ====
    print("\n=== Tracing denoising loop ===")

    # Reset scheduler
    sigmas_input = np.linspace(1.0, 1.0 / steps, steps)
    pipe.scheduler.set_timesteps(steps, device=device, sigmas=sigmas_input.tolist(), mu=mu)
    sigmas = pipe.scheduler.sigmas
    timesteps = pipe.scheduler.timesteps

    # Initial latents = pure noise (no sigma scaling!)
    latents = noise_packed.clone()

    # RoPE
    frame_rate = 24
    rope_interpolation_scale = (vae_temporal / frame_rate, vae_spatial, vae_spatial)

    denoising_trace = []

    for step_idx, t in enumerate(timesteps):
        step_trace = {"step": step_idx}

        # What goes INTO the transformer
        sigma_current = sigmas[step_idx].item()
        sigma_next = sigmas[step_idx + 1].item()
        dt = sigma_next - sigma_current

        step_trace["sigma_current"] = sigma_current
        step_trace["sigma_next"] = sigma_next
        step_trace["dt"] = dt
        step_trace["timestep_value"] = t.item()

        # Input latents
        step_trace["input_latents"] = {
            "shape": list(latents.shape),
            "first_5": latents[0, 0, :5].cpu().tolist(),
            "last_5": latents[0, -1, -5:].cpu().tolist(),
            "mean": float(latents.mean()),
            "std": float(latents.std()),
            "min": float(latents.min()),
            "max": float(latents.max()),
        }

        # KEY: Is input scaled by sigma? NO!
        step_trace["input_scaling"] = {
            "scaled_by_sigma": False,
            "formula": "latents passed directly to transformer (no division by sigma)",
            "note": "Flow-matching: model predicts velocity v, not noise epsilon",
        }

        # Timestep format
        step_trace["timestep_format"] = {
            "value": t.item(),
            "is_sigma_times_1000": True,
            "formula": "timestep = sigma * num_train_timesteps (= sigma * 1000)",
            "corresponding_sigma": sigma_current,
        }

        # Attention mask format
        step_trace["attention_mask_format"] = {
            "dtype": str(attention_mask.dtype),
            "values": "0 and 1 (int64)",
            "note": "Binary mask: 1 for real tokens, 0 for padding",
        }

        # Run transformer
        print(f"  Step {step_idx}: t={t.item():.4f}, sigma={sigma_current:.6f}")
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

        step_trace["model_output"] = {
            "shape": list(model_output.shape),
            "first_5": model_output[0, 0, :5].cpu().tolist(),
            "last_5": model_output[0, -1, -5:].cpu().tolist(),
            "mean": float(model_output.mean()),
            "std": float(model_output.std()),
        }

        # KEY: Is output scaled? NO!
        step_trace["output_scaling"] = {
            "scaled_by_sigma": False,
            "formula": "model output used directly in Euler step (no multiplication by sigma)",
        }

        # ==== THE EULER STEP ====
        # Manually compute to show exact formula
        sample_f32 = latents.to(torch.float32)
        prev_sample_manual = sample_f32 + dt * model_output.to(torch.float32)

        # Also get scheduler result for verification
        latents_new = pipe.scheduler.step(model_output, t, latents, return_dict=False)[0]

        manual_vs_scheduler_diff = float((prev_sample_manual - latents_new.to(torch.float32)).abs().max())

        step_trace["euler_step"] = {
            "formula": "prev_sample = sample + dt * model_output",
            "expanded": f"prev_sample = sample + ({sigma_next:.8f} - {sigma_current:.8f}) * model_output",
            "dt_value": dt,
            "dt_sign": "NEGATIVE (sigma decreases each step)",
            "note": "dt is negative because sigma goes from ~1.0 toward 0.0",
            "manual_vs_scheduler_max_diff": manual_vs_scheduler_diff,
            "verification": "PASS" if manual_vs_scheduler_diff < 1e-5 else "FAIL",
        }

        # Numerical example for step 0
        if step_idx == 0:
            s_val = sample_f32[0, 0, 0].item()
            o_val = model_output[0, 0, 0].item()
            result = s_val + dt * o_val
            step_trace["euler_step"]["numerical_example_element_0"] = {
                "sample[0,0,0]": s_val,
                "output[0,0,0]": o_val,
                "dt": dt,
                "result": result,
                "formula_with_numbers": f"{s_val:.8f} + {dt:.8f} * {o_val:.8f} = {result:.8f}",
            }

        step_trace["output_latents"] = {
            "first_5": latents_new[0, 0, :5].cpu().tolist(),
            "last_5": latents_new[0, -1, -5:].cpu().tolist(),
            "mean": float(latents_new.mean()),
            "std": float(latents_new.std()),
        }

        denoising_trace.append(step_trace)
        latents = latents_new

    trace["denoising_loop"] = denoising_trace

    # ==== POST-PROCESSING: UNPACK + DENORMALIZE ====
    print("\n=== Tracing post-processing ===")

    # Unpack (patch_size=1, so just reshape)
    unpacked = latents.reshape(1, latent_frames, latent_h, latent_w, latent_channels)
    unpacked = unpacked.permute(0, 4, 1, 2, 3).contiguous()  # [1, C, F, H, W]

    trace["unpack"] = {
        "formula": "reshape [1, T*H*W, C] -> [1, T, H, W, C] then permute to [1, C, T, H, W]",
        "patch_size": 1,
        "patch_size_t": 1,
        "note": "With patch_size=1, packing/unpacking is just a transpose",
        "unpacked_shape": list(unpacked.shape),
        "first_5_channel_0": unpacked[0, 0, 0, 0, :5].cpu().tolist(),
    }

    # Denormalization
    latents_mean = pipe.vae.latents_mean
    latents_std = pipe.vae.latents_std
    scaling_factor = pipe.vae.config.scaling_factor

    trace["denormalization"] = {
        "formula": "latents = latents * latents_std / scaling_factor + latents_mean",
        "scaling_factor": float(scaling_factor),
        "latents_mean_first_5": latents_mean[:5].cpu().tolist(),
        "latents_mean_last_5": latents_mean[-5:].cpu().tolist(),
        "latents_std_first_5": latents_std[:5].cpu().tolist(),
        "latents_std_last_5": latents_std[-5:].cpu().tolist(),
        "latents_mean_shape": list(latents_mean.shape),
        "latents_std_shape": list(latents_std.shape),
        "note": "Applied per-channel: mean/std are [128] broadcast over [1, 128, T, H, W]",
        "CRITICAL": "Swift pipeline is MISSING this step!",
    }

    # Apply denormalization
    latents_mean_5d = latents_mean.view(1, -1, 1, 1, 1).to(unpacked.device, unpacked.dtype)
    latents_std_5d = latents_std.view(1, -1, 1, 1, 1).to(unpacked.device, unpacked.dtype)
    denormed = unpacked * latents_std_5d / scaling_factor + latents_mean_5d

    trace["denormalization"]["before_denorm_stats"] = {
        "mean": float(unpacked.mean()),
        "std": float(unpacked.std()),
        "min": float(unpacked.min()),
        "max": float(unpacked.max()),
    }
    trace["denormalization"]["after_denorm_stats"] = {
        "mean": float(denormed.mean()),
        "std": float(denormed.std()),
        "min": float(denormed.min()),
        "max": float(denormed.max()),
    }

    # VAE decode
    print("\n=== Tracing VAE decode ===")
    with torch.no_grad():
        # LTX VAE has no timestep conditioning for this model
        decoded = pipe.vae.decode(denormed, return_dict=False)[0]

    trace["vae_decode"] = {
        "input_shape": list(denormed.shape),
        "output_shape": list(decoded.shape),
        "timestep_conditioning": False,
        "pixel_range_before_clamp": {
            "min": float(decoded.min()),
            "max": float(decoded.max()),
            "mean": float(decoded.mean()),
        },
        "note": "Output is already in [0, 1] range (approximately)",
    }

    # Final pixel processing
    pixels_clamped = decoded.clamp(0, 1)
    trace["final_output"] = {
        "clamp_range": [0.0, 1.0],
        "to_uint8": "pixels * 255",
        "output_shape": list(pixels_clamped.shape),
    }

    # ==== SUMMARY OF CRITICAL FINDINGS ====
    trace["CRITICAL_FINDINGS"] = {
        "1_scheduler_sigma_source": {
            "python": "Pipeline passes sigmas=np.linspace(1.0, 1/steps, steps) to set_timesteps",
            "swift_current": "DiscreteFlowScheduler uses linspace(1.0, 1/stepCount, stepCount) -- MATCHES",
        },
        "2_mu_calculation": {
            "python": f"mu = seq_len * m + b = {video_seq_len} * {m:.10f} + {b:.10f} = {mu:.10f}",
            "python_params": f"base_shift={base_shift}, max_shift={max_shift}, base_seq_len={base_seq_len}, max_seq_len={max_seq_len}",
            "swift_current": "mu = min(1.15, 0.5 + (1.15 - 0.5) * numLatentPixels / (512*512))",
            "swift_uses_wrong_params": True,
            "BUG": "Swift uses hardcoded base_shift=0.5, max_shift=1.15 and wrong formula (should use model config: base_shift=0.95, max_shift=2.05, linear interp by seq_len)",
        },
        "3_shift_terminal": {
            "python": f"After dynamic shift, applies stretch_shift_to_terminal with shift_terminal={shift_terminal}",
            "swift_current": "NOT IMPLEMENTED",
            "BUG": "Swift scheduler is MISSING stretch_shift_to_terminal!",
        },
        "4_initial_latent_scaling": {
            "value": "NO SCALING",
            "note": "Initial latents = pure noise. The parity test (line 174) has 'latents = noise * sigmas[0]' which is WRONG vs the actual pipeline.",
        },
        "5_euler_step_formula": {
            "formula": "prev_sample = sample + (sigma_next - sigma_current) * model_output",
            "dt_sign": "NEGATIVE (sigma decreases)",
            "equivalent": "prev_sample = sample - |dt| * model_output (since dt < 0)",
            "swift_formula": "sample + d * dt where d = output (after simplification) -- MATCHES",
        },
        "6_denormalization_before_vae": {
            "formula": "latents = latents * latents_std / scaling_factor + latents_mean",
            "scaling_factor": float(scaling_factor),
            "applied": "After unpack, before VAE decode",
            "BUG": "Swift pipeline SKIPS this step!",
        },
        "7_transformer_input": {
            "latent_scaling": "NONE (latents passed directly)",
            "timestep_format": "sigma * 1000 (float)",
            "attention_mask": "int64, values 0 and 1",
        },
    }

    # ==== SIGMA COMPARISON TABLE ====
    # Show exactly what Swift SHOULD compute vs what it currently computes
    swift_sigmas_no_stretch = post_shift_sigmas = (np.exp(mu) / (np.exp(mu) + (1.0 / sigmas_input - 1.0)))
    trace["sigma_comparison"] = {
        "note": "Compare what Python produces vs what Swift currently produces",
        "python_final_sigmas": scheduler_trace["actual_scheduler_sigmas"],
        "swift_without_stretch_terminal": np.append(swift_sigmas_no_stretch, 0.0).tolist(),
        "difference_without_stretch": (
            np.array(scheduler_trace["actual_scheduler_sigmas"]) -
            np.append(swift_sigmas_no_stretch, 0.0)
        ).tolist(),
    }

    # Also trace what the WRONG mu gives
    wrong_base_shift = 0.5
    wrong_max_shift = 1.15
    latent_channels_count = latent_channels
    num_latent_pixels = video_seq_len * latent_channels_count
    wrong_mu = min(wrong_max_shift, wrong_base_shift + (wrong_max_shift - wrong_base_shift) * num_latent_pixels / (512 * 512))
    wrong_exp_mu = np.exp(wrong_mu)
    wrong_shifted = wrong_exp_mu / (wrong_exp_mu + (1.0 / sigmas_input - 1.0))

    trace["sigma_comparison"]["swift_wrong_mu_value"] = float(wrong_mu)
    trace["sigma_comparison"]["swift_wrong_mu_sigmas"] = np.append(wrong_shifted, 0.0).tolist()
    trace["sigma_comparison"]["correct_mu_value"] = float(mu)

    return trace


def main():
    trace = trace_full_pipeline(None)

    output_path = Path("/tmp/ltx-trace.json")
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Trace written to: {output_path}")
    print(f"{'='*60}")

    # Print critical findings summary
    print("\n=== CRITICAL BUGS IN SWIFT PIPELINE ===\n")
    findings = trace["CRITICAL_FINDINGS"]

    print("1. MU CALCULATION:")
    print(f"   Python: {findings['2_mu_calculation']['python']}")
    print(f"   Swift:  {findings['2_mu_calculation']['swift_current']}")
    print(f"   FIX: Use model config (base_shift=0.95, max_shift=2.05) and linear interp by seq_len\n")

    print("2. SHIFT_TERMINAL (missing in Swift):")
    print(f"   Python: {findings['3_shift_terminal']['python']}")
    print(f"   Swift:  {findings['3_shift_terminal']['swift_current']}")
    print(f"   FIX: After dynamic shift, apply stretch formula\n")

    print("3. LATENT DENORMALIZATION (missing in Swift):")
    print(f"   Python: {findings['6_denormalization_before_vae']['formula']}")
    print(f"   FIX: Apply per-channel denorm using vae.latents_mean/std before VAE decode\n")

    print("=== SIGMA SCHEDULE ===")
    sched = trace["scheduler"]
    print(f"   Pre-shift:    {sched['pre_shift_sigmas']['values']}")
    print(f"   Post-shift:   {sched['post_dynamic_shift_sigmas']['values']}")
    if "stretch_shift_to_terminal" in sched and "stretched_sigmas" in sched["stretch_shift_to_terminal"]:
        print(f"   Post-stretch: {sched['stretch_shift_to_terminal']['stretched_sigmas']}")
    print(f"   Final+0:      {sched['actual_scheduler_sigmas']}")

    print(f"\n=== EULER STEP (step 0) ===")
    step0 = trace["denoising_loop"][0]
    euler = step0["euler_step"]
    print(f"   Formula: {euler['formula']}")
    print(f"   {euler['expanded']}")
    if "numerical_example_element_0" in euler:
        ex = euler["numerical_example_element_0"]
        print(f"   Example: {ex['formula_with_numbers']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

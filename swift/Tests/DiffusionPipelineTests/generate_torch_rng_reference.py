#!/usr/bin/env python3
"""Generate reference values for TorchRandomSource parity tests.

Produces exact values from PyTorch's CPU generator for multiple seeds and sizes.
PyTorch uses MT19937 + Box-Muller, with a batch-16 optimization for arrays >= 16.

Usage:
    uv run --with torch generate_torch_rng_reference.py

The output is ready to paste into Swift test assertions.
"""

import torch


def reference_scalar(seed: int, count: int) -> list[float]:
    """Scalar path: generate `count` normal samples one at a time.

    This matches TorchRandomSource.nextNormal() called in a loop.
    PyTorch equivalent: [torch.randn(1).item() for _ in range(count)]
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return [torch.randn(1, generator=g, dtype=torch.float64).item() for _ in range(count)]


def reference_batch(seed: int, shape: list[int]) -> list[float]:
    """Batch path: generate a tensor of normal samples.

    This matches TorchRandomSource.normalArray() for count >= 16.
    PyTorch equivalent: torch.randn(shape).tolist()
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).flatten().tolist()


def fmt(values: list[float], per_line: int = 4) -> str:
    lines = []
    for i in range(0, len(values), per_line):
        chunk = values[i:i + per_line]
        lines.append(", ".join(f"{v: .10f}" for v in chunk))
    return ",\n    ".join(lines)


print("=" * 72)
print("TorchRandomSource Parity Reference Values")
print("=" * 72)
print()

# --- 1. Scalar path (count < 16): nextNormal() loop ---
# This is what generateNoise() currently uses
print("// MARK: - Scalar path (nextNormal loop, count < 16)")
print("// Generated with: torch.randn(1, generator=g, dtype=torch.float64) × N")
print()

for seed in [0, 42, 123]:
    vals = reference_scalar(seed, 8)
    print(f"// seed={seed}, count=8 (scalar)")
    print(f"let scalarRef_{seed}: [Double] = [")
    print(f"    {fmt(vals, 4)}")
    print(f"]")
    print()

# --- 2. Batch path (count >= 16): normalArray ---
# This is what torch.randn(shape) does internally
print("// MARK: - Batch path (normalArray, count >= 16)")
print("// Generated with: torch.randn(shape, generator=g, dtype=torch.float32)")
print()

for seed in [0, 42, 123]:
    vals = reference_batch(seed, [32])
    print(f"// seed={seed}, shape=[32] (batch)")
    print(f"let batchRef_{seed}: [Float] = [")
    print(f"    {fmt(vals, 4)}")
    print(f"]")
    print()

# --- 3. Realistic diffusion shape (64x64 latent = 4096 elements) ---
print("// MARK: - Realistic shape (first 8 + last 8 of 4096-element tensor)")
print("// Generated with: torch.randn([1, 16, 16, 16], generator=g, dtype=torch.float32)")
print()

for seed in [42]:
    vals = reference_batch(seed, [1, 16, 16, 16])
    first8 = vals[:8]
    last8 = vals[-8:]
    print(f"// seed={seed}, shape=[1,16,16,16] = 4096 elements")
    print(f"let realisticFirst8: [Float] = [")
    print(f"    {fmt(first8, 4)}")
    print(f"]")
    print(f"let realisticLast8: [Float] = [")
    print(f"    {fmt(last8, 4)}")
    print(f"]")
    print()

# --- 4. Edge case: exactly 16 elements (boundary of batch path) ---
print("// MARK: - Edge case: exactly 16 elements")
print()
for seed in [42]:
    vals = reference_batch(seed, [16])
    print(f"// seed={seed}, shape=[16]")
    print(f"let boundary16: [Float] = [")
    print(f"    {fmt(vals, 4)}")
    print(f"]")
    print()

# --- 5. Count=17 (batch path with remainder handling) ---
print("// MARK: - Edge case: 17 elements (batch path + remainder)")
print()
for seed in [42]:
    vals = reference_batch(seed, [17])
    print(f"// seed={seed}, shape=[17]")
    print(f"let remainder17: [Float] = [")
    print(f"    {fmt(vals, 4)}")
    print(f"]")
    print()

# --- 6. Verify scalar vs batch diverge at count=16 ---
print("// MARK: - Scalar vs batch divergence proof")
print()
seed = 42
scalar_16 = reference_scalar(seed, 16)
batch_16 = reference_batch(seed, [16])
print(f"// seed={seed}: scalar[0]={scalar_16[0]:.10f}, batch[0]={batch_16[0]:.10f}")
print(f"// These SHOULD differ — scalar uses float64 Box-Muller per pair,")
print(f"// batch uses float32 uniforms in 16-element blocks.")
if abs(scalar_16[0] - batch_16[0]) > 1e-6:
    print("// CONFIRMED: scalar and batch paths produce different values.")
else:
    print("// WARNING: scalar and batch paths produced the same value — investigate.")
print()

# --- 7. Verify the internal uint32 stream ---
print("// MARK: - Raw MT19937 uint32 stream (first 8)")
print("// Generated with CPython's _random module (same MT19937)")
print()

# Use numpy since it exposes the raw MT state
import numpy as np
for seed in [42]:
    # NumPy's RandomState uses the same MT19937 seeding as torch
    # but the uint32 draw order is different.
    # For torch: each nextUInt32() call draws one value from MT
    # We can verify by checking the seeding algorithm is identical
    # (both use the same Knuth recurrence).
    #
    # The actual raw uint32 values from torch's MT are not easily
    # accessible from Python, so we verify end-to-end via randn output.
    pass

print("// End-to-end verification is sufficient — the raw MT stream")
print("// is the same algorithm, just verify randn output matches.")
print()
print("=" * 72)
print("Copy the reference arrays above into SchedulerTests.swift")
print("=" * 72)

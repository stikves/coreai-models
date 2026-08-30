// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

/// Which random number generator to use for noise generation.
public enum RandomSourceType: Sendable {
    case numPy
    case torch
    case nvidia
}

/// Generate Gaussian noise (mean 0, stdev 1) using the specified random source.
///
/// For `.torch`, uses `normalArray` to match PyTorch's `torch.randn` batch behavior
/// (batch-16 Box-Muller on float32 uniforms). The scalar `nextNormal()` loop produces
/// a different sequence — see `generate_torch_rng_reference.py` for proof.
public func generateNoise(count: Int, seed: UInt32, sourceType: RandomSourceType = .numPy) -> [Float] {
    switch sourceType {
    case .numPy:
        var rng = NumPyRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    case .torch:
        var rng = TorchRandomSource(seed: seed)
        return rng.normalArray([count])
    case .nvidia:
        var rng = NvRandomSource(seed: seed)
        return (0..<count).map { _ in Float(rng.nextNormal()) }
    }
}

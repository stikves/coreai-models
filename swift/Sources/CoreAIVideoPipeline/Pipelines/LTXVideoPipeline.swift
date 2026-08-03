// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAI
import CoreAIDiffusionPipeline
import CoreAIShared
import CoreGraphics
import Foundation
import Tokenizers

/// LTX Video pipeline using Core AI backend.
///
/// Orchestrates: tokenize -> T5 encode -> 3D RoPE compute -> noise ->
/// denoise loop (flow-match Euler) -> unpack -> 3D VAE decode -> frame extraction.
///
/// Key design: 3D RoPE embeddings are pre-computed in Swift and passed as model inputs
/// (not computed in-graph) to avoid graph optimizer issues. The transformer operates on
/// packed latents [1, seq_len, 128], while the VAE decoder takes unpacked 5D latents
/// [1, 128, T, H, W].
public struct LTXVideoPipeline: VideoPipeline {
    // MARK: - Components

    let transformer: CoreAIDiffusionModelFunction
    let textEncoder: CoreAIDiffusionModelFunction
    let decoder: CoreAIDiffusionModelFunction
    let tokenizer: any Tokenizer

    // MARK: - Architecture Constants (from metadata.json)

    let latentChannels: Int
    let numAttentionHeads: Int
    let attentionHeadDim: Int
    let innerDim: Int
    let captionChannels: Int
    let ropeTheta: Float
    let defaultSteps: Int
    let defaultGuidanceScale: Float

    /// Whether to unload each component after its stage completes.
    public var lazyModelLoading: Bool

    // MARK: - Video defaults

    private let configDefaultFrameCount: Int
    private let configDefaultFPS: Int

    public var defaultVideoSize: (width: Int, height: Int) { (512, 320) }
    public var defaultFrameCount: Int { configDefaultFrameCount }

    // MARK: - Spatial / Temporal Compression

    /// LTX Video spatial downsampling factor (pixel -> latent).
    private static let spatialCompression = 32
    /// LTX Video temporal downsampling factor (frames -> latent frames).
    private static let temporalCompression = 8

    /// T5 text encoder sequence length.
    private static let textSeqLen = 128

    /// Base resolution / frame counts for RoPE interpolation scaling.
    private static let baseNumFrames = 20
    private static let baseHeight = 2048
    private static let baseWidth = 2048

    // MARK: - Init

    public init(
        transformer: CoreAIDiffusionModelFunction,
        textEncoder: CoreAIDiffusionModelFunction,
        decoder: CoreAIDiffusionModelFunction,
        tokenizer: any Tokenizer,
        latentChannels: Int = 128,
        numAttentionHeads: Int = 32,
        attentionHeadDim: Int = 64,
        captionChannels: Int = 4096,
        ropeTheta: Float = 10000.0,
        defaultSteps: Int = 50,
        defaultGuidanceScale: Float = 3.0,
        defaultFrameCount: Int = 49,
        defaultFPS: Int = 24,
        lazyModelLoading: Bool = true
    ) {
        self.transformer = transformer
        self.textEncoder = textEncoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.latentChannels = latentChannels
        self.numAttentionHeads = numAttentionHeads
        self.attentionHeadDim = attentionHeadDim
        self.innerDim = numAttentionHeads * attentionHeadDim
        self.captionChannels = captionChannels
        self.ropeTheta = ropeTheta
        self.defaultSteps = defaultSteps
        self.defaultGuidanceScale = defaultGuidanceScale
        self.configDefaultFrameCount = defaultFrameCount
        self.configDefaultFPS = defaultFPS
        self.lazyModelLoading = lazyModelLoading
    }

    /// Load an LTX Video pipeline from a directory containing .aimodel files and tokenizer/.
    public init(from url: URL, lazyModelLoading: Bool = true) async throws {
        // Parse metadata.json
        let metadataURL = url.appendingPathComponent("metadata.json")
        let metadataData = try Data(contentsOf: metadataURL)
        guard let json = try JSONSerialization.jsonObject(with: metadataData) as? [String: Any],
            let diffusion = json["diffusion"] as? [String: Any]
        else {
            throw LTXVideoError.invalidMetadata("metadata.json missing 'diffusion' block")
        }

        let latentChannels = diffusion["latent_channels"] as? Int ?? 128
        let numAttentionHeads = diffusion["num_attention_heads"] as? Int ?? 32
        let attentionHeadDim = diffusion["attention_head_dim"] as? Int ?? 64
        let captionChannels = diffusion["caption_channels"] as? Int ?? 4096
        let ropeTheta = (diffusion["rope_theta"] as? NSNumber)?.floatValue ?? 10000.0
        let defaultSteps = diffusion["default_steps"] as? Int ?? 50
        let defaultGuidanceScale = (diffusion["default_guidance_scale"] as? NSNumber)?.floatValue ?? 3.0
        let defaultFrameCount = diffusion["default_num_frames"] as? Int ?? 49
        let defaultFPS = diffusion["default_fps"] as? Int ?? 24

        // Resolve model assets
        let transformerURL = Self.resolveAsset(at: url, name: "Transformer")
        let textEncoderURL = Self.resolveAsset(at: url, name: "TextEncoder")
        let decoderURL = Self.resolveAsset(at: url, name: "VAEDecoder")

        guard let transformerURL else {
            throw LTXVideoError.missingComponent("Transformer.aimodel")
        }
        guard let textEncoderURL else {
            throw LTXVideoError.missingComponent("TextEncoder.aimodel")
        }
        guard let decoderURL else {
            throw LTXVideoError.missingComponent("VAEDecoder.aimodel")
        }

        let transformer = CoreAIDiffusionModelFunction(modelURL: transformerURL)
        let textEncoder = CoreAIDiffusionModelFunction(modelURL: textEncoderURL)
        let decoder = CoreAIDiffusionModelFunction(modelURL: decoderURL)

        // Load T5 tokenizer
        let tokenizerDir = url.appendingPathComponent("tokenizer")
        let tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)

        self.init(
            transformer: transformer,
            textEncoder: textEncoder,
            decoder: decoder,
            tokenizer: tokenizer,
            latentChannels: latentChannels,
            numAttentionHeads: numAttentionHeads,
            attentionHeadDim: attentionHeadDim,
            captionChannels: captionChannels,
            ropeTheta: ropeTheta,
            defaultSteps: defaultSteps,
            defaultGuidanceScale: defaultGuidanceScale,
            defaultFrameCount: defaultFrameCount,
            defaultFPS: defaultFPS,
            lazyModelLoading: lazyModelLoading
        )
    }

    // MARK: - Generation

    public func generateVideo(
        configuration: VideoConfiguration,
        progressHandler: @Sendable (VideoProgress) -> Bool
    ) async throws -> VideoGenerationResult {
        let steps = configuration.stepCount
        let numFrames = configuration.numFrames
        let width = configuration.width
        let height = configuration.height
        let fps = configuration.fps

        // 1. Encode text
        let progressContinue = progressHandler(VideoProgress(step: 0, totalSteps: steps, phase: .encoding))
        guard progressContinue else { return VideoGenerationResult(frames: [], fps: fps) }

        let (textEmbeddings, attentionMask) = try await encodeText(configuration.prompt)
        if lazyModelLoading { await textEncoder.unloadResources() }

        let dumpDir = configuration.dumpDirectory

        if let dir = dumpDir {
            dumpFloatArray(
                textEmbeddings, shape: [1, Self.textSeqLen, captionChannels], to: "\(dir)/02_text_embeddings.npy")
        }

        // 2. Compute latent dimensions
        let latentFrames = (numFrames - 1) / Self.temporalCompression + 1
        let latentH = height / Self.spatialCompression
        let latentW = width / Self.spatialCompression
        let videoSeqLen = latentFrames * latentH * latentW

        // 3. Generate noise in packed format [1, seq_len, C]
        let noiseCount = videoSeqLen * latentChannels
        var latents: [Float]
        if let noisePath = configuration.loadNoisePath {
            let noiseData = try Data(contentsOf: URL(fileURLWithPath: noisePath))
            latents = noiseData.withUnsafeBytes { ptr in
                Array(ptr.bindMemory(to: Float.self))
            }
            precondition(latents.count == noiseCount,
                         "Loaded noise has \(latents.count) elements, expected \(noiseCount)")
        } else {
            var rng = TorchRandomSource(seed: configuration.seed)
            latents = (0..<noiseCount).map { _ in Float(rng.nextNormal()) }
        }

        if let dir = dumpDir {
            dumpFloatArray(latents, shape: [1, videoSeqLen, latentChannels], to: "\(dir)/04_noise.npy")
        }

        // 4. Compute 3D RoPE embeddings
        let (ropeCos, ropeSin) = computeLTXVideoRoPE(
            numFrames: latentFrames, height: latentH, width: latentW, dim: innerDim, fps: fps)

        if let dir = dumpDir {
            dumpFloatArray(ropeCos, shape: [1, videoSeqLen, innerDim], to: "\(dir)/05_rope_cos.npy")
            dumpFloatArray(ropeSin, shape: [1, videoSeqLen, innerDim], to: "\(dir)/05_rope_sin.npy")
        }

        // 5. Setup scheduler (flow matching Euler with dynamic shift)
        // mu = base_shift + (max_shift - base_shift) * (num_latent_pixels / (512*512))
        let numLatentPixels = videoSeqLen * latentChannels
        let baseShift: Float = 0.5
        let maxShift: Float = 1.15
        let mu = min(maxShift, baseShift + (maxShift - baseShift) * Float(numLatentPixels) / Float(512 * 512))
        let scheduler = DiscreteFlowScheduler(
            stepCount: steps,
            trainStepCount: 1000,
            timeStepShift: 1.0,
            mu: mu
        )

        // 6. Denoising loop
        let textSeqLen = Self.textSeqLen
        let ropeShape = [1, videoSeqLen, innerDim]
        let textShape = [1, textSeqLen, captionChannels]
        let maskShape = [1, textSeqLen]

        let int32Mask = attentionMask.map { Float($0) }

        for (step, t) in scheduler.timeSteps.enumerated() {
            let timestepValue = Float(t)

            if let dir = dumpDir {
                dumpFloatArray(
                    latents, shape: [1, videoSeqLen, latentChannels],
                    to: "\(dir)/07_step\(String(format: "%02d", step))_input_latents.npy")
            }

            let output = try await transformer.run(floatInputs: [
                (latents, [1, videoSeqLen, latentChannels]),
                (textEmbeddings, textShape),
                ([timestepValue], [1]),
                (int32Mask, maskShape),
                (ropeCos, ropeShape),
                (ropeSin, ropeShape),
            ])

            if let dir = dumpDir {
                dumpFloatArray(
                    output, shape: [1, videoSeqLen, latentChannels],
                    to: "\(dir)/07_step\(String(format: "%02d", step))_model_output.npy")
            }

            latents = scheduler.step(output: output, timeStep: t, sample: latents)

            if let dir = dumpDir {
                dumpFloatArray(
                    latents, shape: [1, videoSeqLen, latentChannels],
                    to: "\(dir)/07_step\(String(format: "%02d", step))_output_latents.npy")
            }

            let progress = VideoProgress(step: step + 1, totalSteps: steps, phase: .denoising)
            if !progressHandler(progress) { break }
        }

        if lazyModelLoading { await transformer.unloadResources() }

        // 7. Unpack latents: [1, seq_len, C] -> [1, C, T, H, W]
        let unpackedLatents = unpackLatents3D(
            latents, channels: latentChannels,
            frames: latentFrames, height: latentH, width: latentW)

        if let dir = dumpDir {
            dumpFloatArray(
                unpackedLatents, shape: [1, latentChannels, latentFrames, latentH, latentW],
                to: "\(dir)/08_unpacked_latents.npy")
        }

        // 8. VAE decode
        _ = progressHandler(VideoProgress(step: 0, totalSteps: 1, phase: .decoding))
        let vaeShape = [1, latentChannels, latentFrames, latentH, latentW]
        let pixels = try await decoder.run(floatInputs: [(unpackedLatents, vaeShape)])
        if lazyModelLoading { await decoder.unloadResources() }

        if let dir = dumpDir {
            dumpFloatArray(
                pixels, shape: [1, 3, numFrames, height, width],
                to: "\(dir)/09_decoded_video.npy")
        }

        // 9. Convert to frames
        // VAE output: [1, 3, output_frames, output_h, output_w]
        let outputFrames = numFrames
        let outputH = height
        let outputW = width
        let frames = try extractFrames(
            pixels, numFrames: outputFrames, height: outputH, width: outputW)

        _ = progressHandler(VideoProgress(step: 1, totalSteps: 1, phase: .assembling))

        return VideoGenerationResult(frames: frames, fps: fps)
    }

    // MARK: - Text Encoding

    private func encodeText(_ text: String) async throws -> ([Float], [Int32]) {
        let seqLen = Self.textSeqLen

        var ids = tokenizer.encode(text: text)
        if ids.count > seqLen {
            ids = Array(ids.prefix(seqLen))
        }

        let realTokenCount = ids.count

        // T5 uses pad_token_id = 0
        while ids.count < seqLen {
            ids.append(0)
        }

        let int32Ids = ids.map { Int32($0) }
        var maskValues = [Int32](repeating: 0, count: seqLen)
        for i in 0..<realTokenCount { maskValues[i] = 1 }

        let hiddenStates = try await textEncoder.run(intInputs: [
            (int32Ids, [1, seqLen]),
            (maskValues, [1, seqLen]),
        ])

        return (hiddenStates, maskValues)
    }

    // MARK: - 3D RoPE Pre-computation

    /// Compute 3D RoPE (cos, sin) embeddings for LTX Video transformer.
    ///
    /// Replicates `LTXVideoRotaryPosEmbed.forward()` logic from the Python reference:
    /// - Build a 3D grid of (frame, height, width) coordinates
    /// - Scale each axis by interpolation scale and base resolution
    /// - Compute log-linear frequency bands from theta
    /// - Apply `grid * 2 - 1` position mapping
    /// - Return (cos, sin) each of shape [1, seq_len, dim]
    func computeLTXVideoRoPE(
        numFrames: Int, height: Int, width: Int, dim: Int, fps: Int
    ) -> ([Float], [Float]) {
        let seqLen = numFrames * height * width
        let freqDim = dim / 6

        // Build 3D coordinate grid and apply interpolation scaling
        // rope_interpolation_scale = (vae_temporal/fps, vae_spatial, vae_spatial)
        let ropeScaleT = Float(Self.temporalCompression) / Float(fps)
        let ropeScaleH = Float(Self.spatialCompression)
        let ropeScaleW = Float(Self.spatialCompression)

        var gridCoords = [(Float, Float, Float)](repeating: (0, 0, 0), count: seqLen)
        for f in 0..<numFrames {
            for h in 0..<height {
                for w in 0..<width {
                    let idx = f * height * width + h * width + w
                    let fCoord = Float(f) * ropeScaleT * 1.0 / Float(Self.baseNumFrames)
                    let hCoord = Float(h) * ropeScaleH * 1.0 / Float(Self.baseHeight)
                    let wCoord = Float(w) * ropeScaleW * 1.0 / Float(Self.baseWidth)
                    gridCoords[idx] = (fCoord, hCoord, wCoord)
                }
            }
        }

        // Compute frequency bands: theta^linspace(log_theta(1), log_theta(theta), dim//6)
        // Then scale by pi/2
        var freqs = [Double](repeating: 0, count: freqDim)
        if freqDim > 1 {
            for i in 0..<freqDim {
                let t = Double(i) / Double(freqDim - 1)
                // linspace from log_theta(1)=0 to log_theta(theta)=1
                let exponent = t  // log_theta(1)=0, log_theta(theta)=1
                freqs[i] = pow(Double(ropeTheta), exponent)
            }
        } else if freqDim == 1 {
            freqs[0] = 1.0
        }
        // Scale by pi/2
        let piOver2 = Double.pi / 2.0
        for i in 0..<freqDim {
            freqs[i] *= piOver2
        }

        // Apply frequencies to grid: freqs * (grid * 2 - 1)
        // For each position, compute angles for all 3 axes x freqDim frequencies
        // Layout: [S, 3, freqDim] -> transpose -> [S, freqDim, 3] -> flatten -> [S, dim/2]
        // Then repeat_interleave(2) -> [S, dim]
        let halfDim = freqDim * 3
        var cosOut = [Float](repeating: 0, count: seqLen * dim)
        var sinOut = [Float](repeating: 0, count: seqLen * dim)

        for s in 0..<seqLen {
            let (fCoord, hCoord, wCoord) = gridCoords[s]
            let coords = [fCoord, hCoord, wCoord]

            // Compute angles for each (freq, axis) pair
            // Python order: freqs * (grid * 2 - 1)  -> shape [S, 3, freqDim]
            // Then transpose(-1, -2) -> [S, freqDim, 3]
            // Then flatten(2) -> [S, freqDim*3] = [S, dim/2]
            for fi in 0..<freqDim {
                for axis in 0..<3 {
                    let scaledCoord = Double(coords[axis]) * 2.0 - 1.0
                    let angle = freqs[fi] * scaledCoord
                    let c = Float(cos(angle))
                    let sn = Float(sin(angle))

                    // After transpose: index = fi * 3 + axis in the half-dim
                    let halfIdx = fi * 3 + axis

                    // repeat_interleave(2): each half-dim value maps to two consecutive positions
                    let outIdx0 = s * dim + halfIdx * 2
                    let outIdx1 = outIdx0 + 1
                    cosOut[outIdx0] = c
                    cosOut[outIdx1] = c
                    sinOut[outIdx0] = sn
                    sinOut[outIdx1] = sn
                }
            }
        }

        // Handle non-divisible dimensions: if dim % 6 != 0, pad at the front
        let remainder = dim % 6
        if remainder != 0 {
            // Shift existing values right by `remainder` positions and pad front with cos=1, sin=0
            var cosPadded = [Float](repeating: 0, count: seqLen * dim)
            var sinPadded = [Float](repeating: 0, count: seqLen * dim)
            for s in 0..<seqLen {
                let base = s * dim
                // Padding at front
                for p in 0..<remainder {
                    cosPadded[base + p] = 1.0
                    sinPadded[base + p] = 0.0
                }
                // Copy computed values
                let computedLen = halfDim * 2
                for i in 0..<computedLen {
                    cosPadded[base + remainder + i] = cosOut[base + i]
                    sinPadded[base + remainder + i] = sinOut[base + i]
                }
            }
            return (cosPadded, sinPadded)
        }

        return (cosOut, sinOut)
    }

    // MARK: - Latent Packing/Unpacking

    /// Unpack latents from [1, T*H*W, C] to [1, C, T, H, W] for the 3D VAE decoder.
    private func unpackLatents3D(
        _ packed: [Float], channels: Int, frames: Int, height: Int, width: Int
    ) -> [Float] {
        var unpacked = [Float](repeating: 0, count: channels * frames * height * width)
        for f in 0..<frames {
            for h in 0..<height {
                for w in 0..<width {
                    let token = f * height * width + h * width + w
                    for c in 0..<channels {
                        let srcIdx = token * channels + c
                        let dstIdx = c * frames * height * width + f * height * width + h * width + w
                        unpacked[dstIdx] = packed[srcIdx]
                    }
                }
            }
        }
        return unpacked
    }

    // MARK: - Frame Extraction

    /// Extract individual frames from VAE output [1, 3, T, H, W] as CGImages.
    private func extractFrames(
        _ pixels: [Float], numFrames: Int, height: Int, width: Int
    ) throws -> [CGImage] {
        let spatialSize = height * width
        var frames: [CGImage] = []
        frames.reserveCapacity(numFrames)

        for t in 0..<numFrames {
            // Extract CHW slice for frame t
            var framePixels = [Float](repeating: 0, count: 3 * spatialSize)
            for c in 0..<3 {
                let srcOffset = c * numFrames * spatialSize + t * spatialSize
                let dstOffset = c * spatialSize
                for i in 0..<spatialSize {
                    framePixels[dstOffset + i] = pixels[srcOffset + i]
                }
            }
            let image = try DiffusionUtilities.pixelsToCGImage(
                framePixels, height: height, width: width)
            frames.append(image)
        }

        return frames
    }

    // MARK: - Asset Resolution

    private static func resolveAsset(at url: URL, name: String) -> URL? {
        let fm = FileManager.default
        let aimodel = url.appendingPathComponent("\(name).aimodel")
        let aimodelc = url.appendingPathComponent("\(name).aimodelc")
        if fm.fileExists(atPath: aimodel.path) {
            return aimodel
        } else if fm.fileExists(atPath: aimodelc.path) {
            return aimodelc
        }
        return nil
    }
}

// MARK: - Errors

public enum LTXVideoError: Error, LocalizedError {
    case invalidMetadata(String)
    case missingComponent(String)
    case invalidDimensions(String)

    public var errorDescription: String? {
        switch self {
        case .invalidMetadata(let detail):
            return "Invalid LTX Video metadata: \(detail)"
        case .missingComponent(let name):
            return "Required component '\(name)' not found in model directory"
        case .invalidDimensions(let detail):
            return "Invalid dimensions for LTX Video: \(detail)"
        }
    }
}

// MARK: - Parity Dump Helpers

/// Write a Float array as a raw .npy file (numpy format) for parity comparison.
func dumpFloatArray(_ data: [Float], shape: [Int], to path: String) {
    // Write numpy .npy format: magic + header + raw data
    let header = numpyHeader(shape: shape, dtype: "<f4")
    var fileData = Data()
    fileData.append(contentsOf: [0x93])  // magic
    fileData.append("NUMPY".data(using: .ascii)!)
    fileData.append(contentsOf: [0x01, 0x00])  // version 1.0
    let headerBytes = header.data(using: .ascii)!
    let paddedLen = ((headerBytes.count + 10 + 63) / 64) * 64 - 10
    let padCount = paddedLen - headerBytes.count
    var paddedHeader = header + String(repeating: " ", count: padCount - 1) + "\n"
    let headerData = paddedHeader.data(using: .ascii)!
    let headerLen = UInt16(headerData.count)
    fileData.append(contentsOf: withUnsafeBytes(of: headerLen.littleEndian) { Array($0) })
    fileData.append(headerData)
    data.withUnsafeBufferPointer { buf in
        fileData.append(Data(buffer: buf))
    }
    try? fileData.write(to: URL(fileURLWithPath: path))
}

private func numpyHeader(shape: [Int], dtype: String) -> String {
    let shapeStr =
        shape.count == 1
        ? "(\(shape[0]),)"
        : "(" + shape.map(String.init).joined(separator: ", ") + ")"
    return "{'descr': '\(dtype)', 'fortran_order': False, 'shape': \(shapeStr), }"
}

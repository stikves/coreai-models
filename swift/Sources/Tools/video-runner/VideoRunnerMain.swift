// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIShared
import CoreAIVideoPipeline
import Foundation

@main
struct VideoRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video-runner",
        abstract: "Generate videos using text-to-video diffusion models"
    )

    @Option(help: "Path to model directory containing .aimodel components")
    var model: String

    @Option(help: "Text prompt for video generation")
    var prompt: String

    @Option(help: "Number of output frames (default: model default)")
    var numFrames: Int?

    @Option(help: "Output FPS")
    var fps: Int?

    @Option(help: "Output resolution as WxH, e.g. 512x320")
    var resolution: String?

    @Option(help: "Number of denoising steps")
    var steps: Int?

    @Option(name: .customLong("guidance-scale"), help: "Classifier-free guidance scale")
    var guidanceScale: Float?

    @Option(help: "Random seed")
    var seed: UInt32 = 42

    @Option(help: "Output file path")
    var output: String = "output.mp4"

    @Option(
        name: .customLong("output-format"),
        help: "Output format: mp4, gif, apng, webp, or frames")
    var outputFormat: String = "mp4"

    @Flag(help: "Disable lazy model loading (keep all models in memory)")
    var noLazyLoading: Bool = false

    @Flag(help: "Enable verbose logging")
    var verbose: Bool = false

    @Option(
        name: .customLong("dump-intermediates"),
        help: "Dump intermediate tensors to directory for parity testing")
    var dumpIntermediates: String?

    @Option(
        name: .customLong("load-noise"),
        help: "Load initial noise from raw float32 binary file (for parity testing)")
    var loadNoise: String?

    func run() async throws {
        let modelURL = URL(fileURLWithPath: model)

        // Parse metadata to determine pipeline type
        let metadataURL = modelURL.appendingPathComponent("metadata.json")
        guard FileManager.default.fileExists(atPath: metadataURL.path) else {
            print("Error: metadata.json not found in \(model)")
            throw ExitCode.failure
        }

        let metadataData = try Data(contentsOf: metadataURL)
        guard let json = try JSONSerialization.jsonObject(with: metadataData) as? [String: Any],
            let diffusion = json["diffusion"] as? [String: Any],
            let pipelineType = diffusion["type"] as? String
        else {
            print("Error: metadata.json missing 'diffusion.type'")
            throw ExitCode.failure
        }

        let parsedWidth: Int?
        let parsedHeight: Int?
        if let resolution {
            let parts = resolution.split(separator: "x")
            guard parts.count == 2,
                let w = Int(parts[0]),
                let h = Int(parts[1])
            else {
                print("Error: --resolution must be in WxH format (e.g. 512x320)")
                throw ExitCode.failure
            }
            parsedWidth = w
            parsedHeight = h
        } else {
            parsedWidth = nil
            parsedHeight = nil
        }

        switch pipelineType {
        case "ltx-video":
            try await runLTXVideo(
                modelURL: modelURL,
                diffusion: diffusion,
                parsedWidth: parsedWidth,
                parsedHeight: parsedHeight
            )
        default:
            print("Error: unsupported pipeline type '\(pipelineType)'")
            throw ExitCode.failure
        }
    }

    // MARK: - LTX Video

    private func runLTXVideo(
        modelURL: URL,
        diffusion: [String: Any],
        parsedWidth: Int?,
        parsedHeight: Int?
    ) async throws {
        print("Loading LTX Video pipeline...")
        let pipeline = try await LTXVideoPipeline(
            from: modelURL,
            lazyModelLoading: !noLazyLoading
        )

        let width = parsedWidth ?? pipeline.defaultVideoSize.width
        let height = parsedHeight ?? pipeline.defaultVideoSize.height
        let frameCount = numFrames ?? pipeline.defaultFrameCount
        let outputFPS = fps ?? (diffusion["default_fps"] as? Int ?? 24)
        let stepCount = steps ?? (diffusion["default_steps"] as? Int ?? 50)
        let guidance =
            guidanceScale
            ?? (diffusion["default_guidance_scale"] as? NSNumber)?.floatValue ?? 3.0

        let config = VideoConfiguration(
            prompt: prompt,
            seed: seed,
            stepCount: stepCount,
            guidanceScale: guidance,
            numFrames: frameCount,
            fps: outputFPS,
            width: width,
            height: height,
            dumpDirectory: dumpIntermediates,
            loadNoisePath: loadNoise
        )

        if let dumpDir = dumpIntermediates {
            try FileManager.default.createDirectory(
                atPath: dumpDir, withIntermediateDirectories: true)
            print("  Dump Dir:       \(dumpDir)")
        }

        print("Video Generation Configuration")
        print("  Model:          \(model)")
        print("  Pipeline:       LTX Video")
        print("  Prompt:         \(prompt)")
        print("  Frames:         \(frameCount)")
        print("  FPS:            \(outputFPS)")
        print("  Resolution:     \(width)x\(height)")
        print("  Steps:          \(stepCount)")
        print("  Guidance Scale: \(guidance)")
        print("  Seed:           \(seed)")
        print("  Output:         \(output)")
        print("  Format:         \(outputFormat)")
        print("  Lazy Loading:   \(!noLazyLoading)")
        print()

        let startTime = CFAbsoluteTimeGetCurrent()

        let result = try await pipeline.generateVideo(configuration: config) { progress in
            switch progress.phase {
            case .encoding:
                print("Encoding text...")
            case .denoising:
                print("Denoising step \(progress.step)/\(progress.totalSteps)")
            case .decoding:
                print("Decoding video...")
            case .assembling:
                print("Assembling frames...")
            }
            return true
        }

        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
        print()
        print("Generated \(result.frames.count) frames in \(String(format: "%.1f", elapsed))s")

        // Write output
        let outputURL = URL(fileURLWithPath: output)

        switch outputFormat.lowercased() {
        case "mp4":
            try await VideoWriter.writeMP4(frames: result.frames, fps: result.fps, to: outputURL)
            print("Saved MP4 to \(output)")

        case "gif":
            try VideoWriter.writeGIF(frames: result.frames, fps: result.fps, to: outputURL)
            print("Saved GIF to \(output)")

        case "apng":
            try VideoWriter.writeAPNG(frames: result.frames, fps: result.fps, to: outputURL)
            print("Saved APNG to \(output)")

        case "webp":
            try VideoWriter.writeWebP(frames: result.frames, fps: result.fps, to: outputURL)
            print("Saved WebP to \(output)")

        case "frames":
            let dir = outputURL.deletingPathExtension()
            try VideoWriter.writeFrames(frames: result.frames, to: dir)
            print("Saved \(result.frames.count) frames to \(dir.path)")

        default:
            print("Error: unknown output format '\(outputFormat)'")
            throw ExitCode.failure
        }
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIShared
import CoreAIVideoPipeline
import Foundation

@main struct Main {
    static func main() async throws { await VideoBenchmark.main() }
}

struct VideoBenchmark: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "video-benchmark",
        abstract: "Video generation benchmark for CoreAI models"
    )

    @Option(name: .customLong("model"), help: "Path to a model bundle directory")
    var model: String
    @Option(name: .customLong("num-frames"), help: "Number of output frames")
    var numFrames: Int = 25
    @Option(name: .customLong("resolution"), help: "Output resolution as WxH")
    var resolution: String = "512x320"
    @Option(name: .customLong("steps"), help: "Number of denoising steps")
    var steps: Int = 20
    @Option(name: [.customShort("n"), .customLong("num-trials")], help: "Number of timing trials")
    var numTrials: Int = 3
    @Option(name: .long, help: "Random seed")
    var seed: UInt32 = 42

    func validate() throws {
        if numFrames < 1 { throw ValidationError("--num-frames must be >= 1") }
        if steps < 1 { throw ValidationError("--steps must be >= 1") }
        if numTrials < 1 { throw ValidationError("--num-trials must be >= 1") }
        guard FileManager.default.fileExists(atPath: model) else {
            throw ValidationError("Model path not found: \(model)")
        }
        let p = resolution.split(separator: "x")
        guard p.count == 2, Int(p[0]) != nil, Int(p[1]) != nil else {
            throw ValidationError("--resolution must be WxH (e.g. 512x320)")
        }
    }

    func run() async throws {
        #if DEBUG
        print("Note: built in Debug mode. For more reliable results, build with -c release.")
        #endif
        let rp = resolution.split(separator: "x")
        let (width, height) = (Int(rp[0])!, Int(rp[1])!)

        print("\n⏳ Loading pipeline from: \(model)")
        let pipeline = try await LTXVideoPipeline(
            from: URL(fileURLWithPath: model), lazyModelLoading: true)
        print("   done")

        let config = VideoConfiguration(
            prompt: "a benchmark prompt for timing only", seed: seed,
            stepCount: steps, guidanceScale: 3.0,
            numFrames: numFrames, fps: 24, width: width, height: height)

        print("\n⚙️  Warming up pipeline...")
        _ = try await runTrial(pipeline: pipeline, config: config)

        print("\n🔄 Benchmarking: \(numFrames) frames, \(width)x\(height), \(steps) steps\n")
        var trials: [TrialResult] = []
        for i in 0..<numTrials {
            let r = try await runTrial(pipeline: pipeline, config: config)
            trials.append(r)
            if i > 0 { print() }
            print("🧪 Trial \(i + 1)")
            print("   Text encode:  \(fmt(r.encodeTime))s")
            print("   Denoise:      \(fmt(r.denoiseTime))s  (\(fmt(r.denoiseTime / Double(steps)))s/step)")
            print("   VAE decode:   \(fmt(r.decodeTime))s")
            print("   Total:        \(fmt(r.totalTime))s")
        }

        print("\n📊 Benchmark Summary:")
        print(String(repeating: "=", count: 55))
        stat("Text encode", trials.map(\.encodeTime), "s")
        stat("Denoise", trials.map(\.denoiseTime), "s")
        stat("Denoise/step", trials.map { $0.denoiseTime / Double(steps) }, "s")
        stat("VAE decode", trials.map(\.decodeTime), "s")
        stat("Total", trials.map(\.totalTime), "s")
        print(String(repeating: "-", count: 55))
        stat("Denoise tput", trials.map { Double(numFrames * steps) / $0.denoiseTime }, " tok/s")
        stat("Pipeline fps", trials.map { Double(numFrames) / $0.totalTime }, " fr/s")
        print(String(repeating: "=", count: 55))
    }

    // MARK: - Trial

    private func runTrial(
        pipeline: LTXVideoPipeline, config: VideoConfiguration
    ) async throws -> TrialResult {
        var encodeEnd: CFAbsoluteTime = 0
        var denoiseEnd: CFAbsoluteTime = 0
        var decodeEnd: CFAbsoluteTime = 0
        let start = CFAbsoluteTimeGetCurrent()
        let result = try await pipeline.generateVideo(configuration: config) { progress in
            let now = CFAbsoluteTimeGetCurrent()
            switch progress.phase {
            case .encoding: break
            case .denoising: if encodeEnd == 0 { encodeEnd = now }
            case .decoding: denoiseEnd = now
            case .assembling: decodeEnd = now
            }
            return true
        }
        let end = CFAbsoluteTimeGetCurrent()
        if encodeEnd == 0 { encodeEnd = start }
        if denoiseEnd == 0 { denoiseEnd = encodeEnd }
        if decodeEnd == 0 { decodeEnd = end }
        _ = result
        return TrialResult(
            encodeTime: encodeEnd - start, denoiseTime: denoiseEnd - encodeEnd,
            decodeTime: decodeEnd - denoiseEnd, totalTime: end - start)
    }

    // MARK: - Helpers

    private func fmt(_ v: Double) -> String { String(format: "%.3f", v) }

    private func stat(_ label: String, _ vals: [Double], _ sfx: String) {
        let (lo, hi) = (vals.min()!, vals.max()!)
        let avg = vals.reduce(0, +) / Double(vals.count)
        let pad = String(repeating: " ", count: max(0, 14 - label.count))
        print("\(label)\(pad) min=\(fmt(lo))\(sfx)  avg=\(fmt(avg))\(sfx)  max=\(fmt(hi))\(sfx)")
    }
}

struct TrialResult {
    let encodeTime: Double
    let denoiseTime: Double
    let decodeTime: Double
    let totalTime: Double
}

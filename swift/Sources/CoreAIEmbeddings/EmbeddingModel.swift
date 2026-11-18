// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Accelerate
import CoreAIShared
import CoreGraphics
import Foundation

/// Runs a `kind: "embedding"` bundle: text and images in, one vector per input out.
///
/// ```swift
/// let model = try await EmbeddingModel(resourcesAt: bundleURL)
/// let query = try await model.embed(["a dog on the beach"])[0]
/// let photos = try await model.embed(images: images)
/// let best = try photos.max { try query.cosine($0) < query.cosine($1) }
/// ```
///
/// Each tower loads on first use, or up front with `load(_:)`. The first load of a Neural Engine
/// program specializes it, which takes minutes for EmbeddingGemma 2; later loads use the Core AI
/// cache.
public actor EmbeddingModel {
    private nonisolated let block: EmbeddingBlock
    private let programs: ProgramCache
    private let drivers: [EmbeddingModality: PipelineDriver]
    private let textLoader: @Sendable () async throws -> TextPreparer
    private var text: Task<TextPreparer, Error>?

    public nonisolated var space: EmbeddingSpace { block.space }
    /// The native vector width.
    public nonisolated var dimension: Int { block.dimension }
    /// The widths `embed` can return, native first.
    public nonisolated var matryoshkaDimensions: [Int] { block.matryoshkaDimensions }
    public nonisolated var modalities: Set<EmbeddingModality> { Set(block.towers.map(\.modality)) }
    /// `logits = logitScale * cosine`, for models trained with one (CLIP).
    public nonisolated var logitScale: Double? { block.logitScale }

    /// Reads and checks the bundle at `url`. Loads no program.
    ///
    /// - Throws: `ModelBundle.BundleError` for a missing or invalid asset, `EmbeddingError` for a
    ///   block that breaks the contract or a tower this runtime cannot run.
    public init(resourcesAt url: URL) async throws {
        let bundle = try EmbeddingBundle(at: url)
        try bundle.bundle.validateModelAssets()
        let folder = bundle.tokenizerURL
        let config = bundle.block.text
        let name = url.lastPathComponent
        try self.init(
            bundle: bundle, programs: ProgramCache.coreAI,
            text: {
                try await InstrumentsProfiler.profile(InstrumentsProfiler.beginTokenizerLoad(id: name)) {
                    try await TextPreparer.load(from: folder, config: config)
                }
            })
    }

    /// A model over `bundle` with programs from `programs` and token ids from `text`.
    init(
        bundle: EmbeddingBundle, programs: @escaping ProgramCache.Factory,
        text: @escaping @Sendable () async throws -> TextPreparer
    ) throws {
        let cache = ProgramCache(bundle: bundle.bundle, factory: programs)
        var drivers: [EmbeddingModality: PipelineDriver] = [:]
        for tower in bundle.block.towers {
            drivers[tower.modality] = try PipelineDriver(tower: tower, block: bundle.block, programs: cache)
        }
        self.block = bundle.block
        self.programs = cache
        self.drivers = drivers
        self.textLoader = text
    }

    /// Loads the programs of `modalities` (default: every tower), and the tokenizer for text.
    /// Functions load on the first call that needs them; `warmup(_:)` loads them too.
    public func load(_ modalities: Set<EmbeddingModality>? = nil) async throws {
        let modalities = modalities ?? self.modalities
        if modalities.contains(.text) { _ = try await textPreparer() }
        for modality in modalities.sorted(by: { $0.rawValue < $1.rawValue }) {
            for stage in try driver(for: modality).stages { _ = try await programs.program(for: stage.component) }
        }
    }

    /// Loads `modalities` (default: every tower) and runs each of their functions once on
    /// placeholder inputs, so later calls do not pay for loading functions.
    public func warmup(_ modalities: Set<EmbeddingModality>? = nil) async throws {
        try await load(modalities)
        try await InstrumentsProfiler.profile(InstrumentsProfiler.beginWarmup()) {
            for modality in (modalities ?? self.modalities).sorted(by: { $0.rawValue < $1.rawValue }) {
                try Task.checkCancellation()
                let driver = try driver(for: modality)
                switch modality {
                case .text:
                    for rows in driver.warmupTokenRows() { _ = try await driver.run(driver.rows(tokens: rows)) }
                case .image:
                    let gray = RGB8Image(width: 64, height: 64, pixels: [UInt8](repeating: 128, count: 64 * 64 * 3))
                    _ = try await driver.run(driver.rows(images: [gray]))
                }
            }
        }
    }

    /// Releases the loaded programs; the next call loads them again.
    public func unload() async {
        await programs.removeAll()
    }

    /// Embeds `inputs`, texts and images in any mix, and returns one vector per input in order.
    /// Texts go in after the bundle's default prompt (`default_prompt_name`), if any.
    ///
    /// - Parameter dimensions: one of `matryoshkaDimensions`; the native width when `nil`.
    /// - Throws: `EmbeddingError.modalityNotInBundle`, `.unsupportedDimension`,
    ///   `.imageConversionFailed`, a program failure, or `CancellationError` when the task is
    ///   cancelled; cancellation takes effect before the next program call.
    public func embed(_ inputs: [EmbeddingInput], dimensions: Int? = nil) async throws -> [Embedding] {
        try await InstrumentsProfiler.profile(InstrumentsProfiler.beginEmbed(inputs: inputs.count)) {
            try await run(inputs, dimensions: dimensions)
        }
    }

    /// The body of `embed(_:dimensions:)`.
    private func run(_ inputs: [EmbeddingInput], dimensions: Int?) async throws -> [Embedding] {
        try check(dimensions: dimensions)
        let textDriver = try inputs.contains(where: \.isText) ? driver(for: .text) : nil
        let imageDriver = try inputs.contains(where: { !$0.isText }) ? driver(for: .image) : nil
        var vectors = [[Float]](repeating: [], count: inputs.count)
        var texts: [(index: Int, tokens: [Int32])] = []
        var images: [(index: Int, image: RGB8Image)] = []
        for (index, input) in inputs.enumerated() {
            try Task.checkCancellation()
            switch input.content {
            case .text(let text):
                texts.append((index, try await textPreparer().tokenize(text)))
            case .image(let image):
                images.append((index, try RGB8Image(cgImage: image)))
            }
        }
        if let driver = textDriver {
            for (vector, item) in zip(try await driver.run(driver.rows(tokens: texts.map(\.tokens))), texts) {
                vectors[item.index] = vector
            }
        }
        if let driver = imageDriver {
            try Task.checkCancellation()
            for (vector, item) in zip(try await driver.run(driver.rows(images: images.map(\.image))), images) {
                vectors[item.index] = vector
            }
        }
        return try vectors.map { try finish($0, width: dimensions) }
    }

    /// Embeds `texts`.
    public func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(texts.map { .text($0) }, dimensions: dimensions)
    }

    /// Embeds `images`.
    public func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(images.map { .image($0) }, dimensions: dimensions)
    }

    /// `embedding` cut to `width` leading values, L2-normalized again when the bundle says so: the
    /// vector `embed` returns at that width, without running the model.
    ///
    /// - Throws: `EmbeddingError.unsupportedDimension` unless `width` is one of
    ///   `matryoshkaDimensions` and at most `embedding.dimension`; `.incompatibleEmbeddings` for an
    ///   embedding of another space.
    public nonisolated func truncate(_ embedding: Embedding, to width: Int) throws -> Embedding {
        guard embedding.space == space else {
            throw EmbeddingError.incompatibleEmbeddings("the embedding is from \(embedding.space.id), not \(space.id)")
        }
        guard matryoshkaDimensions.contains(width), width <= embedding.dimension else {
            throw EmbeddingError.unsupportedDimension(
                width, supported: matryoshkaDimensions.filter { $0 <= embedding.dimension })
        }
        return try finish(embedding.vector, width: width)
    }

    /// Throws `unsupportedDimension` unless `dimensions` is `nil` or one of `matryoshkaDimensions`.
    nonisolated func check(dimensions: Int?) throws {
        if let dimensions, !matryoshkaDimensions.contains(dimensions) {
            throw EmbeddingError.unsupportedDimension(dimensions, supported: matryoshkaDimensions)
        }
    }

    // MARK: - Private

    /// The tokenizer; concurrent first calls share one load.
    private func textPreparer() async throws -> TextPreparer {
        if let text { return try await text.value }
        let load = Task { [textLoader] in try await textLoader() }
        text = load
        do {
            return try await load.value
        } catch {
            if text == load { text = nil }
            throw error
        }
    }

    private func driver(for modality: EmbeddingModality) throws -> PipelineDriver {
        guard let driver = drivers[modality] else { throw EmbeddingError.modalityNotInBundle(modality) }
        return driver
    }

    /// Cuts `vector` to `width` and L2-normalizes it again when the bundle says so.
    private nonisolated func finish(_ vector: [Float], width: Int?) throws -> Embedding {
        let width = width ?? dimension
        guard vector.allSatisfy(\.isFinite) else {
            throw EmbeddingError.unexpectedOutput(component: "embedding", "non-finite values")
        }
        var v = Array(vector.prefix(width))
        if width < vector.count && block.renormalizeAfterTruncation {
            let norm = vDSP.sumOfSquares(v).squareRoot()
            if norm > 0 { v = vDSP.divide(v, norm) }
        }
        return Embedding(vector: v, space: space)
    }
}

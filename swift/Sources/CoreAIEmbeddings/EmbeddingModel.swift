// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreAIVectorIndex
import CoreGraphics
import Foundation
import Tokenizers

/// A text and image embedding model loaded from an exported embedding bundle.
///
/// Turns text and images into vectors in one shared space for semantic search, deduplication,
/// clustering or classification; a text query can find images. Each bundle component runs on the
/// compute unit the bundle declares (GPU, or the Neural Engine for Neural Engine exports).
///
/// ## Usage
/// ```swift
/// let model = try EmbeddingModel(bundleAt: URL(filePath: "exports/embeddinggemma-2"))
/// let query = try await model.embed(.text("a cat asleep on a sofa"), promptName: "query")
/// let photo = try await model.embed(.image(cgImage))
/// let similarity = try query.cosineSimilarity(to: photo)
///
/// // Smaller vectors: any width in `matryoshkaDimensions`.
/// let compact = try await model.embed(.text("invoice due friday"), promptName: "document", dimensions: 256)
/// ```
///
/// ## See also
/// Export a bundle with `uv run coreai.embedding.export embeddinggemma-2` (see
/// models/embedding_gemma2/README.md). Store and search the vectors with `VectorIndex`.
public actor EmbeddingModel {
    /// Vector space (model id and revision) stamped on every embedding.
    public nonisolated let space: VectorSpace
    /// Native embedding width.
    public nonisolated let dimension: Int
    /// Widths accepted by `dimensions:`, native first.
    public nonisolated let matryoshkaDimensions: [Int]
    /// Modalities this bundle has a tower for.
    public nonisolated let supportedModalities: Set<EmbeddingModality>
    /// Sorted prompt names the bundle defines.
    public nonisolated let promptNames: [String]

    nonisolated let bundleURL: URL
    nonisolated let name: String
    nonisolated let config: EmbeddingConfig
    nonisolated let assets: [String: String]

    package nonisolated var documentTemplate: String? { config.text?.documentTemplate }

    package typealias ComponentFactory =
        @Sendable (_ name: String, _ url: URL, _ computeUnits: [String]?) throws -> any EmbeddingComponent
    package typealias TokenizerFactory = @Sendable (_ folder: URL) async throws -> any Tokenizer

    private let makeComponent: ComponentFactory
    private let makeTokenizer: TokenizerFactory
    private var tokenizer: (any Tokenizer)?
    private var textPreprocessor: TextEmbeddingPreprocessor?
    private var components: [String: any EmbeddingComponent] = [:]
    /// In-flight component prepares and tokenizer load (concurrent `load` calls share them).
    private var componentTasks: [String: Task<PreparedComponent, any Error>] = [:]
    private var tokenizerTask: Task<(any Tokenizer, Duration), any Error>?
    private var preparedFunctions: Set<String> = []
    private var ranFunctions: Set<String> = []
    private var generation = 0

    /// Reads the bundle's metadata; the model assets load on first use.
    public init(bundleAt url: URL) throws {
        try self.init(
            bundleAt: url,
            makeComponent: { name, assetURL, units in
                try CoreAIEmbeddingComponent(name: name, modelURL: assetURL, computeUnits: units)
            },
            makeTokenizer: { folder in
                guard FileManager.default.fileExists(atPath: folder.appending(path: "tokenizer.json").path) else {
                    throw EmbeddingError.invalidBundle("tokenizer not found at \(folder.path)")
                }
                return try await AutoTokenizer.from(modelFolder: folder)
            })
    }

    /// Dependency-injecting init for tests and custom backends.
    package init(
        bundleAt url: URL, makeComponent: @escaping ComponentFactory, makeTokenizer: @escaping TokenizerFactory
    ) throws {
        let bundle = try EmbeddingBundle(at: url)
        let config = bundle.config
        self.bundleURL = url
        self.name = bundle.bundle.name
        self.config = config
        self.assets = bundle.bundle.assets
        self.space = config.space
        self.dimension = config.dimension
        self.matryoshkaDimensions = config.widths
        self.supportedModalities = Set(config.knownTowers.map(\.modality))
        self.promptNames = (config.text?.prompts ?? [:]).keys.sorted()
        self.makeComponent = makeComponent
        self.makeTokenizer = makeTokenizer
    }

    // MARK: - Lifecycle

    /// Loads the tokenizer and components the towers for `modalities` need (filtered to the supported ones).
    ///
    /// Component prepares and the tokenizer load run concurrently. Idempotent; a later call with more
    /// modalities loads only the missing pieces. `embed` calls this for its input modalities.
    public func load(modalities: Set<EmbeddingModality> = [.text, .image]) async throws {
        let towers = config.knownTowers.filter { modalities.contains($0.modality) }.map(\.tower)
        let needed = Set(towers.flatMap(\.pipeline)).filter { assets[$0] != nil && components[$0] == nil }.sorted()
        let needsText = towers.contains { $0.modality == EmbeddingModality.text.rawValue }
        let needsTokenizer = needsText || towers.contains(where: needsTokenizerForImage)
        let needsPreprocessor = needsText && textPreprocessor == nil
        guard !needed.isEmpty || (needsTokenizer && tokenizer == nil) || needsPreprocessor else { return }

        let generation = self.generation
        // Start everything before awaiting anything so the prepares and the tokenizer overlap.
        if needsTokenizer && tokenizer == nil && tokenizerTask == nil {
            let folder = bundleURL.appending(path: config.text?.tokenizer ?? "tokenizer")
            tokenizerTask = Task { [makeTokenizer] in
                let start = ContinuousClock.now
                let tokenizer = try await makeTokenizer(folder)
                return (tokenizer, ContinuousClock.now - start)
            }
        }
        for component in needed where componentTasks[component] == nil {
            let url = bundleURL.appending(path: assets[component]!)
            let first = firstFunction(of: component)
            let units = config.computeUnits(of: component)
            componentTasks[component] = Task { [makeComponent] in
                let start = ContinuousClock.now
                let prepared = try makeComponent(component, url, units)
                try await prepared.load()
                let prepare = ContinuousClock.now - start
                // Load the function a short request runs first, off the critical path.
                var function: (String, Duration)?
                if let first {
                    let t1 = ContinuousClock.now
                    if (try? await prepared.prepareFunction(named: first)) != nil {
                        function = (first, ContinuousClock.now - t1)
                    }
                }
                return PreparedComponent(component: prepared, prepare: prepare, function: function)
            }
        }

        var firstError: (any Error)?
        if needsTokenizer, tokenizer == nil, let task = tokenizerTask {
            do {
                let (loaded, elapsed) = try await task.value
                if self.generation == generation, tokenizer == nil {
                    tokenizer = loaded
                    Self.log("tokenizer", config.text?.tokenizer ?? "tokenizer", elapsed)
                }
            } catch {
                firstError = firstError ?? error
            }
            if tokenizerTask == task { tokenizerTask = nil }
        }
        for name in needed {
            guard let task = componentTasks[name] else { continue }
            do {
                let prepared = try await task.value
                if self.generation == generation, components[name] == nil {
                    components[name] = prepared.component
                    let unit = ComputeUnit.resolve(config.computeUnits(of: name)) ?? .gpu
                    CLILogger.log("\(name) runs on \(unit.displayName)", component: "embedding")
                    Self.log("prepare", name, prepared.prepare, cacheHit: await prepared.component.loadedFromCache == true)
                    if let (function, elapsed) = prepared.function {
                        preparedFunctions.insert("\(name)/\(function)")
                        Self.log("load_function", "\(name).\(function)", elapsed)
                    }
                }
            } catch {
                firstError = firstError ?? error
            }
            if componentTasks[name] == task { componentTasks[name] = nil }
        }
        if let firstError { throw firstError }

        if needsPreprocessor, textPreprocessor == nil, let tower = config.tower(for: .text), let tokenizer {
            textPreprocessor = try await makeTextPreprocessor(tower: tower, tokenizer: tokenizer)
        }
    }

    /// Releases the tokenizer and every loaded component.
    public func unload() async {
        generation += 1
        for task in componentTasks.values { task.cancel() }
        tokenizerTask?.cancel()
        componentTasks = [:]
        tokenizerTask = nil
        for component in components.values { await component.unload() }
        components = [:]
        tokenizer = nil
        textPreprocessor = nil
        preparedFunctions = []
        ranFunctions = []
    }

    private struct PreparedComponent: Sendable {
        var component: any EmbeddingComponent
        var prepare: Duration
        var function: (String, Duration)?
    }

    private static func log(_ phase: String, _ component: String, _ duration: Duration, cacheHit: Bool = false) {
        CLILogger.log(
            String(format: "%@ %@ %.1f ms%@", phase, component, duration.inMilliseconds, cacheHit ? " (cache hit)" : ""),
            component: "embedding")
    }

    /// The function a short request of `component` runs first: the smallest static trace or the single function.
    private func firstFunction(of component: String) -> String? {
        guard let spec = config.component(component) else { return nil }
        if let functions = spec.functions, !functions.isEmpty {
            return functions.min { ($0.batch, $0.length) < ($1.batch, $1.length) }?.function
        }
        return spec.function
    }

    /// The image tower needs the tokenizer only when metadata lacks the special-token ids it adds.
    private func needsTokenizerForImage(_ tower: EmbeddingConfig.Tower) -> Bool {
        guard tower.modality == EmbeddingModality.image.rawValue else { return false }
        let special = config.text?.specialTokens
        return (special?.bos == true && special?.bosTokenId == nil)
            || (special?.eos == true && special?.eosTokenId == nil) || special?.padTokenId == nil
    }

    /// Loads `function` of `component` once, logging its time.
    private func prepare(_ function: String, of component: String, on target: any EmbeddingComponent) async throws {
        let key = "\(component)/\(function)"
        guard !preparedFunctions.contains(key) else { return }
        let start = ContinuousClock.now
        try await target.prepareFunction(named: function)
        preparedFunctions.insert(key)
        Self.log("load_function", "\(component).\(function)", ContinuousClock.now - start)
    }

    /// Runs `function` of `component`, logging its load and first run.
    private func run(
        _ component: String, _ target: any EmbeddingComponent, function: String,
        inputs: [String: EmbeddingTensor], output: String?
    ) async throws -> [Float] {
        try await prepare(function, of: component, on: target)
        let key = "\(component)/\(function)"
        guard !ranFunctions.contains(key) else {
            return try await target.run(function: function, inputs: inputs, output: output)
        }
        let start = ContinuousClock.now
        let result = try await target.run(function: function, inputs: inputs, output: output)
        ranFunctions.insert(key)
        Self.log("first_run", "\(component).\(function)", ContinuousClock.now - start)
        return result
    }

    private func failure(_ component: String, function: String? = nil, _ reason: String) -> EmbeddingError {
        let unit = ComputeUnit.resolve(config.computeUnits(of: component)) ?? .gpu
        return .componentFailed(
            component: component, function: function, computeUnit: unit.rawValue, cached: false, reason: reason)
    }

    // MARK: - Embedding

    /// Embeds one input.
    ///
    /// - Parameters:
    ///   - promptName: Prompt applied to text inputs; one of ``promptNames`` (default: the bundle's default).
    ///   - dimensions: Output width; one of ``matryoshkaDimensions``.
    public func embed(
        _ input: EmbeddingInput, promptName: String? = nil, dimensions: Int? = nil
    ) async throws -> Embedding {
        try await embed([input], promptName: promptName, dimensions: dimensions)[0]
    }

    /// Embeds `inputs`, returning one embedding per input in order; text and images may be mixed.
    public func embed(
        _ inputs: [EmbeddingInput], promptName: String? = nil, dimensions: Int? = nil
    ) async throws -> [Embedding] {
        let width = try resolveWidth(dimensions)
        var modalities: Set<EmbeddingModality> = []
        for input in inputs {
            if case .image = input { modalities.insert(.image) } else { modalities.insert(.text) }
        }
        for modality in modalities where !supportedModalities.contains(modality) {
            throw EmbeddingError.unsupportedModality(modality)
        }
        try await load(modalities: modalities)

        var results = [Embedding?](repeating: nil, count: inputs.count)
        var textIndices: [Int] = []
        var textTokens: [TextEmbeddingPreprocessor.Tokenized] = []
        for (i, input) in inputs.enumerated() {
            guard case .text(let string) = input else { continue }
            let (preprocessor, tokenizer) = try requireText()
            textIndices.append(i)
            textTokens.append(try preprocessor.tokenize(string, promptName: promptName, tokenizer: tokenizer))
        }
        for (k, vector) in try await embedText(textTokens, width: width).enumerated() {
            results[textIndices[k]] = vector
        }
        for (i, input) in inputs.enumerated() {
            guard case .image(let image) = input else { continue }
            let (vector, tokens) = try await runImage(image)
            results[i] = try finish(vector, width: width, truncated: false, tokens: tokens, tower: .image)
        }
        return results.map { $0! }
    }

    /// Embeds fully formed token ids (parity tests); never re-tokenized or truncated.
    package func embed(tokens: [[Int32]], dimensions: Int?) async throws -> [Embedding] {
        let width = try resolveWidth(dimensions)
        try await load(modalities: [.text])
        let (preprocessor, _) = try requireText()
        return try await embedText(try tokens.map(preprocessor.validateTokens), width: width)
    }

    /// Token ids the text tower would feed for `text`.
    package func tokenize(_ text: String, promptName: String? = nil) async throws -> [Int32] {
        try await load(modalities: [.text])
        let (preprocessor, tokenizer) = try requireText()
        return try preprocessor.tokenize(text, promptName: promptName, tokenizer: tokenizer).ids
    }

    /// Fills `document_template` (`{title}` / `{text}`); without a template returns `text`.
    package nonisolated func formatDocument(title: String?, text: String) -> String {
        guard let template = documentTemplate else { return text }
        return template.replacingOccurrences(of: "{title}", with: title ?? "none")
            .replacingOccurrences(of: "{text}", with: text)
    }

    private func resolveWidth(_ dimensions: Int?) throws -> Int {
        let width = dimensions ?? dimension
        guard matryoshkaDimensions.contains(width) else {
            throw EmbeddingError.unsupportedDimension(width, supported: matryoshkaDimensions)
        }
        return width
    }

    // MARK: - Private: text

    private func requireText() throws -> (TextEmbeddingPreprocessor, any Tokenizer) {
        guard let preprocessor = textPreprocessor, let tokenizer else {
            throw EmbeddingError.unsupportedModality(.text)
        }
        return (preprocessor, tokenizer)
    }

    private func makeTextPreprocessor(
        tower: EmbeddingConfig.Tower, tokenizer: any Tokenizer
    ) async throws -> TextEmbeddingPreprocessor {
        var fallback: [EmbeddingConfig.FunctionSpec] = []
        if let name = tower.pipeline.last, let component = components[name] {
            let spec = config.component(name)
            if (spec?.functions ?? []).isEmpty {
                let function = spec?.function ?? "main"
                let idsName = Self.inputNames(tower).ids
                if let shape = try await component.staticShape(function: function, input: idsName), shape.count == 2 {
                    fallback = [.init(function: function, batch: shape[0], length: shape[1])]
                }
            }
        }
        return try TextEmbeddingPreprocessor(
            configuration: config, tower: tower,
            bosTokenId: tokenizer.bosTokenId, eosTokenId: tokenizer.eosTokenId, padTokenId: tokenizerPadId,
            fallbackFunctions: fallback)
    }

    private var tokenizerPadId: Int? {
        guard let tokenizer, let id = tokenizer.convertTokenToId("<pad>"), id != tokenizer.unknownTokenId else {
            return nil
        }
        return id
    }

    static func inputNames(_ tower: EmbeddingConfig.Tower) -> (ids: String, mask: String) {
        let keys = (tower.inputs ?? [:]).keys.sorted()
        let mask = keys.first { $0.contains("mask") } ?? "attention_mask"
        let ids = keys.first { !$0.contains("mask") } ?? "input_ids"
        return (ids, mask)
    }

    private func embedText(_ tokenized: [TextEmbeddingPreprocessor.Tokenized], width: Int) async throws -> [Embedding] {
        guard !tokenized.isEmpty else { return [] }
        guard let tower = config.tower(for: .text), let last = tower.pipeline.last else {
            throw EmbeddingError.unsupportedModality(.text)
        }
        let vectors = try await runText(tokenized, tower: tower)
        return try zip(tokenized, vectors).map {
            try finish($1, width: width, truncated: $0.truncated, tokens: $0.ids.count, tower: .text, component: last)
        }
    }

    private func runText(
        _ tokenized: [TextEmbeddingPreprocessor.Tokenized], tower: EmbeddingConfig.Tower
    ) async throws -> [[Float]] {
        switch tower.pipeline.count {
        case 1: return try await runSingleEncoder(tokenized, tower: tower)
        case 2: return try await runTokenTableText(tokenized, tower: tower)
        default: throw EmbeddingError.invalidBundle("unsupported text pipeline \(tower.pipeline)")
        }
    }

    private func requireComponent(_ name: String) throws -> any EmbeddingComponent {
        guard let component = components[name] else {
            throw EmbeddingError.invalidBundle("the bundle declares no asset '\(name)'")
        }
        return component
    }

    private func runSingleEncoder(
        _ tokenized: [TextEmbeddingPreprocessor.Tokenized], tower: EmbeddingConfig.Tower
    ) async throws -> [[Float]] {
        let name = tower.pipeline[0]
        let component = try requireComponent(name)
        let (preprocessor, _) = try requireText()
        let names = Self.inputNames(tower)
        let defaultFunction = config.component(name)?.function ?? "main"
        var out = [[Float]](repeating: [], count: tokenized.count)
        for plan in try preprocessor.plan(lengths: tokenized.map(\.ids.count)) {
            let rows = plan.indices.map { tokenized[$0].ids }
            let (ids, mask) = preprocessor.pack(rows, batch: plan.batch, length: plan.length)
            let shape = [plan.batch, plan.length]
            let function = plan.function ?? defaultFunction
            let flat = try await run(
                name, component, function: function,
                inputs: [
                    names.ids: EmbeddingTensor(int32: ids, shape: shape),
                    names.mask: EmbeddingTensor(int32: mask, shape: shape),
                ],
                output: tower.output ?? config.component(name)?.output)
            guard flat.count == plan.batch * dimension else {
                throw failure(name, function: function, "returned \(flat.count) values, expected \(plan.batch * dimension)")
            }
            for (row, index) in plan.indices.enumerated() {
                out[index] = Array(flat[(row * dimension)..<((row + 1) * dimension)])
            }
        }
        return out
    }

    /// Text through token_table then encoder, one input per call at the smallest batch-1 encoder function that fits.
    private func runTokenTableText(
        _ tokenized: [TextEmbeddingPreprocessor.Tokenized], tower: EmbeddingConfig.Tower
    ) async throws -> [[Float]] {
        let (tableName, encoderName) = (tower.pipeline[0], tower.pipeline[1])
        let table = try requireComponent(tableName)
        let encoder = try requireComponent(encoderName)
        let (preprocessor, _) = try requireText()
        var out: [[Float]] = []
        out.reserveCapacity(tokenized.count)
        for row in tokenized {
            let (function, length) = try encoderFunction(encoderName, tokens: row.ids.count)
            let (ids, mask) = preprocessor.pack([row.ids], batch: 1, length: length)
            out.append(
                try await runTableAndEncoder(
                    ids: ids, mask: mask, function: function, table: (tableName, table),
                    encoder: (encoderName, encoder), output: tower.output, merge: { table, _ in table }))
        }
        return out
    }

    /// Smallest batch-1 function of `encoder` holding `tokens`, or the single function at `tokens` length.
    private func encoderFunction(_ encoder: String, tokens: Int) throws -> (function: String, length: Int) {
        let spec = config.component(encoder)
        guard let functions = spec?.functions, !functions.isEmpty else {
            return (spec?.function ?? "main", tokens)
        }
        let singles = functions.filter { $0.batch == 1 }
        guard let fit = singles.filter({ $0.length >= tokens }).min(by: { $0.length < $1.length }) else {
            throw EmbeddingError.inputTooLong(tokens: tokens, maximum: singles.map(\.length).max() ?? 0)
        }
        return (fit.function, fit.length)
    }

    /// token_table over padded `ids`, `merge` (image scatter), then the encoder with the mask encoding it declares.
    private func runTableAndEncoder(
        ids: [Int32], mask: [Int32], function: String,
        table: (name: String, component: any EmbeddingComponent),
        encoder: (name: String, component: any EmbeddingComponent), output: String?,
        merge: (_ table: [Float], _ hidden: Int) throws -> [Float]
    ) async throws -> [Float] {
        let length = ids.count
        let tableSpec = config.component(table.name)
        let tableFunction = tableSpec?.function ?? "main"
        let embeds = try await run(
            table.name, table.component, function: tableFunction,
            inputs: [tableSpec?.inputs?.first ?? "input_ids": EmbeddingTensor(int32: ids, shape: [1, length])],
            output: tableSpec?.output)
        guard length > 0, !embeds.isEmpty, embeds.count % length == 0 else {
            throw failure(table.name, function: tableFunction, "returned \(embeds.count) values for \(length) tokens")
        }
        let hidden = embeds.count / length
        let encoderSpec = config.component(encoder.name) ?? .init()
        let vector = try await run(
            encoder.name, encoder.component, function: function,
            inputs: Self.encoderInputs(component: encoderSpec, embeds: try merge(embeds, hidden), mask: mask, hidden: hidden),
            output: output ?? encoderSpec.output)
        guard vector.count == dimension else {
            throw failure(encoder.name, function: function, "returned \(vector.count) values, expected \(dimension)")
        }
        return vector
    }

    /// Additive mask value for padded keys (fp16-safe "minus infinity").
    static let additiveMaskBias: Float = -30000

    /// Encoder inputs for one `[1, L]` row: `inputs_embeds` plus either the int32 `attention_mask` or, for an
    /// additive-bias encoder, `bias` `[1, key, 1, query]` and `pool_weights` `[1, L]` (mask / valid count).
    static func encoderInputs(
        component: EmbeddingConfig.Component, embeds: [Float], mask: [Int32], hidden: Int
    ) -> [String: EmbeddingTensor] {
        let length = mask.count
        var inputs = ["inputs_embeds": EmbeddingTensor(float32: embeds, shape: [1, length, hidden])]
        switch component.maskEncoding ?? .attentionMask {
        case .attentionMask:
            inputs["attention_mask"] = EmbeddingTensor(int32: mask, shape: [1, length])
        case .additiveBias:
            let valid = Float(max(mask.reduce(0) { $0 + Int($1) }, 1))
            var bias = [Float](repeating: 0, count: length * length)
            for key in 0..<length where mask[key] == 0 {
                bias.replaceSubrange(
                    (key * length)..<((key + 1) * length), with: repeatElement(additiveMaskBias, count: length))
            }
            inputs["bias"] = EmbeddingTensor(float32: bias, shape: [1, length, 1, length])
            inputs["pool_weights"] = EmbeddingTensor(float32: mask.map { Float($0) / valid }, shape: [1, length])
        }
        return inputs
    }

    private func finish(
        _ vector: [Float], width: Int, truncated: Bool, tokens: Int, tower: EmbeddingModality,
        component: String? = nil
    ) throws -> Embedding {
        guard EmbeddingMath.isFinite(vector) else {
            let name = component ?? config.tower(for: tower)?.pipeline.last ?? ""
            throw failure(name, "produced a non-finite embedding (NaN or Inf)")
        }
        var values = vector
        if width < values.count {
            values = config.renormalize ? EmbeddingMath.truncate(values, to: width) : Array(values.prefix(width))
        }
        return Embedding(space: space, vector: values, truncatedInput: truncated, tokenCount: tokens)
    }

    // MARK: - Private: image

    /// Image tower: patchify, image encoder to soft tokens, token table, scatter, then the encoder.
    private func runImage(_ image: CGImage) async throws -> (vector: [Float], tokens: Int) {
        guard let tower = config.tower(for: .image) else { throw EmbeddingError.unsupportedModality(.image) }
        guard tower.pipeline.count == 3 else {
            throw EmbeddingError.invalidBundle("unsupported image pipeline \(tower.pipeline)")
        }
        let (imageName, tableName, encoderName) = (tower.pipeline[0], tower.pipeline[1], tower.pipeline[2])
        let imageEncoder = try requireComponent(imageName)
        let table = try requireComponent(tableName)
        let encoder = try requireComponent(encoderName)
        _ = try imageTokenIds(numSoftTokens: 0)  // fail on missing token ids before running components

        let patcher = ImagePatcher(config.preprocessing?.image ?? .init())
        let patches = try patcher.patches(from: image)
        let imageSpec = config.component(imageName)
        let imageFunction = imageSpec?.function ?? "main"
        let soft = try await run(
            imageName, imageEncoder, function: imageFunction,
            inputs: [
                "pixel_values": EmbeddingTensor(
                    float32: patches.pixelValues, shape: [1, patcher.maxPatches, patcher.patchDimension]),
                "position_ids": EmbeddingTensor(int32: patches.positionIds, shape: [1, patcher.maxPatches, 2]),
            ],
            output: imageSpec?.output)
        guard patcher.maxSoftTokens > 0, soft.count % patcher.maxSoftTokens == 0, !soft.isEmpty else {
            throw failure(imageName, function: imageFunction, "returned \(soft.count) values for \(patcher.maxSoftTokens) soft tokens")
        }
        let hidden = soft.count / patcher.maxSoftTokens

        let tokenIds = try imageTokenIds(numSoftTokens: patches.numSoftTokens)
        let length = tokenIds.count
        // Static encoder traces: run at the smallest batch-1 function that fits, right-padded.
        let (function, paddedLength) = try encoderFunction(encoderName, tokens: length)
        let padded = tokenIds + [Int32](repeating: padTokenId, count: paddedLength - length)
        let mask = (0..<paddedLength).map { Int32($0 < length ? 1 : 0) }
        let placeholder = Int32(config.merge?.placeholderIds?["image"] ?? 0)
        let vector = try await runTableAndEncoder(
            ids: padded, mask: mask, function: function, table: (tableName, table),
            encoder: (encoderName, encoder), output: tower.output,
            merge: { embeds, tableHidden in
                guard tableHidden == hidden else {
                    throw failure(
                        tableName, "hidden width \(tableHidden) differs from the image encoder's \(hidden)")
                }
                return Self.scatter(embeds, ids: padded, placeholder: placeholder, softTokens: soft, hidden: hidden)
            })
        return (vector, length)
    }

    /// Pad id for right-padded encoder rows: `special_tokens.pad_token_id`, else the tokenizer's, else 0.
    private var padTokenId: Int32 {
        Int32(config.text?.specialTokens?.padTokenId ?? tokenizerPadId ?? 0)
    }

    /// `<bos> <boi> <image>*N <eoi> <eos>` (BOS/EOS per `text.special_tokens`).
    func imageTokenIds(numSoftTokens: Int) throws -> [Int32] {
        func required(_ value: Int?, _ label: String) throws -> Int32 {
            guard let value else { throw EmbeddingError.invalidBundle("no token id for '\(label)' in metadata") }
            return Int32(value)
        }
        let image = try required(config.merge?.placeholderIds?["image"], "image placeholder")
        let boi = try required(config.merge?.tokenIds?["boi"], "boi")
        let eoi = try required(config.merge?.tokenIds?["eoi"], "eoi")
        let special = config.text?.specialTokens
        var ids: [Int32] = []
        if special?.bos == true { ids.append(try required(special?.bosTokenId ?? tokenizer?.bosTokenId, "bos")) }
        ids.append(boi)
        ids += [Int32](repeating: image, count: numSoftTokens)
        ids.append(eoi)
        if special?.eos == true { ids.append(try required(special?.eosTokenId ?? tokenizer?.eosTokenId, "eos")) }
        return ids
    }

    /// Replaces placeholder rows of `embeds` (`[L, H]`) with soft tokens in order.
    static func scatter(_ embeds: [Float], ids: [Int32], placeholder: Int32, softTokens: [Float], hidden: Int)
        -> [Float]
    {
        var out = embeds
        var next = 0
        for (row, id) in ids.enumerated() where id == placeholder {
            let src = next * hidden
            guard src + hidden <= softTokens.count else { break }
            out.replaceSubrange((row * hidden)..<((row + 1) * hidden), with: softTokens[src..<(src + hidden)])
            next += 1
        }
        return out
    }
}

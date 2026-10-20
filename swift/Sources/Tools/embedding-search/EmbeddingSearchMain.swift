// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIEmbeddings
import CoreAIShared
import CoreAIVectorIndex
import Foundation

@main
struct EmbeddingSearch: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embedding-search",
        abstract: "Image + text semantic search with a Core AI embedding bundle.",
        subcommands: [IndexCommand.self, QueryCommand.self, EmbedCommand.self]
    )
}

/// Options shared by every subcommand.
struct ModelOptions: ParsableArguments {
    @Option(name: .long, help: "Path to the embedding bundle (metadata.json kind \"embedding\").")
    var model: String

    @Flag(
        name: .customLong("clear-coreai-cache"),
        help: "Clear Core AI cached specialization for this bundle before loading (forces re-specialization)")
    var clearCoreAICache = false

    @Flag(help: "Print load / embed / search timings to stderr.")
    var verbose = false

    var bundleURL: URL { expand(model) }

    /// Reads the bundle metadata (no asset is touched), clearing the specialization cache first if asked.
    func openModel() throws -> EmbeddingModel {
        if verbose { CLILogger.level = 1 }
        if clearCoreAICache {
            let cleared = try PreparedModel.clearCache(at: bundleURL)
            FileHandle.standardError.write(
                Data("embedding-search: cleared specialization cache for \(cleared.count) asset(s)\n".utf8))
        }
        return try EmbeddingModel(bundleAt: bundleURL)
    }

    /// Loads what `modalities` need and returns the time it took.
    func load(_ model: EmbeddingModel, modalities: Set<EmbeddingModality>) async throws -> Duration {
        let start = ContinuousClock.now
        try await model.load(modalities: modalities)
        return ContinuousClock.now - start
    }
}

func expand(_ path: String) -> URL { URL(fileURLWithPath: (path as NSString).expandingTildeInPath) }

extension QueryKind: ExpressibleByArgument {}

/// Storage element type for `index --storage`.
enum StorageType: String, CaseIterable, ExpressibleByArgument {
    case float32
    case float16

    var elementType: VectorElementType { self == .float16 ? .float16 : .float32 }
}

/// Runs `body`, printing errors as one clear line on stderr and exiting non-zero.
private func runReportingErrors(_ body: () async throws -> Void) async throws {
    do {
        try await body()
    } catch let error as ExitCode {
        throw error
    } catch {
        FileHandle.standardError.write(Data("embedding-search: error: \(error.localizedDescription)\n".utf8))
        throw ExitCode.failure
    }
}

// MARK: - index

struct IndexCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "index",
        abstract: "Embed a folder of images and/or a text file into a vector index (.cvi)."
    )

    @OptionGroup var modelOptions: ModelOptions

    @Option(name: .long, help: "Output index path (.cvi). Overwritten atomically.")
    var out: String

    @Option(name: .long, help: "Folder searched recursively for jpg/jpeg/png/heic/heif images (or one image file).")
    var images: String?

    @Option(
        name: .long,
        help: "Text documents: .jsonl ({\"text\", \"id\"?, \"title\"?} per line) or plain text (one per line).")
    var texts: String?

    @Option(name: .long, help: "Embedding width; one of the bundle's Matryoshka dimensions (default: native).")
    var dimensions: Int?

    @Option(
        name: .long,
        help: "Prompt applied to documents (default 'document' when the bundle defines it; '' for none).")
    var promptName: String?

    @Option(name: .long, help: "Vector storage: float32 or float16.")
    var storage: StorageType = .float32

    func validate() throws {
        guard images != nil || texts != nil else {
            throw ValidationError("Provide --images and/or --texts.")
        }
        if let dimensions, dimensions < 1 { throw ValidationError("--dimensions must be positive.") }
    }

    func run() async throws {
        try await runReportingErrors {
            let model = try modelOptions.openModel()
            let imageURLs = try images.map { try ImageDiscovery.discover(at: expand($0)) } ?? []
            let documents = try texts.map { try TextCorpus.load(from: expand($0)) } ?? []
            print("found \(imageURLs.count) images, \(documents.count) text documents")
            let builder = IndexBuilder(
                model: model, bundle: modelOptions.model, dimensions: dimensions ?? model.dimension,
                documentPrompt: model.resolvePrompt(requested: promptName, defaultName: "document"),
                elementType: storage.elementType)
            _ = try builder.preflight(imageCount: imageURLs.count, textCount: documents.count)
            // Only the towers this corpus needs, loaded concurrently.
            var modalities: Set<EmbeddingModality> = []
            if !imageURLs.isEmpty { modalities.insert(.image) }
            if !documents.isEmpty { modalities.insert(.text) }
            let loadTime = try await modelOptions.load(model, modalities: modalities)
            print(String(format: "model loaded in %.2f s", loadTime.inSeconds))
            let (index, summary) = try await builder.build(images: imageURLs, texts: documents)
            let outURL = expand(out)
            try index.save(to: outURL)
            let imageRate = summary.imageSeconds > 0 ? Double(summary.images) / summary.imageSeconds : 0
            let textRate = summary.textSeconds > 0 ? Double(summary.texts) / summary.textSeconds : 0
            let d = index.descriptor
            print(
                String(
                    format: "indexed %d images (%.1f/s), %d texts (%.1f/s), %d skipped -> %@",
                    summary.images, imageRate, summary.texts, textRate, summary.skippedImages.count, outURL.path))
            print("descriptor: space \(d.space), \(d.dimension)-d, \(d.elementType.rawValue)")
        }
    }
}

// MARK: - query

struct QueryCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "query",
        abstract: "Search an index with a text or image query, or with many queries in one process (one model load).",
        discussion: """
            --queries-json takes a JSON array of {"id"?, "text" | "image", "k"?, "kind"?, "prompt_name"?} objects \
            ('-' reads stdin), and --output-json writes one {"id", "output", "timing", "error"} record per query.
            """
    )

    @OptionGroup var modelOptions: ModelOptions

    @Option(name: .long, help: "Index path (.cvi) written by `index`.")
    var index: String

    @Option(name: .long, help: "Text query.")
    var text: String?

    @Option(name: .long, help: "Image query (file path).")
    var image: String?

    @Option(
        name: .customLong("queries-json"),
        help:
            "JSON array of queries ({\"id\"?, \"text\" | \"image\", \"k\"?, \"kind\"?, \"prompt_name\"?}); '-' = stdin."
    )
    var queriesJson: String?

    @Option(name: .customLong("output-json"), help: "Write the result records (JSON array) to this path.")
    var outputJson: String?

    @Option(name: .long, help: "Number of results.")
    var k: Int = 10

    @Option(name: .long, help: "Restrict results to one kind: image, text or all.")
    var kind: QueryKind = .all

    @Option(
        name: .long,
        help: "Prompt for text queries (default 'query' when the bundle defines it; '' for none).")
    var promptName: String?

    @Option(name: .long, help: "Query width; must equal the index width (default: the index width).")
    var dimensions: Int?

    func validate() throws {
        guard [text != nil, image != nil, queriesJson != nil].filter({ $0 }).count == 1 else {
            throw ValidationError("Provide exactly one of --text, --image or --queries-json.")
        }
        guard k >= 1 else { throw ValidationError("--k must be >= 1.") }
    }

    func run() async throws {
        var failed = 0
        try await runReportingErrors {
            let start = ContinuousClock.now
            let model = try modelOptions.openModel()
            let runner = QueryRunner(model: model, bundle: modelOptions.model)
            let vectorIndex = try runner.openIndex(at: expand(index), dimensions: dimensions)
            let batch = QueryBatch(
                runner: runner, index: vectorIndex, k: k, kind: kind,
                promptName: model.resolvePrompt(requested: promptName, defaultName: "query"))

            let queries: [Result<QueryRecord, RecordFailure>]
            if let queriesJson {
                let contents: String
                if queriesJson == "-" {
                    contents = String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
                } else {
                    let url = expand(queriesJson)
                    guard let read = try? String(contentsOf: url, encoding: .utf8) else {
                        throw EmbeddingSearchError.notFound(url.path)
                    }
                    contents = read
                }
                queries = try QueryInputParser.parse(contents: contents)
                // Every tower the queries need, loaded concurrently before the first one runs.
                let modalities = Set(queries.compactMap { try? $0.get().modality })
                    .filter { model.supportedModalities.contains($0) }
                if !modalities.isEmpty { _ = try await modelOptions.load(model, modalities: modalities) }
            } else {
                let record = QueryRecord(id: nil, source: image.map { .image(path: $0) } ?? .text(text ?? ""))
                try model.require(record.modality, bundle: modelOptions.model)
                queries = [.success(record)]
            }

            var records: [JSONRecord<QueryOutput>] = []
            for (number, query) in queries.enumerated() {
                let (record, hits) = await batch.run(query)
                records.append(record)
                if record.error != nil { failed += 1 }
                guard outputJson == nil else { continue }
                if queriesJson != nil {
                    print("# query \(record.id ?? String(number + 1))")
                    print(hits.map { OutputFormatter.table($0) } ?? "error: \(record.error?.message ?? "")")
                } else if let hits {
                    print(OutputFormatter.table(hits))
                    print(
                        String(
                            format: "(%d vectors, %.2f s incl. model load%@)", vectorIndex.count,
                            (ContinuousClock.now - start).inSeconds,
                            record.timing["load_ms"].map { String(format: " %.2f s", $0 / 1000) } ?? ""))
                } else {
                    throw RecordFailure(id: nil, message: record.error?.message ?? "query failed")
                }
            }
            if let outputJson {
                let url = expand(outputJson)
                do {
                    try Data((try JSONOutput.encode(records) + "\n").utf8).write(to: url, options: .atomic)
                } catch {
                    throw EmbeddingSearchError.notWritable(url.path)
                }
                FileHandle.standardError.write(
                    Data("embedding-search: wrote \(records.count) record(s) to \(url.path)\n".utf8))
            }
            CLILogger.log(
                String(format: "process total %.1f ms", (ContinuousClock.now - start).inMilliseconds),
                component: "embedding-search")
        }
        if failed > 0 {
            FileHandle.standardError.write(
                Data("embedding-search: error: \(EmbeddingSearchError.recordsFailed(failed))\n".utf8))
            throw ExitCode.failure
        }
    }
}

// MARK: - embed

struct EmbedCommand: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "embed",
        abstract: "Print the embedding of one text or image."
    )

    @OptionGroup var modelOptions: ModelOptions

    @Option(name: .long, help: "Text to embed.")
    var text: String?

    @Option(name: .long, help: "Image to embed (file path).")
    var image: String?

    @Option(name: .long, help: "Prompt for text (default: none; e.g. 'query' or 'document').")
    var promptName: String?

    @Option(name: .long, help: "Embedding width; one of the bundle's Matryoshka dimensions (default: native).")
    var dimensions: Int?

    func validate() throws {
        guard (text == nil) != (image == nil) else {
            throw ValidationError("Provide exactly one of --text or --image.")
        }
        if let dimensions, dimensions < 1 { throw ValidationError("--dimensions must be positive.") }
    }

    func run() async throws {
        try await runReportingErrors {
            let model = try modelOptions.openModel()
            let modality: EmbeddingModality = image == nil ? .text : .image
            try model.require(modality, bundle: modelOptions.model)
            _ = try await modelOptions.load(model, modalities: [modality])
            let input: EmbeddingInput =
                if let image { .image(try CGImageUtils.load(from: image)) } else { .text(text ?? "") }
            let embedStart = ContinuousClock.now
            let embedding = try await model.embed(
                input, promptName: promptName.flatMap { $0.isEmpty ? nil : $0 }, dimensions: dimensions)
            CLILogger.log(
                String(format: "embed %.1f ms", (ContinuousClock.now - embedStart).inMilliseconds),
                component: "embedding-search")
            print(
                "# space \(embedding.space) dimension \(embedding.dimension) tokens \(embedding.tokenCount)"
                    + " truncated \(embedding.truncatedInput)")
            print(embedding.vector.map { String($0) }.joined(separator: " "))
        }
    }
}

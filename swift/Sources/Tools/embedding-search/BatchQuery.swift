// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIEmbeddings
import CoreAIShared
import CoreAIVectorIndex
import Foundation

/// One query: from `--text` / `--image`, or one object of `query --queries-json`
/// (`{"id"?, "text" | "image", "k"?, "kind"?, "prompt_name"?}`; unset fields take the CLI values).
struct QueryRecord: Equatable, Sendable {
    enum Source: Equatable, Sendable {
        case text(String)
        case image(path: String)
    }

    var id: String?
    var source: Source
    var k: Int?
    var kind: QueryKind?
    /// `nil`: CLI default; `""`: no prompt.
    var promptName: String?

    var modality: EmbeddingModality {
        if case .image = source { return .image }
        return .text
    }
}

/// `{"hits": [...]}` under a record's `output`.
struct QueryOutput: Encodable, Equatable {
    var hits: [QueryHit]
}

/// Parses `query --queries-json`: a JSON array of query objects (like image-segmenter's `--queries-json`).
/// A malformed query becomes a failed record (it still gets an output record, in input order).
enum QueryInputParser {
    /// The queries of `contents`. Throws only when the whole document is unreadable.
    static func parse(contents: String) throws -> [Result<QueryRecord, RecordFailure>] {
        let array: [Any]
        do {
            guard let parsed = try JSONSerialization.jsonObject(with: Data(contents.utf8)) as? [Any] else {
                throw EmbeddingSearchError.invalidQueries("expected a JSON array of query objects")
            }
            array = parsed
        } catch let error as EmbeddingSearchError {
            throw error
        } catch {
            throw EmbeddingSearchError.invalidQueries("invalid JSON (\(error.localizedDescription))")
        }
        return array.enumerated().map { parse(value: $1, label: "query \($0 + 1)") }
    }

    /// One query object `{"id"?, "text" | "image", "k"?, "kind"?, "prompt_name"?}`; `label` prefixes errors.
    static func parse(value: Any, label: String) -> Result<QueryRecord, RecordFailure> {
        guard let object = value as? [String: Any] else {
            return .failure(RecordFailure(id: nil, message: "\(label): expected a JSON object"))
        }
        let id: String?
        switch object["id"] {
        case nil, is NSNull: id = nil
        case let value as String: id = value
        case let value as NSNumber where CFGetTypeID(value) != CFBooleanGetTypeID(): id = value.stringValue
        default: return .failure(RecordFailure(id: nil, message: "\(label): \"id\" must be a string or number"))
        }
        func fail(_ reason: String) -> Result<QueryRecord, RecordFailure> {
            .failure(RecordFailure(id: id, message: "\(label): \(reason)"))
        }
        let text = object["text"] as? String
        let image = object["image"] as? String
        guard (text == nil) != (image == nil) else {
            return fail("expected exactly one of \"text\" or \"image\" (string)")
        }
        var record = QueryRecord(id: id, source: text.map { .text($0) } ?? .image(path: image ?? ""))
        if let value = object["k"] {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let k = value as? Int,
                k >= 1
            else { return fail("\"k\" must be an integer >= 1") }
            record.k = k
        }
        if let value = object["kind"] {
            guard let name = value as? String, let kind = QueryKind(rawValue: name) else {
                return fail("\"kind\" must be one of \(QueryKind.allCases.map(\.rawValue).joined(separator: ", "))")
            }
            record.kind = kind
        }
        if let value = object["prompt_name"], !(value is NSNull) {
            guard let name = value as? String else { return fail("\"prompt_name\" must be a string") }
            record.promptName = name
        }
        return .success(record)
    }
}

struct RecordFailure: Error, Equatable, LocalizedError {
    var id: String?
    var message: String

    var errorDescription: String? { message }
}

/// Runs query records against one loaded model and index; one output record per input record.
struct QueryBatch {
    var runner: QueryRunner
    var index: VectorIndex
    var k: Int
    var kind: QueryKind
    var promptName: String?

    /// Process `record`; failures become records with `error` set.
    func run(_ parsed: Result<QueryRecord, RecordFailure>) async -> (JSONRecord<QueryOutput>, [QueryHit]?) {
        let record: QueryRecord
        switch parsed {
        case .failure(let failure):
            return (JSONRecord(id: failure.id, output: nil, timing: [:], error: .init(message: failure.message)), nil)
        case .success(let value): record = value
        }
        let start = ContinuousClock.now
        var timing: [String: Double] = [:]
        do {
            try runner.model.require(record.modality, bundle: runner.bundle)
            let loadStart = ContinuousClock.now
            try await runner.model.load(modalities: [record.modality])
            let loaded = ContinuousClock.now - loadStart
            if loaded > .milliseconds(1) { timing["load_ms"] = JSONOutput.milliseconds(loaded) }

            let input: EmbeddingInput
            switch record.source {
            case .text(let text): input = .text(text)
            case .image(let path):
                input = .image(try CGImageUtils.load(from: path))
            }
            let prompt: String? =
                switch record.promptName {
                case nil: promptName
                case let name?: name.isEmpty ? nil : name
                }
            let embedStart = ContinuousClock.now
            let query = try await runner.embedQuery(input, promptName: prompt, dimensions: index.descriptor.dimension)
            let searchStart = ContinuousClock.now
            let hits = try runner.search(query, in: index, k: record.k ?? k, kind: record.kind ?? kind)
            let end = ContinuousClock.now
            timing["embed_ms"] = JSONOutput.milliseconds(searchStart - embedStart)
            timing["search_ms"] = JSONOutput.milliseconds(end - searchStart)
            timing["total_ms"] = JSONOutput.milliseconds(end - start)
            CLILogger.log(
                String(
                    format: "query %@: embed %.1f ms, search %.2f ms, total %.1f ms", record.id ?? "-",
                    timing["embed_ms"]!, timing["search_ms"]!, timing["total_ms"]!),
                component: "embedding-search")
            return (JSONRecord(id: record.id, output: QueryOutput(hits: hits), timing: timing, error: nil), hits)
        } catch {
            timing["total_ms"] = JSONOutput.milliseconds(ContinuousClock.now - start)
            return (
                JSONRecord(
                    id: record.id, output: nil, timing: timing, error: .init(message: error.localizedDescription)), nil
            )
        }
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIEmbeddings
import CoreAIVectorIndex
import Foundation

/// Kind of an indexed item; stored as the `kind` attribute and in the payload.
enum ItemKind: String, Codable, Sendable {
    case image
    case text
}

/// JSON payload stored with every vector and returned with search results.
struct SearchPayload: Codable, Equatable, Sendable {
    var kind: ItemKind
    var path: String?
    var text: String?
    var title: String?
}

/// One text document to index.
struct TextDocument: Equatable, Sendable {
    var id: String
    var text: String
    var title: String?
}

/// Builds ids, payloads and attributes for index entries.
enum IndexEntryFactory {
    static let kindAttribute = "kind"

    static func imageEntry(url: URL, vector: [Float]) throws -> VectorIndexEntry {
        let path = url.standardizedFileURL.path
        return VectorIndexEntry(
            id: path, vector: vector, payload: try jsonPayload(SearchPayload(kind: .image, path: path)),
            attributes: [kindAttribute: ItemKind.image.rawValue])
    }

    static func textEntry(document: TextDocument, vector: [Float]) throws -> VectorIndexEntry {
        VectorIndexEntry(
            id: document.id, vector: vector,
            payload: try jsonPayload(SearchPayload(kind: .text, text: document.text, title: document.title)),
            attributes: [kindAttribute: ItemKind.text.rawValue])
    }

    /// Attribute filter for `--kind`; empty searches everything.
    static func filter(for kind: QueryKind) -> [String: String] {
        kind == .all ? [:] : [kindAttribute: kind.rawValue]
    }

    /// The payload stored with a search result, when it decodes.
    static func payload(of result: SearchResult) -> SearchPayload? {
        result.payload.flatMap { try? JSONDecoder().decode(SearchPayload.self, from: $0) }
    }

    private static func jsonPayload(_ value: some Encodable) throws -> Data {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        return try encoder.encode(value)
    }
}

/// Recursive image discovery.
enum ImageDiscovery {
    static let extensions: Set<String> = ["jpg", "jpeg", "png", "heic", "heif"]

    static func isImage(_ url: URL) -> Bool { extensions.contains(url.pathExtension.lowercased()) }

    /// Image files under `root` (or `root` itself when it is an image file), hidden files skipped,
    /// sorted by path so runs are reproducible.
    static func discover(at root: URL) throws -> [URL] {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory) else {
            throw EmbeddingSearchError.notFound(root.path)
        }
        guard isDirectory.boolValue else {
            guard isImage(root) else { throw EmbeddingSearchError.notAnImage(root.path) }
            return [root.standardizedFileURL]
        }
        guard
            let enumerator = FileManager.default.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey], options: [.skipsHiddenFiles])
        else { throw EmbeddingSearchError.notFound(root.path) }
        var found: [URL] = []
        for case let url as URL in enumerator where isImage(url) {
            let isFile = (try? url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) ?? false
            if isFile { found.append(url.standardizedFileURL) }
        }
        return found.sorted { $0.path < $1.path }
    }
}

/// Text corpus ingestion: `.jsonl` (one `{"text", "id"?, "title"?}` object per line) or plain text
/// (one document per non-empty line). Default ids are `<file name>:<line number>` (1-based).
enum TextCorpus {
    static func load(from url: URL) throws -> [TextDocument] {
        let contents: String
        do {
            contents = try String(contentsOf: url, encoding: .utf8)
        } catch {
            throw EmbeddingSearchError.notFound(url.path)
        }
        let name = url.lastPathComponent
        return try url.pathExtension.lowercased() == "jsonl"
            ? parseJSONLines(contents, sourceName: name) : parseLines(contents, sourceName: name)
    }

    static func parseLines(_ contents: String, sourceName: String) -> [TextDocument] {
        var documents: [TextDocument] = []
        for (index, line) in lines(contents).enumerated() {
            let text = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { continue }
            documents.append(TextDocument(id: "\(sourceName):\(index + 1)", text: text))
        }
        return documents
    }

    static func parseJSONLines(_ contents: String, sourceName: String) throws -> [TextDocument] {
        var documents: [TextDocument] = []
        var seen: [String: Int] = [:]
        for (index, line) in lines(contents).enumerated() {
            let lineNumber = index + 1
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { continue }
            let object: [String: Any]
            do {
                guard let parsed = try JSONSerialization.jsonObject(with: Data(trimmed.utf8)) as? [String: Any]
                else {
                    throw EmbeddingSearchError.invalidTextRecord(
                        file: sourceName, line: lineNumber, reason: "expected a JSON object")
                }
                object = parsed
            } catch let error as EmbeddingSearchError {
                throw error
            } catch {
                throw EmbeddingSearchError.invalidTextRecord(
                    file: sourceName, line: lineNumber, reason: "invalid JSON (\(error.localizedDescription))")
            }
            guard let text = object["text"] as? String,
                !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw EmbeddingSearchError.invalidTextRecord(
                    file: sourceName, line: lineNumber, reason: "missing or empty \"text\" string")
            }
            let id: String
            switch object["id"] {
            case nil, is NSNull: id = "\(sourceName):\(lineNumber)"
            case let value as String where !value.isEmpty: id = value
            case let value as NSNumber where CFGetTypeID(value) != CFBooleanGetTypeID():
                id = value.stringValue
            default:
                throw EmbeddingSearchError.invalidTextRecord(
                    file: sourceName, line: lineNumber, reason: "\"id\" must be a non-empty string or a number")
            }
            if let first = seen[id] {
                throw EmbeddingSearchError.invalidTextRecord(
                    file: sourceName, line: lineNumber, reason: "duplicate id '\(id)' (first on line \(first))")
            }
            seen[id] = lineNumber
            var title: String?
            switch object["title"] {
            case nil, is NSNull: title = nil
            case let value as String: title = value
            default:
                throw EmbeddingSearchError.invalidTextRecord(
                    file: sourceName, line: lineNumber, reason: "\"title\" must be a string")
            }
            documents.append(TextDocument(id: id, text: text, title: title))
        }
        return documents
    }

    /// Splits on newlines (`\r\n` is one `Character`) keeping empty lines so line numbers match the file.
    private static func lines(_ contents: String) -> [Substring] {
        contents.split(omittingEmptySubsequences: false, whereSeparator: \.isNewline)
    }
}

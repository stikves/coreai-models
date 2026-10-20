// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import ArgumentParser
import CoreAIEmbeddings
import CoreAIVectorIndex
import Foundation
import TestUtilities
import Testing

@testable import embedding_search

@Suite("embedding-search argument parsing")
struct EmbeddingSearchArgumentTests {
    @Test("index parses options and defaults")
    func indexDefaults() throws {
        let cmd = try IndexCommand.parse(["--model", "/b", "--out", "/o.cvi", "--images", "/imgs"])
        #expect(cmd.modelOptions.model == "/b")
        #expect(cmd.out == "/o.cvi")
        #expect(cmd.images == "/imgs")
        #expect(cmd.texts == nil)
        #expect(cmd.dimensions == nil)
        #expect(cmd.storage == .float32)

        let full = try IndexCommand.parse([
            "--model", "/b", "--out", "/o.cvi", "--texts", "/t.jsonl", "--dimensions", "256",
            "--storage", "float16", "--prompt-name", "",
        ])
        #expect(full.dimensions == 256)
        #expect(full.storage.elementType == .float16)
        #expect(full.promptName == "")
    }

    @Test("index validation rejects missing inputs and bad numbers")
    func indexValidation() {
        #expect(throws: (any Error).self) { try IndexCommand.parse(["--model", "/b", "--out", "/o.cvi"]) }
        #expect(throws: (any Error).self) {
            try IndexCommand.parse(["--model", "/b", "--out", "/o", "--texts", "/t", "--dimensions", "-1"])
        }
        #expect(throws: (any Error).self) { try IndexCommand.parse(["--out", "/o", "--texts", "/t"]) }
    }

    @Test("query requires exactly one of --text / --image; --kind and --k parse")
    func queryValidation() throws {
        let cmd = try QueryCommand.parse(["--model", "/b", "--index", "/i.cvi", "--text", "cats"])
        #expect(cmd.k == 10)
        #expect(cmd.kind == .all)
        let img = try QueryCommand.parse([
            "--model", "/b", "--index", "/i", "--image", "/x.png", "--k", "3", "--kind", "image",
        ])
        #expect(img.kind == .image)
        #expect(img.k == 3)

        #expect(throws: (any Error).self) { try QueryCommand.parse(["--model", "/b", "--index", "/i"]) }
        #expect(throws: (any Error).self) {
            try QueryCommand.parse(["--model", "/b", "--index", "/i", "--text", "a", "--image", "/x.png"])
        }
        #expect(throws: (any Error).self) {
            try QueryCommand.parse(["--model", "/b", "--index", "/i", "--text", "a", "--k", "0"])
        }
        #expect(throws: (any Error).self) {
            try QueryCommand.parse(["--model", "/b", "--index", "/i", "--text", "a", "--kind", "audio"])
        }
    }

    @Test("embed requires exactly one input; root dispatches subcommands")
    func embedAndRoot() throws {
        let cmd = try EmbedCommand.parse(["--model", "/b", "--text", "hi", "--dimensions", "128"])
        #expect(cmd.dimensions == 128)
        #expect(throws: (any Error).self) { try EmbedCommand.parse(["--model", "/b"]) }

        let root = try EmbeddingSearch.parseAsRoot(["query", "--model", "/b", "--index", "/i", "--text", "x"])
        #expect(root is QueryCommand)
        #expect(try EmbeddingSearch.parseAsRoot(["embed", "--model", "/b", "--image", "/x.png"]) is EmbedCommand)
    }
}

@Suite("embedding-search text ingestion")
struct TextCorpusTests {
    @Test("plain text: one document per non-empty line, ids carry file line numbers")
    func plainLines() {
        let docs = TextCorpus.parseLines("first doc\n\n  second doc  \r\nthird\n", sourceName: "notes.txt")
        #expect(
            docs == [
                TextDocument(id: "notes.txt:1", text: "first doc"),
                TextDocument(id: "notes.txt:3", text: "second doc"),
                TextDocument(id: "notes.txt:4", text: "third"),
            ])
    }

    @Test("jsonl: id string / number / default, title, blank lines")
    func jsonLines() throws {
        let contents = """
            {"id": "a", "text": "alpha", "title": "A"}

            {"id": 7, "text": "beta"}
            {"text": "gamma", "title": null}
            """
        let docs = try TextCorpus.parseJSONLines(contents, sourceName: "c.jsonl")
        #expect(
            docs == [
                TextDocument(id: "a", text: "alpha", title: "A"),
                TextDocument(id: "7", text: "beta"),
                TextDocument(id: "c.jsonl:4", text: "gamma"),
            ])
    }

    @Test(
        "jsonl: malformed records fail with file:line",
        arguments: [
            ("{\"text\": \"a\"}\nnot json", 2),
            ("{\"id\": \"x\"}", 1),
            ("{\"text\": \"  \"}", 1),
            ("[1, 2]", 1),
            ("{\"id\": true, \"text\": \"a\"}", 1),
            ("{\"id\": \"d\", \"text\": \"a\"}\n{\"id\": \"d\", \"text\": \"b\"}", 2),
            ("{\"text\": \"a\", \"title\": 3}", 1),
        ])
    func jsonLinesErrors(contents: String, line: Int) {
        do {
            _ = try TextCorpus.parseJSONLines(contents, sourceName: "bad.jsonl")
            Issue.record("expected an error")
        } catch let EmbeddingSearchError.invalidTextRecord(file, errorLine, _) {
            #expect(file == "bad.jsonl")
            #expect(errorLine == line)
        } catch {
            Issue.record("unexpected error \(error)")
        }
    }

    @Test("load dispatches on extension")
    func loadByExtension() throws {
        let dir = try TestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        try TestFiles.write("{\"text\": \"x\"}\n", to: dir.appending(path: "t.JSONL"))
        try TestFiles.write("{\"text\": \"x\"}\n", to: dir.appending(path: "t.txt"))
        #expect(try TextCorpus.load(from: dir.appending(path: "t.JSONL")) == [TextDocument(id: "t.JSONL:1", text: "x")])
        #expect(try TextCorpus.load(from: dir.appending(path: "t.txt"))[0].text == "{\"text\": \"x\"}")
        #expect(throws: EmbeddingSearchError.self) { try TextCorpus.load(from: dir.appending(path: "missing.txt")) }
    }
}

@Suite("embedding-search image discovery and entries")
struct ImageDiscoveryAndEntryTests {
    @Test("recursive, extension-filtered, hidden skipped, sorted")
    func discovery() throws {
        let dir = try TestFiles.temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: dir) }
        for name in ["b.PNG", "a/c.jpg", "a/deep/d.heic", "e.jpeg", ".hidden.png", "notes.txt", ".git/x.png"] {
            try TestFiles.write("x", to: dir.appending(path: name))
        }
        try FileManager.default.createDirectory(
            at: dir.appending(path: "folder.png"), withIntermediateDirectories: true)
        let found = try ImageDiscovery.discover(at: dir).map {
            $0.path.replacingOccurrences(of: dir.path + "/", with: "")
        }
        #expect(found.sorted() == found)
        #expect(Set(found) == ["b.PNG", "a/c.jpg", "a/deep/d.heic", "e.jpeg"])

        #expect(try ImageDiscovery.discover(at: dir.appending(path: "e.jpeg")).count == 1)
        #expect(throws: EmbeddingSearchError.self) { try ImageDiscovery.discover(at: dir.appending(path: "notes.txt")) }
        #expect(throws: EmbeddingSearchError.self) { try ImageDiscovery.discover(at: dir.appending(path: "nope")) }
    }

    @Test("ids, payloads, attributes and the kind filter")
    func entries() throws {
        let url = URL(fileURLWithPath: "gallery/x/../pics/cat.png")
        let path = url.standardizedFileURL.path
        let image = try IndexEntryFactory.imageEntry(url: url, vector: [1, 0])
        #expect(image.id == path)
        #expect(image.attributes == ["kind": "image"])
        let imagePayload = try JSONDecoder().decode(SearchPayload.self, from: try #require(image.payload))
        #expect(imagePayload == SearchPayload(kind: .image, path: path))

        let text = try IndexEntryFactory.textEntry(
            document: TextDocument(id: "d1", text: "hello", title: "T"), vector: [0, 1])
        #expect(text.id == "d1")
        #expect(text.attributes == ["kind": "text"])
        let textPayload = try JSONDecoder().decode(SearchPayload.self, from: try #require(text.payload))
        #expect(textPayload == SearchPayload(kind: .text, text: "hello", title: "T"))

        #expect(IndexEntryFactory.filter(for: .all).isEmpty)
        #expect(IndexEntryFactory.filter(for: .image) == ["kind": "image"])
        #expect(IndexEntryFactory.filter(for: .text) == ["kind": "text"])
    }

    @Test("prompt resolution: explicit wins, '' disables, default dropped when undefined")
    func prompts() throws {
        let model = try SearchTestModel.make(image: false, prompts: ["query": "query: "]).model
        #expect(model.resolvePrompt(requested: nil, defaultName: "query") == "query")
        #expect(model.resolvePrompt(requested: nil, defaultName: "document") == nil)
        #expect(model.resolvePrompt(requested: "", defaultName: "query") == nil)
        #expect(model.resolvePrompt(requested: "clustering", defaultName: "query") == "clustering")
    }

    @Test("chunking and an aligned result table")
    func formatting() {
        #expect([1, 2, 3, 4, 5].chunked(by: 2).map(Array.init) == [[1, 2], [3, 4], [5]])
        #expect([Int]().chunked(by: 3).isEmpty)
        let hits = [
            QueryHit(rank: 1, score: 0.9, id: "a.png", kind: "image", path: "a.png"),
            QueryHit(rank: 2, score: 0.5, id: "d", kind: "text", text: "line one\nline two"),
        ]
        #expect(
            OutputFormatter.table(hits).split(separator: "\n").map(String.init) == [
                "rank  score   kind   item",
                "1     0.9000  image  a.png",
                "2     0.5000  text   [d] line one line two",
            ])
        #expect(OutputFormatter.table([]) == "no results")
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIEmbeddings
import CoreAIVectorIndex
import Foundation
import TestUtilities
import Testing

@testable import embedding_search

@Suite("embedding-search --queries-json")
struct EmbeddingSearchBatchQueryTests {
    @Test("JSON array: fields, defaults, numeric ids; invalid entries become failed records in order")
    func parseArray() throws {
        let parsed = try QueryInputParser.parse(
            contents: """
                [{"id": "a", "text": "red", "k": 2, "kind": "image", "prompt_name": ""},
                 {"id": 7, "image": "/x.png"},
                 {"id": "bad", "text": "x", "image": "/y.png"},
                 {"text": "y", "k": true},
                 3]
                """)
        #expect(parsed.count == 5)
        #expect(
            try parsed[0].get()
                == QueryRecord(id: "a", source: .text("red"), k: 2, kind: .image, promptName: ""))
        #expect(try parsed[1].get() == QueryRecord(id: "7", source: .image(path: "/x.png")))
        #expect(throws: RecordFailure.self) { try parsed[2].get() }
        if case .failure(let failure) = parsed[2] {
            #expect(failure.id == "bad")
            #expect(failure.message.hasPrefix("query 3:"))
        }
        if case .failure(let failure) = parsed[3] { #expect(failure.message.contains("\"k\"")) }
        if case .failure(let failure) = parsed[4] { #expect(failure.message == "query 5: expected a JSON object") }
    }

    @Test("anything but a JSON array fails the whole file")
    func unreadableFile() {
        let notArray = EmbeddingSearchError.invalidQueries("expected a JSON array of query objects")
        #expect(throws: EmbeddingSearchError.self) { try QueryInputParser.parse(contents: "[{\"text\": 1},") }
        #expect(throws: notArray) { try QueryInputParser.parse(contents: "{\"text\": \"a\"}") }
        #expect(throws: EmbeddingSearchError.self) {
            try QueryInputParser.parse(contents: "{\"text\": \"a\"}\n{\"text\": \"b\"}")
        }
    }

    @Test("records: hits + *_ms timing, null error; failures keep their id and message")
    func records() async throws {
        let fake = try SearchTestModel.make(image: false)
        var index = try VectorIndex(descriptor: VectorIndexDescriptor(model: fake.model))
        let vectors = try await fake.model.embed([.text("red"), .text("blue")])
        try index.append(
            contentsOf: zip(["r", "b"], vectors).map { VectorIndexEntry(id: $0, vector: $1.vector) },
            space: fake.model.space)
        let batch = QueryBatch(
            runner: QueryRunner(model: fake.model, bundle: "bundle"), index: index, k: 2, kind: .all,
            promptName: "query")

        let (ok, hits) = await batch.run(.success(QueryRecord(id: "q1", source: .text("red"), k: 1)))
        #expect(hits?.map(\.id) == ["r"])
        #expect(ok.error == nil)
        #expect(Set(ok.timing.keys).isSuperset(of: ["embed_ms", "search_ms", "total_ms"]))
        #expect(await fake.text.calls.last.map(SearchTestModel.rows) == ["query: red"])

        let (noImage, _) = await batch.run(.success(QueryRecord(id: "q2", source: .image(path: "/x.png"))))
        #expect(noImage.id == "q2")
        #expect(noImage.error?.message.contains("no image tower") == true)
        let (invalid, _) = await batch.run(.failure(RecordFailure(id: "q3", message: "query 3: bad")))
        #expect(invalid.error == .init(message: "query 3: bad"))

        let json = try JSONOutput.encode([ok, invalid])
        let decoded = try #require(JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
        #expect(decoded[0]["error"] is NSNull)
        #expect((decoded[0]["output"] as? [String: Any])?["hits"] is [Any])
        #expect(decoded[1]["output"] is NSNull)
        #expect((decoded[1]["error"] as? [String: Any])?["message"] as? String == "query 3: bad")
        #expect(json.contains("\"embed_ms\""))
    }

    @Test("query accepts --queries-json / --output-json and needs exactly one input")
    func arguments() throws {
        let cmd = try QueryCommand.parse([
            "--model", "bundle", "--index", "i.cvi", "--queries-json", "q.json", "--output-json", "o.json",
            "--verbose", "--clear-coreai-cache",
        ])
        #expect(cmd.queriesJson == "q.json")
        #expect(cmd.outputJson == "o.json")
        #expect(cmd.modelOptions.verbose)
        #expect(cmd.modelOptions.clearCoreAICache)
        #expect(throws: (any Error).self) {
            try QueryCommand.parse(["--model", "bundle", "--index", "i.cvi", "--text", "a", "--queries-json", "q.json"])
        }
        #expect(throws: (any Error).self) {
            try QueryCommand.parse(["--model", "bundle", "--index", "i.cvi", "--text", "a", "--json"])
        }
        #expect(throws: (any Error).self) {
            try IndexCommand.parse(["--model", "bundle", "--out", "o.cvi", "--texts", "t.txt", "--batch", "4"])
        }
    }
}

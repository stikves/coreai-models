// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Testing

@testable import CoreAIEmbeddings

/// Opt-in tests against a real exported bundle. Set `COREAI_EMBEDDING_BUNDLE=/path/to/bundle`
/// (a `kind: "embedding"` directory); skipped otherwise.
@Suite(
    "Real embedding bundle (opt-in)",
    .enabled(if: ProcessInfo.processInfo.environment["COREAI_EMBEDDING_BUNDLE"] != nil)
)
struct RealEmbeddingBundleTests {
    var bundleURL: URL {
        URL(fileURLWithPath: ProcessInfo.processInfo.environment["COREAI_EMBEDDING_BUNDLE"] ?? "bundle")
    }

    @Test("loads, embeds text with prompts, MRL widths, padding and batch invariance")
    func realTextTower() async throws {
        let model = try EmbeddingModel(bundleAt: bundleURL)
        try await model.load()
        let queries = ["Which planet is known as the Red Planet?", "capital of France"]
        let docs = [
            "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
            "Paris is the capital and largest city of France.",
            "Jupiter is the largest planet in the Solar System.",
        ]
        let promptQuery = model.promptNames.contains("query") ? "query" : nil
        let promptDoc = model.promptNames.contains("document") ? "document" : nil
        let q = try await model.embed(queries.map { .text($0) }, promptName: promptQuery)
        let d = try await model.embed(docs.map { .text($0) }, promptName: promptDoc)
        for e in q + d {
            #expect(e.dimension == model.dimension)
            #expect(EmbeddingMath.isFinite(e.vector))
            #expect(abs(EmbeddingMath.l2Norm(e.vector) - 1) < 1e-3)
        }
        // Retrieval sanity: each query's best document.
        #expect(try q[0].cosineSimilarity(to: d[0]) > q[0].cosineSimilarity(to: d[1]))
        #expect(try q[1].cosineSimilarity(to: d[1]) > q[1].cosineSimilarity(to: d[2]))

        // Batch and padding invariance.
        for (i, text) in docs.enumerated() {
            let alone = try await model.embed(.text(text), promptName: promptDoc)
            #expect(try alone.cosineSimilarity(to: d[i]) > 0.9999)
        }
        for dims in model.matryoshkaDimensions {
            let e = try await model.embed(.text(queries[0]), promptName: promptQuery, dimensions: dims)
            #expect(e.dimension == dims)
            #expect(try e.cosineSimilarity(to: q[0].truncated(to: dims)) > 0.9999)
        }
        await model.unload()
    }
}

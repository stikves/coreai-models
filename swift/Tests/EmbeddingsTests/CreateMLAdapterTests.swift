// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

#if canImport(CreateMLComponents)
import CoreImage
import CoreML
import CreateMLComponents
import Foundation
import Testing
import os

@testable import CoreAIEmbeddings

@Suite struct CreateMLAdapterTests {
    /// A CLIP model whose text vector is one-hot at `word count % 512`, and whose image vector
    /// holds the first pixel's normalized red value at index 0 (and 1 at index 1).
    static func model() throws -> SharedEmbeddingModel {
        try SharedEmbeddingModel(
            FakeModels.model("clip-vit-b32") { component, _, b, inputs in
                var v = [Float](repeating: 0, count: 512)
                if component == "text_encoder" {
                    let mask = inputs["attention_mask"]!.ints![(b * 77)..<((b + 1) * 77)]
                    v[Int(mask.reduce(0, +) - 2) % 512] = 1
                } else {
                    v[0] = inputs["pixel_values"]!.floats[b * 3 * 224 * 224]
                    v[1] = 1
                }
                return v
            })
    }

    @Test func extractorsReportEachInput() async throws {
        let origins = OSAllocatedUnfairLock(initialState: [String]())
        let handler: EventHandler = { event in origins.withLock { $0.append("\(event.origin) \(event.itemCount)") } }
        let model = try Self.model()
        let features = try await CoreAIEmbeddingTextExtractor(model: model)
            .applied(to: "one two three", eventHandler: handler)
        #expect(features.shape == [512] && features.scalars[3] == 1)
        let red = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
        let image = try await CoreAIEmbeddingImageExtractor(model: model).applied(to: red, eventHandler: handler)
        #expect(image.shape == [512] && image.scalars[0] > 1 && image.scalars[1] == 1)
        #expect(origins.withLock { $0 } == ["CoreAIEmbeddingTextExtractor 1", "CoreAIEmbeddingImageExtractor 1"])
    }

    @Test func refusals() throws {
        let model = try Self.model()
        #expect(throws: EmbeddingError.self) { try CoreAIEmbeddingTextExtractor(model: model, dimensions: 256) }
        let textOnly = try SharedEmbeddingModel(FakeModels.model("clip-vit-b32-text"))
        #expect(throws: EmbeddingError.self) { try CoreAIEmbeddingImageExtractor(model: textOnly) }
    }
}
#endif

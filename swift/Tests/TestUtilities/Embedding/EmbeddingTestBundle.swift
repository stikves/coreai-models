// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Tokenizers

import CoreAIEmbeddings

/// Temporary embedding bundle directories for tests.
package enum EmbeddingTestBundle {
    /// A model over a temporary bundle whose components are `components` (by asset name).
    ///
    /// The tokenizer is `tokenizer` when given, else loaded from the copied `tokenizerFolder`.
    package static func model(
        metadata: Data,
        components: [String: FakeEmbeddingComponent],
        tokenizerFolder: URL? = nil,
        tokenizer: (any Tokenizer)? = nil
    ) throws -> EmbeddingModel {
        let url = FileManager.default.temporaryDirectory.appending(path: "embedding-bundle-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try metadata.write(to: url.appending(path: "metadata.json"))
        if let tokenizerFolder {
            try FileManager.default.copyItem(at: tokenizerFolder, to: url.appending(path: "tokenizer"))
        }
        return try EmbeddingModel(
            bundleAt: url,
            makeComponent: { name, _, _ in
                guard let component = components[name] else {
                    throw EmbeddingError.invalidBundle("the bundle declares no asset '\(name)'")
                }
                return component
            },
            makeTokenizer: { folder in
                if let tokenizer { return tokenizer }
                return try await AutoTokenizer.from(modelFolder: folder)
            })
    }
}

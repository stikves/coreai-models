// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

// Records follow the llm-server --replay envelope: snake_case keys, null error on success.

/// One output record: `{"id", "output", "timing", "error"}`.
struct JSONRecord<Output: Encodable>: Encodable {
    var id: String?
    var output: Output?
    var timing: [String: Double]
    var error: RecordError?

    struct RecordError: Encodable, Equatable {
        var message: String
    }

    enum CodingKeys: String, CodingKey {
        case id, output, timing, error
    }

    // Explicit so `id`, `output` and `error` are written as `null` instead of being dropped.
    func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(output, forKey: .output)
        try c.encode(timing, forKey: .timing)
        try c.encode(error, forKey: .error)
    }
}

enum JSONOutput {
    /// snake_case, sorted keys, pretty-printed.
    static func encode(_ value: some Encodable) throws -> String {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return String(decoding: try encoder.encode(value), as: UTF8.self)
    }

    /// Milliseconds rounded to 0.001 so records stay short.
    static func milliseconds(_ duration: Duration) -> Double {
        (duration.inMilliseconds * 1000).rounded() / 1000
    }
}

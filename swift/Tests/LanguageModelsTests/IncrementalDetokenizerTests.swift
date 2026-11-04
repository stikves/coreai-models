// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import TestUtilities
import Testing

@testable import CoreAILanguageModels

@Suite("IncrementalDetokenizer")
struct IncrementalDetokenizerTests {
    private let tokenizer = MockTokenizer()

    @Test(
        "Deltas concatenate to the text, one UTF-8 byte per token",
        arguments: [
            "plain ascii",
            "cafe\u{0301} \u{20AC}5",
            "family \u{1F468}\u{200D}\u{1F469}\u{200D}\u{1F467} ok",
            "\u{0B95}\u{0BB3}\u{0BBF}\u{0BB2}\u{0BCD}",
        ])
    func concatenates(text: String) {
        var detokenizer = IncrementalDetokenizer(tokenizer: tokenizer)
        var deltas: [String] = []
        for token in tokenizer.encode(text: text) {
            deltas.append(detokenizer.append(token))
        }
        deltas.append(detokenizer.flush())
        #expect(deltas.joined() == text)
        #expect(!deltas.contains { $0.unicodeScalars.contains("\u{FFFD}") })
    }

    @Test("Incomplete UTF-8 is held, then flushed as U+FFFD")
    func holdsIncompleteSequence() {
        var detokenizer = IncrementalDetokenizer(tokenizer: tokenizer)
        #expect(detokenizer.append(0x61) == "a")
        #expect(detokenizer.append(0xE2) == "")
        #expect(detokenizer.isIncomplete)
        #expect(detokenizer.append(0x82) == "")
        #expect(detokenizer.flush() == "\u{FFFD}")
    }

    @Test("Each step decodes a bounded slice, not the whole sequence")
    func decodesBoundedSlices() {
        var sliceLengths: [Int] = []
        let tokenizer = tokenizer
        var detokenizer = IncrementalDetokenizer(decode: { tokens in
            sliceLengths.append(tokens.count)
            return tokenizer.decode(tokens: tokens)
        })
        let text = String(repeating: "x \u{20AC} \u{1F468}\u{200D}\u{1F469} ", count: 200)
        let tokens = tokenizer.encode(text: text)
        var streamed = ""
        for token in tokens {
            streamed += detokenizer.append(token)
        }
        #expect(streamed == text)
        // Context (one character, at most 4 bytes) plus the pending character.
        #expect(sliceLengths.max()! <= 8)
        #expect(sliceLengths.count <= 2 * tokens.count)
    }

    @Test("Context keeps the leading space of SentencePiece-style tokens")
    func keepsLeadingSpace() {
        // Metaspace-style decode: "▁" becomes a space and the first one is stripped.
        let pieces = ["\u{2581}Hello", "\u{2581}world", "!"]
        var detokenizer = IncrementalDetokenizer(decode: { tokens in
            let text = tokens.map { pieces[$0] }.joined().replacingOccurrences(of: "\u{2581}", with: " ")
            return text.hasPrefix(" ") ? String(text.dropFirst()) : text
        })
        #expect([0, 1, 2].map { detokenizer.append($0) } == ["Hello", " world", "!"])
    }
}

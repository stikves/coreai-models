// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Tokenizers

/// Turns a stream of token IDs into text deltas without re-decoding the whole sequence each step.
///
/// Uses the prefix/read offset scheme of Hugging Face `DecodeStream`: each step decodes only the tokens
/// since the last emitted delta, with the tokens of that delta as context so tokenizers that infer a
/// leading space from the previous token (SentencePiece/Metaspace) keep it. New text is sliced by
/// Unicode scalars, so a token that extends the previous grapheme (ZWJ emoji, combining marks, Indic
/// vowel signs) is emitted once. Text ending in U+FFFD is an incomplete UTF-8 sequence and is held
/// until a later token completes it.
package struct IncrementalDetokenizer {
    private let decode: ([Int]) -> String
    /// Tokens from `prefixOffset` on; everything before has been emitted and is no longer needed.
    private var tokens: [Int] = []
    /// `tokens[..<readOffset]` is the context whose text has already been emitted.
    private var readOffset = 0
    private var prefixScalarCount = 0

    /// Whether the last `append` held text back because it ended in an incomplete UTF-8 sequence.
    package private(set) var isIncomplete = false

    package init(tokenizer: any Tokenizer) {
        self.init(decode: { tokenizer.decode(tokens: $0) })
    }

    package init(decode: @escaping ([Int]) -> String) {
        self.decode = decode
    }

    /// Appends `token` and returns the text it completes, or `""` if none is ready yet.
    package mutating func append(_ token: Int) -> String {
        tokens.append(token)
        let scalars = decode(tokens).unicodeScalars
        isIncomplete = scalars.last == "\u{FFFD}"
        guard !isIncomplete, scalars.count > prefixScalarCount else {
            return ""
        }
        let delta = String(String.UnicodeScalarView(scalars.dropFirst(prefixScalarCount)))

        // The tokens just emitted become the context for the next step.
        tokens.removeFirst(readOffset)
        readOffset = tokens.count
        prefixScalarCount = decode(tokens).unicodeScalars.count
        return delta
    }

    /// Returns any held-back text, including incomplete UTF-8 as U+FFFD, and resets the stream.
    package mutating func flush() -> String {
        let scalars = decode(tokens).unicodeScalars
        let remaining = String(String.UnicodeScalarView(scalars.dropFirst(prefixScalarCount)))
        tokens = []
        readOffset = 0
        prefixScalarCount = 0
        isIncomplete = false
        return remaining
    }
}

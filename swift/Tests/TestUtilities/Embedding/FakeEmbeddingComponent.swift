// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

import CoreAIEmbeddings

/// Deterministic stand-in for an embedding component.
///
/// Without a handler, a text-encoder call returns the L2-normalized masked mean of a per-token
/// pseudo-random vector per row, so each output depends only on the attended ids.
package actor FakeEmbeddingComponent: EmbeddingComponent {
    package struct Call: Sendable {
        package var function: String
        package var shapes: [String: [Int]]
        package var inputs: [String: EmbeddingTensor]
        package var output: String?
    }

    package typealias Handler = @Sendable (String, [String: EmbeddingTensor]) -> [Float]

    package let dimension: Int
    private let handler: Handler?
    package private(set) var calls: [Call] = []
    package private(set) var loadCount = 0
    package private(set) var prepared: [String] = []

    package init(dimension: Int, handler: Handler? = nil) {
        self.dimension = dimension
        self.handler = handler
    }

    /// Number of `run` calls made to `function`.
    package func callCount(function: String) -> Int { calls.filter { $0.function == function }.count }

    package func load() async throws {
        loadCount += 1
        try await Task.sleep(for: .milliseconds(5))  // lets concurrent `load` calls overlap
    }

    package func unload() async {}
    package func prepareFunction(named name: String) async throws { prepared.append(name) }
    package func staticShape(function: String, input: String) async throws -> [Int]? { nil }

    package func run(function: String, inputs: [String: EmbeddingTensor], output: String?) async throws -> [Float] {
        calls.append(Call(function: function, shapes: inputs.mapValues(\.shape), inputs: inputs, output: output))
        if let handler { return handler(function, inputs) }
        guard case .int32(let ids) = inputs["input_ids"]?.storage,
            case .int32(let mask) = inputs["attention_mask"]?.storage,
            let shape = inputs["input_ids"]?.shape
        else { return [] }
        let (batch, length) = (shape[0], shape[1])
        var out: [Float] = []
        for b in 0..<batch {
            var acc = [Float](repeating: 0, count: dimension)
            var n: Float = 0
            for t in 0..<length where mask[b * length + t] == 1 {
                let id = Float(ids[b * length + t])
                for k in 0..<dimension { acc[k] += sin(id * 0.37 + Float(k) * 0.11 + 0.5) }
                n += 1
            }
            out += Self.normalized(acc.map { $0 / max(n, 1) })
        }
        return out
    }

    /// An image encoder returning 280 soft tokens of width `hidden`; token values cycle `$0 % 7`.
    package static func softTokens(hidden: Int) -> FakeEmbeddingComponent {
        FakeEmbeddingComponent(dimension: hidden) { _, _ in
            (0..<(280 * hidden)).map { Float($0 % 7) }
        }
    }

    /// An encoder returning the same normalized `1...dimension` ramp for every call.
    package static func constantEncoder(dimension: Int) -> FakeEmbeddingComponent {
        FakeEmbeddingComponent(dimension: dimension) { _, _ in
            normalized((0..<dimension).map { Float($0 + 1) })
        }
    }

    package static func normalized(_ v: [Float]) -> [Float] {
        let norm = v.reduce(0) { $0 + $1 * $1 }.squareRoot()
        return norm > 0 ? v.map { $0 / norm } : v
    }
}

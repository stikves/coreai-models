// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import CoreGraphics
import Foundation
import Synchronization

/// Many callers, one model: calls run one at a time, in the order they arrive.
///
/// ```swift
/// let shared = SharedEmbeddingModel(try await CoreAIEmbeddingModel(resourcesAt: bundleURL))
/// async let a = shared.embed(texts: ["a cat"])
/// async let b = shared.embed(images: photos)
/// let (texts, images) = try await (a, b)
/// ```
///
/// Each call waits for the calls before it, then takes the model, runs, and puts the model back,
/// also when the call throws. A call cancelled while it waits throws `CancellationError`.
public final class SharedEmbeddingModel: EmbeddingModel, Sendable {
    public let space: EmbeddingSpace
    /// The native vector width.
    public let dimension: Int
    /// The widths `embed` can return, native first.
    public let matryoshkaDimensions: [Int]
    public let modalities: Set<EmbeddingModality>
    /// `logits = logitScale * cosine`, for models trained with one (CLIP).
    public let logitScale: Double?
    /// Whether a vector cut to a smaller width is L2-normalized again.
    public let renormalizesAfterTruncation: Bool

    /// The model between calls; `nil` while a call has it.
    private let model: Mutex<CoreAIEmbeddingModel?>
    private let queue: RequestQueue

    /// Serves `model`. A call made while one runs and `maxQueueDepth` others wait throws
    /// `EmbeddingError.queueFull`.
    public init(_ model: consuming CoreAIEmbeddingModel, maxQueueDepth: Int = .max) {
        space = model.space
        dimension = model.dimension
        matryoshkaDimensions = model.matryoshkaDimensions
        modalities = model.modalities
        logitScale = model.logitScale
        renormalizesAfterTruncation = model.renormalizesAfterTruncation
        self.model = Mutex(.some(model))
        queue = RequestQueue(maxDepth: maxQueueDepth)
    }

    /// `CoreAIEmbeddingModel.embed(_:dimensions:)`, after the calls before it.
    public func embed(_ inputs: [EmbeddingInput], dimensions: Int? = nil) async throws -> [Embedding] {
        try await withModel { try await $0.embed(inputs, dimensions: dimensions) }
    }

    /// Embeds `texts`.
    public func embed(texts: [String], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(texts.map { .text($0) }, dimensions: dimensions)
    }

    /// Embeds `images`.
    public func embed(images: [CGImage], dimensions: Int? = nil) async throws -> [Embedding] {
        try await embed(images.map { .image($0) }, dimensions: dimensions)
    }

    /// `CoreAIEmbeddingModel.load(_:)`, after the calls before it.
    public func load(_ modalities: Set<EmbeddingModality>? = nil) async throws {
        try await withModel { try await $0.load(modalities) }
    }

    /// `CoreAIEmbeddingModel.warmup(_:)`, after the calls before it.
    public func warmup(_ modalities: Set<EmbeddingModality>? = nil) async throws {
        try await withModel { try await $0.warmup(modalities) }
    }

    /// Calls waiting for the one that runs.
    var waitingCount: Int { queue.queuedCount }

    /// Waits for the calls before this one, then runs `body` on the model and puts it back.
    private func withModel<Result: Sendable>(
        _ body: (inout CoreAIEmbeddingModel) async throws -> Result
    ) async throws -> Result {
        let permit: QueuePermit
        do {
            permit = try await queue.acquire()
        } catch RequestQueueError.full(let depth) {
            throw EmbeddingError.queueFull(depth: depth)
        }
        defer { permit.release() }
        var checkedOut = model.withLock { $0.take() }
        defer { model.withLock { swap(&$0, &checkedOut) } }
        return try await body(&checkedOut!)
    }
}

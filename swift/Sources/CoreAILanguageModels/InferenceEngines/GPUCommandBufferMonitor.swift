// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal
import Synchronization

// MARK: - Error

/// A GPU command buffer finished with an error, so the work encoded on it may not have run.
///
/// The pipelined engine raises this when any command buffer on its queue fails, for example
/// when the GPU runs out of memory (`MTLCommandBufferError.outOfMemory`). The logits of a failed
/// step are garbage, so generation stops at the first failure instead of sampling them.
package struct GPUCommandBufferError: Error, LocalizedError, CustomStringConvertible {
    /// The command buffer's `error`, or the error a GPU callback reported. `nil` when the
    /// command buffer only reported `status == .error`.
    package let underlying: (any Error)?
    /// The command buffer's final status.
    package let status: MTLCommandBufferStatus
    /// The label of the queue the command buffer ran on.
    package let queueLabel: String?

    package init(status: MTLCommandBufferStatus, underlying: (any Error)?, queueLabel: String? = nil) {
        self.status = status
        self.underlying = underlying
        self.queueLabel = queueLabel
    }

    /// The failure a completed command buffer reports, or `nil` when it succeeded.
    package init?(status: MTLCommandBufferStatus, error: (any Error)?, queueLabel: String? = nil) {
        guard status == .error || error != nil else { return nil }
        self.init(status: status, underlying: error, queueLabel: queueLabel)
    }

    /// The failure `commandBuffer` reports, or `nil` when it succeeded. Call it once the
    /// command buffer has completed (from its completed handler, or one that runs after it).
    package init?(checking commandBuffer: any MTLCommandBuffer) {
        let status = commandBuffer.status
        let error = commandBuffer.error
        guard status == .error || error != nil else { return nil }
        self.init(status: status, underlying: error, queueLabel: commandBuffer.commandQueue.label)
    }

    /// True when the GPU ran out of memory.
    package var isOutOfMemory: Bool { Self.isOutOfMemory(underlying) }

    package var errorDescription: String? {
        if isOutOfMemory {
            return "GPU out of memory (kIOGPUCommandBufferCallbackErrorOutOfMemory). "
                + "The model needs more memory than this device has available."
        }
        let detail = underlying.map { ($0 as NSError).localizedDescription } ?? "status \(status.rawValue)"
        return "GPU command buffer failed: \(detail)"
    }

    package var description: String { errorDescription ?? "GPU command buffer failed" }

    /// IOGPU's name for the out-of-memory code (8). Metal puts it in the error's description, as in
    /// `Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`.
    static let ioGPUOutOfMemoryName = "kIOGPUCommandBufferCallbackErrorOutOfMemory"

    /// True when `error`, or an error it wraps, is a GPU out-of-memory failure.
    package static func isOutOfMemory(_ error: (any Error)?) -> Bool {
        var next = error
        // Bounded: an underlying-error chain is short, and must not loop.
        for _ in 0..<4 {
            guard let current = next else { return false }
            if let gpu = current as? GPUCommandBufferError {
                next = gpu.underlying
                continue
            }
            let ns = current as NSError
            if ns.domain == MTLCommandBufferErrorDomain && ns.code == MTLCommandBufferError.outOfMemory.rawValue {
                return true
            }
            if ns.localizedDescription.contains(ioGPUOutOfMemoryName)
                || String(describing: current).contains(ioGPUOutOfMemoryName)
            {
                return true
            }
            next = ns.userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return false
    }

    /// `error` as a `GPUCommandBufferError` when it reports a failed command buffer, else `error`.
    ///
    /// GPU callbacks (MPSGraph's completion handler) pass the command buffer's own error through;
    /// this gives it the same type as a failure the queue monitor saw.
    package static func wrapping(_ error: any Error) -> any Error {
        if error is GPUCommandBufferError { return error }
        let ns = error as NSError
        if ns.domain == MTLCommandBufferErrorDomain || isOutOfMemory(error) {
            return GPUCommandBufferError(status: .error, underlying: error)
        }
        return error
    }

    /// The GPU failure behind `error`, looking through the wrappers a caller may add.
    package static func find(in error: any Error) -> GPUCommandBufferError? {
        var next: (any Error)? = error
        for _ in 0..<4 {
            guard let current = next else { return nil }
            if let gpu = current as? GPUCommandBufferError { return gpu }
            if case InferenceRuntimeError.modelLoadingFailed(let underlying) = current {
                next = underlying
                continue
            }
            next = (current as NSError).userInfo[NSUnderlyingErrorKey] as? NSError
        }
        return nil
    }
}

// MARK: - Failure latch

/// The first GPU failure of a generation, shared between the encode loop and Metal callbacks.
///
/// `hasFailed` is one atomic load, cheap enough to check before every yield. The first error
/// recorded wins: a later callback that fails because of the first failure never replaces it.
package final class GPUFailureLatch: Sendable {
    private let failed = Atomic<Bool>(false)
    private let firstError = Mutex<(any Error)?>(nil)

    package init() {}

    package var hasFailed: Bool { failed.load(ordering: .acquiring) }

    /// The first error recorded since the last `reset()`, or `nil`.
    package var failure: (any Error)? {
        guard hasFailed else { return nil }
        return firstError.withLock { $0 }
    }

    /// Record `error`; returns the first error recorded, which is the one to report.
    @discardableResult
    package func record(_ error: any Error) -> any Error {
        let first = firstError.withLock { stored -> any Error in
            if let stored { return stored }
            stored = error
            return error
        }
        failed.store(true, ordering: .releasing)
        return first
    }

    package func throwIfFailed() throws {
        if let failure { throw failure }
    }

    /// Record `commandBuffer`'s failure, if it failed. Call it once the command buffer has
    /// completed. Returns true when it failed.
    @discardableResult
    package func recordFailure(of commandBuffer: any MTLCommandBuffer) -> Bool {
        guard let failure = GPUCommandBufferError(checking: commandBuffer) else { return false }
        record(failure)
        return true
    }

    /// Forget the recorded failure. Call only when no GPU work of the previous run is in flight.
    package func reset() {
        firstError.withLock { $0 = nil }
        failed.store(false, ordering: .releasing)
    }
}

// MARK: - Monitored command queue

/// A command queue that checks the result of every command buffer made from it.
///
/// Core AI's `ComputeStream` and MPSGraph's `runAsync` make and commit their own command
/// buffers on the queue they are given, so the engine never sees them. Handing them this wrapper
/// instead of the real queue puts a completed handler on each command buffer when it is made,
/// before the caller can commit it. A buffer that ends with `status == .error` or an `error`
/// calls `onFailure`.
///
/// Command buffers on one queue complete in commit order, so a failed model step is reported
/// before the sampler callback of that step runs.
///
/// Selectors the wrapper doesn't implement are forwarded to the real queue.
///
/// Limitation: a command buffer made from the real queue directly is not monitored. That is
/// the case for the follow-on command buffers of MPS commit-and-continue (`MPSCommandBuffer`
/// makes them from the committed buffer's `commandQueue`, which is the real queue). For those:
/// - When they belong to an MPSGraph run (the samplers), MPSGraph's completion handler runs on
///   the last one and passes on its error, which the engine records like any other.
/// - Otherwise the failure is caught only if a monitored command buffer that runs later on the
///   queue fails too (the next step's, a sampler's, or the engine's drain sentinel). Metal
///   doesn't promise that, so a failure confined to such a follow-on buffer can go unreported.
package final class MonitoredCommandQueue: NSObject, MTLCommandQueue, @unchecked Sendable {
    // @unchecked: `base` is a Metal command queue (thread-safe, Sendable in the SDK), and
    // `onFailure` is a @Sendable closure. Nothing else is stored.
    package let base: any MTLCommandQueue
    private let onFailure: @Sendable (GPUCommandBufferError) -> Void
    private let madeCount = Atomic<Int>(0)

    package init(
        wrapping base: any MTLCommandQueue,
        onFailure: @escaping @Sendable (GPUCommandBufferError) -> Void
    ) {
        self.base = base
        self.onFailure = onFailure
        super.init()
    }

    /// How many command buffers have been made from this queue. For tests.
    package var _commandBuffersMadeForTesting: Int { madeCount.load(ordering: .relaxed) }

    package var label: String? {
        get { base.label }
        set { base.label = newValue }
    }

    package var device: any MTLDevice { base.device }

    package func makeCommandBuffer() -> (any MTLCommandBuffer)? {
        monitored(base.makeCommandBuffer())
    }

    package func makeCommandBuffer(descriptor: MTLCommandBufferDescriptor) -> (any MTLCommandBuffer)? {
        monitored(base.makeCommandBuffer(descriptor: descriptor))
    }

    package func makeCommandBufferWithUnretainedReferences() -> (any MTLCommandBuffer)? {
        monitored(base.makeCommandBufferWithUnretainedReferences())
    }

    @available(macOS, deprecated: 10.13)
    @available(iOS, deprecated: 11.0)
    package func insertDebugCaptureBoundary() {
        base.insertDebugCaptureBoundary()
    }

    package func addResidencySet(_ residencySet: any MTLResidencySet) {
        base.addResidencySet(residencySet)
    }

    // The Metal protocol requirement's Swift name (the array form is an SDK extension on it).
    // swift-format-ignore: AlwaysUseLowerCamelCase
    package func __addResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
        base.__addResidencySets(residencySets, count: count)
    }

    package func removeResidencySet(_ residencySet: any MTLResidencySet) {
        base.removeResidencySet(residencySet)
    }

    // The Metal protocol requirement's Swift name (the array form is an SDK extension on it).
    // swift-format-ignore: AlwaysUseLowerCamelCase
    package func __removeResidencySets(_ residencySets: UnsafePointer<any MTLResidencySet>, count: Int) {
        base.__removeResidencySets(residencySets, count: count)
    }

    // Anything else a framework calls on the queue (private API included) goes to the real one.
    package override func forwardingTarget(for aSelector: Selector!) -> Any? {
        base
    }

    package override func responds(to aSelector: Selector!) -> Bool {
        super.responds(to: aSelector) || base.responds(to: aSelector)
    }

    package override var description: String { "MonitoredCommandQueue(\(base.description))" }

    /// Check the command buffer when it completes; report it if it failed.
    private func monitored(_ commandBuffer: (any MTLCommandBuffer)?) -> (any MTLCommandBuffer)? {
        guard let commandBuffer else { return nil }
        madeCount.add(1, ordering: .relaxed)
        let onFailure = self.onFailure
        // One handler per command buffer, added before the caller can add its own, so it runs
        // first. On success it only reads `status` and `error`.
        commandBuffer.addCompletedHandler { completed in
            if let failure = GPUCommandBufferError(checking: completed) {
                onFailure(failure)
            }
        }
        return commandBuffer
    }
}

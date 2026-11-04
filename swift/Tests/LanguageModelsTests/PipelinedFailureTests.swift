// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation
import Metal
import MetalPerformanceShaders
import TestUtilities
import Testing

@testable import CoreAILanguageModels

/// The out-of-memory error Metal reports, as seen on device:
/// `Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`.
private func outOfMemoryError() -> NSError {
    NSError(
        domain: MTLCommandBufferErrorDomain,
        code: Int(MTLCommandBufferError.outOfMemory.rawValue),
        userInfo: [
            NSLocalizedDescriptionKey: "Insufficient Memory (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)"
        ])
}

private struct UnrelatedError: Error {}

@Suite("GPU command buffer failures")
struct GPUCommandBufferErrorTests {
    @Test("a completed command buffer is not a failure")
    func successIsNotAFailure() {
        #expect(GPUCommandBufferError(status: .completed, error: nil) == nil)
    }

    @Test("status .error or an error is a failure")
    func failureIsDetected() throws {
        let byStatus = try #require(GPUCommandBufferError(status: .error, error: nil))
        #expect(!byStatus.isOutOfMemory)
        let byError = try #require(GPUCommandBufferError(status: .completed, error: outOfMemoryError()))
        #expect(byError.isOutOfMemory)
    }

    @Test("out of memory is recognised by its Metal code")
    func outOfMemoryByCode() {
        let error = NSError(domain: MTLCommandBufferErrorDomain, code: Int(MTLCommandBufferError.outOfMemory.rawValue))
        #expect(GPUCommandBufferError.isOutOfMemory(error))
    }

    @Test("out of memory is recognised by the IOGPU name in an unrelated domain")
    func outOfMemoryByText() {
        let error = NSError(
            domain: "MPSGraphErrorDomain", code: 1,
            userInfo: [NSLocalizedDescriptionKey: "command buffer failed: kIOGPUCommandBufferCallbackErrorOutOfMemory"])
        #expect(GPUCommandBufferError.isOutOfMemory(error))
    }

    @Test("out of memory is recognised as an underlying error")
    func outOfMemoryUnderlying() {
        let wrapper = NSError(domain: "Wrapper", code: 1, userInfo: [NSUnderlyingErrorKey: outOfMemoryError()])
        #expect(GPUCommandBufferError.isOutOfMemory(wrapper))
    }

    @Test("other command buffer errors are not out of memory")
    func otherErrors() {
        let timeout = NSError(domain: MTLCommandBufferErrorDomain, code: Int(MTLCommandBufferError.timeout.rawValue))
        #expect(!GPUCommandBufferError.isOutOfMemory(timeout))
        #expect(!GPUCommandBufferError.isOutOfMemory(UnrelatedError()))
        #expect(!GPUCommandBufferError.isOutOfMemory(nil))
    }

    @Test("the out-of-memory message names the IOGPU code and the cause")
    func outOfMemoryMessage() {
        let error = GPUCommandBufferError(status: .error, underlying: outOfMemoryError())
        #expect(
            error.description
                == "GPU out of memory (kIOGPUCommandBufferCallbackErrorOutOfMemory). "
                + "The model needs more memory than this device has available.")
    }

    @Test("wrapping types Metal errors and leaves others alone")
    func wrapping() {
        #expect(GPUCommandBufferError.wrapping(outOfMemoryError()) is GPUCommandBufferError)
        let timeout = NSError(domain: MTLCommandBufferErrorDomain, code: Int(MTLCommandBufferError.timeout.rawValue))
        #expect(GPUCommandBufferError.wrapping(timeout) is GPUCommandBufferError)
        #expect(GPUCommandBufferError.wrapping(UnrelatedError()) is UnrelatedError)
    }

    @Test("find looks through modelLoadingFailed")
    func findThroughWrapper() {
        let gpu = GPUCommandBufferError(status: .error, underlying: outOfMemoryError())
        #expect(GPUCommandBufferError.find(in: gpu) != nil)
        #expect(GPUCommandBufferError.find(in: InferenceRuntimeError.modelLoadingFailed(underlying: gpu)) != nil)
        #expect(GPUCommandBufferError.find(in: UnrelatedError()) == nil)
    }
}

@Suite("GPUFailureLatch")
struct GPUFailureLatchTests {
    @Test("the first failure wins")
    func firstFailureWins() throws {
        let latch = GPUFailureLatch()
        #expect(!latch.hasFailed)
        #expect(latch.failure == nil)

        let first = GPUCommandBufferError(status: .error, underlying: outOfMemoryError())
        let reported = latch.record(first)
        #expect(reported is GPUCommandBufferError)

        // A later error (a consequence of the first) never replaces it.
        let second = latch.record(UnrelatedError())
        #expect(second is GPUCommandBufferError)
        #expect(latch.hasFailed)
        #expect(throws: GPUCommandBufferError.self) { try latch.throwIfFailed() }
    }

    @Test("reset forgets the failure")
    func resetClears() throws {
        let latch = GPUFailureLatch()
        latch.record(UnrelatedError())
        latch.reset()
        #expect(!latch.hasFailed)
        try latch.throwIfFailed()
    }
}

@Suite("Pipelined token delivery after a GPU failure")
struct PipelinedTokenDeliveryTests {
    /// Drain a stream into (tokens, terminating error).
    private func collect(
        _ stream: AsyncThrowingStream<InferenceEngine.TokenId, Error>
    ) async -> ([InferenceEngine.TokenId], (any Error)?) {
        var tokens: [InferenceEngine.TokenId] = []
        do {
            for try await token in stream { tokens.append(token) }
            return (tokens, nil)
        } catch {
            return (tokens, error)
        }
    }

    @Test("tokens flow while nothing has failed")
    func deliversTokens() async {
        let latch = GPUFailureLatch()
        let (stream, continuation) = AsyncThrowingStream<InferenceEngine.TokenId, Error>.makeStream()
        deliverSampledToken(1, error: nil, failures: latch, to: continuation)
        deliverSampledToken(2, error: nil, failures: latch, to: continuation)
        continuation.finish()
        let (tokens, error) = await collect(stream)
        #expect(tokens == [1, 2])
        #expect(error == nil)
    }

    @Test("a failed model step stops the stream before its sampled token")
    func failedModelStepYieldsNothing() async throws {
        let latch = GPUFailureLatch()
        let (stream, continuation) = AsyncThrowingStream<InferenceEngine.TokenId, Error>.makeStream()
        deliverSampledToken(7, error: nil, failures: latch, to: continuation)

        // The queue monitor saw the model's command buffer fail; the sampler callback for the
        // same step (and every later one) reports success with a garbage token.
        latch.record(GPUCommandBufferError(status: .error, underlying: outOfMemoryError()))
        deliverSampledToken(99, error: nil, failures: latch, to: continuation)
        deliverSampledToken(98, error: nil, failures: latch, to: continuation)
        continuation.finish()

        let (tokens, error) = await collect(stream)
        #expect(tokens == [7])
        let gpuError = try #require(error as? GPUCommandBufferError)
        #expect(gpuError.isOutOfMemory)
    }

    @Test("a sampler error is recorded and typed, and later tokens are dropped")
    func samplerErrorStopsLaterTokens() async throws {
        let latch = GPUFailureLatch()
        let (stream, continuation) = AsyncThrowingStream<InferenceEngine.TokenId, Error>.makeStream()
        deliverSampledToken(0, error: outOfMemoryError(), failures: latch, to: continuation)
        deliverSampledToken(5, error: nil, failures: latch, to: continuation)

        let (tokens, error) = await collect(stream)
        #expect(tokens.isEmpty)
        #expect(error is GPUCommandBufferError)
        #expect(latch.failure is GPUCommandBufferError)
    }
}

@Suite("MonitoredCommandQueue", .enabled(if: !CIEnvironment.isVM))
struct MonitoredCommandQueueTests {
    static let device: MTLDevice? = MTLCreateSystemDefaultDevice()

    @Test("forwards the queue's label and device")
    func forwardsProperties() throws {
        let device = try #require(Self.device)
        let base = try #require(device.makeCommandQueue())
        base.label = "test.queue"
        let queue = MonitoredCommandQueue(wrapping: base) { _ in }
        #expect(queue.label == "test.queue")
        #expect(queue.device === device)
        queue.label = "renamed"
        #expect(base.label == "renamed")
    }

    @Test("a successful command buffer reports no failure")
    func successfulCommandBuffer() async throws {
        let device = try #require(Self.device)
        let latch = GPUFailureLatch()
        let queue = MonitoredCommandQueue(wrapping: try #require(device.makeCommandQueue())) {
            latch.record($0)
        }
        let commandBuffer = try #require(queue.makeCommandBuffer())
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            // Added after the monitor's handler, so it runs after it.
            commandBuffer.addCompletedHandler { _ in done.resume() }
            commandBuffer.commit()
        }
        #expect(commandBuffer.status == .completed)
        #expect(!latch.hasFailed)
        #expect(queue._commandBuffersMadeForTesting == 1)
    }

    @Test("MPSGraph sampling through the monitored queue still works and is monitored")
    func samplerRunsOnMonitoredQueue() async throws {
        let device = try #require(Self.device)
        let vocabSize = 1024
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: vocabSize)
        let latch = GPUFailureLatch()
        let queue = MonitoredCommandQueue(wrapping: try #require(device.makeCommandQueue())) {
            latch.record($0)
        }

        let logits = try #require(device.makeBuffer(length: vocabSize * 2, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: 64, options: .storageModeShared))
        let pointer = logits.contents().assumingMemoryBound(to: Float16.self)
        for i in 0..<vocabSize { pointer[i] = i == 321 ? 50 : 0 }

        let token: Int32 = try await withCheckedThrowingContinuation { cont in
            sampler.encode(
                to: queue, logitsBuffer: logits, logitsOffset: 0,
                outputBuffer: output, outputOffset: 0
            ) { token, error in
                if let error { cont.resume(throwing: error) } else { cont.resume(returning: token) }
            }
        }
        #expect(token == 321)
        #expect(!latch.hasFailed)
        // MPSGraph made its command buffer from the wrapper, so the monitor saw it.
        #expect(queue._commandBuffersMadeForTesting >= 1)
    }

    @Test("the prefill slice path makes its blit from the monitored queue and checks it")
    func slicePathIsMonitored() async throws {
        let device = try #require(Self.device)
        let vocabSize = 1024
        let queryLength = 3
        let sampler = try MPSGraphArgmaxSampler(device: device, vocabSize: vocabSize)
        let latch = GPUFailureLatch()
        let queue = MonitoredCommandQueue(wrapping: try #require(device.makeCommandQueue())) {
            latch.record($0)
        }

        // Only the last row is sampled; earlier rows peak elsewhere.
        let logits = try #require(
            device.makeBuffer(length: queryLength * vocabSize * 2, options: .storageModeShared))
        let output = try #require(device.makeBuffer(length: 64, options: .storageModeShared))
        let pointer = logits.contents().assumingMemoryBound(to: Float16.self)
        for row in 0..<queryLength {
            for i in 0..<vocabSize { pointer[row * vocabSize + i] = i == (row == queryLength - 1 ? 77 : 5) ? 50 : 0 }
        }

        let token: Int32 = try await withCheckedThrowingContinuation { cont in
            sampler.encodeWithSlice(
                to: queue, logitsBuffer: logits, queryLength: queryLength,
                outputBuffer: output, outputOffset: 0
            ) { token, error in
                if let error { cont.resume(throwing: error) } else { cont.resume(returning: token) }
            }
        }
        #expect(token == 77)
        #expect(!latch.hasFailed)
        // The blit and the sampling graph.
        #expect(queue._commandBuffersMadeForTesting >= 2)
    }

    @Test("a completed command buffer checks as no failure")
    func checkingCompletedBuffer() async throws {
        let device = try #require(Self.device)
        let latch = GPUFailureLatch()
        let base = try #require(device.makeCommandQueue())
        let commandBuffer = try #require(base.makeCommandBuffer())
        let recorded: Bool = await withCheckedContinuation { done in
            commandBuffer.addCompletedHandler { completed in
                done.resume(returning: latch.recordFailure(of: completed))
            }
            commandBuffer.commit()
        }
        #expect(!recorded)
        #expect(GPUCommandBufferError(checking: commandBuffer) == nil)
        #expect(!latch.hasFailed)
    }

    /// Documents the limitation described on `MonitoredCommandQueue`: MPS commit-and-continue
    /// makes its follow-on command buffer from the real queue, so the wrapper never sees it.
    /// If this starts to fail, the framework now goes through the wrapper: update the comment.
    @Test("commit-and-continue follow-on command buffers bypass the wrapper")
    func commitAndContinueBypassesWrapper() async throws {
        let device = try #require(Self.device)
        let queue = MonitoredCommandQueue(wrapping: try #require(device.makeCommandQueue())) { _ in }
        let mpsBuffer = MPSCommandBuffer(from: queue)
        #expect(queue._commandBuffersMadeForTesting == 1)
        let first = mpsBuffer.commandBuffer
        mpsBuffer.commitAndContinue()
        #expect(mpsBuffer.commandBuffer !== first)
        #expect(queue._commandBuffersMadeForTesting == 1)
        await withCheckedContinuation { (done: CheckedContinuation<Void, Never>) in
            mpsBuffer.addCompletedHandler { _ in done.resume() }
            mpsBuffer.commit()
        }
    }
}

@Suite("Sampler blit failures")
struct SamplerBlitFailureTests {
    @Test("a failed blit replaces the sampled token with its failure")
    func failedBlitReportsFailure() throws {
        var received: (Int32, (any Error)?)?
        let wrapped = samplerCompletion(
            { received = ($0, $1) },
            reportingFailure: { GPUCommandBufferError(status: .error, underlying: outOfMemoryError()) })
        wrapped(42, nil)
        let (token, error) = try #require(received)
        #expect(token == 0)
        let gpuError = try #require(error as? GPUCommandBufferError)
        #expect(gpuError.isOutOfMemory)
    }

    @Test("a sampler error is passed on unchanged")
    func samplerErrorPassesThrough() throws {
        var received: (Int32, (any Error)?)?
        let wrapped = samplerCompletion(
            { received = ($0, $1) },
            reportingFailure: { GPUCommandBufferError(status: .error, underlying: nil) })
        wrapped(0, UnrelatedError())
        #expect(try #require(received).1 is UnrelatedError)
    }

    @Test("the token flows when the blit succeeded")
    func successfulBlitPassesToken() throws {
        var received: (Int32, (any Error)?)?
        let wrapped = samplerCompletion({ received = ($0, $1) }, reportingFailure: { nil })
        wrapped(42, nil)
        let (token, error) = try #require(received)
        #expect(token == 42)
        #expect(error == nil)
    }
}

// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Testing

@testable import CoreAILanguageModels

/// Boundary math for splitting a constrained jump-forward batch at the sliding-window ring
/// wrap. The exported RingKVCache requires (offset % capacity) + queryLen <= capacity for a
/// batched write; a batch near the end of the ring must be split so no sub-batch straddles.
@Suite("SlidingWrapSplit")
struct SlidingWrapSplitTests {
    @Test func nonSlidingModelIsNeverSplit() {
        // capacity nil signals a full-context (non-sliding) model — always a single batch.
        #expect(SlidingWrap.splitLengths(offset: 4090, queryLen: 8, capacity: nil) == [8])
        #expect(SlidingWrap.splitLengths(offset: 0, queryLen: 1, capacity: nil) == [1])
    }

    @Test func singleTokenIsNeverSplit() {
        // A plain decode step (queryLen == 1) always satisfies the precondition.
        #expect(SlidingWrap.splitLengths(offset: 2047, queryLen: 1, capacity: 2048) == [1])
    }

    @Test func batchThatFitsBeforeWrapIsNotSplit() {
        // slot = 100, 100 + 8 <= 2048, no wrap.
        #expect(SlidingWrap.splitLengths(offset: 100, queryLen: 8, capacity: 2048) == [8])
    }

    @Test func batchEndingExactlyAtCapacityIsNotSplit() {
        // slot = 2040, 2040 + 8 == 2048, the last slot written is capacity-1, no wrap.
        #expect(SlidingWrap.splitLengths(offset: 2040, queryLen: 8, capacity: 2048) == [8])
    }

    @Test func straddlingBatchIsSplitAtBoundary() {
        // slot = 2044, room = 4, remaining 4 after wrap.
        #expect(SlidingWrap.splitLengths(offset: 2044, queryLen: 8, capacity: 2048) == [4, 4])
    }

    @Test func splitWhenFirstSlotIsLastInRing() {
        // slot = 2047, room = 1, then 7 after wrap.
        #expect(SlidingWrap.splitLengths(offset: 2047, queryLen: 8, capacity: 2048) == [1, 7])
    }

    @Test func offsetPastCapacityUsesModuloSlot() {
        // offset 4094 -> slot 4094 % 2048 = 2046, room = 2, then 62.
        #expect(SlidingWrap.splitLengths(offset: 4094, queryLen: 64, capacity: 2048) == [2, 62])
    }

    @Test func splitLengthsAlwaysSumToQueryLen() {
        let capacity = 128
        for offset in 0..<300 {
            for queryLen in 1...64 {
                let lengths = SlidingWrap.splitLengths(
                    offset: offset, queryLen: queryLen, capacity: capacity)
                #expect(lengths.reduce(0, +) == queryLen)
                // No sub-batch straddles the wrap.
                var slot = offset % capacity
                for len in lengths {
                    #expect(slot + len <= capacity)
                    slot = (slot + len) % capacity
                }
            }
        }
    }
}

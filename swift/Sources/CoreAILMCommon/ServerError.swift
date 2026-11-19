// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import Foundation

public enum ServerError: Error, LocalizedError {
    case badRequest(String)
    case queueFull(depth: Int)

    public var isBadRequest: Bool {
        if case .badRequest = self { return true }
        return false
    }

    public var errorDescription: String? {
        switch self {
        case .badRequest(let msg): return msg
        case .queueFull(let depth): return "Queue full (depth: \(depth))"
        }
    }
}

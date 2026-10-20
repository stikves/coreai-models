// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// A dense host tensor fed to an embedding component.
package struct EmbeddingTensor: Sendable, Equatable {
    package enum Storage: Sendable, Equatable {
        case int32([Int32])
        case float32([Float])
    }

    package var storage: Storage
    package var shape: [Int]

    package init(int32 values: [Int32], shape: [Int]) {
        storage = .int32(values)
        self.shape = shape
    }

    package init(float32 values: [Float], shape: [Int]) {
        storage = .float32(values)
        self.shape = shape
    }

    var count: Int {
        switch storage {
        case .int32(let v): return v.count
        case .float32(let v): return v.count
        }
    }
}

/// One loaded asset of a bundle; tests substitute a fake to run the host pipeline without model files.
package protocol EmbeddingComponent: Sendable {
    /// Load weights and functions. Idempotent.
    func load() async throws
    /// Release everything `load()` acquired.
    func unload() async
    /// Static shape of `input` in `function`, or `nil` when any dimension is dynamic or the input is absent.
    func staticShape(function: String, input: String) async throws -> [Int]?
    /// Run `function` and return output `output` (or the first output when `nil`) as dense fp32.
    func run(function: String, inputs: [String: EmbeddingTensor], output: String?) async throws -> [Float]
    /// Load (and cache) `function` without running it.
    func prepareFunction(named name: String) async throws
    /// Whether the last `load()` was served from the Core AI specialization cache (`nil` when unknown).
    var loadedFromCache: Bool? { get async }
}

extension EmbeddingComponent {
    package var loadedFromCache: Bool? { get async { nil } }
}

/// The compute unit a component is specialized for.
enum ComputeUnit: String, Sendable, Equatable {
    case gpu
    case neuralEngine = "neural_engine"

    /// The GPU when listed, else the Neural Engine (`neural_engine` or `ane`); `nil` when neither is listed.
    /// Missing metadata means the GPU.
    static func resolve(_ units: [String]?) -> ComputeUnit? {
        guard let units else { return .gpu }
        let names = Set(units.map { $0.lowercased() })
        if names.contains("gpu") { return .gpu }
        if names.contains("neural_engine") || names.contains("ane") { return .neuralEngine }
        return nil
    }

    /// `resolve(units)`, or an `invalidBundle` error naming `component`.
    static func require(_ units: [String]?, component: String) throws -> ComputeUnit {
        guard let unit = resolve(units) else {
            throw EmbeddingError.invalidBundle(
                "component '\(component)' lists compute units \(units ?? []); supported: gpu, neural_engine")
        }
        return unit
    }

    var displayName: String { self == .gpu ? "GPU" : "Neural Engine" }
}

/// One .aimodel asset of a bundle, specialized for its compute unit; functions load on first use.
actor CoreAIEmbeddingComponent: EmbeddingComponent {
    let name: String
    let modelURL: URL
    nonisolated let computeUnit: ComputeUnit
    private let options: SpecializationOptions
    private var model: AIModel?
    private var functionCache: [String: InferenceFunction] = [:]
    private(set) var loadedFromCache: Bool?

    init(name: String, modelURL: URL, computeUnits: [String]?) throws {
        let unit = try ComputeUnit.require(computeUnits, component: name)
        self.name = name
        self.modelURL = modelURL
        self.computeUnit = unit
        self.options = SpecializationOptions(preferredComputeUnitKind: unit == .gpu ? .gpu : .neuralEngine)
    }

    func load() async throws {
        guard model == nil else { return }
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            throw EmbeddingError.invalidBundle("model asset '\(name)' not found at \(modelURL.path)")
        }
        // A cache lookup tells hits from misses; a miss specializes and fills the cache.
        if let cached = try? AIModelCache.default.model(for: modelURL, options: options) {
            model = cached
            loadedFromCache = true
        } else {
            do {
                model = try await AIModel(contentsOf: modelURL, options: options)
            } catch {
                throw failure(function: nil, cached: false, reason: error.localizedDescription)
            }
            loadedFromCache = false
        }
    }

    func prepareFunction(named name: String) async throws { _ = try await function(named: name) }

    func unload() {
        functionCache.removeAll()
        model = nil
        loadedFromCache = nil
    }

    func staticShape(function name: String, input: String) async throws -> [Int]? {
        let fn = try await function(named: name)
        guard case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: input) else { return nil }
        let shape = desc.shape
        return shape.allSatisfy { $0 > 0 } ? shape : nil
    }

    func run(function name: String, inputs: [String: EmbeddingTensor], output: String?) async throws -> [Float] {
        let fn = try await function(named: name)
        let inputNames = fn.descriptor.inputNames
        var named: [String: NDArray] = [:]
        for (key, tensor) in inputs {
            guard inputNames.contains(key), case .ndArray(let desc) = fn.descriptor.inputDescriptor(of: key) else {
                throw failure(function: name, reason: "no input '\(key)' (inputs: \(inputNames))")
            }
            if desc.shape.allSatisfy({ $0 > 0 }) && desc.shape != tensor.shape {
                throw failure(function: name, reason: "input '\(key)' has static shape \(desc.shape), got \(tensor.shape)")
            }
            let resolved = desc.resolvingDynamicDimensions(tensor.shape)
            guard resolved.shape.reduce(1, *) == tensor.count else {
                throw failure(function: name, reason: "input '\(key)' expects shape \(resolved.shape), got \(tensor.shape)")
            }
            var array = NDArray(descriptor: resolved)
            do {
                try Self.fill(&array, with: tensor)
            } catch {
                throw failure(function: name, reason: error.localizedDescription)
            }
            named[key] = array
        }
        var outputs: InferenceFunction.Outputs
        do {
            outputs = try await fn.run(inputs: named)
        } catch {
            throw failure(function: name, reason: error.localizedDescription)
        }
        let outputNames = fn.descriptor.outputNames
        let key = output ?? outputNames.first ?? ""
        guard outputNames.contains(key), let array = outputs.remove(key)?.ndArray else {
            throw failure(function: name, reason: "no output '\(key)' (outputs: \(outputNames))")
        }
        switch array.scalarType {
        case .float32: return flattenNDArray(array, as: Float.self)
        case .float16: return flattenNDArray(array, as: Float16.self)
        case .bfloat16: return flattenBFloat16NDArray(array)
        default: throw failure(function: name, reason: "output '\(key)' has an unsupported scalar type")
        }
    }

    private func function(named name: String) async throws -> InferenceFunction {
        try await load()
        if let cached = functionCache[name] { return cached }
        let loaded: InferenceFunction?
        do {
            loaded = try model?.loadFunction(named: name)
        } catch {
            throw failure(function: name, reason: error.localizedDescription)
        }
        guard let fn = loaded else {
            throw failure(function: name, reason: "no function '\(name)' (functions: \(model?.functionNames ?? []))")
        }
        functionCache[name] = fn
        return fn
    }

    private func failure(function: String?, cached: Bool? = nil, reason: String) -> EmbeddingError {
        .componentFailed(
            component: name, function: function, computeUnit: computeUnit.rawValue,
            cached: cached ?? (loadedFromCache == true), reason: reason)
    }

    static func fill(_ array: inout NDArray, with tensor: EmbeddingTensor) throws {
        switch (tensor.storage, array.scalarType) {
        case (.int32(let v), .int32):
            fillNDArray(&array, as: Int32.self, count: v.count) { v[$0] }
        case (.int32(let v), .float32):
            fillNDArray(&array, as: Float.self, count: v.count) { Float(v[$0]) }
        case (.float32(let v), .float32):
            fillNDArray(&array, as: Float.self, count: v.count) { v[$0] }
        case (.float32(let v), .float16):
            fillNDArray(&array, as: Float16.self, count: v.count) { Float16(v[$0]) }
        case (.int32(let v), .float16):
            fillNDArray(&array, as: Float16.self, count: v.count) { Float16(Float(v[$0])) }
        default:
            let kind: String
            switch tensor.storage {
            case .int32: kind = "int32"
            case .float32: kind = "float32"
            }
            throw FillError(message: "cannot feed \(kind) data into a \(array.scalarType) input")
        }
    }
}

private struct FillError: LocalizedError {
    var message: String
    var errorDescription: String? { message }
}

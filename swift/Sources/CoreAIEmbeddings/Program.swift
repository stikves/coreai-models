// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAI
import CoreAIShared
import Foundation

/// A dense row-major tensor on the host.
struct HostTensor: Sendable, Equatable {
    enum Storage: Sendable, Equatable {
        case int32([Int32])
        case float32([Float])
    }

    var storage: Storage
    var shape: [Int]

    init(int32 values: [Int32], shape: [Int]) {
        precondition(values.count == shape.reduce(1, *), "HostTensor: \(values.count) values for shape \(shape)")
        storage = .int32(values)
        self.shape = shape
    }

    init(float32 values: [Float], shape: [Int]) {
        precondition(values.count == shape.reduce(1, *), "HostTensor: \(values.count) values for shape \(shape)")
        storage = .float32(values)
        self.shape = shape
    }

    var count: Int { shape.reduce(1, *) }

    /// The values as `Float`.
    var floats: [Float] {
        switch storage {
        case .int32(let v): return v.map(Float.init)
        case .float32(let v): return v
        }
    }

    /// The values as `Int32`; `nil` for float storage.
    var ints: [Int32]? {
        if case .int32(let v) = storage { return v }
        return nil
    }
}

/// A component's loaded asset. `CoreAIProgram` runs Core AI models; tests use fakes.
protocol Program: Sendable {
    /// Runs `function`. Inputs are converted to the function's declared scalar types; float
    /// outputs come back as `float32`.
    func run(function: String, inputs: [String: HostTensor]) async throws -> [String: HostTensor]
}

/// A `Program` over a Core AI model specialized for one compute unit. Functions load on first use.
actor CoreAIProgram: Program {
    let url: URL
    let unit: ComputeUnit
    private let model: AIModel
    private var functions: [String: InferenceFunction] = [:]

    /// Loads (and on first use specializes) the model at `url` for `unit`.
    init(url: URL, unit: ComputeUnit) async throws {
        self.url = url
        self.unit = unit
        var options: SpecializationOptions
        switch unit {
        case .gpu:
            options = SpecializationOptions(preferredComputeUnitKind: .gpu)
            options.expectFrequentReshapes = true
        case .neuralEngine: options = SpecializationOptions(preferredComputeUnitKind: .neuralEngine)
        case .cpu: options = SpecializationOptions(preferredComputeUnitKind: .cpu)
        }
        do {
            model = try await AIModel(contentsOf: url, options: options)
        } catch {
            throw EmbeddingError.programFailed(
                asset: url.lastPathComponent, function: nil, unit: unit.rawValue, reason: error.localizedDescription)
        }
    }

    func run(function name: String, inputs: [String: HostTensor]) async throws -> [String: HostTensor] {
        let fn = try await function(named: name)
        let descriptor = fn.descriptor
        var arrays: [String: NDArray] = [:]
        for (key, tensor) in inputs {
            guard case .ndArray(let input) = descriptor.inputDescriptor(of: key) else {
                throw EmbeddingError.unexpectedOutput(
                    component: url.lastPathComponent, "\(name) has no input '\(key)' (inputs: \(descriptor.inputNames))"
                )
            }
            var array = NDArray(descriptor: input.resolvingDynamicDimensions(tensor.shape))
            guard array.shape == tensor.shape else {
                throw EmbeddingError.unexpectedOutput(
                    component: url.lastPathComponent, "\(name).\(key) takes \(array.shape), got \(tensor.shape)")
            }
            try Self.fill(&array, with: tensor, name: key)
            arrays[key] = array
        }
        var outputs: InferenceFunction.Outputs
        do {
            outputs = try await fn.run(inputs: arrays)
        } catch {
            throw failure(error, function: name)
        }
        var result: [String: HostTensor] = [:]
        for key in descriptor.outputNames {
            guard let array = outputs.remove(key)?.ndArray else { continue }
            result[key] = HostTensor(float32: flattenAsFloat(array), shape: array.shape)
        }
        return result
    }

    private func function(named name: String) async throws -> InferenceFunction {
        if let fn = functions[name] { return fn }
        let loaded: InferenceFunction?
        do {
            loaded = try model.loadFunction(named: name)
        } catch {
            throw failure(error, function: name)
        }
        guard let fn = loaded else {
            throw EmbeddingError.unexpectedOutput(
                component: url.lastPathComponent, "no function '\(name)' (functions: \(model.functionNames.sorted()))")
        }
        functions[name] = fn
        return fn
    }

    private func failure(_ error: Error, function: String?) -> Error {
        if error is EmbeddingError { return error }
        return EmbeddingError.programFailed(
            asset: url.lastPathComponent, function: function, unit: unit.rawValue, reason: error.localizedDescription)
    }

    static func fill(_ array: inout NDArray, with tensor: HostTensor, name: String) throws {
        let count = tensor.count
        switch (tensor.storage, array.scalarType) {
        case (.int32(let v), .int32):
            fillNDArray(&array, as: Int32.self, count: count) { v[$0] }
        case (.float32(let v), .float32):
            fillNDArray(&array, as: Float.self, count: count) { v[$0] }
        case (.int32(let v), .float32):
            fillNDArray(&array, as: Float.self, count: count) { Float(v[$0]) }
        #if !((os(macOS) || targetEnvironment(macCatalyst)) && arch(x86_64))
        case (.float32(let v), .float16):
            fillNDArray(&array, as: Float16.self, count: count) { Float16(v[$0]) }
        case (.int32(let v), .float16):
            fillNDArray(&array, as: Float16.self, count: count) { Float16(Float(v[$0])) }
        #endif
        default:
            throw EmbeddingError.unexpectedOutput(
                component: name, "cannot feed \(tensor.storage) as \(array.scalarType)")
        }
    }
}

/// One `Program` per asset and compute unit, shared by the components that name it.
actor ProgramCache {
    /// Makes and loads the program of a component's asset.
    typealias Factory = @Sendable (_ url: URL, _ component: Component) async throws -> any Program

    private let bundle: ModelBundle
    private let factory: Factory
    /// Loads in flight or done, so concurrent first calls share one load.
    private var programs: [String: Task<any Program, Error>] = [:]

    init(bundle: ModelBundle, factory: @escaping Factory) {
        self.bundle = bundle
        self.factory = factory
    }

    /// The program of `component`, loading its asset on first use.
    func program(for component: Component) async throws -> any Program {
        let key = component.asset + "/" + component.computeUnits[0].rawValue
        if let load = programs[key] { return try await load.value }
        let url = try bundle.requireModelURL(for: component.asset)
        let load = Task { [factory] in try await factory(url, component) }
        programs[key] = load
        do {
            return try await load.value
        } catch {
            if programs[key] == load { programs[key] = nil }
            throw error
        }
    }

    /// Drops every program.
    func removeAll() { programs.removeAll() }

    /// A factory of loaded `CoreAIProgram`s on the component's first compute unit.
    static let coreAI: Factory = { url, component in
        try await InstrumentsProfiler.profile(InstrumentsProfiler.beginModelLoad(name: component.asset)) {
            try await CoreAIProgram(url: url, unit: component.computeUnits[0])
        }
    }
}

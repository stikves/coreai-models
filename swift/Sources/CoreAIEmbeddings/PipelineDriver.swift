// Copyright 2026 Apple Inc.
//
// Use of this source code is governed by a BSD-3-clause license that can
// be found in the LICENSE file or at https://opensource.org/licenses/BSD-3-Clause

import CoreAIShared
import Foundation

/// Runs one tower: its `pipeline` in order over a table of named tensors per input (R4).
///
/// The tower's inputs seed the table. Each stage reads its declared inputs from the table, or
/// derives them by a rule, and writes its declared outputs back; the last stage's `embedding` is
/// the vector. An input that neither the table nor a rule provides makes the tower unsupported
/// when the driver is made, before anything runs.
///
/// | Bundle | Seed | Stages |
/// |---|---|---|
/// | CLIP text, EG2 macOS text | `input_ids`, `attention_mask` | `text_encoder` |
/// | CLIP image | `pixel_values [3,S,S]` | `image_encoder` |
/// | EG2 iOS text | `input_ids`, `attention_mask` | `token_table`, `encoder` (bias, pool weights derived) |
/// | EG2 image | patches, positions, image token ids | `image_encoder`, `token_table`, `encoder` (soft tokens merged) |
struct PipelineDriver: Sendable {
    /// How the first inputs are made.
    enum Seed: Sendable, Equatable {
        /// `input_ids`, `attention_mask` from token ids.
        case tokens
        /// `pixel_values [3,S,S]` (`center_crop`).
        case croppedPixels
        /// `pixel_values [P,D]`, `position_ids [P,2]`, and `input_ids` / `attention_mask` for
        /// `<bos> <boi> <image>*n <eoi> <eos>` (`aspect_ratio_preserving`).
        case patches
    }

    /// A rule that builds a stage input the table does not hold.
    enum Derivation: Sendable, Equatable {
        /// `inputs_embeds` with `soft_tokens` written over the `image_token_id` positions.
        case mergeSoftTokens(imageTokenId: Int32)
        /// `bias[k, 0, q]` from `attention_mask`.
        case additiveBias(AdditiveBias)
        /// `attention_mask / attention_mask.sum(-1)`.
        case poolWeights
    }

    /// A stage and where each of its inputs comes from (`nil`: the table).
    struct Stage: Sendable {
        let name: String
        let component: Component
        let derived: [String: Derivation]
    }

    let modality: EmbeddingModality
    let seed: Seed
    let stages: [Stage]
    let block: EmbeddingBlock

    /// Plans `tower`: picks the seed from the first stage's inputs and a source for every stage input.
    ///
    /// - Throws: `EmbeddingError.unsupportedPipeline` naming the first input nothing provides.
    init(tower: Tower, block: EmbeddingBlock) throws {
        let first = try Self.component(tower.pipeline.first ?? "", block)
        let pixels = first.inputs["pixel_values"]
        let seed: Seed
        var table: Set<String>
        switch tower.modality {
        case .text:
            seed = .tokens
            table = ["input_ids", "attention_mask"]
        case .image where block.image?.strategy == .centerCrop && pixels?.shape.count == 4 && first.inputs.count == 1:
            seed = .croppedPixels
            table = ["pixel_values"]
        case .image
        where block.image?.strategy == .aspectRatioPreserving && pixels?.shape.count == 3
            && first.inputs["position_ids"] != nil:
            seed = .patches
            table = ["pixel_values", "position_ids", "input_ids", "attention_mask"]
        case .image:
            throw EmbeddingError.unsupportedPipeline(
                .image,
                "no image seed for inputs \(first.inputs.keys.sorted()) with \(block.image?.strategy.rawValue ?? "no image block")"
            )
        }
        var stages: [Stage] = []
        for name in tower.pipeline {
            let component = try Self.component(name, block)
            var derived: [String: Derivation] = [:]
            for input in component.inputs.keys.sorted() {
                if let rule = Self.rule(for: input, component: component, table: table, block: block) {
                    derived[input] = rule
                } else if !table.contains(input) {
                    throw EmbeddingError.unsupportedPipeline(
                        tower.modality, "nothing provides input '\(input)' of '\(name)'")
                }
            }
            stages.append(Stage(name: name, component: component, derived: derived))
            table.formUnion(component.outputs.keys)
        }
        self.modality = tower.modality
        self.seed = seed
        self.stages = stages
        self.block = block
    }

    private static func component(_ name: String, _ block: EmbeddingBlock) throws -> Component {
        guard let component = block.components[name] else {
            throw EmbeddingError.invalidMetadata("unknown component '\(name)'")
        }
        return component
    }

    /// The rule for `input`, or `nil` when it comes from the table as is.
    static func rule(for input: String, component: Component, table: Set<String>, block: EmbeddingBlock) -> Derivation?
    {
        switch input {
        case "inputs_embeds" where table.contains("soft_tokens") && table.contains("inputs_embeds"):
            return block.image?.patches.map { .mergeSoftTokens(imageTokenId: $0.imageTokenId) }
        case "bias" where table.contains("attention_mask"):
            return component.bias.map { .additiveBias($0) }
        case "pool_weights" where table.contains("attention_mask") && component.poolWeights != nil:
            return .poolWeights
        default:
            return nil
        }
    }

    // MARK: - Running

    /// Named tensors of one input, without the batch axis.
    struct Row: Sendable {
        var tensors: [String: HostTensor] = [:]
        /// Real token count, for inputs with tokens.
        var length: Int?
    }

    /// The image token ids `<bos> <boi> <image>*n <eoi> <eos>`, BOS and EOS as the text block adds them.
    func imageTokenIds(softTokens n: Int) -> [Int32] {
        guard let p = block.image?.patches else { return [] }
        let text = block.text
        return (text.addBosToken ? text.bosTokenId.map { [$0] } ?? [] : []) + [p.boiTokenId]
            + [Int32](repeating: p.imageTokenId, count: n) + [p.eoiTokenId]
            + (text.addEosToken ? [text.eosTokenId] : [])
    }

    /// Seed rows from token ids.
    func rows(tokens: [[Int32]]) -> [Row] {
        tokens.map { ids in
            Row(
                tensors: [
                    "input_ids": HostTensor(int32: ids, shape: [ids.count]),
                    "attention_mask": HostTensor(int32: [Int32](repeating: 1, count: ids.count), shape: [ids.count]),
                ], length: ids.count)
        }
    }

    /// Seed rows from images.
    func rows(images: [RGB8Image]) throws -> [Row] {
        guard let config = block.image else { throw EmbeddingError.modalityNotInBundle(.image) }
        return try images.map { image in
            switch seed {
            case .croppedPixels:
                let size = config.size ?? 0
                return Row(tensors: [
                    "pixel_values": HostTensor(
                        float32: try ImagePreparer.centerCrop(image, config), shape: [3, size, size])
                ])
            case .patches:
                let p = try ImagePreparer.patchify(image, config)
                let g = config.patches!
                var row = rows(tokens: [imageTokenIds(softTokens: p.softTokens)])[0]
                row.tensors["pixel_values"] = HostTensor(
                    float32: p.pixelValues, shape: [g.maxPatches, g.patchDimension])
                row.tensors["position_ids"] = HostTensor(int32: p.positionIds, shape: [g.maxPatches, 2])
                return row
            case .tokens:
                throw EmbeddingError.unsupportedPipeline(.image, "the image tower has a token seed")
            }
        }
    }

    /// Placeholder token rows that reach every fixed-length function of the tower: one group per
    /// function, of its smallest batch, each row as long as the function.
    func warmupTokenRows() -> [[[Int32]]] {
        var seen: Set<[Int]> = []
        var groups: [[[Int32]]] = []
        for stage in stages {
            for entry in stage.component.functions {
                guard let length = entry.length, length.lowerBound == length.upperBound,
                    length.upperBound <= block.text.maxTokens,
                    seen.insert([entry.batch.lowerBound, length.upperBound]).inserted
                else { continue }
                let row = [Int32](repeating: block.text.padTokenId, count: length.upperBound)
                groups.append([[Int32]](repeating: row, count: entry.batch.lowerBound))
            }
        }
        return groups
    }

    /// Runs the stages over `rows` with programs from `programs` and returns each row's `embedding`.
    /// Checks for cancellation before each program call.
    func run(_ rows: [Row], programs: inout ProgramCache) async throws -> [[Float]] {
        var rows = rows
        for stage in stages {
            let program = try await programs.program(for: stage.component)
            for call in try stage.component.plan(lengths: rows.map { $0.length ?? 0 }, name: stage.name) {
                try Task.checkCancellation()
                var inputs: [String: HostTensor] = [:]
                for (name, spec) in stage.component.inputs {
                    let values = try call.rows.map { r in
                        if let rule = stage.derived[name] { return try derive(rule, row: rows[r], length: call.length) }
                        guard let value = rows[r].tensors[name] else {
                            throw EmbeddingError.unsupportedPipeline(modality, "no '\(name)' for '\(stage.name)'")
                        }
                        return value
                    }
                    inputs[name] = try stack(
                        values, spec: spec, call: call, pad: name == "input_ids" ? block.text.padTokenId : 0)
                }
                let span = InstrumentsProfiler.beginEmbedInference(
                    component: stage.name, function: call.function, batch: call.rows.count)
                let outputs = try await InstrumentsProfiler.profile(span) {
                    try await program.run(function: call.function, inputs: inputs)
                }
                for (name, spec) in stage.component.outputs {
                    guard let output = outputs[name] else {
                        throw EmbeddingError.unexpectedOutput(component: stage.name, "missing '\(name)'")
                    }
                    for (slot, r) in call.rows.enumerated() {
                        rows[r].tensors[name] = try unstack(
                            output, slot: slot, spec: spec, call: call, length: rows[r].length, stage: stage.name)
                    }
                }
            }
        }
        return try rows.map { row in
            guard let vector = row.tensors["embedding"]?.floats, vector.count == block.dimension else {
                throw EmbeddingError.unexpectedOutput(
                    component: stages.last?.name ?? "", "no \(block.dimension)-wide embedding")
            }
            return vector
        }
    }

    /// Builds a derived input of `row` at the call length `length`.
    func derive(_ rule: Derivation, row: Row, length: Int?) throws -> HostTensor {
        let mask = row.tensors["attention_mask"]?.ints ?? []
        switch rule {
        case .mergeSoftTokens(let imageTokenId):
            guard let embeds = row.tensors["inputs_embeds"], let soft = row.tensors["soft_tokens"],
                let ids = row.tensors["input_ids"]?.ints, let hidden = embeds.shape.last, soft.shape.last == hidden
            else {
                throw EmbeddingError.unsupportedPipeline(
                    modality, "soft-token merge needs inputs_embeds, soft_tokens, input_ids")
            }
            var out = embeds.floats
            let source = soft.floats
            var next = 0
            for (position, id) in ids.enumerated() where id == imageTokenId {
                guard (next + 1) * hidden <= source.count else {
                    throw EmbeddingError.unexpectedOutput(
                        component: "image_encoder", "fewer soft tokens than image tokens")
                }
                out.replaceSubrange(
                    position * hidden..<(position + 1) * hidden, with: source[next * hidden..<(next + 1) * hidden])
                next += 1
            }
            return HostTensor(float32: out, shape: embeds.shape)
        case .additiveBias(let bias):
            let l = length ?? mask.count
            var values = [Float](repeating: bias.unmasked, count: l * l)
            for key in 0..<l where key >= mask.count || mask[key] == 0 {
                values.replaceSubrange(key * l..<(key + 1) * l, with: repeatElement(bias.masked, count: l))
            }
            return HostTensor(float32: values, shape: [l, 1, l])
        case .poolWeights:
            let l = length ?? mask.count
            let total = Float(max(mask.reduce(0) { $0 + Int($1) }, 1))
            let weights = (0..<l).map { $0 < mask.count ? Float(mask[$0]) / total : 0 }
            return HostTensor(float32: weights, shape: [l])
        }
    }

    /// The batch tensor of a call: each row padded along its leading `L` axis to the call length,
    /// then padding rows up to the call batch.
    func stack(_ values: [HostTensor], spec: TensorSpec, call: Call, pad: Int32) throws -> HostTensor {
        let rowShape = try spec.shape.dropFirst().map { axis -> Int in
            switch axis {
            case .length:
                guard let l = call.length else {
                    throw EmbeddingError.invalidMetadata("\(spec) has L but its functions have no length")
                }
                return l
            case .fixed(let n): return n
            case .batch, .secondBatch: throw EmbeddingError.invalidMetadata("\(spec) has a second batch axis")
            }
        }
        guard spec.shape.first == .batch else { throw EmbeddingError.invalidMetadata("\(spec) does not start with B") }
        let rowCount = rowShape.reduce(1, *)
        let integer = values.first?.ints != nil
        var ints: [Int32] = []
        var floats: [Float] = []
        for value in values {
            let padded = try padRow(value, to: rowShape, pad: pad, spec: spec)
            if integer { ints += padded.ints ?? [] } else { floats += padded.floats }
        }
        let missing = (call.batch - values.count) * rowCount
        let shape = [call.batch] + rowShape
        if integer {
            return HostTensor(int32: ints + [Int32](repeating: pad, count: missing), shape: shape)
        }
        return HostTensor(float32: floats + [Float](repeating: Float(pad), count: missing), shape: shape)
    }

    private func padRow(_ value: HostTensor, to shape: [Int], pad: Int32, spec: TensorSpec) throws -> HostTensor {
        if value.shape == shape { return value }
        guard spec.shape.dropFirst().first == .length, value.shape.count == shape.count,
            Array(value.shape.dropFirst()) == Array(shape.dropFirst()), value.shape[0] <= shape[0]
        else {
            throw EmbeddingError.unexpectedOutput(
                component: "\(spec)", "row shape \(value.shape) does not fit \(shape)")
        }
        let extra = (shape[0] - value.shape[0]) * shape.dropFirst().reduce(1, *)
        switch value.storage {
        case .int32(let v): return HostTensor(int32: v + [Int32](repeating: pad, count: extra), shape: shape)
        case .float32(let v): return HostTensor(float32: v + [Float](repeating: Float(pad), count: extra), shape: shape)
        }
    }

    /// Row `slot` of a call output, cut back to the row's `length` along a leading `L` axis.
    func unstack(_ output: HostTensor, slot: Int, spec: TensorSpec, call: Call, length: Int?, stage: String) throws
        -> HostTensor
    {
        guard output.shape.first == call.batch, slot < call.batch else {
            throw EmbeddingError.unexpectedOutput(component: stage, "shape \(output.shape) for batch \(call.batch)")
        }
        var rowShape = Array(output.shape.dropFirst())
        let rowCount = rowShape.reduce(1, *)
        var values = Array(output.floats[slot * rowCount..<(slot + 1) * rowCount])
        if spec.shape.dropFirst().first == .length, let length, length < rowShape[0] {
            values.removeLast((rowShape[0] - length) * rowShape.dropFirst().reduce(1, *))
            rowShape[0] = length
        }
        return HostTensor(float32: values, shape: rowShape)
    }
}

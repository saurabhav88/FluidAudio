@preconcurrency import CoreML
import Foundation

/// One audio window of at most 15 s, prepared once: the preprocessor and encoder
/// have run and their output is kept so several decoder passes (plain and
/// phrase-boosted) can reuse it without running the encoder again (#3338).
///
/// Immutable and `Sendable`: the encoder output is held as a private copy of its
/// bytes, not as the Core ML buffer, so a later prediction cannot recycle or
/// overwrite it and nothing mutable crosses an isolation boundary. Bound to the
/// model set that made it; `decode` refuses it under any other.
public struct PreparedWindow: Sendable {
    /// Number of input samples the window was prepared from (before padding).
    public let sampleCount: Int
    /// The language option every decode of this window uses.
    public let decodeLanguage: Language?

    let encoderBytes: Data
    let encoderShape: [Int]
    let encoderStrides: [Int]
    let encoderDataType: Int
    let encoderSequenceLength: Int
    let actualAudioFrames: Int
    let modelsID: UUID
}

extension AsrManager {

    /// Run the preprocessor and encoder once for `samples` (at most
    /// `ASRConstants.maxModelSamples`) and keep the encoder output.
    ///
    /// Uses the ordinary single-window preparation: frame alignment, padding to
    /// the model length and the shared front end. Requires the Parakeet v3
    /// model. Too-short input throws `ASRError.invalidAudioData`, as ordinary
    /// transcription does; longer-than-one-window input throws
    /// `ASRError.processingFailed` instead of being truncated.
    public func prepareWindow(
        samples: [Float],
        decodeLanguage: Language? = nil
    ) async throws -> PreparedWindow {
        guard isAvailable else { throw ASRError.notInitialized }
        if let version = asrModels?.version, version != .v3 {
            throw ASRError.processingFailed(Self.phraseBoostUnsupportedMessage(version))
        }
        let minimumRequiredSamples = ASRConstants.minimumRequiredSamples(forSampleRate: config.sampleRate)
        guard samples.count >= minimumRequiredSamples else { throw ASRError.invalidAudioData }
        guard samples.count <= ASRConstants.maxModelSamples else {
            throw ASRError.processingFailed(
                "prepareWindow accepts at most \(ASRConstants.maxModelSamples) samples; got \(samples.count)")
        }

        // The identity is captured before the first suspension: a reload or
        // cleanup while this window is being prepared must make it fail, never
        // stamp old encoder output with the new model set's identity.
        let preparingModelsID = loadedModelsID

        let (alignedSamples, frameAlignedLength) = frameAlignedAudio(samples)
        let paddedAudio = padAudioIfNeeded(alignedSamples, targetLength: ASRConstants.maxModelSamples)
        let preprocessorInput = try await preparePreprocessorInput(paddedAudio, actualLength: frameAlignedLength)
        let preprocessorAudioArray = preprocessorInput.featureValue(for: "audio_signal")?.multiArrayValue

        // The input array goes back to the shared cache on every exit, as in
        // `executeMLInferenceWithTimings`.
        let frontEnd: (bytes: Data, output: MLMultiArray, length: Int)
        do {
            let (encoderOutput, encoderSequenceLength) = try await runFrontEnd(preprocessorInput)
            // The encoder's array may be padded between rows (strided), so copy
            // its whole backing span and keep the strides with it.
            let byteCount =
                Self.backingElementCount(
                    shape: encoderOutput.shape.map(\.intValue), strides: encoderOutput.strides.map(\.intValue))
                * Self.byteWidth(of: encoderOutput.dataType)
            guard byteCount > 0 else {
                throw ASRError.processingFailed("Unsupported encoder output data type")
            }
            frontEnd = (Data(bytes: encoderOutput.dataPointer, count: byteCount), encoderOutput, encoderSequenceLength)
            // A cancellation that arrived during the last front-end call must
            // not return a window. Ordinary transcription is not changed.
            try Task.checkCancellation()
        } catch {
            if let preprocessorAudioArray { await sharedMLArrayCache.returnArray(preprocessorAudioArray) }
            throw error
        }
        let encoderOutput = frontEnd.output
        let shape = encoderOutput.shape.map(\.intValue)
        let strides = encoderOutput.strides.map(\.intValue)
        let dataType = encoderOutput.dataType.rawValue
        if let preprocessorAudioArray { await sharedMLArrayCache.returnArray(preprocessorAudioArray) }
        try Task.checkCancellation()
        guard isAvailable, loadedModelsID == preparingModelsID else {
            throw ASRError.processingFailed("The model set changed while preparing this window")
        }

        return PreparedWindow(
            sampleCount: samples.count,
            decodeLanguage: decodeLanguage,
            encoderBytes: frontEnd.bytes,
            encoderShape: shape,
            encoderStrides: strides,
            encoderDataType: dataType,
            encoderSequenceLength: frontEnd.length,
            actualAudioFrames: ASRConstants.calculateEncoderFrames(from: frameAlignedLength),
            modelsID: preparingModelsID)
    }

    /// Decode a prepared window: a fresh decoder state every call, the window's
    /// language option, and, when `phraseBoost` is given, a fresh call-local
    /// boost runtime. The preprocessor and encoder do not run.
    public func decode(
        _ window: PreparedWindow,
        phraseBoost: PhraseBoostConfig? = nil
    ) async throws -> [TokenTiming] {
        try await decode(window, request: phraseBoost.map { PhraseBoostRequest(config: $0) })
    }

    internal func decode(_ window: PreparedWindow, request: PhraseBoostRequest?) async throws -> [TokenTiming] {
        guard isAvailable else { throw ASRError.notInitialized }
        guard window.modelsID == loadedModelsID else {
            throw ASRError.processingFailed("This prepared window was made with a different model set")
        }
        let encoderOutput = try Self.encoderArray(from: window, observer: preparedBufferObserver)
        var decoderState = try TdtDecoderState(decoderLayers: decoderLayerCount)
        let hypothesis = try await tdtDecodeWithTimings(
            encoderOutput: encoderOutput,
            encoderSequenceLength: window.encoderSequenceLength,
            actualAudioFrames: window.actualAudioFrames,
            originalAudioSamples: [],
            decoderState: &decoderState,
            isLastChunk: true,
            language: window.decodeLanguage,
            phraseBoost: request
        )
        let result = processTranscriptionResult(
            tokenIds: hypothesis.ySequence,
            timestamps: hypothesis.timestamps,
            confidences: hypothesis.tokenConfidences,
            tokenDurations: hypothesis.tokenDurations,
            encoderSequenceLength: window.encoderSequenceLength,
            audioSampleCount: window.sampleCount,
            processingTime: 0
        )
        return result.tokenTimings ?? []
    }

    /// A fresh Core ML array over a private copy of the window's encoder bytes,
    /// with the original strides, so a decode can never write into the window.
    ///
    /// The copy is owned by a `PreparedDecodeBuffer`, which frees it exactly
    /// once in its `deinit`. The array's deallocator only releases that owner and
    /// never frees memory itself, so if the initializer throws (and drops the
    /// deallocator) or succeeds (and later calls it), the memory is freed once.
    internal static func encoderArray(
        from window: PreparedWindow,
        observer: AsyncStream<PreparedBufferEvent>.Continuation? = nil,
        makeArray: (UnsafeMutableRawPointer, [NSNumber], MLMultiArrayDataType, [NSNumber], @escaping (UnsafeMutableRawPointer) -> Void) throws -> MLMultiArray = {
            try MLMultiArray(dataPointer: $0, shape: $1, dataType: $2, strides: $3, deallocator: $4)
        }
    ) throws -> MLMultiArray {
        guard let dataType = MLMultiArrayDataType(rawValue: window.encoderDataType) else {
            throw ASRError.processingFailed("Unknown encoder data type")
        }
        let expected =
            backingElementCount(shape: window.encoderShape, strides: window.encoderStrides) * byteWidth(of: dataType)
        guard expected == window.encoderBytes.count, expected > 0 else {
            throw ASRError.processingFailed("Prepared window layout does not match its bytes")
        }
        let owner = PreparedDecodeBuffer(byteCount: expected, observer: observer)
        window.encoderBytes.withUnsafeBytes { raw in
            owner.pointer.copyMemory(from: raw.baseAddress!, byteCount: expected)
        }
        return try makeArray(
            owner.pointer,
            window.encoderShape.map { NSNumber(value: $0) },
            dataType,
            window.encoderStrides.map { NSNumber(value: $0) },
            { _ in withExtendedLifetime(owner) {} })
    }

    /// Elements spanned by a strided array: one past its last addressed element.
    private static func backingElementCount(shape: [Int], strides: [Int]) -> Int {
        guard !shape.contains(0) else { return 0 }
        return zip(shape, strides).reduce(1) { $0 + ($1.0 - 1) * $1.1 }
    }

    private static func byteWidth(of dataType: MLMultiArrayDataType) -> Int {
        switch dataType {
        case .float16: return 2
        case .float32, .int32: return 4
        case .double: return 8
        default: return 0
        }
    }
}

/// Sole owner of one prepared-window decode buffer: allocates in `init`, frees
/// in `deinit`, so every path frees it exactly once (#3338).
final class PreparedDecodeBuffer {
    let pointer: UnsafeMutableRawPointer
    private let observer: AsyncStream<AsrManager.PreparedBufferEvent>.Continuation?

    init(byteCount: Int, observer: AsyncStream<AsrManager.PreparedBufferEvent>.Continuation?) {
        pointer = UnsafeMutableRawPointer.allocate(byteCount: byteCount, alignment: 64)
        self.observer = observer
        observer?.yield(.allocated)
    }

    deinit {
        pointer.deallocate()
        observer?.yield(.freed)
    }
}

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
    /// Where the encoder output came from (#3338).
    public enum Origin: Sendable, Equatable {
        /// `prepareWindow`: one standalone window.
        case prepared
        /// Captured from ordinary single-window batch transcription.
        case batchSingleWindow
        /// Captured from chunk `index` of ordinary multi-chunk batch transcription.
        case batchChunk(index: Int)
    }

    /// Number of input samples the window was prepared from (before padding),
    /// including any prepended context.
    public let sampleCount: Int
    /// The language option every decode of this window uses.
    public let decodeLanguage: Language?
    public let origin: Origin
    /// Global encoder frame of the window's first frame (0 for a standalone window).
    public let globalFrameOffset: Int

    // Decode geometry, exactly as the batch path decoded this window.
    let contextFrameAdjustment: Int
    let isLastChunk: Bool
    let emitTokensAfterGlobalFrame: Int?
    let initialTimeIndexOverride: Int?

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
        let window: PreparedWindow
        do {
            let (encoderOutput, encoderSequenceLength) = try await runFrontEnd(preprocessorInput)
            window = try Self.snapshotWindow(
                encoderOutput: encoderOutput, encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: ASRConstants.calculateEncoderFrames(from: frameAlignedLength),
                sampleCount: samples.count, decodeLanguage: decodeLanguage, origin: .prepared,
                globalFrameOffset: 0, contextFrameAdjustment: 0, isLastChunk: true,
                emitTokensAfterGlobalFrame: nil, initialTimeIndexOverride: nil, modelsID: preparingModelsID)
            // A cancellation that arrived during the last front-end call must
            // not return a window. Ordinary transcription is not changed.
            try Task.checkCancellation()
        } catch {
            if let preprocessorAudioArray { await sharedMLArrayCache.returnArray(preprocessorAudioArray) }
            throw error
        }
        if let preprocessorAudioArray { await sharedMLArrayCache.returnArray(preprocessorAudioArray) }
        try Task.checkCancellation()
        guard isAvailable, loadedModelsID == preparingModelsID else {
            throw ASRError.processingFailed("The model set changed while preparing this window")
        }
        return window
    }

    /// Immutable copy of one encoder output plus the geometry it was (or will
    /// be) decoded with. The single place a window's bytes are taken (#3338).
    /// The encoder's array may be padded between rows (strided), so the whole
    /// backing span is copied and the strides kept with it.
    static func snapshotWindow(
        encoderOutput: MLMultiArray, encoderSequenceLength: Int, actualAudioFrames: Int, sampleCount: Int,
        decodeLanguage: Language?, origin: PreparedWindow.Origin, globalFrameOffset: Int,
        contextFrameAdjustment: Int, isLastChunk: Bool, emitTokensAfterGlobalFrame: Int?,
        initialTimeIndexOverride: Int?, modelsID: UUID
    ) throws -> PreparedWindow {
        let shape = encoderOutput.shape.map(\.intValue)
        let strides = encoderOutput.strides.map(\.intValue)
        let byteCount =
            backingElementCount(shape: shape, strides: strides) * byteWidth(of: encoderOutput.dataType)
        guard byteCount > 0 else {
            throw ASRError.processingFailed("Unsupported encoder output data type")
        }
        return PreparedWindow(
            sampleCount: sampleCount,
            decodeLanguage: decodeLanguage,
            origin: origin,
            globalFrameOffset: globalFrameOffset,
            contextFrameAdjustment: contextFrameAdjustment,
            isLastChunk: isLastChunk,
            emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
            initialTimeIndexOverride: initialTimeIndexOverride,
            encoderBytes: Data(bytes: encoderOutput.dataPointer, count: byteCount),
            encoderShape: shape,
            encoderStrides: strides,
            encoderDataType: encoderOutput.dataType.rawValue,
            encoderSequenceLength: encoderSequenceLength,
            actualAudioFrames: actualAudioFrames,
            modelsID: modelsID)
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
            contextFrameAdjustment: window.contextFrameAdjustment,
            isLastChunk: window.isLastChunk,
            globalFrameOffset: window.globalFrameOffset,
            language: window.decodeLanguage,
            emitTokensAfterGlobalFrame: window.emitTokensAfterGlobalFrame,
            initialTimeIndexOverride: window.initialTimeIndexOverride,
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

/// What `transcribeCapturingWindows` hands back (#3338 C.2b): the ordinary result
/// and the encoder windows that produced it.
public struct CapturedTranscription: Sendable {
    /// Exactly what `transcribe` would have returned.
    public let result: ASRResult
    /// The windows the batch path decoded, in chunk order. Empty when ineligible.
    public let windows: [PreparedWindow]
    /// Why these windows do not explain the result, if they do not (dual-decode
    /// arbitration, a seam repair that changed tokens). Nil means eligible.
    public let ineligibleReason: String?
}

/// Collects captured windows from the single-window path and from concurrent
/// chunk tasks. Sendable through an AsyncStream continuation, so no lock and no
/// `@unchecked Sendable`.
final class WindowCapture: Sendable {
    enum Event: Sendable {
        case window(PreparedWindow, tokens: [Int])
        case ineligible(String)
    }

    private let stream: AsyncStream<Event>
    private let continuation: AsyncStream<Event>.Continuation

    init() {
        (stream, continuation) = AsyncStream<Event>.makeStream()
    }

    func record(window: PreparedWindow, tokens: [Int]) {
        continuation.yield(.window(window, tokens: tokens))
    }

    static let modelSetChanged = "model set changed during capture"

    func markIneligible(_ reason: String) {
        continuation.yield(.ineligible(reason))
    }

    /// Ends the capture and returns everything recorded, windows in chunk order.
    func finish() async -> (windows: [(window: PreparedWindow, tokens: [Int])], ineligibleReason: String?) {
        continuation.finish()
        var windows: [(window: PreparedWindow, tokens: [Int])] = []
        var reason: String?
        for await event in stream {
            switch event {
            case .window(let window, let tokens): windows.append((window, tokens))
            case .ineligible(let why): reason = reason ?? why
            }
        }
        windows.sort { Self.order($0.window.origin) < Self.order($1.window.origin) }
        return (windows, reason)
    }

    private static func order(_ origin: PreparedWindow.Origin) -> Int {
        switch origin {
        case .prepared, .batchSingleWindow: return -1
        case .batchChunk(let index): return index
        }
    }
}

/// A capture bound to the window it will record (single window or chunk index).
struct WindowCaptureBinding: Sendable {
    let capture: WindowCapture
    let origin: PreparedWindow.Origin
}

extension AsrManager {

    /// Ordinary batch transcription that also hands back the encoder windows it
    /// decoded (#3338 C.2b). The transcript is exactly `transcribe`'s; no extra
    /// encoder runs. Requires the Parakeet v3 model; any other version, and
    /// dual-decode arbitration or a token-changing seam repair, return the
    /// ordinary result with no windows and a reason.
    ///
    /// Always starts from a fresh decoder state, because a captured window is
    /// replayed from a fresh state; caller-owned predictor context could not be
    /// reproduced.
    public func transcribeCapturingWindows(
        _ audioSamples: [Float],
        language: Language? = nil
    ) async throws -> CapturedTranscription {
        let run = try await transcribeCapturingWindowsWithTokens(audioSamples, language: language)
        if let reason = run.reason {
            return CapturedTranscription(result: run.result, windows: [], ineligibleReason: reason)
        }
        return CapturedTranscription(result: run.result, windows: run.windows.map(\.window), ineligibleReason: nil)
    }

    /// The capturing run, with the tokens each window produced (tests read them).
    internal func transcribeCapturingWindowsWithTokens(
        _ audioSamples: [Float], language: Language? = nil
    ) async throws -> (result: ASRResult, windows: [(window: PreparedWindow, tokens: [Int])], reason: String?) {
        // The parent's identity at entry: worker windows carry it, and a reload
        // anywhere during the run makes every window ineligible.
        let modelsIDAtEntry = loadedModelsID
        let capture = WindowCapture()
        let eligibleModel = asrModels?.version == .v3
        if !eligibleModel { capture.markIneligible("phrase boosting requires the Parakeet v3 model") }
        var state = try TdtDecoderState(decoderLayers: decoderLayerCount)
        let result = try await transcribe(
            audioSamples, decoderState: &state, language: language, phraseBoost: nil,
            windowCapture: eligibleModel ? capture : nil)
        if loadedModelsID != modelsIDAtEntry { capture.markIneligible(WindowCapture.modelSetChanged) }
        let captured = await capture.finish()
        if captured.windows.contains(where: { $0.window.modelsID != modelsIDAtEntry }) {
            return (result, [], WindowCapture.modelSetChanged)
        }
        guard captured.ineligibleReason == nil else { return (result, [], captured.ineligibleReason) }
        return (result, captured.windows, nil)
    }
}

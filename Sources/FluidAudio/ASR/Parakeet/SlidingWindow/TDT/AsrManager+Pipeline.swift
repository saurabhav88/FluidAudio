@preconcurrency import CoreML
import Foundation

extension AsrManager {

    internal func executeMLInferenceWithTimings(
        _ paddedAudio: [Float],
        originalLength: Int? = nil,
        actualAudioFrames: Int? = nil,
        decoderState: inout TdtDecoderState,
        contextFrameAdjustment: Int = 0,
        isLastChunk: Bool = false,
        globalFrameOffset: Int = 0,
        language: Language? = nil,
        emitTokensAfterGlobalFrame: Int? = nil,
        initialTimeIndexOverride: Int? = nil,
        phraseBoost: PhraseBoostRequest? = nil,
        windowCapture: WindowCaptureBinding? = nil
    ) async throws -> (hypothesis: TdtHypothesis, encoderSequenceLength: Int) {
        // Identity before the first suspension, so a reload while this window
        // is being decoded can never relabel its encoder output (#3338).
        let modelsIDAtEntry = loadedModelsID

        let preprocessorInput = try await preparePreprocessorInput(
            paddedAudio, actualLength: originalLength)

        let preprocessorAudioArray = preprocessorInput.featureValue(for: "audio_signal")?.multiArrayValue

        do {
            let (rawEncoderOutput, encoderSequenceLength) = try await runFrontEnd(preprocessorInput)

            // Calculate actual audio frames if not provided using shared constants
            let actualFrames =
                actualAudioFrames ?? ASRConstants.calculateEncoderFrames(from: originalLength ?? paddedAudio.count)

            let hypothesis = try await tdtDecodeWithTimings(
                encoderOutput: rawEncoderOutput,
                encoderSequenceLength: encoderSequenceLength,
                actualAudioFrames: actualFrames,
                originalAudioSamples: paddedAudio,
                decoderState: &decoderState,
                contextFrameAdjustment: contextFrameAdjustment,
                isLastChunk: isLastChunk,
                globalFrameOffset: globalFrameOffset,
                language: language,
                emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
                initialTimeIndexOverride: initialTimeIndexOverride,
                phraseBoost: phraseBoost
            )

            // #3338 C.2b: hand over this decode's own encoder output, with the
            // geometry it was decoded with. No second encoder run; nothing is
            // copied unless a caller asked for capture.
            if let windowCapture {
                let window = try Self.snapshotWindow(
                    encoderOutput: rawEncoderOutput, encoderSequenceLength: encoderSequenceLength,
                    actualAudioFrames: actualFrames, sampleCount: originalLength ?? paddedAudio.count,
                    decodeLanguage: language, origin: windowCapture.origin, globalFrameOffset: globalFrameOffset,
                    contextFrameAdjustment: contextFrameAdjustment, isLastChunk: isLastChunk,
                    emitTokensAfterGlobalFrame: emitTokensAfterGlobalFrame,
                    initialTimeIndexOverride: initialTimeIndexOverride, modelsID: modelsIDAtEntry)
                if loadedModelsID == modelsIDAtEntry {
                    windowCapture.capture.record(window: window, tokens: hypothesis.ySequence)
                } else {
                    windowCapture.capture.markIneligible(WindowCapture.modelSetChanged)
                }
            }

            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }

            return (hypothesis, encoderSequenceLength)
        } catch {
            if let preprocessorAudioArray {
                await sharedMLArrayCache.returnArray(preprocessorAudioArray)
            }
            throw error
        }
    }

    /// Preprocessor then encoder (or the fused front end) for one prepared
    /// input: the single owner of front-end inference, shared by ordinary
    /// decoding and `prepareWindow` (#3338). Counts each model call for tests.
    internal func runFrontEnd(
        _ preprocessorInput: MLFeatureProvider
    ) async throws -> (encoderOutput: MLMultiArray, encoderSequenceLength: Int) {
        guard let preprocessorModel = preprocessorModel else {
            throw ASRError.notInitialized
        }

        try Task.checkCancellation()
        let preprocessorOutput = try await preprocessorModel.compatPrediction(
            from: preprocessorInput,
            options: predictionOptions
        )
        frontEndCounters.preprocessor += 1
        if let frontEndObserver { await frontEndObserver(.afterPreprocessor) }

        let encoderOutputProvider: MLFeatureProvider
        if let encoderModel = encoderModel {
            // Split frontend: run separate encoder
            let encoderInput = try prepareEncoderInput(
                encoder: encoderModel,
                preprocessorOutput: preprocessorOutput,
                originalInput: preprocessorInput
            )

            try Task.checkCancellation()
            encoderOutputProvider = try await encoderModel.compatPrediction(
                from: encoderInput,
                options: predictionOptions
            )
            frontEndCounters.encoder += 1
            if let frontEndObserver { await frontEndObserver(.afterEncoder) }
        } else {
            // Fused frontend: preprocessor output already contains encoder features
            encoderOutputProvider = preprocessorOutput
        }

        let rawEncoderOutput = try extractFeatureValue(
            from: encoderOutputProvider, key: "encoder", errorMessage: "Invalid encoder output")
        let encoderLength = try extractFeatureValue(
            from: encoderOutputProvider, key: "encoder_length",
            errorMessage: "Invalid encoder output length")

        return (rawEncoderOutput, encoderLength[0].intValue)
    }

    private func prepareEncoderInput(
        encoder: MLModel,
        preprocessorOutput: MLFeatureProvider,
        originalInput: MLFeatureProvider
    ) throws -> MLFeatureProvider {
        let inputDescriptions = encoder.modelDescription.inputDescriptionsByName

        let missingNames = inputDescriptions.keys.filter { name in
            preprocessorOutput.featureValue(for: name) == nil
        }

        if missingNames.isEmpty {
            return preprocessorOutput
        }

        var features: [String: MLFeatureValue] = [:]

        for name in inputDescriptions.keys {
            if let value = preprocessorOutput.featureValue(for: name) {
                features[name] = value
                continue
            }

            if let fallback = originalInput.featureValue(for: name) {
                features[name] = fallback
                continue
            }

            let availableInputs = preprocessorOutput.featureNames.sorted().joined(separator: ", ")
            let fallbackInputs = originalInput.featureNames.sorted().joined(separator: ", ")
            throw ASRError.processingFailed(
                "Missing required encoder input: \(name). Available inputs: \(availableInputs), "
                    + "fallback inputs: \(fallbackInputs)"
            )
        }

        return try MLDictionaryFeatureProvider(dictionary: features)
    }

    /// Align audio samples to encoder frame boundaries by zero-padding to the next frame boundary.
    /// Returns the aligned samples and the frame-aligned length.
    /// - Parameters:
    ///   - audioSamples: Raw audio samples
    ///   - allowAlignment: When false, skip alignment (e.g. when previous context exists)
    nonisolated internal func frameAlignedAudio(
        _ audioSamples: [Float], allowAlignment: Bool = true
    ) -> (samples: [Float], frameAlignedLength: Int) {
        let originalLength = audioSamples.count
        let frameAlignedCandidate =
            ((originalLength + ASRConstants.samplesPerEncoderFrame - 1)
                / ASRConstants.samplesPerEncoderFrame) * ASRConstants.samplesPerEncoderFrame
        if allowAlignment && frameAlignedCandidate > originalLength
            && frameAlignedCandidate <= ASRConstants.maxModelSamples
        {
            let aligned = audioSamples + Array(repeating: 0, count: frameAlignedCandidate - originalLength)
            return (aligned, frameAlignedCandidate)
        }
        return (audioSamples, originalLength)
    }

    nonisolated internal func padAudioIfNeeded(_ audioSamples: [Float], targetLength: Int) -> [Float] {
        guard audioSamples.count < targetLength else { return audioSamples }
        return audioSamples + Array(repeating: 0, count: targetLength - audioSamples.count)
    }

}

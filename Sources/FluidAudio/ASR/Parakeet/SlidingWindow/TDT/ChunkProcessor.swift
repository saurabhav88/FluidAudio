import Foundation

struct ChunkProcessor {
    let sampleSource: AudioSampleSource
    let totalSamples: Int

    private let logger = AppLogger(category: "ChunkProcessor")
    typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float, duration: Int)
    private struct TaskResult: Sendable {
        let index: Int
        let tokens: [TokenWindow]
        let workerIndex: Int
    }
    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }
    struct ChunkStartDecision {
        let start: Int
        let useWarmupPrefix: Bool
    }

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (frame-aligned) to stay under encoder limit
    // - 2.0s overlap (frame-aligned) to give the decoder slack when merging windows
    let overlapSeconds: Double = 2.0

    /// Context samples prepended from previous chunk for mel spectrogram stability (80ms = 1 encoder frame).
    /// The FastConformer encoder's depthwise convolutions need left context for stable output.
    /// Without this, the first frames of a chunk may produce features that cause all-blank predictions.
    ///
    /// Issue #594: on `parakeet-tdt-0.6b-v3-coreml` multilingual long-form
    /// audio this prepend can shift the encoder's first-frame distribution
    /// enough to make the SOS-primed decoder drift to its English-biased prior.
    /// Callers can opt out via `ASRConfig.melChunkContext = false` to
    /// use the v3/no-mel boundary warmup path below.
    private let melContextSamples: Int = ASRConstants.samplesPerEncoderFrame  // 1280 samples = 80ms

    /// Default v3/no-mel path warmup size. v42 intentionally keeps the
    /// non-arbitrated path warmup-free; the opt-in arbitration path's path B
    /// owns the explicit 7-frame warmup probe.
    private let noMelWarmupPrefixFrames: Int = 0

    private var maxModelSamples: Int { ASRConstants.maxModelSamples }

    /// #1237 empty-chunk recovery (EnviousWispr fork carry, upstream #746 still open):
    /// minimum encoder frames a window must have produced before an all-blank decode is
    /// treated as recoverable speech (rather than genuine trailing silence / no speech).
    /// 25 frames = 2.0s at `secondsPerEncoderFrame` (0.08s).
    static let recoverableEmptyMinFrames: Int = 25

    /// Static logger for the (static) #1237 recovery path — the instance
    /// `logger` is not reachable from `static transcribeChunk`.
    private static let recoveryLogger = AppLogger(category: "ChunkProcessor")

    private var noMelWarmupPrefixSamples: Int {
        noMelWarmupPrefixFrames * ASRConstants.samplesPerEncoderFrame
    }

    /// Effective per-chunk mel-context size based on the runtime flag.
    private func effectiveMelContextSamples(melChunkContext: Bool) -> Int {
        melChunkContext ? melContextSamples : 0
    }

    private func effectiveWarmupPrefixSamples(melChunkContext: Bool, modelVersion: AsrModelVersion?) -> Int {
        guard !melChunkContext, case .v3? = modelVersion else { return 0 }
        return noMelWarmupPrefixSamples
    }

    /// Frame-aligned chunk size that reserves space for the context prepend
    /// (or fills the encoder window when context is disabled).
    private func chunkSamples(melChunkContext: Bool, modelVersion: AsrModelVersion?) -> Int {
        let reserved = effectiveMelContextSamples(melChunkContext: melChunkContext)
        let maxActualChunk = maxModelSamples - reserved
        let raw = max(maxActualChunk - ASRConstants.melHopSize, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    private func overlapSamples(forChunkSamples chunkSamples: Int) -> Int {
        let requested = Int(overlapSeconds * Double(ASRConstants.sampleRate))
        let capped = min(requested, chunkSamples / 2)
        return capped / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    private func strideSamples(forChunkSamples chunkSamples: Int) -> Int {
        let raw = max(chunkSamples - overlapSamples(forChunkSamples: chunkSamples), ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    func chunkLayout(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) -> (
        chunkSamples: Int,
        strideSamples: Int,
        melContextSamples: Int,
        warmupPrefixSamples: Int
    ) {
        let chunkSamples = self.chunkSamples(melChunkContext: melChunkContext, modelVersion: modelVersion)
        let warmupPrefixSamples = effectiveWarmupPrefixSamples(
            melChunkContext: melChunkContext,
            modelVersion: modelVersion
        )
        let stride = strideSamples(forChunkSamples: chunkSamples)
        return (
            chunkSamples: chunkSamples,
            strideSamples: stride,
            melContextSamples: effectiveMelContextSamples(melChunkContext: melChunkContext),
            warmupPrefixSamples: warmupPrefixSamples
        )
    }

    private func chunkStarts(
        warmupPrefixSamples: Int,
        chunkSamples: Int,
        strideSamples: Int,
        preferSilenceAlignment: Bool
    ) throws -> [ChunkStartDecision] {
        guard preferSilenceAlignment || warmupPrefixSamples > 0 else {
            return regularChunkStarts(strideSamples: strideSamples)
        }
        return try silenceAlignedChunkStarts(
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            canUseWarmupPrefix: warmupPrefixSamples > 0
        )
    }

    func regularChunkStarts(strideSamples: Int) -> [ChunkStartDecision] {
        var starts = [ChunkStartDecision(start: 0, useWarmupPrefix: false)]
        var start = strideSamples
        while start < totalSamples {
            starts.append(ChunkStartDecision(start: start, useWarmupPrefix: false))
            start += strideSamples
        }
        return starts
    }

    func silenceAlignedChunkStarts(
        chunkSamples: Int,
        strideSamples: Int,
        canUseWarmupPrefix: Bool
    ) throws -> [ChunkStartDecision] {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let silenceSearchRadiusFrames = max(1, Int((4.0 * Double(ASRConstants.sampleRate)) / Double(frameSamples)))
        let valleySearchRadiusFrames = max(1, Int((0.5 * Double(ASRConstants.sampleRate)) / Double(frameSamples)))
        let halfEnergyWindowSamples = frameSamples
        let minimumOverlapSamples = frameSamples * 6

        var starts = [ChunkStartDecision(start: 0, useWarmupPrefix: false)]
        var previousStart = 0
        var target = strideSamples

        while target < totalSamples {
            let targetFrame = target / frameSamples
            let latestCoveredStart = previousStart + chunkSamples - minimumOverlapSamples
            let targetStart = min(max(targetFrame * frameSamples, previousStart + frameSamples), latestCoveredStart)

            let silenceCandidate = try bestBoundaryCandidate(
                targetFrame: targetFrame,
                searchRadiusFrames: silenceSearchRadiusFrames,
                previousStart: previousStart,
                latestCoveredStart: latestCoveredStart,
                halfEnergyWindowSamples: halfEnergyWindowSamples
            )
            let foundNearSilence = isNearSilenceBoundary(silenceCandidate)

            var bestStart: Int
            var useWarmupPrefix = false
            if foundNearSilence {
                let shouldWarmup =
                    canUseWarmupPrefix ? (try shouldUseWarmupPrefix(at: silenceCandidate.start)) : false
                let compressesSpeechTail: Bool
                if shouldWarmup && silenceCandidate.start < targetStart {
                    compressesSpeechTail = try wouldCompressSpeechTail(
                        candidateStart: silenceCandidate.start,
                        targetStart: targetStart,
                        chunkSamples: chunkSamples,
                        minimumOverlapSamples: minimumOverlapSamples,
                        medianScore: silenceCandidate.medianScore,
                        halfEnergyWindowSamples: halfEnergyWindowSamples
                    )
                } else {
                    compressesSpeechTail = false
                }
                if compressesSpeechTail {
                    bestStart = targetStart
                } else {
                    bestStart = silenceCandidate.start
                    useWarmupPrefix = shouldWarmup
                }
            } else {
                let valleyCandidate = try bestBoundaryCandidate(
                    targetFrame: targetFrame,
                    searchRadiusFrames: valleySearchRadiusFrames,
                    previousStart: previousStart,
                    latestCoveredStart: latestCoveredStart,
                    halfEnergyWindowSamples: halfEnergyWindowSamples
                )
                bestStart = isUsableValleyBoundary(valleyCandidate) ? valleyCandidate.start : targetStart
            }

            if bestStart <= previousStart {
                bestStart = min(previousStart + strideSamples, totalSamples)
            }

            starts.append(
                ChunkStartDecision(
                    start: bestStart,
                    useWarmupPrefix: useWarmupPrefix
                )
            )
            previousStart = bestStart
            target += strideSamples
        }

        return starts
    }

    private func bestBoundaryCandidate(
        targetFrame: Int,
        searchRadiusFrames: Int,
        previousStart: Int,
        latestCoveredStart: Int,
        halfEnergyWindowSamples: Int
    ) throws -> (start: Int, score: Float, medianScore: Float) {
        let frameSamples = ASRConstants.samplesPerEncoderFrame
        let lowerFrame = max(1, targetFrame - searchRadiusFrames)
        let upperFrame = min((totalSamples - 1) / frameSamples, targetFrame + searchRadiusFrames)
        let targetStart = min(max(targetFrame * frameSamples, previousStart + frameSamples), latestCoveredStart)

        var bestStart = targetStart
        var bestScore = Float.greatestFiniteMagnitude
        var scores: [Float] = []

        if lowerFrame <= upperFrame {
            for frameIndex in lowerFrame...upperFrame {
                let candidate = frameIndex * frameSamples
                if candidate <= previousStart { continue }
                if candidate > latestCoveredStart { continue }
                let score = try boundaryEnergyScore(
                    centeredAt: candidate,
                    halfWindowSamples: halfEnergyWindowSamples
                )
                scores.append(score)
                if score < bestScore {
                    bestScore = score
                    bestStart = candidate
                }
            }
        }

        guard !scores.isEmpty else {
            return (targetStart, Float.greatestFiniteMagnitude, 0)
        }

        let sortedScores = scores.sorted()
        let medianScore = sortedScores[sortedScores.count / 2]
        return (bestStart, bestScore, medianScore)
    }

    private func isNearSilenceBoundary(_ candidate: (start: Int, score: Float, medianScore: Float)) -> Bool {
        candidate.score <= adaptiveBoundaryThreshold(medianScore: candidate.medianScore, ratio: 0.05)
    }

    private func isUsableValleyBoundary(_ candidate: (start: Int, score: Float, medianScore: Float)) -> Bool {
        candidate.score <= adaptiveBoundaryThreshold(medianScore: candidate.medianScore, ratio: 0.35)
    }

    private func adaptiveBoundaryThreshold(medianScore: Float, ratio: Float) -> Float {
        guard medianScore > 0 else { return 0 }
        return medianScore * ratio
    }

    private func wouldCompressSpeechTail(
        candidateStart: Int,
        targetStart: Int,
        chunkSamples: Int,
        minimumOverlapSamples: Int,
        medianScore: Float,
        halfEnergyWindowSamples: Int
    ) throws -> Bool {
        guard medianScore > 0 else { return false }

        let forcedNextBoundary = candidateStart + chunkSamples - minimumOverlapSamples
        guard forcedNextBoundary < totalSamples else { return false }

        let speechLikeThreshold = medianScore * 0.8
        let targetScore = try boundaryEnergyScore(
            centeredAt: targetStart,
            halfWindowSamples: halfEnergyWindowSamples
        )
        let forcedScore = try boundaryEnergyScore(
            centeredAt: forcedNextBoundary,
            halfWindowSamples: halfEnergyWindowSamples
        )
        return targetScore > speechLikeThreshold && forcedScore > speechLikeThreshold
    }

    private func shouldUseWarmupPrefix(at centerSample: Int) throws -> Bool {
        let lookaheadSamples = Int(0.5 * Double(ASRConstants.sampleRate))
        let minimumStableQuietSamples = Int(0.2 * Double(ASRConstants.sampleRate))
        let windowSamples = max(1, ASRConstants.sampleRate / 50)  // 20ms
        let quietRmsThreshold: Float = 0.003

        var offset = 0
        var quietSamples = 0

        while offset < lookaheadSamples {
            let start = centerSample + offset
            guard start < totalSamples else { break }

            let count = min(windowSamples, totalSamples - start, lookaheadSamples - offset)
            guard count > 0 else { break }

            let samples = try readSamples(offset: start, count: count)
            var sum: Float = 0
            for sample in samples {
                sum += sample * sample
            }
            let rms = sqrt(sum / Float(samples.count))
            guard rms < quietRmsThreshold else { break }

            quietSamples += samples.count
            if quietSamples >= minimumStableQuietSamples {
                return false
            }
            offset += samples.count
        }

        return true
    }

    private func boundaryEnergyScore(centeredAt centerSample: Int, halfWindowSamples: Int) throws -> Float {
        let start = max(0, centerSample - halfWindowSamples)
        let end = min(totalSamples, centerSample + halfWindowSamples)
        let count = end - start
        guard count > 0 else { return 0 }

        let samples = try readSamples(offset: start, count: count)
        var sum: Float = 0
        for sample in samples {
            sum += sample * sample
        }
        return sum / Float(count)
    }

    #if DEBUG
    internal func chunkLayoutForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) -> (
        chunkSamples: Int,
        strideSamples: Int,
        melContextSamples: Int,
        warmupPrefixSamples: Int
    ) {
        chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
    }

    internal func chunkStartsForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) throws -> [Int] {
        try chunkStartDecisionsForTesting(
            melChunkContext: melChunkContext,
            modelVersion: modelVersion
        ).map(\.start)
    }

    internal func chunkStartDecisionsForTesting(
        melChunkContext: Bool,
        modelVersion: AsrModelVersion?
    ) throws -> [(start: Int, useWarmupPrefix: Bool)] {
        let layout = chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
        return try chunkStarts(
            warmupPrefixSamples: layout.warmupPrefixSamples,
            chunkSamples: layout.chunkSamples,
            strideSamples: layout.strideSamples,
            preferSilenceAlignment: !melChunkContext && modelVersion == .v3
        ).map { ($0.start, $0.useWarmupPrefix) }
    }

    internal func mergeTokenWindowsForTesting(
        left: [(token: Int, timestamp: Int, confidence: Float, duration: Int)],
        right: [(token: Int, timestamp: Int, confidence: Float, duration: Int)],
        spliceSafeTokenIds: Set<Int>? = nil,
        caseVariantIds: [Int: Int]? = nil
    ) -> [(token: Int, timestamp: Int, confidence: Float, duration: Int)] {
        mergeChunks(left, right, spliceSafeTokenIds: spliceSafeTokenIds, caseVariantIds: caseVariantIds)
    }
    #endif

    /// Initialize with a streaming audio sample source for memory-efficient processing.
    init(sampleSource: AudioSampleSource) {
        self.sampleSource = sampleSource
        self.totalSamples = sampleSource.sampleCount
    }

    /// Convenience initializer for in-memory audio samples.
    init(audioSamples: [Float]) {
        self.init(sampleSource: ArrayAudioSampleSource(samples: audioSamples))
    }

    func process(
        using manager: AsrManager,
        startTime: Date,
        progressHandler: ((Double) async -> Void)? = nil,
        language: Language? = nil
    ) async throws -> ASRResult {
        let requestedConcurrency = max(1, await manager.parallelChunkConcurrency)
        let workers = await makeWorkerPool(using: manager, count: requestedConcurrency) ?? [manager]
        let decoderLayers = await manager.decoderLayerCount
        let maxModelSamples = self.maxModelSamples
        // Issue #594: opt-out of PR #264's 80ms mel-context prepend. For v3,
        // no-mel uses real-audio warmup plus silence-aligned chunk starts.
        let melChunkContext = await manager.melChunkContext
        let modelVersion = await manager.modelVersion
        let dualDecodeArbitration = await manager.dualDecodeArbitration

        // Dual-decode opt-in (only effective for v3 + no-mel; other paths
        // are not changed by the flag).
        if dualDecodeArbitration, !melChunkContext, modelVersion == .v3 {
            return try await processWithDualDecodeArbitration(
                using: manager,
                workers: workers,
                decoderLayers: decoderLayers,
                maxModelSamples: maxModelSamples,
                modelVersion: modelVersion,
                startTime: startTime,
                progressHandler: progressHandler,
                language: language
            )
        }

        let layout = chunkLayout(melChunkContext: melChunkContext, modelVersion: modelVersion)
        let melContextSamples = layout.melContextSamples
        let warmupPrefixSamples = layout.warmupPrefixSamples
        let chunkSamples = layout.chunkSamples
        let strideSamples = layout.strideSamples
        let chunkStarts = try self.chunkStarts(
            warmupPrefixSamples: warmupPrefixSamples,
            chunkSamples: chunkSamples,
            strideSamples: strideSamples,
            preferSilenceAlignment: !melChunkContext && modelVersion == .v3
        )

        var chunkOutputs: [[TokenWindow]?] = []
        var availableWorkers = Array(workers.indices)
        var inFlight = 0
        var chunkDecision = chunkStarts.first ?? ChunkStartDecision(start: 0, useWarmupPrefix: false)
        var chunkStart = chunkDecision.start
        var chunkIndex = 0

        func collectNextResult(
            _ group: inout ThrowingTaskGroup<TaskResult, Error>
        ) async throws {
            guard inFlight > 0 else { return }
            guard let finished = try await group.next() else { return }
            chunkOutputs[finished.index] = finished.tokens
            availableWorkers.append(finished.workerIndex)
            inFlight -= 1
        }

        try await withThrowingTaskGroup(of: TaskResult.self) { group in
            while chunkStart < totalSamples {
                try Task.checkCancellation()
                let warmupSamples =
                    chunkIndex > 0 && chunkDecision.useWarmupPrefix
                    ? min(warmupPrefixSamples, chunkStart) : 0
                let visibleChunkSamples = max(
                    ASRConstants.samplesPerEncoderFrame,
                    chunkSamples - warmupSamples
                )
                let candidateEnd = chunkStart + visibleChunkSamples
                let isLastChunk = candidateEnd >= totalSamples
                let chunkEnd = isLastChunk ? totalSamples : candidateEnd

                if chunkEnd <= chunkStart {
                    break
                }

                // In the default path, contextSamples means mel/STFT context
                // and is skipped by the decoder. In v3/no-mel mode, the
                // warmup prefix is decoded from frame 0 and only its emitted
                // tokens are suppressed.
                let contextSamples = warmupSamples > 0 ? 0 : (chunkIndex > 0 ? melContextSamples : 0)
                let contextStart = chunkStart - max(warmupSamples, contextSamples)
                let chunkLengthWithContext = chunkEnd - contextStart
                let chunkSamplesArray = try readSamples(offset: contextStart, count: chunkLengthWithContext)
                let emitTokensAfterFrame =
                    warmupSamples > 0 ? chunkStart / ASRConstants.samplesPerEncoderFrame : nil

                if availableWorkers.isEmpty {
                    try await collectNextResult(&group)
                }
                if availableWorkers.isEmpty {
                    availableWorkers.append(0)
                }

                let workerIndex = availableWorkers.removeFirst()
                let worker = workers[workerIndex]
                let index = chunkIndex
                let chunkStartOffset = warmupSamples > 0 ? contextStart : chunkStart
                chunkOutputs.append(nil)

                group.addTask {
                    var decoderState = TdtDecoderState.make(decoderLayers: decoderLayers)
                    decoderState.reset()

                    let (windowTokens, windowTimestamps, windowConfidences, windowDurations) =
                        try await Self
                        .transcribeChunk(
                            samples: chunkSamplesArray,
                            contextSamples: contextSamples,
                            chunkStart: chunkStartOffset,
                            isLastChunk: isLastChunk,
                            using: worker,
                            decoderState: &decoderState,
                            maxModelSamples: maxModelSamples,
                            language: language,
                            emitTokensAfterFrame: emitTokensAfterFrame,
                            initialTimeIndexOverride: emitTokensAfterFrame == nil ? nil : 0
                        )

                    guard
                        windowTokens.count == windowTimestamps.count
                            && windowTokens.count == windowConfidences.count
                    else {
                        throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
                    }

                    let durations =
                        windowDurations.count == windowTokens.count
                        ? windowDurations : Array(repeating: 0, count: windowTokens.count)

                    let windowData: [TokenWindow] = zip(
                        zip(zip(windowTokens, windowTimestamps), windowConfidences), durations
                    ).map {
                        (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
                    }

                    return TaskResult(index: index, tokens: windowData, workerIndex: workerIndex)
                }
                inFlight += 1
                chunkIndex += 1

                if let progressHandler, !isLastChunk {
                    let progress = min(1.0, max(0.0, Double(chunkEnd) / Double(totalSamples)))
                    await progressHandler(progress)
                }

                if isLastChunk {
                    break
                }

                if chunkIndex < chunkStarts.count {
                    chunkDecision = chunkStarts[chunkIndex]
                    chunkStart = chunkDecision.start
                } else {
                    chunkStart += strideSamples
                    chunkDecision = ChunkStartDecision(start: chunkStart, useWarmupPrefix: false)
                }

                if availableWorkers.isEmpty && inFlight > 0 {
                    try await collectNextResult(&group)
                }
            }

            while inFlight > 0 {
                try Task.checkCancellation()
                try await collectNextResult(&group)
            }
        }

        let orderedChunkOutputs = chunkOutputs.compactMap { $0 }

        guard var mergedTokens = orderedChunkOutputs.first else {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if orderedChunkOutputs.count > 1 {
            let vocabulary = await manager.vocabulary
            let spliceSafeTokenIds = Self.spliceSafeTokenIds(vocabulary: vocabulary)
            let caseVariantIds = Self.caseVariantCanonicalIds(vocabulary: vocabulary)
            for chunk in orderedChunkOutputs.dropFirst() {
                mergedTokens = mergeChunks(
                    mergedTokens,
                    chunk,
                    spliceSafeTokenIds: spliceSafeTokenIds,
                    caseVariantIds: caseVariantIds
                )
            }
            if mergedTokens.count > 1 {
                mergedTokens.sort { $0.timestamp < $1.timestamp }
            }
            mergedTokens = collapseSeamWordDuplicates(mergedTokens, vocabulary: vocabulary)
        } else if mergedTokens.count > 1 {
            mergedTokens.sort { $0.timestamp < $1.timestamp }
        }

        let allTokens = mergedTokens.map { $0.token }
        let allTimestamps = mergedTokens.map { $0.timestamp }
        let allConfidences = mergedTokens.map { $0.confidence }
        let allDurations = mergedTokens.map { $0.duration }

        return await manager.processTranscriptionResult(
            tokenIds: allTokens,
            timestamps: allTimestamps,
            confidences: allConfidences,
            tokenDurations: allDurations,
            encoderSequenceLength: 0,  // Not relevant for chunk processing
            audioSamples: [],
            processingTime: Date().timeIntervalSince(startTime)
        )
    }

    private func makeWorkerPool(using manager: AsrManager, count: Int) async -> [AsrManager]? {
        guard count > 0 else { return nil }
        var workers: [AsrManager] = [manager]
        if count == 1 {
            return workers
        }
        for _ in 1..<count {
            guard let clone = await manager.makeWorkerClone() else {
                return nil
            }
            workers.append(clone)
        }
        logger.debug("ChunkProcessor using worker pool of size \(workers.count)")
        return workers
    }

    func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    static func transcribeChunk(
        samples: [Float],
        contextSamples: Int,
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState,
        maxModelSamples: Int,
        language: Language? = nil,
        emitTokensAfterFrame: Int? = nil,
        initialTimeIndexOverride: Int? = nil
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        guard !samples.isEmpty else { return ([], [], [], []) }

        let paddedChunk = manager.padAudioIfNeeded(samples, targetLength: maxModelSamples)

        // Calculate frame count for the ACTUAL audio (excluding prepended context)
        let actualAudioSamples = samples.count - contextSamples
        let actualFrameCount = ASRConstants.calculateEncoderFrames(from: actualAudioSamples)

        // Global frame offset is based on original chunkStart (not context-adjusted start)
        let globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame

        // Context frame adjustment tells decoder to skip the prepended context frames
        let contextFrames = contextSamples / ASRConstants.samplesPerEncoderFrame

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,  // Full length including context
            actualAudioFrames: actualFrameCount,  // Only actual audio frames (excluding context)
            decoderState: &decoderState,
            contextFrameAdjustment: contextFrames,  // Skip context frames in decoder
            isLastChunk: isLastChunk,
            globalFrameOffset: globalFrameOffset,
            language: language,
            emitTokensAfterGlobalFrame: emitTokensAfterFrame,
            initialTimeIndexOverride: initialTimeIndexOverride
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            // #1237 tail-clip recovery (EnviousWispr fork carry; upstream #746 open):
            // a window with substantial encoder frames that decoded to ZERO tokens
            // still carries real speech the TDT decoder blanked (typically the end of
            // a dictation after a mid-sentence pause). Recover it by re-decoding the
            // window's internal speech islands. Genuine trailing-silence / no-speech
            // windows (few real frames) fall below the threshold and keep returning
            // empty as today.
            if encoderSequenceLength >= recoverableEmptyMinFrames {
                if let recovered = try await recoverEmptyChunk(
                    samples: samples,
                    contextSamples: contextSamples,
                    chunkStart: chunkStart,
                    isLastChunk: isLastChunk,
                    using: manager,
                    maxModelSamples: maxModelSamples
                ) {
                    return recovered
                }
            }
            return ([], [], [], [])
        }

        return (hypothesis.ySequence, hypothesis.timestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations)
    }

    /// Global frame offset for a recovered island (v0.15.4 convention: offsets are
    /// passed INTO the inference, so timestamps come out global). The samples array
    /// begins at `chunkStart - contextSamples`; the detector snaps island starts to
    /// encoder-frame multiples so the division is exact for frame-aligned chunk starts.
    static func islandGlobalFrameOffset(
        chunkStart: Int,
        contextSamples: Int,
        islandStartSampleWithinWindow: Int
    ) -> Int {
        max(0, chunkStart - contextSamples + islandStartSampleWithinWindow)
            / ASRConstants.samplesPerEncoderFrame
    }

    /// Recover a window that decoded to ZERO tokens despite carrying substantial speech
    /// (#1237 end-of-dictation tail clip; EnviousWispr fork carry). Splits the window's
    /// `samples` at internal silence into speech islands, decodes each ONCE with a fresh
    /// decoder state, and returns the assembled token tuple for `process()` to merge
    /// exactly as a normal window.
    ///
    /// Timestamp basis (v0.15.4 convention): the normal path passes
    /// `globalFrameOffset = chunkStart / samplesPerEncoderFrame` INTO the inference, so
    /// returned timestamps are already global. Islands do the same with the island's own
    /// absolute start: the samples array begins at `chunkStart - contextSamples`
    /// (context-prefixed; the warmup path passes the array start as `chunkStart` with
    /// `contextSamples == 0`), so an island's absolute start is
    /// `chunkStart - contextSamples + island.start`. The detector snaps island starts down
    /// to encoder-frame multiples so the division is exact for frame-aligned chunk starts.
    ///
    /// Heart-path safe and fail-open: returns `nil` (the caller falls back to today's
    /// exact empty result) when there are no usable islands, the island count exceeds the
    /// cap, or every island re-blanks. A single island that re-blanks is DROPPED while the
    /// others are kept (partial recovery recovers strictly more real speech than
    /// abandoning the window — never worse than today). A genuine decode error returns
    /// `nil`; cancellation is rethrown so it aborts the whole transcription. Each island
    /// decodes once — no recursion, no retry — so a blank can never loop.
    static func recoverEmptyChunk(
        samples: [Float],
        contextSamples: Int,
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        maxModelSamples: Int
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int])? {
        let detector = SilenceIslandDetector()
        let islands = detector.islands(in: samples)
        guard !islands.isEmpty, islands.count <= detector.maxIslands else {
            recoveryLogger.warning(
                "[ChunkDiag] recovery: \(islands.isEmpty ? "no speech islands" : "island count \(islands.count) > cap \(detector.maxIslands)") — fail-open"
            )
            return nil
        }

        let decoderLayers = await manager.decoderLayerCount
        var tokens: [Int] = []
        var timestamps: [Int] = []
        var confidences: [Float] = []
        var durations: [Int] = []

        for island in islands {
            let islandSamples = Array(samples[island.start..<island.end])
            let padded = manager.padAudioIfNeeded(islandSamples, targetLength: maxModelSamples)
            var state = TdtDecoderState.make(decoderLayers: decoderLayers)
            state.reset()
            let islandGlobalFrameOffset = Self.islandGlobalFrameOffset(
                chunkStart: chunkStart,
                contextSamples: contextSamples,
                islandStartSampleWithinWindow: island.start)
            do {
                // Decode the island with a fresh decoder state. Pass through the OUTER
                // window's isLastChunk: only the final window may fire the decoder's
                // last-chunk finalization (flush the true tail); an interior empty window
                // must NOT finalize, or it injects EOF boundary tokens/punctuation before
                // the next chunk merges (Codex code-diff review, #1237).
                let (hypothesis, islandEncoderLength) = try await manager.executeMLInferenceWithTimings(
                    padded,
                    originalLength: islandSamples.count,
                    actualAudioFrames: nil,
                    decoderState: &state,
                    contextFrameAdjustment: 0,
                    isLastChunk: isLastChunk,
                    globalFrameOffset: islandGlobalFrameOffset
                )
                guard !hypothesis.isEmpty, islandEncoderLength > 0 else {
                    // Drop this one island and KEEP the others (partial recovery) — a
                    // dropped island is never worse than today, which loses the entire
                    // window. Each island decodes once, so a deterministic blank can
                    // never loop. (Semantics carried from the reviewed #1237 patch.)
                    recoveryLogger.info(
                        "[ChunkDiag] recovery: island [\(island.start),\(island.end)) re-blanked — dropped, keeping other islands"
                    )
                    continue
                }
                tokens.append(contentsOf: hypothesis.ySequence)
                timestamps.append(contentsOf: hypothesis.timestamps)
                confidences.append(contentsOf: hypothesis.tokenConfidences)
                durations.append(contentsOf: hypothesis.tokenDurations)
            } catch is CancellationError {
                // Cancellation must abort the whole transcription, NOT degrade to a
                // partial/empty result — rethrow so process()'s caller sees it.
                throw CancellationError()
            } catch {
                recoveryLogger.warning(
                    "[ChunkDiag] recovery: island decode failed (\(error.localizedDescription)) — fail-open")
                return nil
            }
        }

        guard !tokens.isEmpty else { return nil }
        // Islands are processed left-to-right and each island's tokens are already
        // time-ordered on the global basis, so the concatenation is in ascending
        // timestamp order; process() merges it normally.
        recoveryLogger.info(
            "[ChunkDiag] recovery: recovered \(tokens.count) token(s) across \(islands.count) island(s)")
        return (tokens, timestamps, confidences, durations)
    }

    /// Token IDs whose vocabulary piece may safely start the portion spliced
    /// in from the `right` window at a seam: SentencePiece word-initial pieces
    /// (`▁` prefix) or punctuation-only pieces (which attach to the previous
    /// word by design). Returns nil for an empty vocabulary so merge behavior
    /// is unchanged when no vocabulary is available (issue #683).
    static func spliceSafeTokenIds(vocabulary: [Int: String]) -> Set<Int>? {
        guard !vocabulary.isEmpty else { return nil }
        var ids = Set<Int>()
        for (id, piece) in vocabulary where isSpliceSafePiece(piece) {
            ids.insert(id)
        }
        return ids
    }

    /// Maps every token ID that has a case-only twin in the vocabulary to a
    /// shared canonical ID, so the overlap matcher can treat e.g. `▁Meeting`
    /// and `▁meeting` as the same word (issue #706).
    ///
    /// A window that begins mid-sentence biases the RNNT decoder to capitalize
    /// its first word as if it started a sentence. In the 2 s overlap the
    /// previous (left) window already heard that word lower-cased with real
    /// left context, but the exact-ID matcher misses the seam pair because the
    /// IDs differ — so the word survives in both windows and decodes twice,
    /// the second copy spuriously capitalized ("the meeting Meeting was").
    /// Folding case at match time lets the seam word anchor and collapse to the
    /// left window's contextually-correct casing.
    ///
    /// Only IDs that actually share a folded piece with another ID are
    /// included, so the map stays small and exact-ID matching is unchanged for
    /// every token without a case twin. Returns nil for an empty vocabulary so
    /// behavior is unchanged when no vocabulary is available.
    static func caseVariantCanonicalIds(vocabulary: [Int: String]) -> [Int: Int]? {
        guard !vocabulary.isEmpty else { return nil }
        var groups: [String: [Int]] = [:]
        for (id, piece) in vocabulary {
            groups[piece.lowercased(), default: []].append(id)
        }
        var canon: [Int: Int] = [:]
        for (folded, ids) in groups where ids.count > 1 {
            // Only groups with a genuine case twin survive; pure-lowercase,
            // punctuation and numeric pieces are unique and stay singletons.
            // Make the all-lower-case variant the canonical ID so a later
            // collapse can tell which copy of a seam duplicate to keep.
            let canonical = ids.first { vocabulary[$0] == folded } ?? ids.min()!
            for id in ids { canon[id] = canonical }
        }
        return canon.isEmpty ? nil : canon
    }

    /// Issue #706: drop an adjacent case-only duplicate of a seam *word* left by
    /// a window that re-emitted the seam word as a false sentence start — e.g.
    /// the previous window ended `...we don't have` and the next emitted
    /// `Have a...`, leaving `we don't have Have a`. Works at the word level
    /// (reconstructing SentencePiece words from the token stream) so it catches
    /// multi-token words too — essential for the small Unified subword vocab,
    /// where whole words like `have`/`Have` are several pieces and a token-level
    /// check never sees them as a unit.
    ///
    /// A pair collapses only when the two words are equal up to case, differ in
    /// case, start within the overlap window, and the earlier word does not end
    /// a sentence — so genuine repeats (`that that`), same-case duplicates, and
    /// legitimate sentence boundaries (`...thank you. You said...`) are left
    /// alone. The lower-cased copy is kept (it is the one with real left
    /// context); if neither is lower-case the earlier copy wins.
    func collapseSeamWordDuplicates(
        _ tokens: [TokenWindow],
        vocabulary: [Int: String]
    ) -> [TokenWindow] {
        guard !vocabulary.isEmpty, tokens.count > 1 else { return tokens }
        let overlapFrames = Int((overlapSeconds / ASRConstants.secondsPerEncoderFrame).rounded())

        func piece(_ id: Int) -> String { vocabulary[id] ?? "" }
        func startsWord(_ id: Int) -> Bool {
            let p = piece(id)
            return p.hasPrefix(ASRConstants.sentencePieceWordBoundary) || p.hasPrefix(" ")
        }

        struct Word {
            var tokens: [TokenWindow]
            var core: String
            var startTimestamp: Int
            var endsSentence: Bool
        }

        // Segment the token stream into words on word-initial pieces.
        var words: [Word] = []
        for token in tokens {
            if words.isEmpty || startsWord(token.token) {
                words.append(Word(tokens: [token], core: "", startTimestamp: token.timestamp, endsSentence: false))
            } else {
                words[words.count - 1].tokens.append(token)
            }
        }

        let strippable = CharacterSet.punctuationCharacters.union(.whitespaces)
        for index in words.indices {
            var text = ""
            for token in words[index].tokens {
                text += stripWordBoundaryPrefix(piece(token.token))
            }
            words[index].core = text.trimmingCharacters(in: strippable)
            if let last = text.last { words[index].endsSentence = ".?!:".contains(last) }
        }

        var keep = [Bool](repeating: true, count: words.count)
        var lastKept = -1
        for index in words.indices {
            guard lastKept >= 0 else {
                lastKept = index
                continue
            }
            let previous = words[lastKept]
            let current = words[index]
            let previousCore = previous.core
            let currentCore = current.core

            let isSeamDuplicate =
                !previousCore.isEmpty && !currentCore.isEmpty
                && previousCore != currentCore
                && previousCore.lowercased() == currentCore.lowercased()
                && currentCore.first?.isLetter == true
                && !previous.endsSentence
                && current.startTimestamp - previous.startTimestamp <= overlapFrames

            guard isSeamDuplicate else {
                lastKept = index
                continue
            }

            // Keep the lower-cased copy; if neither is lower-case keep the
            // earlier (left-context) one.
            if currentCore == currentCore.lowercased(), previousCore != previousCore.lowercased() {
                keep[lastKept] = false
                lastKept = index
            } else {
                keep[index] = false
            }
        }

        var result: [TokenWindow] = []
        result.reserveCapacity(tokens.count)
        for index in words.indices where keep[index] {
            result.append(contentsOf: words[index].tokens)
        }
        return result
    }

    /// A piece is splice-safe when decoding it right after another word does
    /// not glue two words together: it either starts a new word (`▁`/space
    /// prefix) or is pure punctuation/symbols.
    static func isSpliceSafePiece(_ piece: String) -> Bool {
        guard !piece.isEmpty else { return false }
        if isWordBoundary(piece) { return true }
        return piece.unicodeScalars.allSatisfy { scalar in
            CharacterSet.punctuationCharacters.contains(scalar)
                || CharacterSet.symbols.contains(scalar)
        }
    }

    func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow],
        spliceSafeTokenIds: Set<Int>? = nil,
        caseVariantIds: [Int: Int]? = nil
    ) -> [TokenWindow] {
        if left.isEmpty { return right }
        if right.isEmpty { return left }

        let frameDuration = ASRConstants.secondsPerEncoderFrame
        let overlapDuration = overlapSeconds
        let halfOverlapWindow = overlapDuration / 2

        func startTime(of token: TokenWindow) -> Double {
            Double(token.timestamp) * frameDuration
        }

        func endTime(of token: TokenWindow) -> Double {
            startTime(of: token) + frameDuration
        }

        let leftEndTime = endTime(of: left.last!)
        let rightStartTime = startTime(of: right.first!)

        if leftEndTime <= rightStartTime {
            return left + right
        }

        let overlapLeft: [IndexedToken] = left.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            let end = start + frameDuration
            guard end > rightStartTime - overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: end)
        }

        let overlapRight: [IndexedToken] = right.enumerated().compactMap { offset, token in
            let start = startTime(of: token)
            guard start < leftEndTime + overlapDuration else { return nil }
            return IndexedToken(index: offset, token: token, start: start, end: start + frameDuration)
        }

        guard overlapLeft.count >= 2 && overlapRight.count >= 2 else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration, spliceSafeTokenIds: spliceSafeTokenIds)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        // EXTRACTED: Contiguous matching using SequenceMatcher
        let timeTolerantMatcher: (IndexedToken, IndexedToken) -> Bool = { [self] l, r in
            tokensMatch(l, r, tolerance: halfOverlapWindow, caseVariantIds: caseVariantIds)
        }

        let contiguousMatches = SequenceMatcher.findContiguousMatches(
            left: overlapLeft,
            right: overlapRight,
            matcher: timeTolerantMatcher
        )

        // Convert SequenceMatch results to index pairs
        let contiguousPairs = contiguousMatches.map { ($0.leftStartIndex, $0.rightStartIndex) }

        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right,
                spliceSafeTokenIds: spliceSafeTokenIds
            )
        }

        // EXTRACTED: LCS fallback using SequenceMatcher
        let lcsMatches = SequenceMatcher.findLongestCommonSubsequence(
            left: overlapLeft,
            right: overlapRight,
            matcher: timeTolerantMatcher
        )

        guard !lcsMatches.isEmpty else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration, spliceSafeTokenIds: spliceSafeTokenIds)
        }

        // Map LCS matches directly to pairs (no consolidation)
        // mergeUsingMatches requires one pair per matched element to function correctly
        let lcsPairs = lcsMatches.map { ($0.leftStartIndex, $0.rightStartIndex) }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right,
            spliceSafeTokenIds: spliceSafeTokenIds
        )
    }

    private func tokensMatch(
        _ left: IndexedToken,
        _ right: IndexedToken,
        tolerance: Double,
        caseVariantIds: [Int: Int]? = nil
    ) -> Bool {
        guard tokenIdsMatch(left.token.token, right.token.token, caseVariantIds: caseVariantIds) else {
            return false
        }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    /// Two token IDs match when they are equal, or — issue #706 — when they are
    /// case-only variants of the same vocabulary piece (e.g. `▁Meeting`/
    /// `▁meeting`), so a seam word the right window capitalized as a false
    /// sentence start still anchors against the left window's lower-cased copy.
    private func tokenIdsMatch(_ left: Int, _ right: Int, caseVariantIds: [Int: Int]?) -> Bool {
        if left == right { return true }
        guard let caseVariantIds, let lhs = caseVariantIds[left], let rhs = caseVariantIds[right] else {
            return false
        }
        return lhs == rhs
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow],
        spliceSafeTokenIds: Set<Int>?
    ) -> [TokenWindow] {
        let leftIndices = matches.map { overlapLeft[$0.0].index }
        let rightIndices = matches.map { overlapRight[$0.1].index }

        var result: [TokenWindow] = []

        if let firstLeft = leftIndices.first, firstLeft > 0 {
            result.append(contentsOf: left[..<firstLeft])
        }

        for idx in 0..<matches.count {
            let leftIndex = leftIndices[idx]
            let rightIndex = rightIndices[idx]

            result.append(left[leftIndex])

            guard idx < matches.count - 1 else { continue }

            let nextLeftIndex = leftIndices[idx + 1]
            let nextRightIndex = rightIndices[idx + 1]

            let gapLeft = nextLeftIndex > leftIndex + 1 ? Array(left[(leftIndex + 1)..<nextLeftIndex]) : []
            let gapRight = nextRightIndex > rightIndex + 1 ? Array(right[(rightIndex + 1)..<nextRightIndex]) : []

            if gapRight.count > gapLeft.count {
                result.append(contentsOf: gapRight)
            } else {
                result.append(contentsOf: gapLeft)
            }
        }

        if let lastRight = rightIndices.last, lastRight + 1 < right.count {
            let tail = right[(lastRight + 1)...]
            if let safeIds = spliceSafeTokenIds,
                let firstTail = tail.first,
                !safeIds.contains(firstTail.token)
            {
                // Issue #683: the splice lands mid-word — right's first
                // post-match piece continues the word containing the matched
                // anchor, so splicing here can decode a left-prefix +
                // right-suffix hybrid or glue two words together. Re-splice
                // at a word boundary so exactly one window segments the
                // seam word.
                if let wordStart = wordInitialIndex(in: right, endingAt: lastRight, safeIds: safeIds),
                    popSeamWord(from: &result, safeIds: safeIds)
                {
                    // The right window heard the seam word from its start —
                    // adopt its segmentation of the whole word. (The left
                    // window's chunk often ends mid-word here, so its view
                    // of the word is the truncated one.)
                    result.append(contentsOf: right[wordStart...])
                } else {
                    // The right window was cut mid-word at its stream start
                    // (no word-initial piece before the anchor): the left
                    // window owns the seam word. Complete it with left's own
                    // continuation pieces and resume right at its next
                    // word-initial piece instead of gluing.
                    if let lastLeft = leftIndices.last {
                        var cursor = lastLeft + 1
                        while cursor < left.count, !safeIds.contains(left[cursor].token) {
                            result.append(left[cursor])
                            cursor += 1
                        }
                    }
                    if let resume = tail.firstIndex(where: { safeIds.contains($0.token) }) {
                        result.append(contentsOf: tail[resume...])
                    }
                }
            } else {
                result.append(contentsOf: tail)
            }
        }

        return result
    }

    /// Index of the word-initial (or punctuation) piece starting the word
    /// that contains `anchor`, or nil when the stream begins mid-word.
    private func wordInitialIndex(
        in stream: [TokenWindow],
        endingAt anchor: Int,
        safeIds: Set<Int>
    ) -> Int? {
        var index = anchor
        while index >= 0 {
            if safeIds.contains(stream[index].token) { return index }
            index -= 1
        }
        return nil
    }

    /// Remove the trailing seam word (continuation pieces plus its
    /// word-initial piece) from `result` so the right window's segmentation
    /// of the same word can replace it. Returns false — leaving `result`
    /// untouched — when no word-initial piece exists within a plausible
    /// word length.
    private func popSeamWord(from result: inout [TokenWindow], safeIds: Set<Int>) -> Bool {
        let maxPiecesPerWord = 12
        var cursor = result.count - 1
        var inspected = 0
        while cursor >= 0, inspected < maxPiecesPerWord {
            if safeIds.contains(result[cursor].token) {
                result.removeLast(result.count - cursor)
                return true
            }
            cursor -= 1
            inspected += 1
        }
        return false
    }

    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double,
        spliceSafeTokenIds: Set<Int>?
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        // Token streams are emitted in timestamp order, so the cutoff filter
        // is equivalent to a prefix/suffix split.
        var leftEnd = left.firstIndex { Double($0.timestamp) * frameDuration >= cutoff } ?? left.count
        var rightStart = right.firstIndex { Double($0.timestamp) * frameDuration >= cutoff } ?? right.count
        if let safeIds = spliceSafeTokenIds {
            // Issue #683: a pure time cutoff can split a word. Extend the
            // left stream until the word it started is complete, and drop
            // orphaned continuation pieces (whose word-initial piece was
            // trimmed away) from the head of the right stream.
            if leftEnd > 0 {
                while leftEnd < left.count, !safeIds.contains(left[leftEnd].token) {
                    leftEnd += 1
                }
            }
            while rightStart < right.count, !safeIds.contains(right[rightStart].token) {
                rightStart += 1
            }
        }
        return Array(left[..<leftEnd]) + Array(right[rightStart...])
    }
}

/// Energy-based speech-island detector used ONLY by `ChunkProcessor.recoverEmptyChunk`
/// (#1237 EnviousWispr fork carry) to split an already-failed (empty) ASR window at
/// internal silence into 1..N speech islands. Energy-only (no VAD-model dependency) is
/// acceptable because this runs solely on a window that ALREADY decoded empty: a wrong
/// split costs at worst a slightly-worse re-decode, never dropped audio. The threshold is
/// scale-invariant (works on normalized float or raw audio) so it needs no per-scale tuning.
struct SilenceIslandDetector {
    /// RMS analysis frame length in samples (100ms at 16kHz).
    let analysisFrameSamples: Int
    /// Minimum run of silent analysis frames that separates two islands.
    let minSilenceFrames: Int
    /// Minimum island length (samples) kept after padding.
    let minIslandSamples: Int
    /// Padding (samples) added on each side of a detected island.
    let paddingSamples: Int
    /// Hard cap on island count; above this the caller fails open.
    let maxIslands: Int

    init(minSilenceMs: Int = 300, minIslandMs: Int = 320, paddingMs: Int = 100, maxIslands: Int = 8) {
        let sr = ASRConstants.sampleRate
        self.analysisFrameSamples = max(1, sr / 10)  // 1600 = 100ms
        self.minSilenceFrames = max(1, minSilenceMs / 100)
        self.minIslandSamples = max(1, minIslandMs * sr / 1000)
        self.paddingSamples = max(0, paddingMs * sr / 1000)
        self.maxIslands = max(1, maxIslands)
    }

    /// Returns frame-aligned `[start, end)` sample ranges of speech islands within `samples`.
    /// Island STARTS are snapped down to encoder-frame multiples so the caller's timestamp
    /// offset (`islandStart / samplesPerEncoderFrame`) is exact; ends are clamped to the
    /// window length (`padAudioIfNeeded` + `originalLength` handle a non-aligned tail).
    func islands(in samples: [Float]) -> [(start: Int, end: Int)] {
        let n = samples.count
        guard n >= analysisFrameSamples else { return [] }
        let frame = ASRConstants.samplesPerEncoderFrame

        // 1. RMS per analysis frame.
        var frameRMS: [Float] = []
        frameRMS.reserveCapacity(n / analysisFrameSamples + 1)
        var i = 0
        while i < n {
            let end = min(i + analysisFrameSamples, n)
            var sumSquares: Float = 0
            for s in i..<end { sumSquares += samples[s] * samples[s] }
            frameRMS.append((sumSquares / Float(end - i)).squareRoot())
            i += analysisFrameSamples
        }
        guard !frameRMS.isEmpty else { return [] }

        // 2. Voiced threshold.
        //    - Absolute silence floor: a window whose loudest frame is near-silent has no real
        //      speech to recover → no islands (recovery fails open). This rejects all-silence.
        //    - Otherwise threshold above the noise floor (20th-pct RMS), but CAPPED at half the
        //      peak so a uniform all-speech window (noise floor ≈ peak) can't exclude its own
        //      speech; floored at a small fraction of the peak so a low-noise-floor bimodal
        //      window (the #1237 silence-then-speech shape) still splits at the speech edge.
        let sortedRMS = frameRMS.sorted()
        let noiseFloor = sortedRMS[min(sortedRMS.count - 1, (sortedRMS.count * 20) / 100)]
        let peak = sortedRMS.last ?? 0
        let speechFloor: Float = 0.005  // normalized-audio RMS below this peak = no speech
        guard peak >= speechFloor else { return [] }
        let threshold = max(peak * 0.05, min(noiseFloor * 2.5, peak * 0.5))
        var voiced = frameRMS.map { $0 >= threshold }

        // 3. Bridge silence gaps shorter than minSilenceFrames so micro-pauses don't split.
        var g = 0
        while g < voiced.count {
            if !voiced[g] {
                var j = g
                while j < voiced.count && !voiced[j] { j += 1 }
                if (j - g) < minSilenceFrames {
                    for k in g..<j { voiced[k] = true }
                }
                g = j
            } else {
                g += 1
            }
        }

        // 4. Extract voiced runs → padded, frame-aligned, non-overlapping islands.
        var result: [(start: Int, end: Int)] = []
        var lastEnd = 0
        var f = 0
        while f < voiced.count {
            if voiced[f] {
                var j = f
                while j < voiced.count && voiced[j] { j += 1 }
                var start = max(0, f * analysisFrameSamples - paddingSamples)
                let end = min(n, j * analysisFrameSamples + paddingSamples)
                // Frame-align the START down so islandStart / frame is exact, and keep islands
                // disjoint (padding can never bridge a >=minSilence gap, but guard anyway).
                start = (start / frame) * frame
                start = max(start, lastEnd)
                if end - start >= minIslandSamples {
                    result.append((start: start, end: end))
                    lastEnd = end
                }
                f = j
            } else {
                f += 1
            }
        }
        return result
    }
}

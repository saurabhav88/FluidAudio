import Foundation

struct ChunkProcessor {
    let sampleSource: AudioSampleSource
    let totalSamples: Int

    private let logger = AppLogger(category: "ChunkProcessor")
    private typealias TokenWindow = (token: Int, timestamp: Int, confidence: Float, duration: Int)
    private struct IndexedToken {
        let index: Int
        let token: TokenWindow
        let start: Double
        let end: Double
    }

    // Stateless chunking aligned with CoreML reference:
    // - process ~14.96s of audio per window (frame-aligned) to stay under encoder limit
    // - 2.0s overlap (frame-aligned) to give the decoder slack when merging windows
    private let overlapSeconds: Double = 2.0

    /// Context samples prepended from previous chunk for mel spectrogram stability (80ms = 1 encoder frame).
    /// The FastConformer encoder's depthwise convolutions need left context for stable output.
    /// Without this, the first frames of a chunk may produce features that cause all-blank predictions.
    private let melContextSamples: Int = ASRConstants.samplesPerEncoderFrame  // 1280 samples = 80ms

    private var maxModelSamples: Int { ASRConstants.maxModelSamples }

    private var chunkSamples: Int {
        // Reserve space for context samples that will be prepended to non-first chunks.
        // This ensures chunkSamples + melContextSamples <= maxModelSamples.
        let maxActualChunk = maxModelSamples - melContextSamples  // 240000 - 1280 = 238720
        let raw = max(maxActualChunk - ASRConstants.melHopSize, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }
    private var overlapSamples: Int {
        let requested = Int(overlapSeconds * Double(ASRConstants.sampleRate))
        let capped = min(requested, chunkSamples / 2)
        return capped / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }
    private var strideSamples: Int {
        let raw = max(chunkSamples - overlapSamples, ASRConstants.samplesPerEncoderFrame)
        return raw / ASRConstants.samplesPerEncoderFrame * ASRConstants.samplesPerEncoderFrame
    }

    /// #1237 empty-chunk recovery: minimum encoder frames a window must have produced
    /// before an all-blank decode is treated as recoverable speech (rather than genuine
    /// trailing silence / no speech). 25 frames = 2.0s at `secondsPerEncoderFrame` (0.08s).
    private static let recoverableEmptyMinFrames: Int = 25

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
        progressHandler: ((Double) async -> Void)? = nil
    ) async throws -> ASRResult {
        var chunkOutputs: [[TokenWindow]] = []

        var chunkStart = 0
        var chunkIndex = 0
        var chunkDecoderState = TdtDecoderState.make(
            decoderLayers: await manager.decoderLayerCount
        )

        while chunkStart < totalSamples {
            try Task.checkCancellation()
            let candidateEnd = chunkStart + chunkSamples
            let isLastChunk = candidateEnd >= totalSamples
            let chunkEnd = isLastChunk ? totalSamples : candidateEnd

            if chunkEnd <= chunkStart {
                break
            }

            chunkDecoderState.reset()

            // For chunks after the first, prepend context samples from the overlap region.
            // This provides left context for the mel spectrogram STFT window and encoder convolutions.
            let contextSamples = chunkIndex > 0 ? melContextSamples : 0
            let contextStart = chunkStart - contextSamples
            let chunkLengthWithContext = chunkEnd - contextStart
            let chunkSamplesArray = try readSamples(offset: contextStart, count: chunkLengthWithContext)

            logger.info(
                "[ChunkDiag] Chunk \(chunkIndex): start=\(chunkStart), end=\(chunkEnd), isLast=\(isLastChunk), samples=\(chunkLengthWithContext), context=\(contextSamples), timeJump=\(String(describing: chunkDecoderState.timeJump))"
            )

            let (windowTokens, windowTimestamps, windowConfidences, windowDurations) = try await transcribeChunk(
                samples: chunkSamplesArray,
                contextSamples: contextSamples,
                chunkStart: chunkStart,
                isLastChunk: isLastChunk,
                using: manager,
                decoderState: &chunkDecoderState
            )

            logger.info(
                "[ChunkDiag] Chunk \(chunkIndex) result: tokens=\(windowTokens.count), firstTs=\(windowTimestamps.first.map(String.init) ?? "nil"), lastTs=\(windowTimestamps.last.map(String.init) ?? "nil"), timeJumpAfter=\(String(describing: chunkDecoderState.timeJump))"
            )

            // Combine tokens, timestamps, and confidences into aligned tuples
            guard windowTokens.count == windowTimestamps.count && windowTokens.count == windowConfidences.count else {
                throw ASRError.processingFailed("Token, timestamp, and confidence arrays are misaligned")
            }

            // Default to 0 per token if durations array is misaligned (shouldn't happen in practice)
            let durations =
                windowDurations.count == windowTokens.count
                ? windowDurations : Array(repeating: 0, count: windowTokens.count)

            let windowData: [TokenWindow] = zip(
                zip(zip(windowTokens, windowTimestamps), windowConfidences), durations
            ).map {
                (token: $0.0.0.0, timestamp: $0.0.0.1, confidence: $0.0.1, duration: $0.1)
            }
            chunkOutputs.append(windowData)

            chunkIndex += 1

            if isLastChunk {
                break
            }

            if let progressHandler {
                let progress = min(1.0, max(0.0, Double(chunkEnd) / Double(totalSamples)))
                await progressHandler(progress)
            }

            chunkStart += strideSamples
        }

        guard var mergedTokens = chunkOutputs.first else {
            return await manager.processTranscriptionResult(
                tokenIds: [],
                timestamps: [],
                confidences: [],
                encoderSequenceLength: 0,
                audioSamples: [],
                processingTime: Date().timeIntervalSince(startTime)
            )
        }

        if chunkOutputs.count > 1 {
            for chunk in chunkOutputs.dropFirst() {
                mergedTokens = mergeChunks(mergedTokens, chunk)
            }
        }

        if mergedTokens.count > 1 {
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

    private func readSamples(offset: Int, count: Int) throws -> [Float] {
        var buffer = [Float](repeating: 0, count: count)
        try buffer.withUnsafeMutableBufferPointer { pointer in
            try sampleSource.copySamples(into: pointer.baseAddress!, offset: offset, count: count)
        }
        return buffer
    }

    private func transcribeChunk(
        samples: [Float],
        contextSamples: Int,
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager,
        decoderState: inout TdtDecoderState
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int]) {
        guard !samples.isEmpty else { return ([], [], [], []) }

        let paddedChunk = manager.padAudioIfNeeded(samples, targetLength: maxModelSamples)

        // Decode the full local window with zero adjustments, matching the streaming contract.
        // The decoder's hardcoded 25-frame context skip handles overlap internally when
        // state is fresh. Passing non-zero contextFrameAdjustment causes double-skipping
        // (proven in streaming fix ablation: -13.7% text loss vs -11.8% baseline).
        // Global frame offsets are applied to timestamps AFTER decode, not during.
        let globalFrameOffset = chunkStart / ASRConstants.samplesPerEncoderFrame

        logger.info(
            "[ChunkDiag] transcribeChunk: samples=\(samples.count), contextSamples=\(contextSamples), globalOffset=\(globalFrameOffset), isLast=\(isLastChunk)"
        )

        let (hypothesis, encoderSequenceLength) = try await manager.executeMLInferenceWithTimings(
            paddedChunk,
            originalLength: samples.count,
            actualAudioFrames: nil,
            decoderState: &decoderState,
            contextFrameAdjustment: 0,
            isLastChunk: isLastChunk,
            globalFrameOffset: 0
        )

        logger.info(
            "[ChunkDiag] inference result: encLen=\(encoderSequenceLength), tokens=\(hypothesis.ySequence.count), isEmpty=\(hypothesis.isEmpty)"
        )

        if hypothesis.isEmpty || encoderSequenceLength == 0 {
            logger.warning(
                "[ChunkDiag] EMPTY CHUNK: hypothesis.isEmpty=\(hypothesis.isEmpty), encLen=\(encoderSequenceLength)")
            // #1237 tail-clip recovery: a window with substantial encoder frames that decoded
            // to ZERO tokens still carries real speech the TDT decoder blanked (typically the
            // end of a dictation after a mid-sentence pause). Recover it by re-decoding the
            // window's internal speech islands. Genuine trailing-silence / no-speech windows
            // (few real frames) fall below the threshold and keep returning empty as today.
            if encoderSequenceLength >= Self.recoverableEmptyMinFrames {
                if let recovered = try await recoverEmptyChunk(
                    samples: samples,
                    chunkStart: chunkStart,
                    isLastChunk: isLastChunk,
                    using: manager
                ) {
                    return recovered
                }
            }
            return ([], [], [], [])
        }

        // Apply global frame offset to timestamps externally (decoder returned chunk-local timestamps)
        let globalTimestamps = hypothesis.timestamps.map { $0 + globalFrameOffset }

        return (hypothesis.ySequence, globalTimestamps, hypothesis.tokenConfidences, hypothesis.tokenDurations)
    }

    /// Recover a window that decoded to ZERO tokens despite carrying substantial speech
    /// (#1237 end-of-dictation tail clip). Splits the window's `samples` at internal silence
    /// into speech islands, decodes each ONCE with a fresh decoder state, offsets the
    /// island-local timestamps onto the window's timestamp basis, and returns the assembled
    /// token tuple for `process()` to merge exactly as a normal window.
    ///
    /// Heart-path safe and fail-open: returns `nil` (the caller falls back to today's exact empty
    /// result) when there are no usable islands, the island count exceeds the cap, or every island
    /// re-blanks. A single island that re-blanks is DROPPED while the others are kept (partial
    /// recovery recovers strictly more real speech than abandoning the window — never worse than
    /// today). A genuine decode error returns `nil`; cancellation is rethrown so it aborts the whole
    /// transcription. Each island decodes once — no recursion, no retry — so a blank can never loop.
    private func recoverEmptyChunk(
        samples: [Float],
        chunkStart: Int,
        isLastChunk: Bool,
        using manager: AsrManager
    ) async throws -> (tokens: [Int], timestamps: [Int], confidences: [Float], durations: [Int])? {
        let detector = SilenceIslandDetector()
        let islands = detector.islands(in: samples)
        guard !islands.isEmpty, islands.count <= detector.maxIslands else {
            logger.warning(
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
            do {
                // Decode the island with a fresh decoder state. Pass through the OUTER window's
                // isLastChunk: only the final window may fire the decoder's last-chunk
                // finalization (flush the true tail); an interior empty window must NOT finalize,
                // or it injects EOF boundary tokens/punctuation before the next chunk merges
                // (Codex code-diff review, #1237). globalFrameOffset 0: timestamps are offset
                // externally below so the basis matches the normal window convention.
                let (hypothesis, islandEncoderLength) = try await manager.executeMLInferenceWithTimings(
                    padded,
                    originalLength: islandSamples.count,
                    actualAudioFrames: nil,
                    decoderState: &state,
                    contextFrameAdjustment: 0,
                    isLastChunk: isLastChunk,
                    globalFrameOffset: 0
                )
                guard !hypothesis.isEmpty, islandEncoderLength > 0 else {
                    // Drop this one island and KEEP the others (partial recovery). Empirically this
                    // recovers strictly more real speech than abandoning the whole window: on the real
                    // clip A8171A1F one island re-blanks while the rest yield "...that you should be
                    // using today" — partial keeps that tail; all-or-nothing would discard it and
                    // regress to today's full-window drop. A dropped island is never worse than today
                    // (which loses the entire window). Each island decodes once, so a deterministic
                    // blank can never loop. (Codex flagged a doc/code mismatch here; resolved toward
                    // partial — the plan's design — backed by the A8171A1F regression, not toward
                    // all-or-nothing which the data shows loses a real recovery.)
                    logger.info(
                        "[ChunkDiag] recovery: island [\(island.start),\(island.end)) re-blanked — dropped, keeping other islands"
                    )
                    continue
                }
                let offsetTimestamps = Self.offsetIslandTimestamps(
                    hypothesis.timestamps,
                    chunkStart: chunkStart,
                    islandStartSampleWithinWindow: island.start)
                tokens.append(contentsOf: hypothesis.ySequence)
                timestamps.append(contentsOf: offsetTimestamps)
                confidences.append(contentsOf: hypothesis.tokenConfidences)
                durations.append(contentsOf: hypothesis.tokenDurations)
            } catch is CancellationError {
                // Cancellation must abort the whole transcription, NOT degrade to a partial/empty
                // result — rethrow so process()'s caller sees it (Codex code-diff review, #1237).
                throw CancellationError()
            } catch {
                logger.warning(
                    "[ChunkDiag] recovery: island decode failed (\(error.localizedDescription)) — fail-open")
                return nil
            }
        }

        guard !tokens.isEmpty else { return nil }
        // Islands are processed left-to-right and each island's tokens are already time-ordered,
        // so the concatenation is in ascending timestamp order; process() merges it normally.
        logger.info(
            "[ChunkDiag] recovery: recovered \(tokens.count) token(s) across \(islands.count) island(s)")
        return (tokens, timestamps, confidences, durations)
    }

    /// Offset a recovered island's decoder-local frame timestamps onto the window's timestamp
    /// basis so `mergeChunks` dedups them against prior windows. Matches the EXISTING
    /// ChunkProcessor convention (chunkStart-based, deliberately NOT true sample-global —
    /// `transcribeChunk` applies `chunkStart / samplesPerEncoderFrame` at decode return).
    /// `islandStartSampleWithinWindow` is measured within the context-prefixed window `samples`
    /// array and MUST be encoder-frame-aligned (the detector snaps island starts down to a
    /// frame multiple) so the division is exact.
    static func offsetIslandTimestamps(
        _ localTimestamps: [Int],
        chunkStart: Int,
        islandStartSampleWithinWindow: Int
    ) -> [Int] {
        let frame = ASRConstants.samplesPerEncoderFrame
        let offset = (chunkStart / frame) + (islandStartSampleWithinWindow / frame)
        return localTimestamps.map { $0 + offset }
    }

    private func mergeChunks(
        _ left: [TokenWindow],
        _ right: [TokenWindow]
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
                frameDuration: frameDuration)
        }

        let minimumPairs = max(overlapLeft.count / 2, 1)

        let contiguousPairs = findBestContiguousPairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        if contiguousPairs.count >= minimumPairs {
            return mergeUsingMatches(
                matches: contiguousPairs,
                overlapLeft: overlapLeft,
                overlapRight: overlapRight,
                left: left,
                right: right
            )
        }

        let lcsPairs = findLongestCommonSubsequencePairs(
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            tolerance: halfOverlapWindow
        )

        guard !lcsPairs.isEmpty else {
            return mergeByMidpoint(
                left: left, right: right, leftEndTime: leftEndTime, rightStartTime: rightStartTime,
                frameDuration: frameDuration)
        }

        return mergeUsingMatches(
            matches: lcsPairs,
            overlapLeft: overlapLeft,
            overlapRight: overlapRight,
            left: left,
            right: right
        )
    }

    private func findBestContiguousPairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        var best: [(Int, Int)] = []

        for i in 0..<overlapLeft.count {
            for j in 0..<overlapRight.count {
                let leftToken = overlapLeft[i]
                let rightToken = overlapRight[j]

                if tokensMatch(leftToken, rightToken, tolerance: tolerance) {
                    var current: [(Int, Int)] = []
                    var k = i
                    var l = j

                    while k < overlapLeft.count && l < overlapRight.count {
                        let nextLeft = overlapLeft[k]
                        let nextRight = overlapRight[l]

                        if tokensMatch(nextLeft, nextRight, tolerance: tolerance) {
                            current.append((k, l))
                            k += 1
                            l += 1
                        } else {
                            break
                        }
                    }

                    if current.count > best.count {
                        best = current
                    }
                }
            }
        }

        return best
    }

    private func findLongestCommonSubsequencePairs(
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        tolerance: Double
    ) -> [(Int, Int)] {
        let leftCount = overlapLeft.count
        let rightCount = overlapRight.count

        var dp = Array(repeating: Array(repeating: 0, count: rightCount + 1), count: leftCount + 1)

        for i in 1...leftCount {
            for j in 1...rightCount {
                if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                    dp[i][j] = dp[i - 1][j - 1] + 1
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
                }
            }
        }

        var pairs: [(Int, Int)] = []
        var i = leftCount
        var j = rightCount

        while i > 0 && j > 0 {
            if tokensMatch(overlapLeft[i - 1], overlapRight[j - 1], tolerance: tolerance) {
                pairs.append((i - 1, j - 1))
                i -= 1
                j -= 1
            } else if dp[i - 1][j] > dp[i][j - 1] {
                i -= 1
            } else {
                j -= 1
            }
        }

        return pairs.reversed()
    }

    private func tokensMatch(_ left: IndexedToken, _ right: IndexedToken, tolerance: Double) -> Bool {
        guard left.token.token == right.token.token else { return false }
        let timeDifference = abs(left.start - right.start)
        return timeDifference < tolerance
    }

    private func mergeUsingMatches(
        matches: [(Int, Int)],
        overlapLeft: [IndexedToken],
        overlapRight: [IndexedToken],
        left: [TokenWindow],
        right: [TokenWindow]
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
            result.append(contentsOf: right[(lastRight + 1)...])
        }

        return result
    }

    private func mergeByMidpoint(
        left: [TokenWindow],
        right: [TokenWindow],
        leftEndTime: Double,
        rightStartTime: Double,
        frameDuration: Double
    ) -> [TokenWindow] {
        let cutoff = (leftEndTime + rightStartTime) / 2
        let trimmedLeft = left.filter { Double($0.timestamp) * frameDuration < cutoff }
        let trimmedRight = right.filter { Double($0.timestamp) * frameDuration >= cutoff }
        return trimmedLeft + trimmedRight
    }
}

/// Energy-based speech-island detector used ONLY by `ChunkProcessor.recoverEmptyChunk`
/// to split an already-failed (empty) ASR window at internal silence into 1..N speech
/// islands. Energy-only (no VAD-model dependency) is acceptable because this runs solely
/// on a window that ALREADY decoded empty: a wrong split costs at worst a slightly-worse
/// re-decode, never dropped audio. The threshold is scale-invariant (works on normalized
/// float or raw audio) so it needs no per-scale tuning.
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

#if DEBUG
extension ChunkProcessor {
    /// Test-only seam exposing the private overlap merge with a plain tuple type so seam
    /// tests (e.g. the #1237 divergent-overlap test) can exercise the REAL dedup logic.
    func mergeChunksForTesting(
        _ left: [(token: Int, timestamp: Int, confidence: Float, duration: Int)],
        _ right: [(token: Int, timestamp: Int, confidence: Float, duration: Int)]
    ) -> [(token: Int, timestamp: Int, confidence: Float, duration: Int)] {
        let l = left.map {
            TokenWindow(token: $0.token, timestamp: $0.timestamp, confidence: $0.confidence, duration: $0.duration)
        }
        let r = right.map {
            TokenWindow(token: $0.token, timestamp: $0.timestamp, confidence: $0.confidence, duration: $0.duration)
        }
        return mergeChunks(l, r).map {
            (token: $0.token, timestamp: $0.timestamp, confidence: $0.confidence, duration: $0.duration)
        }
    }
}
#endif

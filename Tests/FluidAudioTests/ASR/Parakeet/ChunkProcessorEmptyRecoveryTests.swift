import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for #1237 empty-chunk recovery: the silence-island splitter, the recovered-island
/// timestamp offset, and the overlap merge collapsing a divergent good/recovered seam.
/// The end-to-end recovery (`recoverEmptyChunk`, which needs the CoreML model) is covered by the
/// offline `fluidaudiocli` fixture regression, not here.
final class ChunkProcessorEmptyRecoveryTests: XCTestCase {

    private let frame = ASRConstants.samplesPerEncoderFrame  // 1280
    private let sr = ASRConstants.sampleRate  // 16000

    // MARK: - Audio fixture helpers

    /// Pseudo-speech: deterministic mid-amplitude waveform (no Float.random for reproducibility).
    private func speech(_ seconds: Double, amplitude: Float = 0.3) -> [Float] {
        let n = Int(seconds * Double(sr))
        return (0..<n).map { i in
            // Mix two tones so RMS is stable and clearly above the silence floor.
            let t = Float(i) / Float(sr)
            return amplitude * (sin(2 * .pi * 180 * t) + 0.5 * sin(2 * .pi * 320 * t))
        }
    }

    private func silence(_ seconds: Double) -> [Float] {
        Array(repeating: 0, count: Int(seconds * Double(sr)))
    }

    // MARK: - SilenceIslandDetector

    func testAllSpeechYieldsSingleIsland() {
        let islands = SilenceIslandDetector().islands(in: speech(1.0))
        XCTAssertEqual(islands.count, 1, "Continuous speech should be one island")
    }

    func testSpeechSilenceSpeechYieldsTwoIslands() {
        let buf = speech(1.0) + silence(0.5) + speech(1.0)  // 0.5s gap > 300ms minSilence
        let islands = SilenceIslandDetector().islands(in: buf)
        XCTAssertEqual(islands.count, 2, "A >300ms internal pause should split into two islands")
        guard islands.count == 2 else { return }
        XCTAssertLessThanOrEqual(islands[0].end, islands[1].start, "Islands must be disjoint")
    }

    func testMicroPauseDoesNotSplit() {
        let buf = speech(1.0) + silence(0.15) + speech(1.0)  // 150ms gap < 300ms minSilence
        let islands = SilenceIslandDetector().islands(in: buf)
        XCTAssertEqual(islands.count, 1, "A sub-300ms micro-pause must not split the island")
    }

    func testLeadingSilenceProducesIslandAfterSilence() {
        // The #1237 shape: a window whose speech is bracketed by leading low energy.
        let buf = silence(0.6) + speech(1.0)
        let islands = SilenceIslandDetector().islands(in: buf)
        XCTAssertEqual(islands.count, 1)
        XCTAssertGreaterThan(islands[0].start, 0, "Island should start after the leading silence")
    }

    func testIslandStartsAreFrameAligned() {
        let buf = silence(0.6) + speech(1.0) + silence(0.5) + speech(1.0)
        let islands = SilenceIslandDetector().islands(in: buf)
        XCTAssertGreaterThanOrEqual(islands.count, 1)
        for island in islands {
            XCTAssertEqual(
                island.start % frame, 0,
                "Island start \(island.start) must be encoder-frame-aligned so the timestamp offset is exact")
        }
    }

    func testPureSilenceYieldsNoIslands() {
        let islands = SilenceIslandDetector().islands(in: silence(2.0))
        XCTAssertTrue(islands.isEmpty, "Pure silence should yield no speech islands (recovery fails open)")
    }

    func testTinyBufferYieldsNoIslands() {
        // Shorter than one analysis frame → nothing to analyze.
        let islands = SilenceIslandDetector().islands(in: speech(0.05))
        XCTAssertTrue(islands.isEmpty)
    }

    func testIslandCountCapIsRespectedBySplitter() {
        // Many short speech bursts separated by clear pauses; assert the detector with a low cap
        // is the value the caller uses to fail open (the detector reports the true islands; the
        // caller compares to maxIslands).
        var buf: [Float] = []
        for _ in 0..<10 { buf += speech(0.4) + silence(0.4) }
        let detector = SilenceIslandDetector(maxIslands: 8)
        let islands = detector.islands(in: buf)
        // The recovery guard is `islands.count <= detector.maxIslands`; prove the splitter can
        // exceed the cap so the guard is meaningful.
        XCTAssertGreaterThan(islands.count, detector.maxIslands)
    }

    // MARK: - offsetIslandTimestamps

    func testOffsetFirstWindowIslandAtZeroIsIdentity() {
        // chunkStart 0, island at window start → no shift (true for window 0's first island).
        let out = ChunkProcessor.offsetIslandTimestamps([0, 5, 12], chunkStart: 0, islandStartSampleWithinWindow: 0)
        XCTAssertEqual(out, [0, 5, 12])
    }

    func testOffsetNonFirstWindowIsland() {
        // Codex-required: a recovered island inside a non-first window must land at the right
        // global position. chunkStart = one stride (frame-aligned), island starts 2 frames in.
        let chunkStart = 161 * frame  // 206_080, a representative stride (frame-aligned)
        let islandStart = 2 * frame  // 2560
        let out = ChunkProcessor.offsetIslandTimestamps(
            [0, 3, 7], chunkStart: chunkStart, islandStartSampleWithinWindow: islandStart)
        // offset = chunkStart/frame + islandStart/frame = 161 + 2 = 163
        XCTAssertEqual(out, [163, 166, 170])
    }

    func testOffsetMatchesNormalWindowConventionAtIslandStartZero() {
        // With islandStart 0, the recovered offset must equal the normal window's
        // `chunkStart / samplesPerEncoderFrame` convention exactly (so windows merge consistently).
        let chunkStart = 206_080
        let out = ChunkProcessor.offsetIslandTimestamps([10], chunkStart: chunkStart, islandStartSampleWithinWindow: 0)
        XCTAssertEqual(out, [10 + chunkStart / frame])
    }

    // MARK: - Divergent-overlap seam merge (empirically earned, 2026-06-30)

    func testDivergentOverlapSeamDoesNotDuplicate() {
        // The offline experiment showed a naive word-stitch DUPLICATES the seam when the good
        // window and the recovered window overlap with divergent words
        // ("...that you might" + "that you should..." → "...that you might that you should...").
        // Assert the REAL timestamp-aware merge collapses the shared "that you" (token ids 100,101)
        // exactly once.
        let processor = ChunkProcessor(audioSamples: [])
        typealias TW = (token: Int, timestamp: Int, confidence: Float, duration: Int)

        // Shared prefix "that"(100)@163, "you"(101)@164 in BOTH windows; then they diverge.
        let left: [TW] = [
            (token: 90, timestamp: 158, confidence: 0.9, duration: 1),
            (token: 91, timestamp: 160, confidence: 0.9, duration: 1),
            (token: 100, timestamp: 163, confidence: 0.9, duration: 1),  // "that"
            (token: 101, timestamp: 164, confidence: 0.9, duration: 1),  // "you"
            (token: 102, timestamp: 165, confidence: 0.9, duration: 1),  // "might"
        ]
        let right: [TW] = [
            (token: 100, timestamp: 163, confidence: 0.9, duration: 1),  // "that" (overlap)
            (token: 101, timestamp: 164, confidence: 0.9, duration: 1),  // "you"  (overlap)
            (token: 103, timestamp: 165, confidence: 0.9, duration: 1),  // "should" (divergent)
            (token: 104, timestamp: 167, confidence: 0.9, duration: 1),  // "be"
            (token: 105, timestamp: 169, confidence: 0.9, duration: 1),  // "using"
        ]

        let merged = processor.mergeChunksForTesting(left, right)
        let ids = merged.map { $0.token }

        XCTAssertEqual(ids.filter { $0 == 100 }.count, 1, "shared 'that' must appear once, not duplicated")
        XCTAssertEqual(ids.filter { $0 == 101 }.count, 1, "shared 'you' must appear once, not duplicated")
        // The recovered divergent tail must survive into the merge.
        XCTAssertTrue(ids.contains(103), "recovered 'should' must survive the seam")
        XCTAssertTrue(ids.contains(105), "recovered tail 'using' must survive the seam")
        // Timestamps remain sorted-able (no negative/garbage).
        XCTAssertEqual(merged.map { $0.timestamp }, merged.map { $0.timestamp }.sorted())
    }

    func testNonOverlappingWindowsConcatenate() {
        // Recovered island that starts well after the prior window (a real post-pause tail) must
        // simply append, losing nothing.
        let processor = ChunkProcessor(audioSamples: [])
        typealias TW = (token: Int, timestamp: Int, confidence: Float, duration: Int)
        let left: [TW] = [(token: 100, timestamp: 10, confidence: 0.9, duration: 1)]
        let right: [TW] = [(token: 200, timestamp: 180, confidence: 0.9, duration: 1)]  // far past overlap
        let merged = processor.mergeChunksForTesting(left, right)
        XCTAssertEqual(merged.map { $0.token }, [100, 200], "Disjoint windows concatenate, no loss")
    }
}

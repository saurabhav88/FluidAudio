import Foundation
import XCTest

@testable import FluidAudio

/// Unit tests for #1237 empty-chunk recovery (EnviousWispr fork carry onto v0.15.4):
/// the silence-island splitter, the recovered-island global frame offset, and the
/// overlap merge collapsing a divergent good/recovered seam. The end-to-end recovery
/// (`recoverEmptyChunk`, which needs the CoreML model) is covered by the offline
/// `fluidaudiocli` fixture regression, not here.
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
        let buf = silence(0.6) + speech(1.0) + silence(0.5) + speech(0.8)
        for island in SilenceIslandDetector().islands(in: buf) {
            XCTAssertEqual(
                island.start % frame, 0,
                "Island starts must be encoder-frame aligned for exact offset math")
        }
    }

    func testPureSilenceYieldsNoIslands() {
        let islands = SilenceIslandDetector().islands(in: silence(3.0))
        XCTAssertTrue(islands.isEmpty, "All-silence window has nothing to recover — fail open")
    }

    func testTinyBufferYieldsNoIslands() {
        let islands = SilenceIslandDetector().islands(in: speech(0.05))
        XCTAssertTrue(islands.isEmpty, "Sub-analysis-frame buffer yields no islands")
    }

    func testIslandCountCapIsRespectedBySplitter() {
        // 10 speech bursts separated by long silences → more islands than the cap.
        var buf: [Float] = []
        for _ in 0..<10 {
            buf += speech(0.5) + silence(0.5)
        }
        let detector = SilenceIslandDetector()
        let islands = detector.islands(in: buf)
        // The DETECTOR returns them all; the CALLER (recoverEmptyChunk) enforces the cap.
        XCTAssertGreaterThan(islands.count, detector.maxIslands, "Fixture must exceed the cap")
    }

    // MARK: - islandGlobalFrameOffset (v0.15.4 convention: offset passed INTO inference)

    func testOffsetFirstWindowIslandAtZeroIsIdentity() {
        // chunkStart 0, no context, island at window start → offset 0.
        XCTAssertEqual(
            ChunkProcessor.islandGlobalFrameOffset(
                chunkStart: 0, contextSamples: 0, islandStartSampleWithinWindow: 0),
            0)
    }

    func testOffsetNonFirstWindowIsland() {
        // A recovered island inside a non-first window must land at the right global
        // position. chunkStart = 161 frames (frame-aligned stride), island 2 frames in.
        let chunkStart = 161 * frame
        let islandStart = 2 * frame
        XCTAssertEqual(
            ChunkProcessor.islandGlobalFrameOffset(
                chunkStart: chunkStart, contextSamples: 0, islandStartSampleWithinWindow: islandStart),
            163)
    }

    func testOffsetAccountsForContextPrefix() {
        // The samples array begins at chunkStart - contextSamples: an island found at
        // array position `contextSamples` sits exactly at the chunk start globally.
        let chunkStart = 161 * frame
        let context = frame  // mel context = 1 encoder frame
        XCTAssertEqual(
            ChunkProcessor.islandGlobalFrameOffset(
                chunkStart: chunkStart, contextSamples: context, islandStartSampleWithinWindow: context),
            161,
            "Island at the context boundary must match the normal window's chunkStart/frame convention")
    }

    func testOffsetMatchesNormalWindowConventionAtIslandStartZeroNoContext() {
        let chunkStart = 206_080
        XCTAssertEqual(
            ChunkProcessor.islandGlobalFrameOffset(
                chunkStart: chunkStart, contextSamples: 0, islandStartSampleWithinWindow: 0),
            chunkStart / frame)
    }

    // MARK: - Divergent-overlap seam merge (empirically earned, 2026-06-30)

    func testDivergentOverlapSeamDoesNotDuplicate() {
        // The offline #1237 experiment showed a naive word-stitch DUPLICATES the seam when
        // the good window and the recovered window overlap with divergent words
        // ("...that you might" + "that you should..." → "...that you might that you should...").
        // Assert the REAL timestamp-aware merge collapses the shared "that you" (ids 100,101)
        // exactly once.
        let processor = ChunkProcessor(audioSamples: [])
        typealias TW = (token: Int, timestamp: Int, confidence: Float, duration: Int)

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

        let merged = processor.mergeTokenWindowsForTesting(left: left, right: right)
        let ids = merged.map { $0.token }

        XCTAssertEqual(ids.filter { $0 == 100 }.count, 1, "shared 'that' must appear once")
        XCTAssertEqual(ids.filter { $0 == 101 }.count, 1, "shared 'you' must appear once")
        XCTAssertTrue(ids.contains(103), "recovered 'should' must survive the seam")
        XCTAssertTrue(ids.contains(105), "recovered tail 'using' must survive the seam")
        XCTAssertEqual(merged.map { $0.timestamp }, merged.map { $0.timestamp }.sorted())
    }

    func testNonOverlappingWindowsConcatenate() {
        // Recovered island that starts well after the prior window (a real post-pause tail)
        // must simply append, losing nothing.
        let processor = ChunkProcessor(audioSamples: [])
        typealias TW = (token: Int, timestamp: Int, confidence: Float, duration: Int)
        let left: [TW] = [(token: 100, timestamp: 10, confidence: 0.9, duration: 1)]
        let right: [TW] = [(token: 200, timestamp: 180, confidence: 0.9, duration: 1)]
        let merged = processor.mergeTokenWindowsForTesting(left: left, right: right)
        XCTAssertEqual(merged.map { $0.token }, [100, 200], "Disjoint windows concatenate, no loss")
    }
}

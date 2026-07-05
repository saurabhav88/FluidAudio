import XCTest

@testable import FluidAudio

/// Characterization tests for the EW fork's finish()-time chunk text assembly
/// (ab05466c carry). This stitcher is the KNOWN #1329 culprit layer — it only
/// removes EXACT word overlaps, so a garbled re-decode of the overlap survives
/// as a phantom duplicate. It is carried onto v0.15.4 for strict behavior
/// parity (fresh-state-per-window depends on it); the #1329 PR-3/PR-4 winning
/// engine replaces it. These tests pin CURRENT behavior, including the known
/// failure mode, so any accidental drift in the carry is visible.
final class AssembleChunkTextsTests: XCTestCase {

    func testExactOverlapRemoved() {
        let out = SlidingWindowAsrManager.assembleChunkTexts([
            "we compare Gemini 3.5 Flash and",
            "Flash and Apple Intelligence today",
        ])
        XCTAssertEqual(out, "we compare Gemini 3.5 Flash and Apple Intelligence today")
    }

    func testPartialWordFragmentSkipped() {
        // prev ends "corrections." / next starts "ctions." — fragment dropped.
        let out = SlidingWindowAsrManager.assembleChunkTexts([
            "please apply the corrections.",
            "ctions. and then continue",
        ])
        XCTAssertEqual(out, "please apply the corrections. and then continue")
    }

    func testIntentionalRepetitionWithinChunkPreserved() {
        let out = SlidingWindowAsrManager.assembleChunkTexts([
            "GitHub GitHub GitHub is what I said",
            "said and nothing else",
        ])
        XCTAssertEqual(out, "GitHub GitHub GitHub is what I said and nothing else")
    }

    func testKnownLimitationDivergentOverlapSurvivesAsDuplicate() {
        // THE #1329 FAILURE MODE, pinned deliberately: the second window re-decodes
        // the overlap DIFFERENTLY ("Gemini 3.5 Flash" → "Knight three point five
        // flash"), exact matching whiffs, and the garbled duplicate SURVIVES.
        // If this test starts failing because the duplicate is gone, the carry
        // drifted (or a real fix landed) — either way, investigate; do not
        // silently accept.
        let out = SlidingWindowAsrManager.assembleChunkTexts([
            "compare Gemini 3.5 Flash and",
            "Knight three point five flash and Apple Intelligence",
        ])
        XCTAssertEqual(
            out,
            "compare Gemini 3.5 Flash and Knight three point five flash and Apple Intelligence",
            "divergent overlap is NOT deduped by design limitation (#1329)")
    }

    func testEmptyAndSingleChunkPassthrough() {
        XCTAssertEqual(SlidingWindowAsrManager.assembleChunkTexts([]), "")
        XCTAssertEqual(SlidingWindowAsrManager.assembleChunkTexts(["only chunk"]), "only chunk")
        XCTAssertEqual(
            SlidingWindowAsrManager.assembleChunkTexts(["a b c", "", "c d"]), "a b c d",
            "empty middle chunk skipped, overlap still collapses")
    }
}

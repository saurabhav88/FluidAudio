import XCTest

@testable import FluidAudio

/// Phrase boosting's greedy re-pick (#3338, carried from the #2610 prototype).
/// Pure runtime logic over a joint shortlist; the decoder-level parity run on
/// real clips lives in the EnviousWispr bench, not here.
final class TdtPhraseBoostTests: XCTestCase {

    private let blank = 8192

    private func runtime(
        alpha: Float, _ phrases: [[Int]], start: Bool = true, startScale: Float = 1
    ) -> PhraseBoostRuntime {
        PhraseBoostRuntime(
            config: PhraseBoostConfig(
                alpha: alpha,
                phrases: phrases.enumerated().map { PhraseBoostPhrase(text: "p\($0.offset)", tokenIds: $0.element) },
                allowPhraseStart: start, startScale: startScale))
    }

    func testNeverOverrulesABlank() {
        let r = runtime(alpha: 100, [[5]])
        XCTAssertNil(r.pick(predicted: blank, blankId: blank, topKIds: [blank, 5], topKLogits: [1, 0.9]))
    }

    func testAlphaZeroIsIdentity() {
        let r = runtime(alpha: 0, [[5]])
        XCTAssertNil(r.pick(predicted: 7, blankId: blank, topKIds: [7, 5], topKLogits: [2, 1.9]))
    }

    func testMissingOrEmptyShortlistChangesNothing() {
        let r = runtime(alpha: 10, [[5]])
        XCTAssertNil(r.pick(predicted: 7, blankId: blank, topKIds: nil, topKLogits: nil))
        XCTAssertNil(r.pick(predicted: 7, blankId: blank, topKIds: [], topKLogits: []))
    }

    func testOpeningWinsWithEnoughReward() {
        let r = runtime(alpha: 2, [[5, 6]])
        let picked = r.pick(predicted: 7, blankId: blank, topKIds: [7, 5], topKLogits: [3, 2])
        XCTAssertEqual(picked?.token, 5)
    }

    func testStartScaleShrinksTheOpeningReward() {
        // Opening reward = alpha * startScale = 0.5 cannot cover a gap of 1.
        let r = runtime(alpha: 1, [[5, 6]], startScale: 0.5)
        XCTAssertNil(r.pick(predicted: 7, blankId: blank, topKIds: [7, 5], topKLogits: [3, 2]))
        // The full continuation reward applies once the phrase is open.
        r.advance(consumed: 5)
        XCTAssertEqual(r.pick(predicted: 7, blankId: blank, topKIds: [7, 6], topKLogits: [3, 2.2])?.token, 6)
    }

    func testOpeningNotHelpedWhenStartsAreDisallowed() {
        let r = runtime(alpha: 10, [[5, 6]], start: false)
        XCTAssertNil(r.pick(predicted: 7, blankId: blank, topKIds: [7, 5], topKLogits: [3, 2]))
        r.advance(consumed: 5)  // the recogniser chose the opening itself
        XCTAssertEqual(r.pick(predicted: 7, blankId: blank, topKIds: [7, 6], topKLogits: [3, 2])?.token, 6)
    }

    func testUnreachableContinuationIsCountedAndChangesNothing() {
        let r = runtime(alpha: 10, [[5, 6]], start: false)
        r.advance(consumed: 5)
        XCTAssertNil(r.pick(predicted: 7, blankId: blank, topKIds: [7, 8], topKLogits: [3, 2]))
        XCTAssertEqual(r.stats.unreachableSteps, 1)
    }

    func testOverlappingPrefixesKeepEveryLivePath() {
        // [5, 6] continues while [6, 9] opens on the same token.
        let r = runtime(alpha: 10, [[5, 6, 7], [6, 9]])
        r.advance(consumed: 5)
        r.advance(consumed: 6)
        XCTAssertEqual(r.pick(predicted: 1, blankId: blank, topKIds: [1, 9], topKLogits: [3, 2])?.token, 9)
        XCTAssertEqual(r.pick(predicted: 1, blankId: blank, topKIds: [1, 7], topKLogits: [3, 2])?.token, 7)
    }

    func testAbandonedPrefixRestartsFromTheRoot() {
        let r = runtime(alpha: 10, [[5, 6]], start: false)
        r.advance(consumed: 5)
        r.advance(consumed: 8)  // path fails
        XCTAssertEqual(r.stats.abandonedPrefixes, 1)
        XCTAssertNil(r.pick(predicted: 7, blankId: blank, topKIds: [7, 6], topKLogits: [3, 2]))
        r.advance(consumed: 5)  // reopened
        XCTAssertEqual(r.pick(predicted: 7, blankId: blank, topKIds: [7, 6], topKLogits: [3, 2])?.token, 6)
    }

    func testCompletionIsCounted() {
        let r = runtime(alpha: 10, [[5, 6]])
        r.advance(consumed: 5)
        r.advance(consumed: 6)
        XCTAssertEqual(r.stats.completions, 1)
        XCTAssertEqual(r.stats.completedPhrases, ["p0"])
    }

    func testReturnedProbabilityIsTheUnboostedShortlistSoftmax() throws {
        let r = runtime(alpha: 10, [[5]])
        let logits: [Float] = [3, 2, 1]
        let picked = try XCTUnwrap(r.pick(predicted: 7, blankId: blank, topKIds: [7, 5, 4], topKLogits: logits))
        let sum = logits.map { expf($0 - 3) }.reduce(0, +)
        XCTAssertEqual(picked.token, 5)
        XCTAssertEqual(picked.probability, expf(2 - 3) / sum, accuracy: 1e-6)
    }

    func testRuntimesAreIndependent() {
        let config = PhraseBoostConfig(
            alpha: 10, phrases: [PhraseBoostPhrase(text: "p", tokenIds: [5, 6])], allowPhraseStart: false)
        let a = PhraseBoostRuntime(config: config)
        let b = PhraseBoostRuntime(config: config)
        a.advance(consumed: 5)
        XCTAssertEqual(a.pick(predicted: 7, blankId: blank, topKIds: [7, 6], topKLogits: [3, 2])?.token, 6)
        XCTAssertNil(b.pick(predicted: 7, blankId: blank, topKIds: [7, 6], topKLogits: [3, 2]))
    }
}

import XCTest

@testable import FluidAudio

final class EnglishBlocklistTests: XCTestCase {

    // Minimal vocab: a few English-exclusive and Latin-script French tokens.
    // Token IDs match the SentencePiece vocabulary used by Parakeet TDT v3.
    private let vocab: [Int: String] = [
        506: " the",
        575: " and",
        1868: " with",
        481: " le",
        393: " la",
        453: " et",
        999: " rendre",
        8192: "<blank>",
    ]
    private let blankId = 8192

    func testNoSubstitutionWhenLabelIsBlank() {
        var label = blankId
        var score: Float = 1.0
        TdtDecoderV3.applyEnglishBlocklist(
            label: &label,
            score: &score,
            topKIds: [481, 393],
            topKLogits: [1.0, 0.5],
            vocabulary: vocab,
            blankId: blankId
        )
        XCTAssertEqual(label, blankId)
        XCTAssertEqual(score, 1.0)
    }

    func testNoSubstitutionWhenLabelNotInBlocklist() {
        var label = 999  // ' rendre' — not an English-exclusive token
        var score: Float = 0.9
        TdtDecoderV3.applyEnglishBlocklist(
            label: &label,
            score: &score,
            topKIds: [999, 481],
            topKLogits: [2.0, 1.0],
            vocabulary: vocab,
            blankId: blankId
        )
        XCTAssertEqual(label, 999)
        XCTAssertEqual(score, 0.9)
    }

    func testNoSubstitutionWhenNoValidAlternativeInTopK() {
        var label = 506  // ' the'
        var score: Float = 0.8
        // Top-K contains only blocked English tokens and blank — nothing to substitute.
        TdtDecoderV3.applyEnglishBlocklist(
            label: &label,
            score: &score,
            topKIds: [506, 575, blankId],
            topKLogits: [3.0, 2.0, 1.0],
            vocabulary: vocab,
            blankId: blankId
        )
        XCTAssertEqual(label, 506)
        XCTAssertEqual(score, 0.8)
    }

    func testSubstitutesEnglishTokenWithBestLatinAlternative() {
        var label = 506  // ' the' — in blocklist
        var score: Float = 0.7
        // Top-K: ' the' wins, but ' le' (481) and ' et' (453) are valid French alternatives.
        TdtDecoderV3.applyEnglishBlocklist(
            label: &label,
            score: &score,
            topKIds: [506, 481, 453],
            topKLogits: [5.0, 4.0, 2.0],
            vocabulary: vocab,
            blankId: blankId
        )
        // Should pick ' le' (highest logit among non-blocked Latin alternatives)
        XCTAssertEqual(label, 481)
        XCTAssertGreaterThan(score, 0)
        XCTAssertLessThanOrEqual(score, 1)
    }

    func testSubstitutedScoreIsNormalisedProbability() {
        var label = 575  // ' and' — in blocklist
        var score: Float = 0.5
        // Two alternatives; substitution picks ' la' (393, logit 3.0).
        TdtDecoderV3.applyEnglishBlocklist(
            label: &label,
            score: &score,
            topKIds: [575, 393, 453],
            topKLogits: [4.0, 3.0, 1.0],
            vocabulary: vocab,
            blankId: blankId
        )
        XCTAssertEqual(label, 393)
        // Softmax over [4, 3, 1]: exp(3-4) / (exp(0) + exp(-1) + exp(-3))
        let maxL: Float = 4.0
        let sumExp = exp(4.0 - maxL) + exp(3.0 - maxL) + exp(1.0 - maxL)
        let expected = exp(3.0 - maxL) / sumExp
        XCTAssertEqual(score, expected, accuracy: 1e-5)
    }

    func testBlocklistContainsExpectedTokens() {
        XCTAssertTrue(TdtDecoderV3.englishBlocklistIds.contains(506))  // ' the'
        XCTAssertTrue(TdtDecoderV3.englishBlocklistIds.contains(575))  // ' and'
        XCTAssertTrue(TdtDecoderV3.englishBlocklistIds.contains(1868))  // ' with'
        XCTAssertFalse(TdtDecoderV3.englishBlocklistIds.contains(481))  // ' le'
        XCTAssertFalse(TdtDecoderV3.englishBlocklistIds.contains(453))  // ' et'
    }

    // MARK: - Scope
    //
    // The tests above drive `applyEnglishBlocklist` directly, so they pass no
    // matter which languages the decoder actually applies it to. That is why an
    // over-broad scope survived: the mechanism was tested and its APPLICABILITY
    // was not. These pin the predicate itself.

    func testBlocklistAppliesToFrench() {
        XCTAssertTrue(TdtDecoderV3.shouldApplyEnglishBlocklist(.french))
    }

    func testBlocklistDoesNotApplyToOtherLatinLanguages() {
        // Every one of these is Latin-script, and every one of them was subject
        // to the blocklist before this was scoped. German is the measured case:
        // the list bans ' was', ' will', ' so', ' we', ' her', ' not', which are
        // ordinary German words and German compound prefixes.
        for language in Language.allCases where language.script == .latin && language != .french {
            XCTAssertFalse(
                TdtDecoderV3.shouldApplyEnglishBlocklist(language),
                "\(language.rawValue) is Latin-script but the blocklist was never "
                    + "validated for it; widening needs that language's own evidence")
        }
    }

    func testBlocklistDoesNotApplyToEnglishOrNonLatinOrNil() {
        XCTAssertFalse(TdtDecoderV3.shouldApplyEnglishBlocklist(.english))
        XCTAssertFalse(TdtDecoderV3.shouldApplyEnglishBlocklist(.russian))  // Cyrillic
        XCTAssertFalse(TdtDecoderV3.shouldApplyEnglishBlocklist(.greek))
        // nil means "no language was requested", which disables language
        // conditioning entirely — the default the app ships today.
        XCTAssertFalse(TdtDecoderV3.shouldApplyEnglishBlocklist(nil))
    }

    /// Guards the property the two decoder call sites rely on: they both consult
    /// the same predicate, so exactly one language can ever reach the blocklist.
    /// A future edit that admits a second language fails here rather than
    /// silently corrupting that language's transcripts.
    func testExactlyOneLanguageIsAdmitted() {
        let admitted = Language.allCases.filter { TdtDecoderV3.shouldApplyEnglishBlocklist($0) }
        XCTAssertEqual(admitted, [.french])
    }
}

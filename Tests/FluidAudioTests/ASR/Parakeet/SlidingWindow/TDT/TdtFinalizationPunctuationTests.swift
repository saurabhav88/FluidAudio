import Foundation
import XCTest

@testable import FluidAudio

/// EnviousWispr #1792: the last-chunk finalization block may flush pending PUNCTUATION
/// after the encoder frames are exhausted, but must not emit lexical content there —
/// with no frames left to advance through, the prediction network free-runs and completes
/// the speaker's unfinished sentence.
///
/// These cover the classifier that draws that line. The emission-path behaviour itself is
/// covered by the corpus replay recorded in the EnviousWispr plan for #1792.
final class TdtFinalizationPunctuationTests: XCTestCase {

    private func classify(_ pieces: [Int: String], _ token: Int) -> Bool {
        TdtDecoderV3.isPunctuationOnlyPiece(token, vocabulary: pieces)
    }

    // MARK: - Punctuation is allowed through

    func testSentencePunctuationIsAllowed() {
        // `?` and `!` matter specifically: ASRConstants.punctuationTokens does NOT contain
        // them (its v3 ids map to ".", "й", "ó"), and gating on that list stranded the
        // question mark on 14 of 500 measured recordings.
        let vocab = [0: ".", 1: ",", 2: "?", 3: "!", 4: ":", 5: ";", 6: "-", 7: "--", 8: "'", 9: "¿", 10: "¡"]
        for id in vocab.keys.sorted() {
            XCTAssertTrue(
                classify(vocab, id),
                "\(vocab[id]!.debugDescription) is punctuation and must still flush")
        }
    }

    func testLeadingSpaceAndWordMarkerDoNotDefeatClassification() {
        // The shipped v3 vocabulary uses a literal leading space; older exports use U+2581.
        let vocab = [0: " .", 1: "\u{2581}.", 2: " ?"]
        for id in vocab.keys.sorted() {
            XCTAssertTrue(classify(vocab, id), "boundary marker must not change the verdict")
        }
    }

    // MARK: - Lexical content is blocked

    func testWordsAndDigitsAreBlocked() {
        let vocab = [
            0: " know", 1: "le", 2: " U", 3: " a", 4: "3", 5: "20", 6: " café", 7: "日本",
            8: "don't", 9: " U.S",
        ]
        for id in vocab.keys.sorted() {
            XCTAssertFalse(
                classify(vocab, id),
                "\(vocab[id]!.debugDescription) carries lexical content and must be suppressed")
        }
    }

    func testMixedPieceWithASingleLetterIsBlocked() {
        // A piece is lexical if ANY character is a letter or digit — `.S` must not slip
        // through on the strength of its leading period.
        XCTAssertFalse(classify([0: ".S"], 0))
        XCTAssertFalse(classify([0: "3."], 0))
    }

    // MARK: - Fail closed

    func testUnknownAndEmptyFailClosed() {
        XCTAssertFalse(classify([:], 42), "unknown vocabulary id must suppress, not emit")
        XCTAssertFalse(classify([0: ""], 0), "empty piece must suppress")
        XCTAssertFalse(classify([0: "\u{2581}"], 0), "marker-only piece must suppress")
        XCTAssertFalse(classify([0: "x"], 1), "id absent from the table must suppress")
    }

    // MARK: - Documented over-admission

    func testKnownNonPunctuationSymbolsAreAdmitted() {
        // Documented limitation rather than intent: these carry no letter or digit, so the
        // classifier admits them. An instrumented trace over 556 clips — including 56 built
        // specifically to provoke currency and percent endings — recorded zero such
        // emissions, so this is left as-is rather than tightened on speculation.
        for piece in [" ", "$", "%", "/", "£", "€"] {
            XCTAssertTrue(
                classify([0: piece], 0),
                "\(piece.debugDescription) is admitted today; change this test deliberately")
        }
    }
}

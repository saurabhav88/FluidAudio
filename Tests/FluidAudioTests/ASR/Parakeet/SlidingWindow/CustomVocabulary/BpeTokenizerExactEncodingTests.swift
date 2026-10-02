import Foundation
import XCTest

@testable import FluidAudio

/// `BpeTokenizer` loading of array-form merges, the explicit normalization policy and
/// `encodeExact` (EnviousWispr #3338), against NVIDIA's own `parakeet-tdt-0.6b-v3`
/// `tokenizer.json` (revision 541d1f99c6b0c3cd0b11a95167540bb8edefd82b, array-form merges)
/// and the CTC 110m tokenizer (string-form merges). Expected ids come from the Hugging Face
/// `tokenizers` library 0.23.2, not from this implementation.
///
/// The v3 file is read from the Hugging Face cache, or from `EW_PARAKEET_V3_TOKENIZER`.
/// Like `BpeTokenizerVariantsTests`, these skip when the files are absent.
final class BpeTokenizerExactEncodingTests: XCTestCase {

    private static let v3Revision = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"

    private var v3TokenizerFile: URL? {
        if let override = ProcessInfo.processInfo.environment["EW_PARAKEET_V3_TOKENIZER"] {
            return URL(fileURLWithPath: override)
        }
        let file = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub/models--nvidia--parakeet-tdt-0.6b-v3/snapshots")
            .appendingPathComponent(Self.v3Revision)
            .appendingPathComponent("tokenizer.json")
        return FileManager.default.fileExists(atPath: file.path) ? file : nil
    }

    private var ctcTokenizerFolder: URL? {
        guard
            let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
                .first
        else { return nil }
        let dir = appSupport.appendingPathComponent("FluidAudio/Models/parakeet-ctc-110m-coreml", isDirectory: true)
        return FileManager.default.fileExists(atPath: dir.appendingPathComponent("tokenizer.json").path) ? dir : nil
    }

    private func v3Tokenizer() throws -> BpeTokenizer {
        guard let file = v3TokenizerFile else {
            throw XCTSkip("parakeet-tdt-0.6b-v3 tokenizer.json not in the Hugging Face cache")
        }
        return try BpeTokenizer.load(tokenizerFile: file)
    }

    /// Hugging Face `tokenizers` 0.23.2 on the official v3 file, `encode(text, add_special_tokens=False)`.
    private static let v3Expected: [(String, [Int])] = [
        ("Müller", [356, 1144, 960]),
        ("François", [2914, 5642, 290]),
        ("Łukasz", [7863, 8159, 1174, 3627]),
        ("Dvořák", [360, 528, 8004, 1716]),
        ("Москва", [3796, 3698, 392]),
        ("Αθήνα", [1335, 5623, 456]),
        ("O'Brien", [452, 7931, 7963, 330, 274]),
        ("Jean-Luc", [1031, 284, 7918, 7968, 7873, 7880]),
        ("New York", [2634, 4722]),
        ("Ursula von der Leyen", [517, 6832, 2867, 1473, 623, 1333, 3135]),
        ("iPhone", [312, 7946, 7882, 1116]),
        ("NASA", [382, 7924, 5090]),
    ]

    func testArrayFormMergesLoadAndSpellExactly() throws {
        let tokenizer = try v3Tokenizer()
        var executed = 0
        for (text, expected) in Self.v3Expected {
            XCTAssertEqual(
                try tokenizer.encodeExact(text, normalization: .canonicalCasePreserving).get(), expected, text)
            executed += 1
        }
        XCTAssertEqual(executed, Self.v3Expected.count)
    }

    func testCanonicalPolicyComposesDecomposedAccentsAndKeepsCase() throws {
        let tokenizer = try v3Tokenizer()
        let decomposed = "Mu\u{0308}ller"
        XCTAssertNotEqual(Array(decomposed.unicodeScalars), Array("Müller".unicodeScalars))
        XCTAssertEqual(
            try tokenizer.encodeExact(decomposed, normalization: .canonicalCasePreserving).get(), [356, 1144, 960])
        let upper = try tokenizer.encodeExact("NASA", normalization: .canonicalCasePreserving).get()
        let lower = try tokenizer.encodeExact("nasa", normalization: .canonicalCasePreserving).get()
        XCTAssertNotEqual(upper, lower)
        XCTAssertEqual(try tokenizer.encodeExact("NASA", normalization: .lowercasedCompatibility).get(), lower)
    }

    func testExactEncodingRefusesInsteadOfSubstituting() throws {
        let tokenizer = try v3Tokenizer()
        // Greek final sigma and "&" have no v3 piece (Hugging Face gives the unknown id 0).
        XCTAssertEqual(
            tokenizer.encodeExact("Σωκράτης", normalization: .canonicalCasePreserving), .failure(.unknownPiece("ς")))
        XCTAssertEqual(
            tokenizer.encodeExact("AT&T", normalization: .canonicalCasePreserving), .failure(.unknownPiece("&")))
        // The lenient encoder substitutes the unknown id for the same input.
        XCTAssertTrue(tokenizer.encode("AT&T").contains(0))
        // Hugging Face: "▁" -> [], "New▁York" -> [2634, 4722]; encodeExact refuses rather than doubling the mark.
        for text in ["\u{2581}", "\u{2581}New", "New\u{2581}York"] {
            XCTAssertEqual(
                tokenizer.encodeExact(text, normalization: .canonicalCasePreserving), .failure(.boundaryMarkInInput), text)
        }
        for text in ["", " New", "New ", "New  York"] {
            XCTAssertEqual(
                tokenizer.encodeExact(text, normalization: .canonicalCasePreserving), .failure(.emptyWord), text)
        }
        XCTAssertEqual(
            tokenizer.encodeExact("<unk>", normalization: .canonicalCasePreserving).isSuccess, false,
            "special-token text never spells as a special id")
    }

    func testStringAndArrayMergeFormsGiveIdenticalIds() throws {
        guard let file = v3TokenizerFile else {
            throw XCTSkip("parakeet-tdt-0.6b-v3 tokenizer.json not in the Hugging Face cache")
        }
        // Rewrite NVIDIA's array-form merges as "a b" strings, in order, into a temporary copy.
        var json = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        var model = try XCTUnwrap(json["model"] as? [String: Any])
        let pairs = try XCTUnwrap(model["merges"] as? [[String]])
        XCTAssertGreaterThan(pairs.count, 0)
        XCTAssertFalse(pairs.contains { $0.contains { $0.contains(" ") } }, "a space inside a piece has no string form")
        model["merges"] = pairs.map { $0.joined(separator: " ") }
        json["model"] = model
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try JSONSerialization.data(withJSONObject: json).write(to: dir.appendingPathComponent("tokenizer.json"))

        let arrayForm = try BpeTokenizer.load(tokenizerFile: file)
        let stringForm = try BpeTokenizer.load(from: dir)
        var comparisons = 0
        for (text, _) in Self.v3Expected {
            XCTAssertEqual(
                arrayForm.encodeExact(text, normalization: .canonicalCasePreserving),
                stringForm.encodeExact(text, normalization: .canonicalCasePreserving), text)
            XCTAssertEqual(arrayForm.encode(text), stringForm.encode(text), text)
            comparisons += 2
        }
        XCTAssertEqual(comparisons, Self.v3Expected.count * 2)
    }

    func testMalformedArrayMergeIsRejected() throws {
        guard let file = v3TokenizerFile else {
            throw XCTSkip("parakeet-tdt-0.6b-v3 tokenizer.json not in the Hugging Face cache")
        }
        var json = try XCTUnwrap(try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any])
        var model = try XCTUnwrap(json["model"] as? [String: Any])
        var pairs = try XCTUnwrap(model["merges"] as? [[String]])
        pairs[3] = [pairs[3][0]]
        model["merges"] = pairs
        json["model"] = model
        let bad = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: bad) }
        try JSONSerialization.data(withJSONObject: json).write(to: bad)
        XCTAssertThrowsError(try BpeTokenizer.load(tokenizerFile: bad)) { error in
            guard case BpeTokenizer.Error.malformedMerge(index: 3) = error else {
                return XCTFail("expected malformedMerge(index: 3), got \(error)")
            }
        }
    }

    /// The CTC default path is unchanged: lowercase + NFKC, `<unk>` substitution kept.
    /// Expected ids: Hugging Face `tokenizers` on the CTC 110m `tokenizer.json`, input lowercased then NFKC.
    func testCtcDefaultEncodingIsUnchanged() throws {
        guard let folder = ctcTokenizerFolder else {
            throw XCTSkip("parakeet-ctc-110m tokenizer not present locally")
        }
        let tokenizer = try BpeTokenizer.load(from: folder)
        let expected: [(String, [Int])] = [
            ("hello world", [80, 30, 969, 632]),
            ("nvidia", [43, 989, 69, 457]),
            ("Livmarli", [105, 989, 980, 38, 749]),
            ("Codex", [16, 220, 967, 996]),
            ("café", [804, 984, 0]),
            ("Saurabh Vaish", [282, 78, 184, 975, 173, 970, 376]),
            ("kubernetes", [102, 978, 351, 971, 50, 25]),
            ("\u{FB01}le", [22, 441]),
            ("\u{FF26}\u{FF55}\u{FF4C}\u{FF4C}", [22, 511]),
        ]
        var executed = 0
        for (text, ids) in expected {
            XCTAssertEqual(tokenizer.encode(text), ids, text)
            executed += 1
        }
        XCTAssertEqual(executed, expected.count)
    }
}

extension Result {
    fileprivate var isSuccess: Bool {
        if case .success = self { return true }
        return false
    }
}

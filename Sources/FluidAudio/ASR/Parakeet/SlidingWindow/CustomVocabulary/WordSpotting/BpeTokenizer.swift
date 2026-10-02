import Foundation

/// BPE tokenizer for CTC vocabulary boosting.
/// Only implements encoding - no decoding, chat templates, or other features.
/// Supports the specific tokenizer.json format used by Parakeet models.
///
/// Text normalization: `encode` applies lowercasing + NFKC normalization before BPE
/// encoding, matching the standard NeMo CTC tokenization pipeline. `encodeExact`
/// takes an explicit `Normalization` and never substitutes `<unk>`.
public final class BpeTokenizer: Sendable {
    private let vocab: [String: Int]
    private let merges: [(String, String)]
    private let addedTokens: [String: Int]
    private let specialTokenIDs: Set<Int>

    public enum Error: Swift.Error, LocalizedError {
        case fileNotFound(URL)
        case invalidJSON(String)
        case missingField(String)
        case unsupportedTokenizerType(String)
        case malformedMerge(index: Int)

        public var errorDescription: String? {
            switch self {
            case .fileNotFound(let url):
                return "tokenizer.json not found at \(url.path)"
            case .invalidJSON(let message):
                return "Invalid JSON: \(message)"
            case .missingField(let field):
                return "Missing required field: \(field)"
            case .unsupportedTokenizerType(let type):
                return "Unsupported tokenizer type: \(type). Only 'BPE' is supported."
            case .malformedMerge(let index):
                return "Merge \(index) is neither \"a b\" nor a two-element array"
            }
        }
    }

    /// Text normalization applied before BPE.
    public enum Normalization: Sendable {
        /// Lowercase, then NFKC. The CTC keyword-spotting default (`encode`).
        case lowercasedCompatibility
        /// NFC only: case and characters are kept; canonically decomposed accents compose.
        case canonicalCasePreserving
    }

    /// Why `encodeExact` refused to spell a text.
    public enum ExactEncodingFailure: Swift.Error, Equatable, Sendable {
        /// Empty text, or a leading, trailing or repeated space (an empty word).
        case emptyWord
        /// The text contains the `▁` word-boundary mark itself (a Metaspace pre-tokenizer splits on
        /// it rather than spelling it, which `encodeExact` does not model).
        case boundaryMarkInInput
        /// A BPE piece is not in the vocabulary.
        case unknownPiece(String)
        /// A BPE piece resolves only to an added (special) token.
        case specialTokenPiece(String)
    }

    /// Load tokenizer from a folder containing tokenizer.json
    public static func load(from modelFolder: URL) throws -> BpeTokenizer {
        try load(tokenizerFile: modelFolder.appendingPathComponent("tokenizer.json"))
    }

    /// Load tokenizer from a tokenizer.json-shaped file (`model.type` "BPE", `model.vocab`,
    /// `model.merges` as "a b" strings or two-element arrays, optional `added_tokens`).
    public static func load(tokenizerFile tokenizerPath: URL) throws -> BpeTokenizer {
        guard FileManager.default.fileExists(atPath: tokenizerPath.path) else {
            throw Error.fileNotFound(tokenizerPath)
        }

        let data = try Data(contentsOf: tokenizerPath)

        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Error.invalidJSON("Root is not a dictionary")
        }

        // Parse model section
        guard let model = json["model"] as? [String: Any] else {
            throw Error.missingField("model")
        }

        guard let modelType = model["type"] as? String else {
            throw Error.missingField("model.type")
        }

        guard modelType == "BPE" else {
            throw Error.unsupportedTokenizerType(modelType)
        }

        // Parse vocabulary: {"token": id, ...}
        guard let vocabDict = model["vocab"] as? [String: Int] else {
            throw Error.missingField("model.vocab")
        }

        // Parse merges in file order: ["a b", ...] or [["a", "b"], ...]
        let merges: [(String, String)]
        if let mergesArray = model["merges"] as? [String] {
            merges = mergesArray.compactMap { mergeStr -> (String, String)? in
                let parts = mergeStr.split(separator: " ", maxSplits: 1)
                guard parts.count == 2 else { return nil }
                return (String(parts[0]), String(parts[1]))
            }
        } else if let mergePairs = model["merges"] as? [[String]] {
            merges = try mergePairs.enumerated().map { index, pair in
                guard pair.count == 2 else { throw Error.malformedMerge(index: index) }
                return (pair[0], pair[1])
            }
        } else {
            throw Error.missingField("model.merges")
        }

        // Parse added_tokens (special tokens like <unk>, <pad>)
        var addedTokensDict: [String: Int] = [:]
        let addedTokensList = (json["added_tokens"] as? [[String: Any]]) ?? []
        for token in addedTokensList {
            guard let content = token["content"] as? String,
                let id = token["id"] as? Int
            else { continue }
            addedTokensDict[content] = id
        }

        return BpeTokenizer(
            vocab: vocabDict,
            merges: merges,
            addedTokens: addedTokensDict
        )
    }

    private init(vocab: [String: Int], merges: [(String, String)], addedTokens: [String: Int]) {
        self.vocab = vocab
        self.merges = merges
        self.addedTokens = addedTokens
        self.specialTokenIDs = Set(addedTokens.values)
    }

    /// Encode text to token IDs using BPE.
    ///
    /// - Parameters:
    ///   - text: Input text.
    ///   - addSpecialTokens: Currently unused; kept for API compatibility.
    ///   - prependWordBoundary: If `true` (default), prepend a SentencePiece
    ///     `▁` word-boundary marker before encoding, matching standard NeMo
    ///     CTC tokenization. Set to `false` to produce a "mid-utterance"
    ///     tokenization that omits the leading boundary, which is useful
    ///     for keyword-spotting against audio where the keyword does not
    ///     start at a word boundary (e.g. compound matches like
    ///     `Liv` + `marli` → `Livmarli`).
    public func encode(
        _ text: String,
        addSpecialTokens: Bool = false,
        prependWordBoundary: Bool = true
    ) -> [Int] {
        // Normalize: lowercase + NFKC normalization (matches NeMo CTC models)
        let normalized = normalize(text, .lowercasedCompatibility)

        // Pre-tokenize: replace spaces with ▁ (sentencepiece style)
        let leading = prependWordBoundary ? ASRConstants.sentencePieceWordBoundary : ""
        let preprocessed =
            leading + normalized.replacingOccurrences(of: " ", with: ASRConstants.sentencePieceWordBoundary)

        // Split into characters
        let word = applyMerges(preprocessed.map { String($0) })

        // Convert tokens to IDs
        return word.compactMap { token -> Int? in
            // Check added tokens first (special tokens)
            if let id = addedTokens[token] {
                return id
            }
            // Then check vocabulary
            if let id = vocab[token] {
                return id
            }
            // Unknown token - return <unk> ID if available
            return addedTokens["<unk>"] ?? vocab["<unk>"] ?? 0
        }
    }

    /// Encode text to exact token IDs, or refuse.
    ///
    /// Splits on U+0020 like a `Metaspace` pre-tokenizer with `split: true`, prepends
    /// the `▁` word boundary to each word, runs the same merges as `encode` on each
    /// word's Unicode scalars, and resolves every piece in the vocabulary. Never
    /// substitutes `<unk>`, never drops a piece and never returns a partial spelling.
    public func encodeExact(
        _ text: String,
        normalization: Normalization
    ) -> Result<[Int], ExactEncodingFailure> {
        let normalized = normalize(text, normalization)
        let words = normalized.split(separator: " ", omittingEmptySubsequences: false)
        var ids: [Int] = []
        for word in words {
            guard !word.isEmpty else { return .failure(.emptyWord) }
            guard !word.contains(ASRConstants.sentencePieceWordBoundary) else {
                return .failure(.boundaryMarkInInput)
            }
            let symbols = (ASRConstants.sentencePieceWordBoundary + word).unicodeScalars.map { String($0) }
            for piece in applyMerges(symbols) {
                if let id = vocab[piece], !specialTokenIDs.contains(id), addedTokens[piece] == nil {
                    ids.append(id)
                } else if addedTokens[piece] != nil || vocab[piece] != nil {
                    return .failure(.specialTokenPiece(piece))
                } else {
                    return .failure(.unknownPiece(piece))
                }
            }
        }
        return .success(ids)
    }

    private func normalize(_ text: String, _ normalization: Normalization) -> String {
        switch normalization {
        case .lowercasedCompatibility:
            return text.lowercased().precomposedStringWithCompatibilityMapping
        case .canonicalCasePreserving:
            return text.precomposedStringWithCanonicalMapping
        }
    }

    /// Apply BPE merges iteratively: the highest-priority (earliest) merge first, to every
    /// occurrence, until no merge applies.
    private func applyMerges(_ symbols: [String]) -> [String] {
        var word = symbols

        // Apply BPE merges iteratively
        while word.count >= 2 {
            // Find the highest priority merge (earliest in merges list)
            var bestMergeIndex: Int? = nil
            var bestMergePair: (String, String)? = nil

            for i in 0..<word.count - 1 {
                let pair = (word[i], word[i + 1])

                // Check if this pair has a merge rule
                guard let mergeIndex = merges.firstIndex(where: { $0.0 == pair.0 && $0.1 == pair.1 }) else {
                    continue
                }

                // Update best merge if this is higher priority (lower index)
                if bestMergeIndex.map({ mergeIndex < $0 }) ?? true {
                    bestMergeIndex = mergeIndex
                    bestMergePair = pair
                }
            }

            // No more merges possible
            guard let (first, second) = bestMergePair else { break }

            // Apply the merge to ALL occurrences of the winning pair (standard BPE)
            var newWord: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1 && word[i] == first && word[i + 1] == second {
                    newWord.append(first + second)
                    i += 2  // Skip the next token since we merged it
                } else {
                    newWord.append(word[i])
                    i += 1
                }
            }
            word = newWord
        }
        return word
    }
}

import Foundation

/// Phrase boosting for the TDT decoder: adds a reward to the next piece of any
/// user phrase the decoder is part-way through, or could begin, and lets that
/// reward decide the greedy pick.
///
/// This is the published shape (NVIDIA GPU-PB, WeNet/icefall context graphs)
/// applied at the one place this decoder chooses a token.
///
/// **Three limits are structural. They are stated here so no reader has to
/// rediscover them, and each is measured rather than assumed.**
///
/// 1. **The reward is applied inside the joint's top-K shortlist, not the full
///    vocabulary.** The compiled joint exports 64 ids and 64 raw logits, so a
///    piece outside that shortlist cannot be reached at any reward.
/// 2. **Greedy decoding cannot take a token back.** The context-graph
///    literature repairs a failed partial phrase by subtracting the reward it
///    granted; that fixes a cumulative SCORE, and a greedy decoder has already
///    emitted the text. So the caller must decide, after the fact, whether to
///    keep the boosted transcript at all.
/// 3. **This runtime tracks the PREDICTOR's token stream, which is not the
///    emitted transcript.** Tokens suppressed by the emission window still move
///    the language context, so the trie must consume them, and therefore a
///    completion counted here can be one the user never sees. Never use these
///    counters as the safety gate; decide that on the final text.
public struct PhraseBoostPhrase: Sendable {
    public let text: String
    public let tokenIds: [Int]

    public init(text: String, tokenIds: [Int]) {
        self.text = text
        self.tokenIds = tokenIds
    }
}

public struct PhraseBoostConfig: Sendable {
    /// Reward added to a candidate continuation's raw logit. Zero is the
    /// identity: no comparison changes.
    public let alpha: Float
    public let phrases: [PhraseBoostPhrase]

    /// When false, a phrase can only be helped once the recogniser has chosen
    /// its first piece by itself.
    public let allowPhraseStart: Bool

    /// Fraction of the reward a phrase's FIRST piece gets. Openings are where
    /// the damage is, because the recogniser is not part-way through anything
    /// and a promoted first piece is an invention; continuations are where the
    /// wins are, because the beginning of the word was heard correctly.
    public let startScale: Float

    public init(
        alpha: Float,
        phrases: [PhraseBoostPhrase],
        allowPhraseStart: Bool = true,
        startScale: Float = 1
    ) {
        self.alpha = alpha
        self.phrases = phrases
        self.allowPhraseStart = allowPhraseStart
        self.startScale = startScale
    }
}

/// Research counters. **None of these is a safety signal.** They describe what
/// the decoder did internally; whether the USER got a custom word is a property
/// of the final text and is decided by the caller.
struct PhraseBoostStats: Sendable, Codable {
    /// Phrase ends reached in the predictor's token stream.
    var completions: Int = 0
    /// Times the reward changed the greedy pick.
    var overrides: Int = 0
    /// Active trie paths that failed on the next consumed token. A
    /// path-transition diagnostic, NOT a count of transcript damage: it also
    /// counts paths entered without any reward, paths that already completed at
    /// an end-with-continuations, and predictor-only tokens the user never saw.
    var abandonedPrefixes: Int = 0
    /// Steps where no piece of any live path was in the shortlist.
    var unreachableSteps: Int = 0
    /// Steps taken while at least one path was live.
    var insidePhraseSteps: Int = 0
    /// Deepest shortlist rank at which a rewarded piece won.
    var deepestWinningRank: Int = 0
    /// Phrase ends reached, in order.
    var completedPhrases: [String] = []
    /// Steps where only an opening could have been chosen.
    var rootSteps: Int = 0
    /// Of those, steps where no opening of any phrase was in the shortlist.
    var rootStepsWithNoCandidate: Int = 0
    /// Smallest amount by which the best available opening trailed the
    /// recogniser's own choice, over the whole clip. **Not the reward required
    /// at the intended word**, which has to be measured at an aligned position.
    var smallestStartDeficit: Float?

    mutating func merge(_ other: PhraseBoostStats) {
        completions += other.completions
        overrides += other.overrides
        abandonedPrefixes += other.abandonedPrefixes
        unreachableSteps += other.unreachableSteps
        insidePhraseSteps += other.insidePhraseSteps
        deepestWinningRank = max(deepestWinningRank, other.deepestWinningRank)
        completedPhrases += other.completedPhrases
        rootSteps += other.rootSteps
        rootStepsWithNoCandidate += other.rootStepsWithNoCandidate
        if let d = other.smallestStartDeficit {
            smallestStartDeficit = min(smallestStartDeficit ?? d, d)
        }
    }
}

/// Trie over phrase token sequences, plus every position a greedy decoder could
/// currently be in.
///
/// One runtime belongs to ONE serial decode trajectory. `TdtDecoderV3` builds it
/// inside `decodeWithTimings` from the caller's immutable, `Sendable`
/// `PhraseBoostConfig`, so it never crosses an isolation boundary and needs no
/// lock: concurrent chunks each decode in their own call and get their own
/// runtime (#3338; replaces the #2610 prototype's lock + `@unchecked Sendable`,
/// which this repository forbids).
final class PhraseBoostRuntime {
    private struct Node {
        var children: [Int: Int] = [:]
        var terminalPhrase: Int?
    }

    private var nodes: [Node] = [Node()]
    private let phraseNames: [String]
    let alpha: Float
    private let startScale: Float
    private let allowPhraseStart: Bool

    /// Every live path, not just the deepest. A single position forgets a phrase
    /// that opened while another was in progress, which any list where one
    /// phrase's continuation is another phrase's opening will produce, and it
    /// only notices when the surviving path fails, by which point the forgotten
    /// one cannot be resumed.
    private var active: Set<Int> = [0]

    private var _stats = PhraseBoostStats()

    var stats: PhraseBoostStats { _stats }

    init(config: PhraseBoostConfig) {
        alpha = config.alpha
        startScale = config.startScale
        allowPhraseStart = config.allowPhraseStart
        phraseNames = config.phrases.map(\.text)
        for (index, phrase) in config.phrases.enumerated() {
            guard !phrase.tokenIds.isEmpty else { continue }
            var current = 0
            for token in phrase.tokenIds {
                if let next = nodes[current].children[token] {
                    current = next
                } else {
                    nodes.append(Node())
                    let next = nodes.count - 1
                    nodes[current].children[token] = next
                    current = next
                }
            }
            nodes[current].terminalPhrase = index
        }
    }

    private var startAlpha: Float { alpha * startScale }

    /// `true` marks a piece that continues a path already in progress; `false`
    /// marks one that would open a new phrase. A piece that does both is a
    /// continuation, which is the stronger claim about what was heard.
    private func candidates() -> [Int: Bool] {
        var out: [Int: Bool] = [:]
        if allowPhraseStart {
            for token in nodes[0].children.keys { out[token] = false }
        }
        for node in active where node != 0 {
            for token in nodes[node].children.keys { out[token] = true }
        }
        return out
    }

    private var isInsideAPhrase: Bool { active.contains { $0 != 0 } }

    /// Re-pick the label from the shortlist with the reward applied.
    ///
    /// The reward decides which token wins. The returned probability is
    /// computed from unboosted logits, normalized over the available
    /// shortlist. It excludes the phrase reward but is not a calibrated
    /// full-vocabulary probability and may exceed that probability.
    /// This follows the existing shortlist replacement convention.
    func pick(
        predicted: Int,
        blankId: Int,
        topKIds: [Int]?,
        topKLogits: [Float]?
    ) -> (token: Int, probability: Float)? {
        guard alpha != 0 else { return nil }

        // NEVER overrule silence. Measured on 500 real recordings: seven clips
        // that correctly transcribed to nothing came back as hundreds of words
        // of babble, because an opening was rewarded past the no-token choice
        // and the decoder then had to keep producing. Silence is not a word the
        // user got wrong; it is the absence of a word.
        guard predicted != blankId else { return nil }

        guard let ids = topKIds, let logits = topKLogits else { return nil }
        let count = min(ids.count, logits.count)
        guard count > 0 else { return nil }

        var unboostedBest: Float = -.infinity
        for i in 0..<count where logits[i] > unboostedBest { unboostedBest = logits[i] }

        let boostable = candidates()

        if isInsideAPhrase {
            _stats.insidePhraseSteps += 1
            let reachable = (0..<count).contains { boostable[ids[$0]] == true }
            if !reachable { _stats.unreachableSteps += 1 }
        } else if !nodes[0].children.isEmpty {
            _stats.rootSteps += 1
            var best: Float?
            for i in 0..<count where nodes[0].children[ids[i]] != nil {
                if best == nil || logits[i] > best! { best = logits[i] }
            }
            if let best {
                let deficit = unboostedBest - best
                if _stats.smallestStartDeficit == nil || deficit < _stats.smallestStartDeficit! {
                    _stats.smallestStartDeficit = deficit
                }
            } else {
                _stats.rootStepsWithNoCandidate += 1
            }
        }

        guard !boostable.isEmpty else { return nil }

        var bestIdx = -1
        var bestValue: Float = -.infinity
        for i in 0..<count {
            var reward: Float = 0
            if let isContinuation = boostable[ids[i]] {
                reward = isContinuation ? alpha : startAlpha
            }
            let value = logits[i] + reward
            if bestIdx < 0 || value > bestValue {
                bestValue = value
                bestIdx = i
            }
        }
        guard bestIdx >= 0 else { return nil }
        let chosen = ids[bestIdx]
        guard chosen != predicted else { return nil }

        _stats.overrides += 1
        if boostable[chosen] != nil {
            let rank = bestIdx + 1
            if rank > _stats.deepestWinningRank { _stats.deepestWinningRank = rank }
        }

        var maxLogit = -Float.infinity
        for i in 0..<count where logits[i] > maxLogit { maxLogit = logits[i] }
        guard maxLogit.isFinite else { return (chosen, 0) }
        var sumExp: Float = 0
        for i in 0..<count { sumExp += expf(logits[i] - maxLogit) }
        guard sumExp > 0 else { return (chosen, 0) }
        let probability = expf(logits[bestIdx] - maxLogit) / sumExp
        return (chosen, max(0, min(1, probability)))
    }

    /// Move every live path with a token the decoder CONSUMED.
    ///
    /// Named for the predictor stream on purpose: this includes tokens the
    /// emission window suppresses, because those still move the language
    /// context, and a phrase spanning that boundary could otherwise never close.
    ///
    /// Blanks never reach here: a blank is silence, and silence occurs inside a
    /// spoken word, so resetting on one would break every multi-piece phrase.
    func advance(consumed token: Int) {
        var next: Set<Int> = []
        var anyFailed = false
        for node in active {
            if let child = nodes[node].children[token] {
                next.insert(child)
                if let phrase = nodes[child].terminalPhrase {
                    _stats.completions += 1
                    _stats.completedPhrases.append(phraseNames[phrase])
                }
            } else if node != 0 {
                anyFailed = true
            }
        }
        if anyFailed { _stats.abandonedPrefixes += 1 }
        // The root is always live, so a new phrase can open on the very next
        // token even while another path is running.
        next.insert(0)
        active = next
    }
}

/// Test-only observation of a boosted decode (#3338). Production callers pass
/// no trace; the decoder emits nothing and does no extra work without one.
enum PhraseBoostTraceEvent {
    enum Site { case main, inner }
    /// A main-site step about to consult the runtime, with its counters so far.
    case step(PhraseBoostStats)
    /// The reward changed the greedy pick at `site` and encoder `frame`;
    /// `afterFilters` is the label the script filter and the French blocklist
    /// left standing.
    case override(site: Site, frame: Int, picked: Int, afterFilters: Int)
    /// The call is returning or throwing; the runtime's final counters.
    case finished(PhraseBoostStats)
}

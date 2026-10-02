import Foundation

/// A phrase-boost request as it travels from the public entry to every decoder
/// call (#3338). It carries only immutable configuration: each decoder call
/// builds its own call-local `PhraseBoostRuntime`, so concurrent chunks never
/// share trie state.
struct PhraseBoostRequest: Sendable {
    let config: PhraseBoostConfig
    /// Test-only observation. `trajectory` is nil on the single-window path and
    /// the chunk index on the chunked path.
    let trace: (@Sendable (_ trajectory: Int?, _ event: PhraseBoostTraceEvent) -> Void)?
    let trajectory: Int?

    init(
        config: PhraseBoostConfig,
        trace: (@Sendable (_ trajectory: Int?, _ event: PhraseBoostTraceEvent) -> Void)? = nil,
        trajectory: Int? = nil
    ) {
        self.config = config
        self.trace = trace
        self.trajectory = trajectory
    }

    /// The same request, attributed to one chunk's decode.
    func forTrajectory(_ index: Int) -> PhraseBoostRequest {
        PhraseBoostRequest(config: config, trace: trace, trajectory: index)
    }

    /// The decoder-level trace for this trajectory, if anyone is observing.
    var decoderTrace: ((PhraseBoostTraceEvent) -> Void)? {
        guard let trace else { return nil }
        let trajectory = self.trajectory
        return { event in trace(trajectory, event) }
    }
}

extension AsrManager {

    /// Transcribe with phrase boosting: a reward on the next piece of any
    /// phrase in `boost` may change the decoder's greedy pick (#3338).
    ///
    /// Uses the ordinary transcription path, its decoder-state initialization
    /// and `language` option. Requires the Parakeet v3 model; any other loaded
    /// version throws `ASRError.processingFailed` instead of returning an
    /// unboosted transcript. `alpha: 0` is the identity.
    ///
    /// Multi-chunk transcription with dual-decode arbitration enabled and
    /// mel chunk context disabled is unsupported and throws
    /// `ASRError.processingFailed`.
    public func transcribeWithPhraseBoost(
        _ samples: [Float],
        boost: PhraseBoostConfig,
        language: Language? = nil
    ) async throws -> ASRResult {
        try await transcribeWithPhraseBoost(samples, request: PhraseBoostRequest(config: boost), language: language)
    }

    internal func transcribeWithPhraseBoost(
        _ samples: [Float],
        request: PhraseBoostRequest,
        language: Language? = nil
    ) async throws -> ASRResult {
        if let version = asrModels?.version, version != .v3 {
            throw ASRError.processingFailed(Self.phraseBoostUnsupportedMessage(version))
        }
        var state = try TdtDecoderState(decoderLayers: decoderLayerCount)
        return try await transcribe(samples, decoderState: &state, language: language, phraseBoost: request)
    }

    static func phraseBoostUnsupportedMessage(_ version: AsrModelVersion) -> String {
        "Phrase boosting requires the Parakeet v3 model; the loaded model is \(version)"
    }
}

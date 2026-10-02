@preconcurrency import AVFoundation
@preconcurrency import CoreML
import XCTest

@testable import FluidAudio

/// Phrase boosting inside the real TDT decoder (#3338 PR-2, chunk 1).
///
/// Runs the shipped Parakeet v3 models on the real `SlidingWindow/Fixtures`
/// recordings (founder speech cleared for release, #855) and drives
/// `TdtDecoderV3.decodeWithTimings` directly. Needs the v3 models: runs when they
/// are cached or `FLUIDAUDIO_RUN_ASR_E2E=1` allows a download; otherwise skips.
@available(macOS 14.0, iOS 17.0, *)
final class TdtPhraseBoostDecoderTests: XCTestCase {

    private static let fixtures = [
        "01-validation-request-21.4s.wav", "02-release-readiness-19.8s.wav", "03-diff-explanation-16.9s.wav",
    ]

    private struct Window {
        let encoder: MLMultiArray
        let length: Int
        let actualFrames: Int
    }

    private final class Trace {
        var steps = 0
        var overrides: [(site: PhraseBoostTraceEvent.Site, frame: Int, picked: Int, after: Int)] = []
        var finished: PhraseBoostStats?
        var cancelAtStep: Int?

        func record(_ event: PhraseBoostTraceEvent) {
            switch event {
            case .step:
                steps += 1
                if steps == cancelAtStep { withUnsafeCurrentTask { $0?.cancel() } }
            case .override(let site, let frame, let picked, let after):
                overrides.append((site, frame, picked, after))
            case .finished(let stats):
                finished = stats
            }
        }
    }

    private func loadModels() async throws -> AsrModels {
        let cached = AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory())
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(cached || allowDownload, "Parakeet v3 models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1")
        return try await AsrModels.downloadAndLoad()
    }

    /// 16 kHz mono samples of a fixture, decoded from the file (real audio).
    private func samples(_ name: String) throws -> [Float] {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else { throw XCTSkip("fixture \(name) missing from test bundle") }
        return try AudioConverter().resampleAudioFile(url)
    }

    /// The quietest 1.5 s of a real recording: real room tone, not generated audio.
    private func quietestStretch(_ audio: [Float]) -> [Float] {
        let width = 24_000
        var best = (energy: Float.infinity, start: 0)
        var start = 0
        while start + width <= audio.count {
            var energy: Float = 0
            for i in stride(from: start, to: start + width, by: 16) { energy += abs(audio[i]) }
            if energy < best.energy { best = (energy, start) }
            start += 1_600
        }
        return Array(audio[best.start..<(best.start + width)])
    }

    /// Preprocessor + encoder for one window of at most 15 s, as the manager runs them.
    private func encode(_ audio: [Float], models: AsrModels) throws -> Window {
        let original = min(audio.count, ASRConstants.maxModelSamples)
        var padded = Array(audio.prefix(original))
        padded += Array(repeating: 0, count: ASRConstants.maxModelSamples - original)
        let signal = try MLMultiArray(shape: [1, NSNumber(value: padded.count)], dataType: .float32)
        padded.withUnsafeBufferPointer { buffer in
            let dest = signal.dataPointer.bindMemory(to: Float.self, capacity: padded.count)
            dest.update(from: buffer.baseAddress!, count: padded.count)
        }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: original)
        let input = try MLDictionaryFeatureProvider(dictionary: ["audio_signal": signal, "audio_length": length])
        let pre = try models.preprocessor.prediction(from: input)
        let encoded: MLFeatureProvider
        if let encoderModel = models.encoder {
            var features: [String: MLFeatureValue] = [:]
            for name in encoderModel.modelDescription.inputDescriptionsByName.keys {
                features[name] = pre.featureValue(for: name) ?? input.featureValue(for: name)
            }
            encoded = try encoderModel.prediction(from: MLDictionaryFeatureProvider(dictionary: features))
        } else {
            encoded = pre
        }
        let encoder = try XCTUnwrap(encoded.featureValue(for: "encoder")?.multiArrayValue)
        let frames = try XCTUnwrap(encoded.featureValue(for: "encoder_length")?.multiArrayValue)
        return Window(
            encoder: encoder, length: frames[0].intValue,
            actualFrames: ASRConstants.calculateEncoderFrames(from: original))
    }

    private func decode(
        _ window: Window, models: AsrModels, boost: PhraseBoostConfig? = nil, language: Language? = nil,
        emitAfter: Int? = nil, trace: Trace? = nil
    ) async throws -> TdtHypothesis {
        var state = TdtDecoderState.make(decoderLayers: 2)
        let decoder = TdtDecoderV3(config: .default)
        return try await decoder.decodeWithTimings(
            encoderOutput: window.encoder, encoderSequenceLength: window.length,
            actualAudioFrames: window.actualFrames, decoderModel: models.decoder, jointModel: models.joint,
            decoderState: &state, isLastChunk: true, language: language, vocabulary: models.vocabulary,
            emitTokensAfterGlobalFrame: emitAfter, phraseBoost: boost,
            phraseBoostTrace: trace.map { t in { t.record($0) } })
    }

    private func singles(_ ids: [Int], alpha: Float, startScale: Float = 1) -> PhraseBoostConfig {
        PhraseBoostConfig(
            alpha: alpha, phrases: ids.map { PhraseBoostPhrase(text: "t\($0)", tokenIds: [$0]) },
            allowPhraseStart: true, startScale: startScale)
    }

    func testAlphaZeroDecodesExactlyLikeNoBoost() async throws {
        let models = try await loadModels()
        var clips = try Self.fixtures.map { ("speech:" + $0, try samples($0)) }
        clips.append(("silence:" + Self.fixtures[0], quietestStretch(clips[0].1)))
        var compared = 0
        for (name, audio) in clips {
            let window = try encode(audio, models: models)
            let plain = try await decode(window, models: models)
            let zero = try await decode(
                window, models: models,
                boost: singles(Array(Set(plain.ySequence)).sorted() + [1, 2, 3], alpha: 0))
            XCTAssertEqual(plain.ySequence, zero.ySequence, name)
            XCTAssertEqual(plain.timestamps, zero.timestamps, name)
            XCTAssertEqual(plain.tokenDurations, zero.tokenDurations, name)
            XCTAssertEqual(plain.tokenConfidences, zero.tokenConfidences, name)
            compared += 1
        }
        print("[phrase-boost-receipt] alpha-zero clips compared=\(compared)")
        XCTAssertEqual(compared, 4)
    }

    func testBoostOverridesAtBothSelectionSites() async throws {
        let models = try await loadModels()
        var main = 0
        var inner = 0
        for name in Self.fixtures {
            let window = try encode(try samples(name), models: models)
            let trace = Trace()
            // Every 7th vocabulary piece: a broad list, so the reward reorders some
            // shortlists and leaves others, at both the main and the inner pick.
            let ids = Array(stride(from: 1, to: 8192, by: 7))
            _ = try await decode(window, models: models, boost: singles(ids, alpha: 12), trace: trace)
            main += trace.overrides.filter { $0.site == .main }.count
            inner += trace.overrides.filter { $0.site == .inner }.count
        }
        print("[phrase-boost-receipt] overrides main=\(main) inner=\(inner)")
        XCTAssertGreaterThan(main, 0, "no boosted override at the main pick")
        XCTAssertGreaterThan(inner, 0, "no boosted override at the inner pick")
    }

    func testScriptFilterStillRefusesAPromotedWrongScriptToken() async throws {
        let models = try await loadModels()
        // Pieces that are clearly Cyrillic: a Cyrillic letter and no Latin one.
        // Punctuation and digits match every script, so they are not wrong-script.
        let cyrillic = models.vocabulary.filter { piece in
            piece.value.unicodeScalars.contains { (0x0400...0x04FF).contains($0.value) }
                && !piece.value.unicodeScalars.contains { $0.properties.isAlphabetic && $0.isASCII }
        }.map(\.key)
        XCTAssertGreaterThan(cyrillic.count, 0)
        let trace = Trace()
        let window = try encode(try samples(Self.fixtures[0]), models: models)
        let out = try await decode(
            window, models: models, boost: singles(cyrillic, alpha: 30), language: .german, trace: trace)
        let refused = trace.overrides.filter { cyrillic.contains($0.picked) && $0.after != $0.picked }
        print("[phrase-boost-receipt] script-filter refusals=\(refused.count) of overrides=\(trace.overrides.count)")
        XCTAssertGreaterThan(refused.count, 0, "the boost never promoted a Cyrillic piece the filter then refused")
        XCTAssertTrue(Set(out.ySequence).isDisjoint(with: cyrillic), "a Cyrillic piece reached the transcript")
    }

    func testFrenchBlocklistStillRefusesAPromotedToken() async throws {
        let models = try await loadModels()
        let blocked = Array(TdtDecoderV3.englishBlocklistIds)
        let trace = Trace()
        let window = try encode(try samples(Self.fixtures[1]), models: models)
        let out = try await decode(
            window, models: models, boost: singles(blocked, alpha: 30), language: .french, trace: trace)
        let refused = trace.overrides.filter {
            TdtDecoderV3.englishBlocklistIds.contains($0.picked) && $0.after != $0.picked
        }
        print("[phrase-boost-receipt] blocklist refusals=\(refused.count) of overrides=\(trace.overrides.count)")
        XCTAssertGreaterThan(
            refused.count, 0, "the boost never promoted a blocklisted piece the blocklist then refused")
        XCTAssertTrue(Set(out.ySequence).isDisjoint(with: Set(blocked)), "a blocklisted piece reached the transcript")
    }

    /// The phrase's first piece is consumed before the emission boundary (so it
    /// is never emitted), and the reward then changes the decoder's own choice
    /// to the phrase's continuation, which survives both filters and is emitted
    /// at or after the boundary. The phrase completes exactly once.
    func testARewardedContinuationAfterASuppressedFirstPiece() async throws {
        let models = try await loadModels()
        let window = try encode(try samples(Self.fixtures[2]), models: models)
        let plain = try await decode(window, models: models)
        let y = plain.ySequence
        let t = plain.timestamps
        var found: (first: Int, continuation: Int, boundary: Int, out: TdtHypothesis, trace: Trace)?
        search: for i in 1..<(y.count - 3) where t[i] < t[i + 1] && y.filter({ $0 == y[i] }).count == 1 {
            // Continuation candidates: pieces the model emits a little later, so
            // they are plausible in the next shortlist but not its own choice.
            for k in (i + 2)..<min(i + 6, y.count) where y[k] != y[i + 1] && y[k] != y[i] {
                let boundary = t[i + 1]
                let trace = Trace()
                let config = PhraseBoostConfig(
                    alpha: 50, phrases: [PhraseBoostPhrase(text: "pair", tokenIds: [y[i], y[k]])],
                    allowPhraseStart: false)
                let out = try await decode(window, models: models, boost: config, emitAfter: boundary, trace: trace)
                if trace.overrides.contains(where: { $0.picked == y[k] && $0.after == y[k] && $0.frame >= boundary }) {
                    found = (y[i], y[k], boundary, out, trace)
                    break search
                }
            }
        }
        let f = try XCTUnwrap(found, "no position where the reward changed the decoder's choice to the continuation")
        let wins = f.trace.overrides.filter { $0.picked == f.continuation && $0.after == f.continuation }
        let emittedAt = zip(f.out.ySequence, f.out.timestamps).filter { $0.0 == f.continuation }.map(\.1)
        let stats = try XCTUnwrap(f.trace.finished)
        print(
            "[phrase-boost-receipt] rewarded continuation: boundary=\(f.boundary) override frames=\(wins.map(\.frame)) "
                + "emitted at=\(emittedAt) completions=\(stats.completions)")
        XCTAssertFalse(f.out.ySequence.contains(f.first), "the suppressed first piece was emitted")
        XCTAssertTrue(f.out.timestamps.allSatisfy { $0 >= f.boundary }, "a piece before the boundary was emitted")
        XCTAssertTrue(wins.contains { $0.frame >= f.boundary })
        XCTAssertTrue(emittedAt.contains { wins.map(\.frame).contains($0) }, "the boosted continuation was not emitted")
        XCTAssertEqual(stats.completions, 1)
        XCTAssertEqual(stats.completedPhrases, ["pair"])
    }

    /// Alpha zero on the founder's private name recordings, read from a path the
    /// runner supplies (`EW_NAME_CLIPS`: a file listing one WAV per line). The
    /// recordings are never copied into this repository.
    func testAlphaZeroDecodesExactlyLikeNoBoostOnSuppliedNameClips() async throws {
        guard let list = ProcessInfo.processInfo.environment["EW_NAME_CLIPS"] else {
            throw XCTSkip("EW_NAME_CLIPS not set")
        }
        let models = try await loadModels()
        let paths = try String(contentsOfFile: list, encoding: .utf8).split(separator: "\n").map(String.init)
        var compared = 0
        var truncated = 0
        for path in paths {
            let audio = try AudioConverter().resampleAudioFile(URL(fileURLWithPath: path))
            if audio.count > ASRConstants.maxModelSamples { truncated += 1 }
            let window = try encode(audio, models: models)
            let plain = try await decode(window, models: models)
            let zero = try await decode(
                window, models: models, boost: singles(Array(Set(plain.ySequence)).sorted() + [1, 2, 3], alpha: 0))
            XCTAssertEqual(plain.ySequence, zero.ySequence, path)
            XCTAssertEqual(plain.timestamps, zero.timestamps, path)
            XCTAssertEqual(plain.tokenDurations, zero.tokenDurations, path)
            XCTAssertEqual(plain.tokenConfidences, zero.tokenConfidences, path)
            compared += 1
        }
        print("[phrase-boost-receipt] name clips alpha-zero compared=\(compared) truncated-to-15s=\(truncated)")
        XCTAssertEqual(compared, paths.count)
        XCTAssertGreaterThan(compared, 0)
    }

    func testCancellationDuringABoostedPassThrows() async throws {
        let models = try await loadModels()
        let window = try encode(try samples(Self.fixtures[0]), models: models)
        let trace = Trace()
        trace.cancelAtStep = 5
        do {
            _ = try await decode(window, models: models, boost: singles([1, 2, 3], alpha: 2), trace: trace)
            XCTFail("decode finished after its task was cancelled")
        } catch is CancellationError {
            print("[phrase-boost-receipt] cancelled after steps=\(trace.steps)")
            XCTAssertGreaterThanOrEqual(trace.steps, 5)
            XCTAssertNotNil(trace.finished, "the runtime's final counters were not reported on the throw")
        }
    }
}

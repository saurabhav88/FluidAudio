@preconcurrency import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

/// Phrase boosting through `AsrManager` (#3338 PR-2, chunk 2): the public entry,
/// the single-window path and the concurrent chunked path, on the real
/// `SlidingWindow/Fixtures` recordings with the shipped v3 models. Skips when the
/// models are not cached and `FLUIDAUDIO_RUN_ASR_E2E` is not 1.
@available(macOS 14.0, iOS 17.0, *)
final class AsrManagerPhraseBoostTests: XCTestCase {

    private static let fixtures = [
        "01-validation-request-21.4s.wav", "02-release-readiness-19.8s.wav", "03-diff-explanation-16.9s.wav",
    ]

    /// A Sendable record of one trace event, with the time it arrived.
    private struct Seen: Sendable {
        enum Kind: Sendable { case step, override, finished }
        let trajectory: Int?
        let kind: Kind
        let picked: Int
        let after: Int
        let nanos: UInt64
    }

    private final class Recorder: Sendable {
        let stream: AsyncStream<Seen>
        let continuation: AsyncStream<Seen>.Continuation
        let cancelOnFirstStep: Bool

        init(cancelOnFirstStep: Bool = false) {
            (stream, continuation) = AsyncStream<Seen>.makeStream()
            self.cancelOnFirstStep = cancelOnFirstStep
        }

        var trace: @Sendable (Int?, PhraseBoostTraceEvent) -> Void {
            let continuation = self.continuation
            let cancel = self.cancelOnFirstStep
            return { trajectory, event in
                let now = DispatchTime.now().uptimeNanoseconds
                switch event {
                case .step:
                    continuation.yield(Seen(trajectory: trajectory, kind: .step, picked: 0, after: 0, nanos: now))
                    // Cancels the task running this decode, from inside an active pass.
                    if cancel { withUnsafeCurrentTask { $0?.cancel() } }
                case .override(_, _, let picked, let after):
                    continuation.yield(
                        Seen(trajectory: trajectory, kind: .override, picked: picked, after: after, nanos: now))
                case .finished:
                    continuation.yield(Seen(trajectory: trajectory, kind: .finished, picked: 0, after: 0, nanos: now))
                }
            }
        }

        func drain() async -> [Seen] {
            continuation.finish()
            var out: [Seen] = []
            for await seen in stream { out.append(seen) }
            return out
        }
    }

    private func manager(concurrency: Int = 4) async throws -> AsrManager {
        let cached = AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory())
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(cached || allowDownload, "Parakeet v3 models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1")
        let manager = AsrManager(config: ASRConfig(parallelChunkConcurrency: concurrency, melChunkContext: true))
        try await manager.loadModels(try await AsrModels.downloadAndLoad())
        return manager
    }

    private func samples(_ name: String) throws -> [Float] {
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else { throw XCTSkip("fixture \(name) missing from test bundle") }
        return try AudioConverter().resampleAudioFile(url)
    }

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

    private func plain(_ m: AsrManager, _ audio: [Float], language: Language? = nil) async throws -> ASRResult {
        var state = try TdtDecoderState(decoderLayers: await m.decoderLayerCount)
        return try await m.transcribe(audio, decoderState: &state, language: language)
    }

    private func assertSame(_ a: ASRResult, _ b: ASRResult, _ name: String) {
        XCTAssertEqual(Data(a.text.utf8), Data(b.text.utf8), name)
        let ta = (a.tokenTimings ?? []).map { "\($0.tokenId):\($0.startTime):\($0.endTime):\($0.confidence)" }
        let tb = (b.tokenTimings ?? []).map { "\($0.tokenId):\($0.startTime):\($0.endTime):\($0.confidence)" }
        XCTAssertEqual(ta, tb, name)
    }

    private func singles(_ ids: [Int], alpha: Float) -> PhraseBoostConfig {
        PhraseBoostConfig(alpha: alpha, phrases: ids.map { PhraseBoostPhrase(text: "t\($0)", tokenIds: [$0]) })
    }

    func testAlphaZeroMatchesOrdinaryTranscription() async throws {
        let m = try await manager()
        var clips: [(String, [Float])] = []
        for name in Self.fixtures {
            let audio = try samples(name)
            clips.append(("multi-chunk speech:" + name, audio))
            clips.append(("single-window speech:" + name, Array(audio.prefix(14 * 16_000))))
        }
        clips.append(("silence", quietestStretch(try samples(Self.fixtures[0]))))
        if let list = ProcessInfo.processInfo.environment["EW_NAME_CLIPS"] {
            for path in try String(contentsOfFile: list, encoding: .utf8).split(separator: "\n") {
                clips.append(
                    ("name:" + path, try AudioConverter().resampleAudioFile(URL(fileURLWithPath: String(path)))))
            }
        }
        var compared = 0
        for (name, audio) in clips {
            let ordinary = try await plain(m, audio)
            let zero = try await m.transcribeWithPhraseBoost(
                audio, boost: singles(Array(1...200), alpha: 0))
            assertSame(ordinary, zero, name)
            compared += 1
        }
        let names = clips.filter { $0.0.hasPrefix("name:") }.count
        print("[phrase-boost-receipt] public alpha-zero compared=\(compared) (names=\(names))")
        XCTAssertGreaterThan(compared, 6)
    }

    func testBoostChangesPicksOnSingleWindowAndChunkedPaths() async throws {
        let m = try await manager()
        let audio = try samples(Self.fixtures[0])
        let ids = Array(stride(from: 1, to: 8192, by: 7))
        for (label, clip) in [("single-window", Array(audio.prefix(14 * 16_000))), ("multi-chunk", audio)] {
            let recorder = Recorder()
            _ = try await m.transcribeWithPhraseBoost(
                clip, request: PhraseBoostRequest(config: singles(ids, alpha: 12), trace: recorder.trace))
            let seen = await recorder.drain()
            let overrides = seen.filter { $0.kind == .override }.count
            let trajectories = Set(seen.filter { $0.kind == .finished }.map { $0.trajectory ?? -1 }).count
            print("[phrase-boost-receipt] \(label) overrides=\(overrides) trajectories=\(trajectories)")
            XCTAssertGreaterThan(overrides, 0, label)
        }
    }

    func testLanguageReachesTheDecoderOnBothPaths() async throws {
        let m = try await manager()
        let vocabulary = await m.vocabulary
        let cyrillic = vocabulary.filter { piece in
            piece.value.unicodeScalars.contains { (0x0400...0x04FF).contains($0.value) }
                && !piece.value.unicodeScalars.contains { $0.properties.isAlphabetic && $0.isASCII }
        }.map(\.key)
        let audio = try samples(Self.fixtures[0])
        for (label, clip) in [("single-window", Array(audio.prefix(14 * 16_000))), ("multi-chunk", audio)] {
            let recorder = Recorder()
            let out = try await m.transcribeWithPhraseBoost(
                clip, request: PhraseBoostRequest(config: singles(cyrillic, alpha: 30), trace: recorder.trace),
                language: .german)
            let refused = await recorder.drain().filter {
                $0.kind == .override && cyrillic.contains($0.picked) && $0.after != $0.picked
            }
            print("[phrase-boost-receipt] \(label) german-lock refusals=\(refused.count)")
            XCTAssertGreaterThan(refused.count, 0, "\(label): the German lock did not reach the decoder")
            let emitted = Set((out.tokenTimings ?? []).map(\.tokenId))
            XCTAssertTrue(emitted.isDisjoint(with: cyrillic), label)
        }
    }

    func testChunkTrajectoriesRunConcurrentlyWithTheirOwnRuntimes() async throws {
        let m = try await manager(concurrency: 4)
        let audio = try samples(Self.fixtures[0]) + (try samples(Self.fixtures[1]))
        let recorder = Recorder()
        _ = try await m.transcribeWithPhraseBoost(
            audio, request: PhraseBoostRequest(config: singles([1, 2, 3], alpha: 2), trace: recorder.trace))
        let seen = await recorder.drain()
        var spans: [Int: (start: UInt64, end: UInt64)] = [:]
        var finishedCount: [Int: Int] = [:]
        for s in seen {
            guard let t = s.trajectory else { continue }
            let span = spans[t] ?? (s.nanos, s.nanos)
            spans[t] = (min(span.start, s.nanos), max(span.end, s.nanos))
            if s.kind == .finished { finishedCount[t, default: 0] += 1 }
        }
        let sorted = spans.sorted { $0.value.start < $1.value.start }
        var overlapping = 0
        for i in 0..<sorted.count {
            for j in (i + 1)..<sorted.count where sorted[j].value.start < sorted[i].value.end { overlapping += 1 }
        }
        print(
            "[phrase-boost-receipt] workers requested=\(await m.parallelChunkConcurrency) trajectories=\(spans.count) "
                + "overlapping pairs=\(overlapping) finished per trajectory=\(Set(finishedCount.values))")
        XCTAssertGreaterThanOrEqual(spans.count, 2)
        XCTAssertGreaterThan(overlapping, 0, "no two chunk trajectories overlapped in time")
        XCTAssertEqual(Set(finishedCount.values), [1], "a trajectory reported more or less than one runtime")
    }

    func testCancellationStopsBoostedSingleWindowAndChunkedPasses() async throws {
        let m = try await manager()
        let audio = try samples(Self.fixtures[0])
        for (label, clip) in [("single-window", Array(audio.prefix(14 * 16_000))), ("multi-chunk", audio)] {
            let recorder = Recorder(cancelOnFirstStep: true)
            let request = PhraseBoostRequest(config: singles([1, 2, 3], alpha: 2), trace: recorder.trace)
            // Each case runs in its own task: cancelling the single-window pass
            // cancels the task that runs it, which must not be the test's own.
            let outcome = await Task { () -> Error? in
                do {
                    _ = try await m.transcribeWithPhraseBoost(clip, request: request)
                    return nil
                } catch {
                    return error
                }
            }.value
            XCTAssertTrue(outcome is CancellationError, "\(label): expected CancellationError, got \(String(describing: outcome))")
            // Every child has drained once the call returns: each started trajectory reported its end.
            let seen = await recorder.drain()
            let started = Set(seen.compactMap { $0.kind == .step ? ($0.trajectory ?? -1) : nil }).count
            let finished = seen.filter { $0.kind == .finished }.count
            print("[phrase-boost-receipt] \(label) cancelled: started=\(started) finished=\(finished)")
            XCTAssertGreaterThan(started, 0, "\(label): no decoder pass was active when cancelled")
            XCTAssertEqual(started, finished, "\(label): a trajectory had not drained when the call threw")
        }
    }

    func testOtherModelVersionsRefuseBoostingExplicitly() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
                || AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory(for: .v2), version: .v2),
            "Parakeet v2 models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1")
        let m = AsrManager(config: .default)
        try await m.loadModels(try await AsrModels.downloadAndLoad(version: .v2))
        do {
            _ = try await m.transcribeWithPhraseBoost(
                Array(try samples(Self.fixtures[0]).prefix(14 * 16_000)), boost: singles([1], alpha: 2))
            XCTFail("v2 accepted a phrase boost")
        } catch ASRError.processingFailed(let message) {
            print("[phrase-boost-receipt] v2 refused: \(message)")
            XCTAssertTrue(message.contains("v3"))
        }
    }

    func testDualDecodeArbitrationRefusesABoostAndKeepsTheOrdinaryPath() async throws {
        let cached = AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory())
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(cached || allowDownload, "Parakeet v3 models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1")
        let m = AsrManager(config: ASRConfig(melChunkContext: false, dualDecodeArbitration: true))
        try await m.loadModels(try await AsrModels.downloadAndLoad())
        let dual = await m.dualDecodeArbitration
        let mel = await m.melChunkContext
        let version = await m.modelVersion
        XCTAssertTrue(dual)
        XCTAssertFalse(mel)
        XCTAssertEqual(version, .v3)
        let audio = try samples(Self.fixtures[0])
        XCTAssertGreaterThan(audio.count, ASRConstants.maxModelSamples)

        let recorder = Recorder()
        do {
            _ = try await m.transcribeWithPhraseBoost(
                audio, request: PhraseBoostRequest(config: singles([1, 2, 3], alpha: 2), trace: recorder.trace))
            XCTFail("a boosted dual-decode request was accepted")
        } catch ASRError.processingFailed(let message) {
            XCTAssertEqual(message, "Phrase boosting is not supported with dual-decode arbitration")
        }
        let steps = await recorder.drain().filter { $0.kind == .step }.count
        XCTAssertEqual(steps, 0, "the decoder ran before the refusal")

        let ordinary = try await plain(m, audio)
        print("[phrase-boost-receipt] dual-decode refused with steps=\(steps); ordinary chars=\(ordinary.text.count)")
        XCTAssertFalse(ordinary.text.isEmpty)
    }
}

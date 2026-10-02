@preconcurrency import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

/// Prepared windows (#3338 PR-2, chunk 3): the encoder runs once per window and
/// every decode, plain or phrase-boosted, reuses its output. Real
/// `SlidingWindow/Fixtures` recordings and the shipped v3 models; skips when the
/// models are not cached and `FLUIDAUDIO_RUN_ASR_E2E` is not 1.
@available(macOS 14.0, iOS 17.0, *)
final class PreparedWindowTests: XCTestCase {

    private static let fixtures = [
        "01-validation-request-21.4s.wav", "02-release-readiness-19.8s.wav", "03-diff-explanation-16.9s.wav",
    ]
    private static let window = 14 * 16_000

    private struct Override: Sendable {
        let picked: Int
        let after: Int
    }

    private final class Overrides: Sendable {
        let stream: AsyncStream<Override>
        let continuation: AsyncStream<Override>.Continuation
        let cancelOnFirstStep: Bool

        init(cancelOnFirstStep: Bool = false) {
            (stream, continuation) = AsyncStream<Override>.makeStream()
            self.cancelOnFirstStep = cancelOnFirstStep
        }

        var trace: @Sendable (Int?, PhraseBoostTraceEvent) -> Void {
            let continuation = self.continuation
            let cancel = self.cancelOnFirstStep
            return { _, event in
                switch event {
                case .step:
                    if cancel { withUnsafeCurrentTask { $0?.cancel() } }
                case .override(_, _, let picked, let after):
                    continuation.yield(Override(picked: picked, after: after))
                case .finished:
                    break
                }
            }
        }

        func drain() async -> [Override] {
            continuation.finish()
            var out: [Override] = []
            for await o in stream { out.append(o) }
            return out
        }
    }

    private func manager(version: AsrModelVersion = .v3) async throws -> AsrManager {
        let cached = AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory(for: version), version: version)
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            cached || allowDownload, "Parakeet \(version) models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1")
        let m = AsrManager(config: ASRConfig(melChunkContext: true))
        try await m.loadModels(try await AsrModels.downloadAndLoad(version: version))
        return m
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

    private func key(_ timings: [TokenTiming]) -> [String] {
        timings.map { "\($0.tokenId):\($0.token):\($0.startTime):\($0.endTime):\($0.confidence)" }
    }

    private func boost(_ ids: [Int], alpha: Float, startScale: Float = 0.5) -> PhraseBoostConfig {
        PhraseBoostConfig(
            alpha: alpha, phrases: ids.map { PhraseBoostPhrase(text: "t\($0)", tokenIds: [$0]) },
            startScale: startScale)
    }

    func testOneEncoderRunServesAPlainAndFourBoostedPasses() async throws {
        let m = try await manager()
        let audio = Array(try samples(Self.fixtures[0]).prefix(Self.window))
        let before = await m.frontEndCounters
        let window = try await m.prepareWindow(samples: audio)
        let prepared = await m.frontEndCounters
        XCTAssertEqual(prepared.preprocessor - before.preprocessor, 1)
        XCTAssertEqual(prepared.encoder - before.encoder, 1)

        let ids = Array(stride(from: 1, to: 8192, by: 7))
        _ = try await m.decode(window)
        var overrides = 0
        for alpha: Float in [2, 4, 6, 10] {
            let recorder = Overrides()
            _ = try await m.decode(
                window, request: PhraseBoostRequest(config: boost(ids, alpha: alpha), trace: recorder.trace))
            overrides += await recorder.drain().count
        }
        let after = await m.frontEndCounters
        print(
            "[phrase-boost-receipt] prepare preprocessor=\(prepared.preprocessor - before.preprocessor) "
                + "encoder=\(prepared.encoder - before.encoder); during 5 decodes preprocessor="
                + "\(after.preprocessor - prepared.preprocessor) encoder=\(after.encoder - prepared.encoder); "
                + "overrides=\(overrides)")
        XCTAssertEqual(after, prepared, "a decode ran the preprocessor or encoder")
        XCTAssertGreaterThan(overrides, 0)
    }

    func testPlainPreparedDecodeMatchesOrdinaryTranscription() async throws {
        let m = try await manager()
        var clips: [(String, [Float])] = []
        for name in Self.fixtures { clips.append(("speech:" + name, Array(try samples(name).prefix(Self.window)))) }
        clips.append(("silence", quietestStretch(try samples(Self.fixtures[0]))))
        if let list = ProcessInfo.processInfo.environment["EW_NAME_CLIPS"] {
            for path in try String(contentsOfFile: list, encoding: .utf8).split(separator: "\n") {
                let audio = try AudioConverter().resampleAudioFile(URL(fileURLWithPath: String(path)))
                clips.append(("name:" + path, Array(audio.prefix(ASRConstants.maxModelSamples))))
            }
        }
        var compared = 0
        for (name, audio) in clips {
            var state = try TdtDecoderState(decoderLayers: await m.decoderLayerCount)
            let ordinary = try await m.transcribe(audio, decoderState: &state)
            let prepared = try await m.decode(try await m.prepareWindow(samples: audio))
            XCTAssertEqual(key(ordinary.tokenTimings ?? []), key(prepared), name)
            compared += 1
        }
        let names = clips.filter { $0.0.hasPrefix("name:") }.count
        print("[phrase-boost-receipt] prepared vs ordinary compared=\(compared) (names=\(names))")
        XCTAssertGreaterThan(compared, 3)
    }

    func testRepeatedAndReorderedPassesDoNotContaminateEachOther() async throws {
        let m = try await manager()
        let window = try await m.prepareWindow(samples: Array(try samples(Self.fixtures[1]).prefix(Self.window)))
        let strong = boost(Array(stride(from: 1, to: 8192, by: 7)), alpha: 10)
        let plain1 = key(try await m.decode(window))
        let boosted1 = key(try await m.decode(window, phraseBoost: strong))
        let zero = key(try await m.decode(window, phraseBoost: boost([1, 2, 3], alpha: 0)))
        let plain2 = key(try await m.decode(window))
        let boosted2 = key(try await m.decode(window, phraseBoost: strong))
        print("[phrase-boost-receipt] isolation: boosted differs from plain=\(boosted1 != plain1)")
        XCTAssertEqual(plain1, plain2)
        XCTAssertEqual(plain1, zero)
        XCTAssertEqual(boosted1, boosted2)
    }

    func testTheWindowsLanguageReachesEveryDecode() async throws {
        let m = try await manager()
        let vocabulary = await m.vocabulary
        let cyrillic = vocabulary.filter { piece in
            piece.value.unicodeScalars.contains { (0x0400...0x04FF).contains($0.value) }
                && !piece.value.unicodeScalars.contains { $0.properties.isAlphabetic && $0.isASCII }
        }.map(\.key)
        let window = try await m.prepareWindow(
            samples: Array(try samples(Self.fixtures[0]).prefix(Self.window)), decodeLanguage: .german)
        XCTAssertEqual(window.decodeLanguage, .german)
        let recorder = Overrides()
        let out = try await m.decode(
            window,
            request: PhraseBoostRequest(config: boost(cyrillic, alpha: 30, startScale: 1), trace: recorder.trace))
        let refused = await recorder.drain().filter { cyrillic.contains($0.picked) && $0.after != $0.picked }
        print("[phrase-boost-receipt] prepared german-lock refusals=\(refused.count)")
        XCTAssertGreaterThan(refused.count, 0)
        XCTAssertTrue(Set(out.map(\.tokenId)).isDisjoint(with: cyrillic))
    }

    func testAWindowIsRefusedByAnotherModelSet() async throws {
        let a = try await manager()
        let b = try await manager()
        let window = try await a.prepareWindow(samples: Array(try samples(Self.fixtures[2]).prefix(Self.window)))
        do {
            _ = try await b.decode(window)
            XCTFail("another manager decoded the window")
        } catch ASRError.processingFailed(let message) {
            XCTAssertEqual(message, "This prepared window was made with a different model set")
        }
        try await a.loadModels(try await AsrModels.downloadAndLoad())
        do {
            _ = try await a.decode(window)
            XCTFail("a reload kept the old window valid")
        } catch ASRError.processingFailed {}
        print("[phrase-boost-receipt] incompatible-model refusals=2")
    }

    func testBoundaryInputs() async throws {
        let m = try await manager()
        for (label, input) in [("empty", [Float]()), ("short", [Float](repeating: 0, count: 100))] {
            do {
                _ = try await m.prepareWindow(samples: input)
                XCTFail("\(label) input was prepared")
            } catch ASRError.invalidAudioData {}
        }
        let oversized = Array(try samples(Self.fixtures[0]).prefix(ASRConstants.maxModelSamples + 1))
        XCTAssertEqual(oversized.count, ASRConstants.maxModelSamples + 1)
        do {
            _ = try await m.prepareWindow(samples: oversized)
            XCTFail("oversized input was prepared")
        } catch ASRError.processingFailed(let message) {
            XCTAssertTrue(message.contains("at most"))
        }
        print("[phrase-boost-receipt] boundaries: empty, short, oversized refused")
    }

    func testCancellationDuringPreparationAndDuringDecoding() async throws {
        let m = try await manager()
        let audio = Array(try samples(Self.fixtures[0]).prefix(Self.window))

        let preparing = Task { try await m.prepareWindow(samples: audio) }
        preparing.cancel()
        do {
            _ = try await preparing.value
            XCTFail("preparation finished after cancellation")
        } catch is CancellationError {}

        let window = try await m.prepareWindow(samples: audio)
        let plainBefore = key(try await m.decode(window))
        let recorder = Overrides(cancelOnFirstStep: true)
        let request = PhraseBoostRequest(config: boost([1, 2, 3], alpha: 2), trace: recorder.trace)
        let outcome = await Task { () -> Error? in
            do {
                _ = try await m.decode(window, request: request)
                return nil
            } catch {
                return error
            }
        }.value
        _ = await recorder.drain()
        XCTAssertTrue(outcome is CancellationError, "decode: \(String(describing: outcome))")
        let plainAfter = key(try await m.decode(window))
        print("[phrase-boost-receipt] cancelled prepare and decode; window reusable=\(plainAfter == plainBefore)")
        XCTAssertEqual(plainAfter, plainBefore, "the window changed after a cancelled decode")
    }

    func testOtherModelVersionsCannotPrepareAWindow() async throws {
        let m = try await manager(version: .v2)
        do {
            _ = try await m.prepareWindow(samples: Array(try samples(Self.fixtures[0]).prefix(Self.window)))
            XCTFail("v2 prepared a window")
        } catch ASRError.processingFailed(let message) {
            XCTAssertTrue(message.contains("v3"))
        }
        print("[phrase-boost-receipt] v2 prepareWindow refused")
    }

    func testCancellationAfterTheEncoderReturnsNoWindow() async throws {
        let m = try await manager()
        let audio = Array(try samples(Self.fixtures[0]).prefix(Self.window))
        // Cancels the preparing task right after the encoder call returns.
        await m.setFrontEndObserver { stage in
            if stage == .afterEncoder { withUnsafeCurrentTask { $0?.cancel() } }
        }
        let before = await m.frontEndCounters
        let outcome = await Task { () -> Result<PreparedWindow, Error> in
            do { return .success(try await m.prepareWindow(samples: audio)) } catch { return .failure(error) }
        }.value
        await m.setFrontEndObserver(nil)
        let after = await m.frontEndCounters
        guard case .failure(let error) = outcome else { return XCTFail("a window was returned after cancellation") }
        XCTAssertTrue(error is CancellationError, "\(error)")
        print(
            "[phrase-boost-receipt] cancelled after encoder: preprocessor=\(after.preprocessor - before.preprocessor) "
                + "encoder=\(after.encoder - before.encoder)")
        XCTAssertEqual(after.preprocessor - before.preprocessor, 1)
        XCTAssertEqual(after.encoder - before.encoder, 1)
    }

    func testAReloadDuringPreparationIsRefused() async throws {
        let m = try await manager()
        let models = try await AsrModels.downloadAndLoad()
        let audio = Array(try samples(Self.fixtures[1]).prefix(Self.window))
        let idBefore = await m.loadedModelsID
        await m.setFrontEndObserver { stage in
            if stage == .afterEncoder { try? await m.loadModels(models) }
        }
        do {
            _ = try await m.prepareWindow(samples: audio)
            XCTFail("a window was prepared across a reload")
        } catch ASRError.processingFailed(let message) {
            XCTAssertEqual(message, "The model set changed while preparing this window")
        }
        await m.setFrontEndObserver(nil)
        let idAfter = await m.loadedModelsID
        print("[phrase-boost-receipt] reload during preparation refused; identity changed=\(idBefore != idAfter)")
        XCTAssertNotEqual(idBefore, idAfter)
    }

    func testDecodeBuffersAreFreedOnSuccessCancellationAndConstructionFailure() async throws {
        let m = try await manager()
        let window = try await m.prepareWindow(samples: Array(try samples(Self.fixtures[2]).prefix(Self.window)))
        let plainBefore = key(try await m.decode(window))

        func counted(_ operation: () async throws -> Void) async -> (allocated: Int, freed: Int) {
            let (stream, continuation) = AsyncStream<AsrManager.PreparedBufferEvent>.makeStream()
            await m.setPreparedBufferObserver(continuation)
            try? await operation()
            await m.setPreparedBufferObserver(nil)
            // The buffer is freed when Core ML releases its array, which can be
            // after the call returns (autorelease). Wait for the matching free,
            // bounded by a deadline, then stop listening.
            let deadline = Task {
                try? await Task.sleep(for: .seconds(3))
                continuation.finish()
            }
            var allocated = 0
            var freed = 0
            for await event in stream {
                switch event {
                case .allocated: allocated += 1
                case .freed: freed += 1
                }
                if allocated > 0 && freed == allocated { break }
            }
            deadline.cancel()
            continuation.finish()
            return (allocated, freed)
        }

        let success = await counted { _ = try await m.decode(window) }
        let recorder = Overrides(cancelOnFirstStep: true)
        let request = PhraseBoostRequest(config: boost([1, 2, 3], alpha: 2), trace: recorder.trace)
        let cancelled = await counted {
            _ = await Task { try? await m.decode(window, request: request) }.value
        }
        _ = await recorder.drain()
        let (stream, continuation) = AsyncStream<AsrManager.PreparedBufferEvent>.makeStream()
        struct Refused: Error {}
        XCTAssertThrowsError(
            try AsrManager.encoderArray(from: window, observer: continuation) { _, _, _, _, _ in throw Refused() })
        continuation.finish()
        var failed = (allocated: 0, freed: 0)
        for await event in stream {
            if case .allocated = event { failed.allocated += 1 } else { failed.freed += 1 }
        }
        print(
            "[phrase-boost-receipt] decode buffers success=\(success) cancelled=\(cancelled) "
                + "construction-failure=\(failed)")
        XCTAssertEqual(success.allocated, 1)
        XCTAssertEqual(success.freed, 1)
        XCTAssertEqual(cancelled.allocated, 1)
        XCTAssertEqual(cancelled.freed, 1)
        XCTAssertEqual(failed.allocated, 1)
        XCTAssertEqual(failed.freed, 1)
        let plainAfter = key(try await m.decode(window))
        XCTAssertEqual(plainAfter, plainBefore, "the window changed")
    }
}

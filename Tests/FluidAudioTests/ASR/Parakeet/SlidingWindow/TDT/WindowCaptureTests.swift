@preconcurrency import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

/// Batch transcription that hands over its own encoder windows (#3338 PR-2,
/// chunk 4, plan C.2b). Real `SlidingWindow/Fixtures` recordings and the shipped
/// v3 models; skips when the models are not cached and
/// `FLUIDAUDIO_RUN_ASR_E2E` is not 1.
@available(macOS 14.0, iOS 17.0, *)
final class WindowCaptureTests: XCTestCase {

    private static let fixtures = [
        "01-validation-request-21.4s.wav", "02-release-readiness-19.8s.wav", "03-diff-explanation-16.9s.wav",
    ]
    private static let window = 14 * 16_000

    private func manager(
        version: AsrModelVersion = .v3, config: ASRConfig = ASRConfig(melChunkContext: true)
    )
        async throws -> AsrManager
    {
        let cached = AsrModels.modelsExist(at: AsrModels.defaultCacheDirectory(for: version), version: version)
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            cached || allowDownload, "Parakeet \(version) models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1")
        let m = AsrManager(config: config)
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

    private func key(_ timings: [TokenTiming]?) -> [String] {
        (timings ?? []).map { "\($0.tokenId):\($0.token):\($0.startTime):\($0.endTime):\($0.confidence)" }
    }

    private func ordinary(_ m: AsrManager, _ audio: [Float], language: Language? = nil) async throws -> ASRResult {
        var state = try TdtDecoderState(decoderLayers: await m.decoderLayerCount)
        return try await m.transcribe(audio, decoderState: &state, language: language)
    }

    private func clips() throws -> [(String, [Float])] {
        var out: [(String, [Float])] = []
        for name in Self.fixtures {
            let audio = try samples(name)
            out.append(("multi-chunk:" + name, audio))
            out.append(("single-window:" + name, Array(audio.prefix(Self.window))))
        }
        if let list = ProcessInfo.processInfo.environment["EW_NAME_CLIPS"] {
            for path in try String(contentsOfFile: list, encoding: .utf8).split(separator: "\n") {
                out.append(("name:" + path, try AudioConverter().resampleAudioFile(URL(fileURLWithPath: String(path)))))
            }
        }
        return out
    }

    func testCaptureLeavesTheTranscriptUnchangedAndAddsNoInference() async throws {
        let m = try await manager()
        var compared = 0
        var singles = 0
        var chunked = 0
        var bytes: [Int] = []
        for (name, audio) in try clips() {
            let c0 = await m.frontEndCounters
            let plain = try await ordinary(m, audio)
            let c1 = await m.frontEndCounters
            let captured = try await m.transcribeCapturingWindows(audio)
            let c2 = await m.frontEndCounters
            XCTAssertEqual(Data(plain.text.utf8), Data(captured.result.text.utf8), name)
            XCTAssertEqual(key(plain.tokenTimings), key(captured.result.tokenTimings), name)
            XCTAssertEqual(c2.encoder - c1.encoder, c1.encoder - c0.encoder, "\(name): capture changed encoder runs")
            XCTAssertEqual(
                c2.preprocessor - c1.preprocessor, c1.preprocessor - c0.preprocessor,
                "\(name): capture changed preprocessor runs")
            XCTAssertNil(captured.ineligibleReason, name)
            // The single-window path runs on this manager, so its counters see
            // every front-end call; chunks also run on worker clones whose
            // counters this test cannot read, so there the equality of the two
            // runs' counter deltas above and the per-window re-decode test carry
            // the "no extra inference" evidence.
            if audio.count <= ASRConstants.maxModelSamples {
                XCTAssertEqual(captured.windows.count, 1, name)
                XCTAssertEqual(c1.encoder - c0.encoder, 1, name)
            } else {
                XCTAssertGreaterThan(captured.windows.count, 1, name)
            }
            if captured.windows.count == 1 { singles += 1 } else { chunked += 1 }
            bytes += captured.windows.map(\.encoderBytes.count)
            compared += 1
        }
        print(
            "[phrase-boost-receipt] capture parity compared=\(compared) single-window=\(singles) multi-chunk=\(chunked) "
                + "window bytes min=\(bytes.min() ?? 0) max=\(bytes.max() ?? 0)")
        XCTAssertGreaterThan(singles, 0)
        XCTAssertGreaterThan(chunked, 0)
    }

    func testEachCapturedWindowDecodesToItsOwnBatchTokens() async throws {
        let m = try await manager()
        var windowsChecked = 0
        var chunkWindows = 0
        for name in Self.fixtures {
            for audio in [try samples(name), Array(try samples(name).prefix(Self.window))] {
                let run = try await m.transcribeCapturingWindowsWithTokens(audio)
                XCTAssertNil(run.reason)
                for (window, tokens) in run.windows {
                    let decoded = try await m.decode(window).map(\.tokenId)
                    XCTAssertEqual(decoded, tokens, "\(name) \(window.origin)")
                    windowsChecked += 1
                    if case .batchChunk = window.origin { chunkWindows += 1 }
                }
            }
        }
        print(
            "[phrase-boost-receipt] captured windows re-decoded=\(windowsChecked) (chunk windows from workers=\(chunkWindows))"
        )
        XCTAssertGreaterThan(chunkWindows, 1)
    }

    func testACapturedWindowServesAPlainAndFourBoostedPasses() async throws {
        let m = try await manager()
        let captured = try await m.transcribeCapturingWindows(
            Array(try samples(Self.fixtures[0]).prefix(Self.window)))
        let window = try XCTUnwrap(captured.windows.first)
        XCTAssertEqual(window.origin, .batchSingleWindow)
        let before = await m.frontEndCounters
        _ = try await m.decode(window)
        let ids = Array(stride(from: 1, to: 8192, by: 7))
        var overrides = 0
        for alpha: Float in [2, 4, 6, 10] {
            let (stream, continuation) = AsyncStream<Int>.makeStream()
            let trace: @Sendable (Int?, PhraseBoostTraceEvent) -> Void = { _, event in
                if case .override = event { continuation.yield(1) }
            }
            _ = try await m.decode(
                window,
                request: PhraseBoostRequest(
                    config: PhraseBoostConfig(
                        alpha: alpha, phrases: ids.map { PhraseBoostPhrase(text: "t\($0)", tokenIds: [$0]) },
                        startScale: 0.5),
                    trace: trace))
            continuation.finish()
            for await _ in stream { overrides += 1 }
        }
        let after = await m.frontEndCounters
        print(
            "[phrase-boost-receipt] captured window reuse: front-end calls during 5 decodes=\(after.encoder - before.encoder + after.preprocessor - before.preprocessor) overrides=\(overrides)"
        )
        XCTAssertEqual(after, before)
        XCTAssertGreaterThan(overrides, 0)
    }

    func testCapturedWindowsKeepTheLanguageAndTheModelIdentity() async throws {
        let m = try await manager()
        let captured = try await m.transcribeCapturingWindows(
            try samples(Self.fixtures[1]), language: .german)
        XCTAssertGreaterThan(captured.windows.count, 1)
        XCTAssertTrue(captured.windows.allSatisfy { $0.decodeLanguage == .german })
        try await m.loadModels(try await AsrModels.downloadAndLoad())
        do {
            _ = try await m.decode(captured.windows[0])
            XCTFail("a reload kept a captured window valid")
        } catch ASRError.processingFailed(let message) {
            XCTAssertEqual(message, "This prepared window was made with a different model set")
        }
        print("[phrase-boost-receipt] captured windows: language german on \(captured.windows.count); reload refused")
    }

    func testIneligiblePathsReturnTheOrdinaryResultAndNoWindows() async throws {
        let dual = try await manager(config: ASRConfig(melChunkContext: false, dualDecodeArbitration: true))
        let audio = try samples(Self.fixtures[0])
        let plain = try await ordinary(dual, audio)
        let captured = try await dual.transcribeCapturingWindows(audio)
        XCTAssertEqual(captured.ineligibleReason, "dual-decode arbitration")
        XCTAssertTrue(captured.windows.isEmpty)
        XCTAssertEqual(Data(plain.text.utf8), Data(captured.result.text.utf8))

        let v2 = try await manager(version: .v2)
        let v2Captured = try await v2.transcribeCapturingWindows(
            Array(audio.prefix(Self.window)))
        XCTAssertEqual(v2Captured.ineligibleReason, "phrase boosting requires the Parakeet v3 model")
        XCTAssertTrue(v2Captured.windows.isEmpty)
        print("[phrase-boost-receipt] ineligible: dual-decode and v2 return no windows with a reason")
    }

    func testCancellationDuringCaptureThrowsAndReturnsNoWindows() async throws {
        let m = try await manager()
        let audio = try samples(Self.fixtures[0])
        await m.setFrontEndObserver { stage in
            if stage == .afterEncoder { withUnsafeCurrentTask { $0?.cancel() } }
        }
        let outcome = await Task { () -> Error? in
            do {
                _ = try await m.transcribeCapturingWindows(audio)
                return nil
            } catch {
                return error
            }
        }.value
        await m.setFrontEndObserver(nil)
        XCTAssertTrue(outcome is CancellationError, "\(String(describing: outcome))")
        print("[phrase-boost-receipt] cancellation during capture: \(String(describing: outcome))")
    }

    private actor Once {
        private var used = false
        func claim() -> Bool {
            defer { used = true }
            return !used
        }
    }

    /// Front-end model calls counted on the parent AND its worker clones: the
    /// test observer is copied into every clone (#3338).
    private func countedFrontEnd(_ m: AsrManager, _ body: () async throws -> Void) async throws -> (pre: Int, enc: Int) {
        let (stream, continuation) = AsyncStream<AsrManager.FrontEndStage>.makeStream()
        await m.setFrontEndObserver { stage in continuation.yield(stage) }
        try await body()
        await m.setFrontEndObserver(nil)
        continuation.finish()
        var pre = 0
        var enc = 0
        for await stage in stream {
            switch stage {
            case .afterPreprocessor: pre += 1
            case .afterEncoder: enc += 1
            }
        }
        return (pre, enc)
    }

    func testCaptureAddsNoInferenceOnParentOrWorkers() async throws {
        let m = try await manager()
        var rows: [String] = []
        for name in Self.fixtures {
            let audio = try samples(name)
            let off = try await countedFrontEnd(m) { _ = try await ordinary(m, audio) }
            var windows: [PreparedWindow] = []
            let on = try await countedFrontEnd(m) {
                windows = try await m.transcribeCapturingWindows(audio).windows
            }
            let replay = try await countedFrontEnd(m) {
                for w in windows { _ = try await m.decode(w) }
            }
            rows.append("\(name) off=\(off) on=\(on) windows=\(windows.count) replay=\(replay)")
            XCTAssertEqual(off.pre, on.pre, name)
            XCTAssertEqual(off.enc, on.enc, name)
            XCTAssertEqual(on.enc, windows.count, "\(name): one window per encoder call, parent and workers")
            XCTAssertGreaterThan(windows.count, 1, name)
            XCTAssertEqual(replay.pre + replay.enc, 0, name)
        }
        print("[phrase-boost-receipt] parent+worker front-end counts: " + rows.joined(separator: "; "))
    }

    func testAReloadDuringCaptureReturnsTheResultAndNoWindows() async throws {
        let m = try await manager()
        let models = try await AsrModels.downloadAndLoad()
        var outcomes: [String] = []
        for (label, audio) in [
            ("single-window", Array(try samples(Self.fixtures[0]).prefix(Self.window))),
            ("multi-chunk", try samples(Self.fixtures[0])),
        ] {
            let once = Once()
            // Reload the parent's models once, from whichever manager (parent or
            // worker) finishes an encoder call first.
            await m.setFrontEndObserver { stage in
                if stage == .afterEncoder, await once.claim() { try? await m.loadModels(models) }
            }
            let captured = try await m.transcribeCapturingWindows(audio)
            await m.setFrontEndObserver(nil)
            XCTAssertFalse(captured.result.text.isEmpty, label)
            XCTAssertTrue(captured.windows.isEmpty, label)
            XCTAssertEqual(captured.ineligibleReason, "model set changed during capture", label)
            outcomes.append("\(label): \(captured.ineligibleReason ?? "nil")")
        }
        print("[phrase-boost-receipt] reload during capture: " + outcomes.joined(separator: "; "))
    }

    func testATokenChangingSeamRepairMakesTheTakeIneligible() async throws {
        // A 0.4 s minimum gap makes seam repair change tokens on fixture 03
        // (found by sweeping 1.5/0.8/0.4/0.2/0.1 s over the fixtures).
        let m = try await manager(config: ASRConfig(melChunkContext: true, seamGapRepairMinGapSeconds: 0.4))
        let audio = try samples(Self.fixtures[2])
        let plain = try await ordinary(m, audio)
        let captured = try await m.transcribeCapturingWindows(audio)
        XCTAssertEqual(captured.ineligibleReason, "seam repair changed the merged tokens")
        XCTAssertTrue(captured.windows.isEmpty)
        XCTAssertEqual(Data(plain.text.utf8), Data(captured.result.text.utf8))
        XCTAssertEqual(key(plain.tokenTimings), key(captured.result.tokenTimings))
        print("[phrase-boost-receipt] seam repair: \(captured.ineligibleReason ?? "nil"), windows=\(captured.windows.count)")
    }
}

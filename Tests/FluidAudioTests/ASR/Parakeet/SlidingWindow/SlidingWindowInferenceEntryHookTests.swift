@preconcurrency import AVFoundation
import Foundation
import XCTest

@testable import FluidAudio

/// EW #3546: the window inference-entry observation fires once per window model call, never
/// when no window is processed, and does not change what the stream transcribes.
final class SlidingWindowInferenceEntryHookTests: XCTestCase {

    private final class Counter: @unchecked Sendable {
        private let lock = NSLock()
        private var value = 0
        func increment() {
            lock.lock()
            value += 1
            lock.unlock()
        }
        var count: Int {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    private func loadModels() async throws -> AsrModels {
        let cacheDir = AsrModels.defaultCacheDirectory()
        let cached = AsrModels.modelsExist(at: cacheDir)
        let allowDownload = ProcessInfo.processInfo.environment["FLUIDAUDIO_RUN_ASR_E2E"] == "1"
        try XCTSkipUnless(
            cached || allowDownload,
            "Parakeet v3 models not cached; set FLUIDAUDIO_RUN_ASR_E2E=1 to download")
        return try await AsrModels.downloadAndLoad()
    }

    private func fixtureSamples() throws -> [Float] {
        let name = "01-validation-request-21.4s.wav"
        guard
            let url = Bundle.module.url(forResource: "Fixtures/\(name)", withExtension: nil)
                ?? Bundle.module.url(forResource: name, withExtension: nil)
        else { throw XCTSkip("fixture \(name) missing from test bundle") }
        let file = try AVAudioFile(forReading: url)
        guard
            let buffer = AVAudioPCMBuffer(
                pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
            (try? file.read(into: buffer)) != nil,
            let channel = buffer.floatChannelData?[0]
        else { throw XCTSkip("could not read fixture") }
        return Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
    }

    private nonisolated static func makeChunk(_ samples: ArraySlice<Float>) -> AVAudioPCMBuffer? {
        guard
            let format = AVAudioFormat(
                commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
            let channel = buffer.floatChannelData?[0]
        else { return nil }
        for (offset, sample) in samples.enumerated() { channel[offset] = sample }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        return buffer
    }

    /// Streams the samples in 1 s buffers; returns the transcript and the number of
    /// transcription updates the manager published (one per successful window).
    private func stream(
        _ samples: [Float], models: AsrModels, hook: (@Sendable () -> Void)?
    ) async throws -> (text: String, updates: Int) {
        let manager = SlidingWindowAsrManager(onWindowInferenceEntered: hook)
        try await manager.loadModels(models)
        let updates = await manager.transcriptionUpdates
        let collector = Task { () -> Int in
            var n = 0
            for await _ in updates { n += 1 }
            return n
        }
        try await manager.startStreaming()
        var position = 0
        while position < samples.count {
            let end = min(position + 16_000, samples.count)
            guard let buffer = Self.makeChunk(samples[position..<end]) else {
                XCTFail("could not allocate chunk buffer")
                break
            }
            await manager.streamAudio(buffer)
            position = end
        }
        let text = try await manager.finish()
        await manager.cancel()
        return (text, await collector.value)
    }

    func testHookFiresOncePerWindowAndLeavesTheTranscriptUnchanged() async throws {
        let models = try await loadModels()
        let samples = try fixtureSamples()
        let counter = Counter()

        let observed = try await stream(samples, models: models, hook: { counter.increment() })
        let plain = try await stream(samples, models: models, hook: nil)

        XCTAssertGreaterThan(observed.updates, 0, "the fixture decodes at least one window")
        XCTAssertEqual(
            counter.count, observed.updates,
            "one entry per window model call; no window failed, so every call published an update")
        XCTAssertEqual(observed.text, plain.text, "the observation never changes the transcript")
    }

    func testHookNeverFiresWhenNoWindowIsProcessed() async throws {
        let models = try await loadModels()
        let counter = Counter()
        let manager = SlidingWindowAsrManager(onWindowInferenceEntered: { counter.increment() })
        try await manager.loadModels(models)
        try await manager.startStreaming()
        _ = try? await manager.finish()
        await manager.cancel()
        XCTAssertEqual(counter.count, 0, "an empty stream enters no window")
    }
}

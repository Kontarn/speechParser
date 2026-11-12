import shutil
import struct
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock, patch

import audio_downloader
import pipeline


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class AudioTimelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.output = str(self.root / "mixed.wav")

    def source(self, name, seconds=1, amplitude=1000):
        path = self.root / name
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(struct.pack("<h", amplitude) * round(seconds * 16000))
        return str(path)

    def duration(self, path):
        with wave.open(str(path), "rb") as stream:
            self.assertEqual(stream.getframerate(), 16000)
            self.assertEqual(stream.getnchannels(), 1)
            return stream.getnframes() / stream.getframerate()

    def amplitude(self, path, seconds):
        with wave.open(str(path), "rb") as stream:
            stream.setpos(round(seconds * 16000))
            data = stream.readframes(160)
        samples = struct.unpack(f"<{len(data) // 2}h", data)
        return sum(samples) / len(samples)

    def download(self, events, sources, duration=None, content_type="audio/wav"):
        metadata = {"eventLogs": events}
        if duration is not None:
            metadata["duration"] = duration
        downloader = Mock(side_effect=lambda url, path, session: shutil.copyfile(sources[url], path))
        job_dir = self.root / "job"
        job_dir.mkdir()
        with (
            patch.object(audio_downloader, "_fetch_event_json", return_value=metadata),
            patch.object(audio_downloader, "_get_content_type", return_value=content_type) as head,
            patch.object(audio_downloader, "_download_chunk", downloader),
        ):
            output = audio_downloader.download_audio_only("event", None, None, str(job_dir))
        return output, downloader, head

    def test_eight_simultaneous_tracks_do_not_multiply_recording_duration(self):
        source = self.source("speaker.wav", seconds=1.5)
        sources = {f"https://example.test/{i}.wav": source for i in range(8)}
        events = [{"relativeTime": 0, "data": {"url": url}} for url in sources]
        output, downloader, _ = self.download(events, sources, duration=1.5)
        self.assertEqual(downloader.call_count, 8)
        self.assertAlmostEqual(self.duration(output), 1.5, places=4)
        self.assertAlmostEqual(pipeline.get_media_duration(output), 1.5, places=3)
        # All eight sources remain audible; this is mixing, not truncating a concat.
        self.assertAlmostEqual(self.amplitude(output, 0.5), 8000, delta=2)

    def test_offsets_overlap_and_trailing_silence_preserve_recording_timeline(self):
        first = self.source("first.wav", seconds=2, amplitude=1000)
        second = self.source("second.wav", seconds=1, amplitude=2000)
        audio_downloader._ffmpeg_mix_timeline([(first, 1), (second, 2)], self.output, 5)
        self.assertEqual(self.duration(self.output), 5)
        for position, expected in ((0.5, 0), (1.5, 1000), (2.5, 3000), (4.5, 0)):
            self.assertAlmostEqual(self.amplitude(self.output, position), expected, delta=2)

    def test_sequential_fragments_keep_gaps_without_duration_metadata(self):
        first = self.source("first.wav", amplitude=1000)
        second = self.source("second.wav", amplitude=2000)
        audio_downloader._ffmpeg_mix_timeline([(first, 0), (second, 3)], self.output, None)
        self.assertEqual(self.duration(self.output), 4)
        for position, expected in ((0.5, 1000), (2, 0), (3.5, 2000)):
            self.assertAlmostEqual(self.amplitude(self.output, position), expected, delta=2)

    def test_repeated_events_are_deduplicated_but_distinct_placements_are_kept(self):
        source = self.source("speaker.wav")
        url = "https://example.test/speaker.wav"
        events = [{"relativeTime": start, "data": {"url": url}} for start in (0, 0, 2)]
        output, downloader, head = self.download(events, {url: source}, duration=3)
        downloader.assert_called_once()
        head.assert_called_once()
        self.assertEqual(self.duration(output), 3)
        for position, expected in ((0.5, 1000), (1.5, 0), (2.5, 1000)):
            self.assertAlmostEqual(self.amplitude(output, position), expected, delta=2)

    def test_fragments_are_trimmed_to_recording_bounds(self):
        source = self.source("speaker.wav", seconds=2)
        audio_downloader._ffmpeg_mix_timeline(
            [(source, -0.5), ("missing-but-outside-recording.wav", 20)], self.output, 2,
        )
        self.assertEqual(self.duration(self.output), 2)
        self.assertAlmostEqual(self.amplitude(self.output, 0.5), 1000, delta=2)
        self.assertAlmostEqual(self.amplitude(self.output, 1.8), 0, delta=2)
        self.assertFalse(list(self.root.glob("audio_mix_*")))

    def test_large_overlaps_are_limited_instead_of_wrapping_pcm_samples(self):
        source = self.source("loud.wav", amplitude=20000)
        audio_downloader._ffmpeg_mix_timeline([(source, 0), (source, 0)], self.output, 1)
        level = self.amplitude(self.output, 0.5)
        self.assertGreater(level, 30000)
        self.assertLess(level, 32767)

    def test_video_fragment_audio_uses_its_relative_time(self):
        source = self.source("speaker.wav")
        video = self.root / "video.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=black:s=16x16:r=1:d=1", "-i", source,
            "-c:v", "mpeg4", "-c:a", "aac", "-shortest", str(video),
        ], check=True, capture_output=True)
        url = "https://example.test/video.mp4"
        output, _, _ = self.download(
            [{"relativeTime": 1, "data": {"url": url}}], {url: str(video)},
            duration=3, content_type="video/mp4",
        )
        self.assertEqual(self.duration(output), 3)
        self.assertAlmostEqual(self.amplitude(output, 0.5), 0, delta=2)
        self.assertGreater(self.amplitude(output, 1.5), 500)
        self.assertAlmostEqual(self.amplitude(output, 2.8), 0, delta=2)

    def test_transcription_duration_follows_audio_not_longer_video(self):
        source = self.source("speaker.wav")
        video = self.root / "long_video.mp4"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            "color=c=black:s=16x16:r=1:d=9", "-i", source,
            "-c:v", "mpeg4", "-c:a", "aac", str(video),
        ], check=True, capture_output=True)
        self.assertAlmostEqual(pipeline.get_media_duration(str(video)), 1, places=2)

    def test_container_timestamps_do_not_add_an_extra_offset(self):
        source = self.source("speaker.wav")
        shifted = self.root / "shifted.m4a"
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-itsoffset", "20",
            "-i", source, "-c:a", "aac", str(shifted),
        ], check=True, capture_output=True)
        audio_downloader._ffmpeg_mix_timeline([(str(shifted), 1)], self.output, 3)
        self.assertEqual(self.duration(self.output), 3)
        self.assertAlmostEqual(self.amplitude(self.output, 0.5), 0, delta=2)
        self.assertGreater(self.amplitude(self.output, 1.5), 500)
        self.assertAlmostEqual(self.amplitude(self.output, 2.8), 0, delta=2)


if __name__ == "__main__":
    unittest.main()

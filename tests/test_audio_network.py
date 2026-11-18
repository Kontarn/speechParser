import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx

import audio_downloader
import pipeline


class AudioNetworkTests(unittest.TestCase):
    def test_head_timeout_falls_back_to_get_without_reading_media(self):
        methods = []

        class UnreadBody(httpx.SyncByteStream):
            def __iter__(self):
                raise AssertionError("Probing must not download the recording")

        def handle(request):
            methods.append(request.method)
            self.assertIn("access=session", request.headers["cookie"])
            if request.method == "HEAD":
                raise httpx.ConnectTimeout("HEAD timed out", request=request)
            return httpx.Response(200, headers={"Content-Type": "audio/mp4"}, stream=UnreadBody())

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            result = audio_downloader._get_content_type(
                "https://media.test/track", "session", client=client,
            )
        self.assertEqual(result, "audio/mp4")
        self.assertEqual(methods, ["HEAD", "GET"])

    def test_get_probe_retries_temporary_connection_timeouts(self):
        get_attempts = 0

        def handle(request):
            nonlocal get_attempts
            if request.method == "HEAD":
                return httpx.Response(405)
            get_attempts += 1
            if get_attempts < 3:
                raise httpx.ConnectTimeout("temporary outage", request=request)
            return httpx.Response(200, headers={"Content-Type": "audio/wav"})

        with (
            httpx.Client(transport=httpx.MockTransport(handle)) as client,
            patch.object(audio_downloader.time, "sleep"),
        ):
            self.assertEqual(audio_downloader._get_content_type(
                "https://media.test/track", None, client=client,
            ), "audio/wav")
        self.assertEqual(get_attempts, 3)

    def test_unavailable_fragment_aborts_selection_with_network_diagnosis(self):
        methods = []
        metadata = {"eventLogs": [
            {"data": {"url": "https://media.test/good"}},
            {"data": {"url": "https://media.test/bad?token=private-token"}},
        ]}

        def handle(request):
            if request.url.path == "/good":
                return httpx.Response(200, headers={"Content-Type": "audio/wav"})
            methods.append(request.method)
            raise httpx.ConnectTimeout("unreachable", request=request)

        with (
            tempfile.TemporaryDirectory() as directory,
            httpx.Client(transport=httpx.MockTransport(handle)) as client,
            patch.object(audio_downloader, "_fetch_event_json", return_value=metadata),
            patch.object(audio_downloader.time, "sleep"),
            patch.object(audio_downloader, "_download_chunk") as download,
            self.assertLogs(audio_downloader.log, level="WARNING") as logs,
        ):
            with self.assertRaises(RuntimeError) as caught:
                audio_downloader._download_audio_only("event", None, None, directory, client)
            download.assert_not_called()
        self.assertEqual(methods, ["HEAD", "GET", "GET", "GET"])
        self.assertIsInstance(caught.exception.__cause__, httpx.ConnectTimeout)
        self.assertIn("media.test (ConnectTimeout)", str(caught.exception))
        self.assertNotIn("Не найдено", str(caught.exception))
        self.assertNotIn("private-token", str(caught.exception) + "".join(logs.output))

    def test_interrupted_download_restarts_without_retaining_partial_data(self):
        attempts = 0

        class BrokenBody(httpx.SyncByteStream):
            def __iter__(self):
                yield b"x" * 65536
                raise httpx.ReadError("connection lost")

        def handle(request):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                return httpx.Response(200, stream=BrokenBody())
            return httpx.Response(200, content=b"complete recording")

        with (
            tempfile.TemporaryDirectory() as directory,
            httpx.Client(transport=httpx.MockTransport(handle)) as client,
            patch.object(audio_downloader.time, "sleep"),
        ):
            target = Path(directory) / "track"
            audio_downloader._download_chunk("https://media.test/track", str(target), None, client=client)
            self.assertEqual(target.read_bytes(), b"complete recording")
        self.assertEqual(attempts, 2)

    def test_forbidden_get_is_not_retried(self):
        methods = []

        def handle(request):
            methods.append(request.method)
            return httpx.Response(403)

        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with self.assertRaises(httpx.HTTPStatusError):
                audio_downloader._get_content_type("https://media.test/track", None, client=client)
        self.assertEqual(methods, ["HEAD", "GET"])

    def test_pipeline_preserves_network_diagnosis_when_fallback_also_fails(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(pipeline, "download_audio_fast", side_effect=RuntimeError("media.test (ConnectTimeout)")),
            patch.object(pipeline, "download_recording_full", side_effect=RuntimeError("fallback failed")),
            self.assertLogs(pipeline.log, level="WARNING"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                pipeline.run_pipeline(
                    "https://my.mts-link.ru/1/2/record-new/3", "medium", None,
                    Mock(), str(Path(directory) / "result.txt"),
                )
        self.assertIn("media.test (ConnectTimeout)", str(caught.exception))
        self.assertIn("fallback failed", str(caught.exception))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from datetime import timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.youtube_capture import YouTubeCapture

FAKE_URL = "https://www.youtube.com/watch?v=faketest"
FAKE_STREAM_URL = "https://cdn.example.com/stream.mp4"


class _FakeVideoCapture:
    """Minimal stand-in for cv2.VideoCapture that replays a list of frames."""

    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = iter(frames)
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self._opened = False


def _stub_yt_dlp(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock_run = MagicMock()
    mock_run.return_value.stdout = FAKE_STREAM_URL + "\n"
    monkeypatch.setattr("src.ingestion.youtube_capture.subprocess.run", mock_run)
    return mock_run


def test_iter_yields_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(2)]
    fake_cap = _FakeVideoCapture(dummy_frames)
    _stub_yt_dlp(monkeypatch)

    with patch("src.ingestion.youtube_capture.cv2.VideoCapture", return_value=fake_cap):
        with YouTubeCapture(url=FAKE_URL) as c:
            frames = list(c)

    assert len(frames) == 2
    assert all(FAKE_URL in f.source_label for f in frames)
    assert all(f.timestamp_utc.tzinfo is timezone.utc for f in frames)


def test_yt_dlp_called_with_correct_args(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_run = _stub_yt_dlp(monkeypatch)
    fake_cap = _FakeVideoCapture([])

    with patch("src.ingestion.youtube_capture.cv2.VideoCapture", return_value=fake_cap):
        with YouTubeCapture(url=FAKE_URL, format_spec="bestvideo+bestaudio"):
            pass

    mock_run.assert_called_once()
    cmd_args = mock_run.call_args[0][0]
    assert "yt-dlp" in cmd_args
    assert FAKE_URL in cmd_args
    assert "bestvideo+bestaudio" in cmd_args


def test_release_on_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_yt_dlp(monkeypatch)
    fake_cap = _FakeVideoCapture([])

    with patch("src.ingestion.youtube_capture.cv2.VideoCapture", return_value=fake_cap):
        with YouTubeCapture(url=FAKE_URL):
            pass

    assert not fake_cap.isOpened()


def test_iter_without_context_manager_raises() -> None:
    capture = YouTubeCapture(url=FAKE_URL)
    with pytest.raises(RuntimeError, match="context manager"):
        list(capture)


def test_from_config_reads_settings() -> None:
    cfg = {
        "ingestion": {
            "path": FAKE_URL,
            "youtube": {
                "format": "bestvideo+bestaudio",
                "frame_rate": 25,
            },
        }
    }
    capture = YouTubeCapture.from_config(cfg)
    assert capture._url == FAKE_URL
    assert capture._format_spec == "bestvideo+bestaudio"
    assert capture._frame_rate == 25


def test_from_config_uses_defaults_when_keys_absent() -> None:
    capture = YouTubeCapture.from_config({})
    assert capture._url == ""
    assert capture._format_spec == "best[ext=mp4]/best"
    assert capture._frame_rate == 30

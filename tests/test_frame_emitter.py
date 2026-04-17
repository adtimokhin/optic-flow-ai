from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest

from src.ingestion.frame_emitter import FrameEmitter
from src.ingestion.frames import Frame


def _make_frames(n: int = 3) -> list[Frame]:
    return [
        Frame(
            image=np.zeros((720, 1280, 3), dtype=np.uint8),
            timestamp_utc=datetime.now(tz=timezone.utc),
            source_label=f"fake:{i}",
        )
        for i in range(n)
    ]


class _FakeCapture:
    """Minimal capture stub that yields a fixed list of Frame objects."""

    def __init__(self, frames: list[Frame]) -> None:
        self._frames = frames

    def __enter__(self) -> _FakeCapture:
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def __iter__(self):
        yield from self._frames


def test_iter_yields_all_frames() -> None:
    frames = _make_frames(4)
    emitter = FrameEmitter(_FakeCapture(frames))
    assert list(emitter) == frames


def test_iter_empty_source() -> None:
    emitter = FrameEmitter(_FakeCapture([]))
    assert list(emitter) == []


def test_from_webcam_returns_frame_emitter() -> None:
    fake_cap = _FakeCapture(_make_frames(1))
    with patch("src.ingestion.frame_emitter.WebcamCapture", return_value=fake_cap):
        emitter = FrameEmitter.from_webcam(device_index=0)
    assert isinstance(emitter, FrameEmitter)


def test_from_youtube_returns_frame_emitter() -> None:
    fake_cap = _FakeCapture(_make_frames(1))
    with patch("src.ingestion.frame_emitter.YouTubeCapture", return_value=fake_cap):
        emitter = FrameEmitter.from_youtube(url="https://youtube.com/watch?v=test")
    assert isinstance(emitter, FrameEmitter)


def test_from_config_webcam(tmp_path: pytest.TempPathFactory) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("ingestion:\n  source: webcam\n  webcam:\n    device_index: 0\n")
    fake_cap = _FakeCapture(_make_frames(1))

    with patch("src.ingestion.frame_emitter.WebcamCapture") as MockCap:
        MockCap.from_config.return_value = fake_cap
        emitter = FrameEmitter.from_config(str(config))

    assert isinstance(emitter, FrameEmitter)
    MockCap.from_config.assert_called_once()


def test_from_config_youtube(tmp_path: pytest.TempPathFactory) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "ingestion:\n  source: youtube\n  path: https://youtube.com/watch?v=test\n"
    )
    fake_cap = _FakeCapture(_make_frames(1))

    with patch("src.ingestion.frame_emitter.YouTubeCapture") as MockCap:
        MockCap.from_config.return_value = fake_cap
        emitter = FrameEmitter.from_config(str(config))

    assert isinstance(emitter, FrameEmitter)
    MockCap.from_config.assert_called_once()


def test_from_config_unknown_source_raises(tmp_path: pytest.TempPathFactory) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("ingestion:\n  source: unknown\n")
    with pytest.raises(ValueError, match="Unknown ingestion source"):
        FrameEmitter.from_config(str(config))


def test_frame_fields_preserved() -> None:
    ts = datetime.now(tz=timezone.utc)
    image = np.ones((480, 640, 3), dtype=np.uint8)
    frame = Frame(image=image, timestamp_utc=ts, source_label="test:0")
    emitter = FrameEmitter(_FakeCapture([frame]))
    result = list(emitter)

    assert result[0].source_label == "test:0"
    assert result[0].timestamp_utc == ts
    np.testing.assert_array_equal(result[0].image, image)

from __future__ import annotations

from datetime import timezone
from unittest.mock import patch

import numpy as np
import pytest

from src.ingestion.webcam_capture import WebcamCapture


class _FakeVideoCapture:
    """Minimal stand-in for cv2.VideoCapture that replays a list of frames."""

    def __init__(self, index: int, frames: list[np.ndarray]) -> None:
        self._frames = iter(frames)
        self._opened = True

    def isOpened(self) -> bool:
        return self._opened

    def set(self, prop_id: int, value: float) -> None:
        pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        try:
            return True, next(self._frames)
        except StopIteration:
            return False, None

    def release(self) -> None:
        self._opened = False


def test_iter_yields_frames() -> None:
    dummy_frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(3)]
    fake_cap = _FakeVideoCapture(0, dummy_frames)

    with patch("src.ingestion.webcam_capture.cv2.VideoCapture", return_value=fake_cap):
        capture = WebcamCapture(device_index=0)
        with capture as c:
            frames = list(c)

    assert len(frames) == 3
    assert all(f.source_label == "webcam:0" for f in frames)
    assert all(f.timestamp_utc.tzinfo is timezone.utc for f in frames)


def test_image_shape_preserved() -> None:
    image = np.ones((720, 1280, 3), dtype=np.uint8) * 128
    fake_cap = _FakeVideoCapture(0, [image])

    with patch("src.ingestion.webcam_capture.cv2.VideoCapture", return_value=fake_cap):
        with WebcamCapture(device_index=0) as c:
            frames = list(c)

    assert frames[0].image.shape == (720, 1280, 3)
    np.testing.assert_array_equal(frames[0].image, image)


def test_release_called_on_exit() -> None:
    fake_cap = _FakeVideoCapture(0, [])

    with patch("src.ingestion.webcam_capture.cv2.VideoCapture", return_value=fake_cap):
        with WebcamCapture(device_index=0):
            pass

    assert not fake_cap.isOpened()


def test_iter_without_context_manager_raises() -> None:
    capture = WebcamCapture(device_index=0)
    with pytest.raises(RuntimeError, match="context manager"):
        list(capture)


def test_from_config_reads_settings() -> None:
    cfg = {
        "ingestion": {
            "webcam": {
                "device_index": 2,
                "frame_width": 640,
                "frame_height": 480,
                "frame_rate": 15,
            }
        }
    }
    capture = WebcamCapture.from_config(cfg)
    assert capture._device_index == 2
    assert capture._frame_width == 640
    assert capture._frame_height == 480
    assert capture._frame_rate == 15


def test_from_config_uses_defaults_when_keys_absent() -> None:
    capture = WebcamCapture.from_config({})
    assert capture._device_index == 0
    assert capture._frame_width == 1280
    assert capture._frame_height == 720
    assert capture._frame_rate == 30

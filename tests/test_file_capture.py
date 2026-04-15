from __future__ import annotations

from datetime import timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ingestion.file_capture import FileCapture


def _write_dummy_image(path: Path) -> np.ndarray:
    """Writes a small BGR image to disk and returns the array."""
    import cv2

    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 2] = 200  # red channel
    cv2.imwrite(str(path), image)
    return image


class _FakeVideoCapture:
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


# ---------------------------------------------------------------------------
# Image file tests
# ---------------------------------------------------------------------------


def test_image_yields_exactly_one_frame(tmp_path: Path) -> None:
    img_path = tmp_path / "ship.jpg"
    _write_dummy_image(img_path)

    with FileCapture(str(img_path)) as cap:
        frames = list(cap)

    assert len(frames) == 1


def test_image_source_label(tmp_path: Path) -> None:
    img_path = tmp_path / "ship.png"
    _write_dummy_image(img_path)

    with FileCapture(str(img_path)) as cap:
        frames = list(cap)

    assert str(img_path) in frames[0].source_label


def test_image_timestamp_is_utc(tmp_path: Path) -> None:
    img_path = tmp_path / "ship.jpg"
    _write_dummy_image(img_path)

    with FileCapture(str(img_path)) as cap:
        frames = list(cap)

    assert frames[0].timestamp_utc.tzinfo is timezone.utc


def test_image_pixel_data_correct(tmp_path: Path) -> None:
    img_path = tmp_path / "ship.jpg"
    original = _write_dummy_image(img_path)

    with FileCapture(str(img_path)) as cap:
        frames = list(cap)

    assert frames[0].image.shape == original.shape


# ---------------------------------------------------------------------------
# Video file tests
# ---------------------------------------------------------------------------


def test_video_yields_all_frames(tmp_path: Path) -> None:
    dummy_frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    fake_cap = _FakeVideoCapture(dummy_frames)
    video_path = tmp_path / "ship.mp4"
    video_path.touch()

    with patch("src.ingestion.file_capture.cv2.VideoCapture", return_value=fake_cap):
        with FileCapture(str(video_path)) as cap:
            frames = list(cap)

    assert len(frames) == 5


def test_video_source_label(tmp_path: Path) -> None:
    fake_cap = _FakeVideoCapture([np.zeros((480, 640, 3), dtype=np.uint8)])
    video_path = tmp_path / "ship.mp4"
    video_path.touch()

    with patch("src.ingestion.file_capture.cv2.VideoCapture", return_value=fake_cap):
        with FileCapture(str(video_path)) as cap:
            frames = list(cap)

    assert str(video_path) in frames[0].source_label


def test_video_release_on_exit(tmp_path: Path) -> None:
    fake_cap = _FakeVideoCapture([])
    video_path = tmp_path / "ship.mp4"
    video_path.touch()

    with patch("src.ingestion.file_capture.cv2.VideoCapture", return_value=fake_cap):
        with FileCapture(str(video_path)):
            pass

    assert not fake_cap.isOpened()


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


def test_missing_file_raises_on_enter(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        with FileCapture(str(tmp_path / "nonexistent.mp4")):
            pass


def test_iter_without_context_manager_raises_for_video(tmp_path: Path) -> None:
    video_path = tmp_path / "ship.mp4"
    video_path.touch()
    cap = FileCapture(str(video_path))
    with pytest.raises(RuntimeError, match="context manager"):
        list(cap)


def test_iter_without_context_manager_raises_for_image(tmp_path: Path) -> None:
    img_path = tmp_path / "ship.jpg"
    img_path.touch()
    cap = FileCapture(str(img_path))
    with pytest.raises(RuntimeError, match="context manager"):
        list(cap)


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_from_config_reads_path(tmp_path: Path) -> None:
    img_path = tmp_path / "ship.jpg"
    _write_dummy_image(img_path)
    cfg = {"ingestion": {"path": str(img_path)}}
    cap = FileCapture.from_config(cfg)
    assert cap._path == img_path


def test_from_file_factory_end_to_end(tmp_path: Path) -> None:
    from src.ingestion.frame_emitter import FrameEmitter

    img_path = tmp_path / "ship.jpg"
    _write_dummy_image(img_path)

    emitter = FrameEmitter.from_file(str(img_path))
    frames = list(emitter)

    assert len(frames) == 1
    assert frames[0].image.shape == (64, 64, 3)

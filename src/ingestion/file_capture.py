from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.ingestion.frames import Frame

_IMAGE_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
)


class FileCapture:
    """Captures frames from a local video file or a single image file via OpenCV.

    For image files (.jpg, .png, etc.) exactly one Frame is yielded.
    For video files (.mp4, .avi, etc.) frames are yielded until the file ends.
    """

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._is_image = self._path.suffix.lower() in _IMAGE_EXTENSIONS
        self._cap: cv2.VideoCapture | None = None
        self._image: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> FileCapture:
        """Returns a FileCapture built from the ingestion.path config value."""
        path = cfg.get("ingestion", {}).get("path", "")
        return cls(path)

    def __enter__(self) -> FileCapture:
        if not self._path.exists():
            raise FileNotFoundError(f"Input file not found: {self._path}")
        if self._is_image:
            self._image = cv2.imread(str(self._path))
            if self._image is None:
                raise RuntimeError(f"OpenCV could not read image: {self._path}")
        else:
            self._cap = cv2.VideoCapture(str(self._path))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open video file: {self._path}")
        return self

    def __exit__(self, *_args: object) -> None:
        self._release()

    def __iter__(self) -> Iterator[Frame]:
        """Yields Frame objects from the file until exhausted."""
        if self._is_image:
            if self._image is None:
                raise RuntimeError(
                    "FileCapture must be used as a context manager before iterating"
                )
            yield Frame(
                image=self._image,
                timestamp_utc=datetime.now(tz=timezone.utc),
                source_label=f"file:{self._path}",
            )
        else:
            if self._cap is None:
                raise RuntimeError(
                    "FileCapture must be used as a context manager before iterating"
                )
            while True:
                ret, image = self._cap.read()
                if not ret:
                    break
                yield Frame(
                    image=image,
                    timestamp_utc=datetime.now(tz=timezone.utc),
                    source_label=f"file:{self._path}",
                )

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._image = None

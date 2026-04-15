from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import cv2
import numpy as np

from src.ingestion.frames import Frame

DEFAULT_DEVICE_INDEX = 0
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720
DEFAULT_FRAME_RATE = 30


class WebcamCapture:
    """Captures frames from a local webcam device via OpenCV."""

    def __init__(
        self,
        device_index: int = DEFAULT_DEVICE_INDEX,
        frame_width: int = DEFAULT_FRAME_WIDTH,
        frame_height: int = DEFAULT_FRAME_HEIGHT,
        frame_rate: int = DEFAULT_FRAME_RATE,
    ) -> None:
        self._device_index = device_index
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._frame_rate = frame_rate
        self._cap: cv2.VideoCapture | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> WebcamCapture:
        """Returns a WebcamCapture built from the ingestion.webcam config section."""
        webcam_cfg = cfg.get("ingestion", {}).get("webcam", {})
        return cls(
            device_index=webcam_cfg.get("device_index", DEFAULT_DEVICE_INDEX),
            frame_width=webcam_cfg.get("frame_width", DEFAULT_FRAME_WIDTH),
            frame_height=webcam_cfg.get("frame_height", DEFAULT_FRAME_HEIGHT),
            frame_rate=webcam_cfg.get("frame_rate", DEFAULT_FRAME_RATE),
        )

    def __enter__(self) -> WebcamCapture:
        self._cap = cv2.VideoCapture(self._device_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._frame_rate)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open webcam device index {self._device_index}")
        return self

    def __exit__(self, *_args: object) -> None:
        self._release()

    def __iter__(self) -> Iterator[Frame]:
        """Yields Frame objects from the webcam until the stream ends."""
        if self._cap is None:
            raise RuntimeError(
                "WebcamCapture must be used as a context manager before iterating"
            )
        while True:
            ret, image = self._cap.read()
            if not ret:
                break
            yield Frame(
                image=image,
                timestamp_utc=datetime.now(tz=timezone.utc),
                source_label=f"webcam:{self._device_index}",
            )

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

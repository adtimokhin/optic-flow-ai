from __future__ import annotations

from collections.abc import Iterator
from typing import Union

import yaml

from src.ingestion.file_capture import FileCapture
from src.ingestion.frames import Frame
from src.ingestion.webcam_capture import WebcamCapture
from src.ingestion.youtube_capture import YouTubeCapture

_Capture = Union[WebcamCapture, YouTubeCapture, FileCapture]


class FrameEmitter:
    """Unified frame iterator over webcam, YouTube, or local file sources."""

    def __init__(self, capture: _Capture) -> None:
        self._capture = capture

    @classmethod
    def from_webcam(
        cls,
        device_index: int = 0,
        frame_width: int = 1280,
        frame_height: int = 720,
        frame_rate: int = 30,
    ) -> FrameEmitter:
        """Returns a FrameEmitter backed by a local webcam device."""
        return cls(
            WebcamCapture(
                device_index=device_index,
                frame_width=frame_width,
                frame_height=frame_height,
                frame_rate=frame_rate,
            )
        )

    @classmethod
    def from_youtube(
        cls,
        url: str,
        format_spec: str = "best[ext=mp4]/best",
        frame_rate: int = 30,
    ) -> FrameEmitter:
        """Returns a FrameEmitter backed by a YouTube stream resolved via yt-dlp."""
        return cls(
            YouTubeCapture(url=url, format_spec=format_spec, frame_rate=frame_rate)
        )

    @classmethod
    def from_file(cls, path: str) -> FrameEmitter:
        """Returns a FrameEmitter backed by a local video or image file."""
        return cls(FileCapture(path))

    @classmethod
    def from_config(cls, config_path: str = "config/settings.yaml") -> FrameEmitter:
        """Returns a FrameEmitter configured from the given settings.yaml path."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        source = cfg.get("ingestion", {}).get("source", "webcam")
        if source == "webcam":
            capture: _Capture = WebcamCapture.from_config(cfg)
        elif source == "youtube":
            capture = YouTubeCapture.from_config(cfg)
        elif source == "file":
            capture = FileCapture.from_config(cfg)
        else:
            raise ValueError(f"Unknown ingestion source: {source!r}")
        return cls(capture)

    def __iter__(self) -> Iterator[Frame]:
        """Yields Frame objects from the underlying capture source."""
        with self._capture:
            yield from self._capture

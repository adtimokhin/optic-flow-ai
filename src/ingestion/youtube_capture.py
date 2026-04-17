from __future__ import annotations

import subprocess
from collections.abc import Iterator
from datetime import datetime, timezone

import cv2
import numpy as np

from src.ingestion.frames import Frame

DEFAULT_FORMAT = "best[ext=mp4]/best"
DEFAULT_FRAME_RATE = 30


class YouTubeCapture:
    """Streams frames from a YouTube URL via yt-dlp and OpenCV."""

    def __init__(
        self,
        url: str,
        format_spec: str = DEFAULT_FORMAT,
        frame_rate: int = DEFAULT_FRAME_RATE,
    ) -> None:
        self._url = url
        self._format_spec = format_spec
        self._frame_rate = frame_rate
        self._cap: cv2.VideoCapture | None = None

    @classmethod
    def from_config(cls, cfg: dict) -> YouTubeCapture:
        """Returns a YouTubeCapture built from the ingestion config section."""
        ingestion_cfg = cfg.get("ingestion", {})
        youtube_cfg = ingestion_cfg.get("youtube", {})
        return cls(
            url=ingestion_cfg.get("path", ""),
            format_spec=youtube_cfg.get("format", DEFAULT_FORMAT),
            frame_rate=youtube_cfg.get("frame_rate", DEFAULT_FRAME_RATE),
        )

    def __enter__(self) -> YouTubeCapture:
        stream_url = self._resolve_stream_url(self._url, self._format_spec)
        self._cap = cv2.VideoCapture(stream_url)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video stream for URL: {self._url}")
        return self

    def __exit__(self, *_args: object) -> None:
        self._release()

    def __iter__(self) -> Iterator[Frame]:
        """Yields Frame objects from the YouTube stream until exhausted."""
        if self._cap is None:
            raise RuntimeError(
                "YouTubeCapture must be used as a context manager before iterating"
            )
        while True:
            ret, image = self._cap.read()
            if not ret:
                break
            yield Frame(
                image=image,
                timestamp_utc=datetime.now(tz=timezone.utc),
                source_label=f"youtube:{self._url}",
            )

    def _resolve_stream_url(self, url: str, format_spec: str) -> str:
        """Returns a direct stream URL for the given YouTube URL using yt-dlp."""
        result = subprocess.run(
            ["yt-dlp", "--get-url", "-f", format_spec, url],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip().splitlines()[0]

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import yaml

from src.ingestion.frames import Frame

DEFAULT_INPUT_SIZE: tuple[int, int] = (512, 512)
DEFAULT_NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD: list[float] = [0.229, 0.224, 0.225]


@dataclass
class PreprocessedFrame:
    """A frame that has been preprocessed and is ready for model inference."""

    tensor: torch.Tensor  # shape (1, 3, H, W), float32, normalised
    original_size: tuple[int, int]  # (height, width) of the source frame
    source_label: str


class FramePreprocessor:
    """Converts raw Frame objects into normalised tensors for a PyTorch segmentation model."""

    def __init__(
        self,
        input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
        normalize_mean: list[float] | None = None,
        normalize_std: list[float] | None = None,
    ) -> None:
        self._input_size = input_size
        self._mean = np.array(
            normalize_mean if normalize_mean is not None else DEFAULT_NORMALIZE_MEAN,
            dtype=np.float32,
        )
        self._std = np.array(
            normalize_std if normalize_std is not None else DEFAULT_NORMALIZE_STD,
            dtype=np.float32,
        )

    @classmethod
    def from_config(
        cls, config_path: str = "config/settings.yaml"
    ) -> FramePreprocessor:
        """Returns a FramePreprocessor configured from the waterline section of settings.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        wl_cfg = cfg.get("detection", {}).get("waterline", {})
        raw_size = wl_cfg.get("input_size", list(DEFAULT_INPUT_SIZE))
        return cls(
            input_size=(raw_size[0], raw_size[1]),
            normalize_mean=wl_cfg.get("normalize_mean", DEFAULT_NORMALIZE_MEAN),
            normalize_std=wl_cfg.get("normalize_std", DEFAULT_NORMALIZE_STD),
        )

    def preprocess(self, frame: Frame) -> PreprocessedFrame:
        """Converts a raw Frame into a normalised, batched tensor ready for inference."""
        original_size = (frame.image.shape[0], frame.image.shape[1])
        rgb = self._bgr_to_rgb(frame.image)
        resized = self._resize(rgb)
        normalised = self._normalise(resized)
        tensor = self._to_tensor(normalised)
        return PreprocessedFrame(
            tensor=tensor,
            original_size=original_size,
            source_label=frame.source_label,
        )

    def _bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        """Converts a BGR uint8 image to RGB uint8."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resizes image to the configured input size using bilinear interpolation."""
        h, w = self._input_size
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)

    def _normalise(self, image: np.ndarray) -> np.ndarray:
        """Scales pixels to [0, 1] then applies per-channel ImageNet normalisation."""
        img = image.astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        return img

    def _to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Converts a normalised (H, W, C) float32 array to a batched (1, C, H, W) tensor."""
        tensor = torch.from_numpy(image).permute(2, 0, 1)
        return tensor.unsqueeze(0)

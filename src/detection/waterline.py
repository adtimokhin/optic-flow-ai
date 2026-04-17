from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml

from src.ingestion.frames import Frame

DEFAULT_INPUT_SIZE: tuple[int, int] = (512, 512)
DEFAULT_NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD: list[float] = [0.229, 0.224, 0.225]
NUM_CLASSES: int = 3  # water=0, hull_above=1, hull_below=2


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


# ---------------------------------------------------------------------------
# Segmentation result
# ---------------------------------------------------------------------------


@dataclass
class SegmentationResult:
    """Pixel-wise classification output from the waterline segmentation model."""

    # Shape (H, W), dtype long — values: 0=water, 1=hull_above, 2=hull_below
    mask: torch.Tensor
    original_size: tuple[int, int]  # (height, width) of the source frame
    source_label: str
    is_stub: bool  # True when produced by the stub model, not real weights


# ---------------------------------------------------------------------------
# Stub model — placeholder until real weights are trained
# ---------------------------------------------------------------------------


class _StubSegmentationModel(nn.Module):
    """Returns a constant all-water mask. Deterministic, zero-dependency placeholder."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits shaped (batch, NUM_CLASSES, H, W) with water class dominant."""
        batch, _, h, w = x.shape
        logits = torch.zeros(batch, NUM_CLASSES, h, w)
        logits[:, 0, :, :] = 1.0  # water class gets highest logit everywhere
        return logits


# ---------------------------------------------------------------------------
# Segmentor
# ---------------------------------------------------------------------------


class WaterlineSegmentor:
    """Runs segmentation on a preprocessed frame and returns a pixel-wise class mask.

    When no model weights are available a deterministic stub is used so the
    rest of the pipeline can be developed and tested independently.
    """

    def __init__(self, model: nn.Module | None = None) -> None:
        self._model: nn.Module = (
            model if model is not None else _StubSegmentationModel()
        )
        self._is_stub = model is None
        self._model.eval()

    @classmethod
    def from_config(
        cls, config_path: str = "config/settings.yaml"
    ) -> WaterlineSegmentor:
        """Returns a WaterlineSegmentor, loading weights from model_path if they exist."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        model_path = cfg.get("detection", {}).get("waterline", {}).get("model_path", "")
        weights_file = f"{model_path}weights.pt" if model_path else ""

        import os

        if weights_file and os.path.exists(weights_file):
            model = torch.load(weights_file, map_location="cpu")
            return cls(model=model)

        return cls(model=None)

    def segment(self, preprocessed: PreprocessedFrame) -> SegmentationResult:
        """Runs the segmentation model and returns a labelled mask at model resolution."""
        with torch.no_grad():
            logits = self._model(preprocessed.tensor)  # (1, C, H, W)
        mask = logits.squeeze(0).argmax(dim=0)  # (H, W), long
        return SegmentationResult(
            mask=mask,
            original_size=preprocessed.original_size,
            source_label=preprocessed.source_label,
            is_stub=self._is_stub,
        )

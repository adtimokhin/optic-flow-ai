from __future__ import annotations

from datetime import datetime, timezone

import pytest
import torch

from src.detection.waterline import (
    NUM_CLASSES,
    FramePreprocessor,
    PreprocessedFrame,
    SegmentationResult,
    WaterlineSegmentor,
    _StubSegmentationModel,
)


def _make_preprocessed(h: int = 512, w: int = 512) -> PreprocessedFrame:
    return PreprocessedFrame(
        tensor=torch.zeros(1, 3, h, w),
        original_size=(720, 1280),
        source_label="test:0",
    )


# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------


def test_stub_model_output_shape() -> None:
    model = _StubSegmentationModel()
    x = torch.zeros(1, 3, 512, 512)
    out = model(x)
    assert out.shape == (1, NUM_CLASSES, 512, 512)


def test_stub_model_water_class_dominant() -> None:
    model = _StubSegmentationModel()
    x = torch.zeros(1, 3, 512, 512)
    out = model(x)
    predicted = out.squeeze(0).argmax(dim=0)
    assert (predicted == 0).all(), "Stub should classify every pixel as water (class 0)"


def test_stub_model_is_deterministic() -> None:
    model = _StubSegmentationModel()
    x = torch.rand(1, 3, 512, 512)
    out1 = model(x)
    out2 = model(x)
    assert torch.equal(out1, out2)


# ---------------------------------------------------------------------------
# WaterlineSegmentor
# ---------------------------------------------------------------------------


def test_segmentor_defaults_to_stub() -> None:
    segmentor = WaterlineSegmentor()
    assert segmentor._is_stub is True


def test_segment_mask_shape() -> None:
    segmentor = WaterlineSegmentor()
    result = segmentor.segment(_make_preprocessed())
    assert result.mask.shape == (512, 512)


def test_segment_mask_dtype_is_long() -> None:
    segmentor = WaterlineSegmentor()
    result = segmentor.segment(_make_preprocessed())
    assert result.mask.dtype == torch.long


def test_segment_mask_values_in_valid_range() -> None:
    segmentor = WaterlineSegmentor()
    result = segmentor.segment(_make_preprocessed())
    assert result.mask.min().item() >= 0
    assert result.mask.max().item() < NUM_CLASSES


def test_segment_stub_produces_all_water_mask() -> None:
    segmentor = WaterlineSegmentor()
    result = segmentor.segment(_make_preprocessed())
    assert (result.mask == 0).all()


def test_segment_preserves_original_size() -> None:
    segmentor = WaterlineSegmentor()
    preprocessed = _make_preprocessed()
    result = segmentor.segment(preprocessed)
    assert result.original_size == (720, 1280)


def test_segment_preserves_source_label() -> None:
    segmentor = WaterlineSegmentor()
    result = segmentor.segment(_make_preprocessed())
    assert result.source_label == "test:0"


def test_segment_is_stub_flag_set() -> None:
    segmentor = WaterlineSegmentor()
    result = segmentor.segment(_make_preprocessed())
    assert result.is_stub is True


def test_from_config_no_weights_returns_stub(tmp_path: pytest.TempPathFactory) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("detection:\n  waterline:\n    model_path: models/waterline/\n")
    segmentor = WaterlineSegmentor.from_config(str(config))
    assert segmentor._is_stub is True


def test_custom_model_is_not_stub() -> None:
    import torch.nn as nn

    class _ConstantModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, _, h, w = x.shape
            out = torch.zeros(b, NUM_CLASSES, h, w)
            out[:, 1, :, :] = 1.0  # hull_above dominant
            return out

    segmentor = WaterlineSegmentor(model=_ConstantModel())
    assert segmentor._is_stub is False
    result = segmentor.segment(_make_preprocessed())
    assert (result.mask == 1).all()

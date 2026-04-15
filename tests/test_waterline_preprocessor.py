from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from src.detection.waterline import (
    DEFAULT_INPUT_SIZE,
    FramePreprocessor,
    PreprocessedFrame,
)
from src.ingestion.frames import Frame


def _make_frame(height: int = 720, width: int = 1280) -> Frame:
    return Frame(
        image=np.random.randint(0, 256, (height, width, 3), dtype=np.uint8),
        timestamp_utc=datetime.now(tz=timezone.utc),
        source_label="webcam:0",
    )


def test_output_tensor_shape() -> None:
    preprocessor = FramePreprocessor()
    result = preprocessor.preprocess(_make_frame())
    h, w = DEFAULT_INPUT_SIZE
    assert result.tensor.shape == (1, 3, h, w)


def test_output_tensor_shape_non_square_input() -> None:
    preprocessor = FramePreprocessor(input_size=(256, 512))
    result = preprocessor.preprocess(_make_frame(height=480, width=640))
    assert result.tensor.shape == (1, 3, 256, 512)


def test_output_tensor_dtype() -> None:
    preprocessor = FramePreprocessor()
    result = preprocessor.preprocess(_make_frame())
    assert result.tensor.dtype == torch.float32


def test_output_tensor_is_batched() -> None:
    preprocessor = FramePreprocessor()
    result = preprocessor.preprocess(_make_frame())
    assert result.tensor.ndim == 4
    assert result.tensor.shape[0] == 1


def test_channel_order_is_rgb_not_bgr() -> None:
    # Construct a frame where R, G, B channels have distinct, known values.
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 10  # B channel in BGR
    image[:, :, 1] = 20  # G channel in BGR
    image[:, :, 2] = 30  # R channel in BGR
    frame = Frame(
        image=image,
        timestamp_utc=datetime.now(tz=timezone.utc),
        source_label="test",
    )
    preprocessor = FramePreprocessor(
        input_size=(64, 64),
        normalize_mean=[0.0, 0.0, 0.0],
        normalize_std=[1.0, 1.0, 1.0],
    )
    result = preprocessor.preprocess(frame)
    # After BGR→RGB conversion channel 0 of the tensor should be the R value (30/255).
    expected_r = 30.0 / 255.0
    assert abs(result.tensor[0, 0, 0, 0].item() - expected_r) < 1e-4


def test_normalisation_zero_mean_unit_std() -> None:
    # With mean=0 and std=1 the tensor values should simply be pixels/255.
    preprocessor = FramePreprocessor(
        input_size=(64, 64),
        normalize_mean=[0.0, 0.0, 0.0],
        normalize_std=[1.0, 1.0, 1.0],
    )
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    frame = Frame(
        image=image,
        timestamp_utc=datetime.now(tz=timezone.utc),
        source_label="test",
    )
    result = preprocessor.preprocess(frame)
    expected = 128.0 / 255.0
    assert torch.allclose(
        result.tensor, torch.full_like(result.tensor, expected), atol=1e-4
    )


def test_normalisation_shifts_values() -> None:
    # A solid-colour image should produce a tensor whose mean is close to
    # (pixel/255 - mean_val) / std_val for each channel.
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    preprocessor = FramePreprocessor(
        input_size=(64, 64), normalize_mean=mean, normalize_std=std
    )
    # All pixels = 255 → normalised per channel = (1.0 - mean[c]) / std[c]
    image = np.full((64, 64, 3), 255, dtype=np.uint8)
    frame = Frame(
        image=image,
        timestamp_utc=datetime.now(tz=timezone.utc),
        source_label="test",
    )
    result = preprocessor.preprocess(frame)
    for c in range(3):
        expected = (1.0 - mean[c]) / std[c]
        actual = result.tensor[0, c, 0, 0].item()
        assert (
            abs(actual - expected) < 1e-4
        ), f"channel {c}: expected {expected}, got {actual}"


def test_original_size_preserved() -> None:
    preprocessor = FramePreprocessor(input_size=(256, 256))
    result = preprocessor.preprocess(_make_frame(height=480, width=852))
    assert result.original_size == (480, 852)


def test_source_label_passed_through() -> None:
    frame = _make_frame()
    frame.source_label = "youtube:https://example.com"
    preprocessor = FramePreprocessor()
    result = preprocessor.preprocess(frame)
    assert result.source_label == "youtube:https://example.com"


def test_from_config_reads_settings(tmp_path: pytest.TempPathFactory) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "detection:\n"
        "  waterline:\n"
        "    input_size: [256, 256]\n"
        "    normalize_mean: [0.5, 0.5, 0.5]\n"
        "    normalize_std: [0.5, 0.5, 0.5]\n"
    )
    preprocessor = FramePreprocessor.from_config(str(config))
    assert preprocessor._input_size == (256, 256)
    np.testing.assert_array_equal(preprocessor._mean, [0.5, 0.5, 0.5])
    np.testing.assert_array_equal(preprocessor._std, [0.5, 0.5, 0.5])


def test_from_config_uses_defaults_when_keys_absent(
    tmp_path: pytest.TempPathFactory,
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("detection:\n  waterline: {}\n")
    preprocessor = FramePreprocessor.from_config(str(config))
    assert preprocessor._input_size == DEFAULT_INPUT_SIZE

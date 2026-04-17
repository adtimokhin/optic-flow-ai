from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.detection.ship_detector import (
    _COCO_BOAT_CLASS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PATH,
    ShipDetection,
    ShipDetector,
)
from src.ingestion.frames import Frame


def _make_frame(height: int = 720, width: int = 1280) -> Frame:
    return Frame(
        image=np.zeros((height, width, 3), dtype=np.uint8),
        timestamp_utc=datetime.now(tz=timezone.utc),
        source_label="test:0",
    )


def _make_yolo_result(
    class_ids: list[int],
    confidences: list[float],
    boxes_xyxy: list[list[float]],
) -> MagicMock:
    """Builds a fake YOLO Results object matching the Ultralytics API."""
    result = MagicMock()
    n = len(class_ids)

    if n == 0:
        result.boxes = None
        return result

    mock_boxes = MagicMock()
    mock_boxes.__len__ = lambda self: n
    mock_boxes.cls = torch.tensor(class_ids, dtype=torch.float32)
    mock_boxes.conf = torch.tensor(confidences, dtype=torch.float32)
    mock_boxes.xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)
    result.boxes = mock_boxes
    return result


def _make_detector(
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ShipDetector:
    """Returns a ShipDetector with a mocked YOLO model."""
    with patch(
        "src.detection.ship_detector.ShipDetector._load_model", return_value=MagicMock()
    ):
        detector = ShipDetector(confidence_threshold=confidence_threshold)
    return detector


# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def test_detect_returns_none_when_no_detections() -> None:
    detector = _make_detector()
    detector._model.predict.return_value = [_make_yolo_result([], [], [])]
    result = detector.detect(_make_frame())
    assert result is None


def test_detect_returns_none_when_no_boat_class() -> None:
    detector = _make_detector()
    # class 0 = person, not a boat
    detector._model.predict.return_value = [
        _make_yolo_result([0], [0.9], [[10.0, 20.0, 200.0, 400.0]])
    ]
    result = detector.detect(_make_frame())
    assert result is None


def test_detect_returns_none_below_confidence_threshold() -> None:
    detector = _make_detector(confidence_threshold=0.5)
    detector._model.predict.return_value = [
        _make_yolo_result([_COCO_BOAT_CLASS], [0.3], [[10.0, 20.0, 200.0, 400.0]])
    ]
    result = detector.detect(_make_frame())
    assert result is None


def test_detect_returns_detection_for_boat_class() -> None:
    detector = _make_detector()
    detector._model.predict.return_value = [
        _make_yolo_result([_COCO_BOAT_CLASS], [0.85], [[50.0, 100.0, 800.0, 600.0]])
    ]
    result = detector.detect(_make_frame())
    assert isinstance(result, ShipDetection)


def test_detect_box_coordinates_are_integers() -> None:
    detector = _make_detector()
    detector._model.predict.return_value = [
        _make_yolo_result([_COCO_BOAT_CLASS], [0.85], [[50.4, 100.7, 800.1, 600.9]])
    ]
    result = detector.detect(_make_frame())
    assert all(isinstance(v, int) for v in result.box)


def test_detect_returns_correct_box_values() -> None:
    detector = _make_detector()
    detector._model.predict.return_value = [
        _make_yolo_result([_COCO_BOAT_CLASS], [0.85], [[50.0, 100.0, 800.0, 600.0]])
    ]
    result = detector.detect(_make_frame())
    assert result.box == (50, 100, 800, 600)


def test_detect_returns_highest_confidence_box() -> None:
    detector = _make_detector()
    detector._model.predict.return_value = [
        _make_yolo_result(
            [_COCO_BOAT_CLASS, _COCO_BOAT_CLASS],
            [0.5, 0.9],
            [[10.0, 10.0, 100.0, 100.0], [200.0, 200.0, 800.0, 600.0]],
        )
    ]
    result = detector.detect(_make_frame())
    assert result.confidence == pytest.approx(0.9)
    assert result.box == (200, 200, 800, 600)


def test_detect_ignores_non_boat_classes_mixed_with_boats() -> None:
    detector = _make_detector()
    # person (0) with high confidence, boat with lower — should still pick boat
    detector._model.predict.return_value = [
        _make_yolo_result(
            [0, _COCO_BOAT_CLASS],
            [0.99, 0.6],
            [[5.0, 5.0, 50.0, 50.0], [100.0, 100.0, 900.0, 700.0]],
        )
    ]
    result = detector.detect(_make_frame())
    assert result.confidence == pytest.approx(0.6)
    assert result.box == (100, 100, 900, 700)


def test_detect_source_label_passed_through() -> None:
    detector = _make_detector()
    frame = _make_frame()
    frame.source_label = "file:ship.mp4"
    detector._model.predict.return_value = [
        _make_yolo_result([_COCO_BOAT_CLASS], [0.7], [[0.0, 0.0, 100.0, 100.0]])
    ]
    result = detector.detect(frame)
    assert result.source_label == "file:ship.mp4"


def test_detect_confidence_stored_correctly() -> None:
    detector = _make_detector()
    detector._model.predict.return_value = [
        _make_yolo_result([_COCO_BOAT_CLASS], [0.76], [[0.0, 0.0, 100.0, 100.0]])
    ]
    result = detector.detect(_make_frame())
    assert result.confidence == pytest.approx(0.76)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_from_config_reads_threshold(tmp_path: pytest.TempPathFactory) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text(
        "detection:\n  ship:\n    model_path: yolov8n.pt\n    confidence_threshold: 0.45\n"
    )
    with patch(
        "src.detection.ship_detector.ShipDetector._load_model", return_value=MagicMock()
    ):
        detector = ShipDetector.from_config(str(config))
    assert detector._confidence_threshold == pytest.approx(0.45)


def test_from_config_uses_defaults_when_keys_absent(
    tmp_path: pytest.TempPathFactory,
) -> None:
    config = tmp_path / "settings.yaml"
    config.write_text("detection:\n  ship: {}\n")
    with patch(
        "src.detection.ship_detector.ShipDetector._load_model", return_value=MagicMock()
    ):
        detector = ShipDetector.from_config(str(config))
    assert detector._confidence_threshold == pytest.approx(DEFAULT_CONFIDENCE_THRESHOLD)

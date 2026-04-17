from __future__ import annotations

from dataclasses import dataclass

import yaml

from src.ingestion.frames import Frame

# COCO class index for "boat" — covers all watercraft including cargo vessels
_COCO_BOAT_CLASS: int = 8

DEFAULT_MODEL_PATH: str = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.3


@dataclass
class ShipDetection:
    """Bounding box of the most confident ship found in a frame."""

    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in original pixel space
    confidence: float
    source_label: str


class ShipDetector:
    """Detects ships in raw frames using a COCO-pretrained YOLOv8 model.

    No training required — the COCO 'boat' class covers cargo vessels, tankers,
    and container ships as seen from shore-based cameras.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._confidence_threshold = confidence_threshold
        self._model = self._load_model(model_path)

    @classmethod
    def from_config(cls, config_path: str = "config/settings.yaml") -> ShipDetector:
        """Returns a ShipDetector configured from the detection.ship section of settings.yaml."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        ship_cfg = cfg.get("detection", {}).get("ship", {})
        return cls(
            model_path=ship_cfg.get("model_path", DEFAULT_MODEL_PATH),
            confidence_threshold=ship_cfg.get(
                "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
            ),
        )

    def detect(self, frame: Frame) -> ShipDetection | None:
        """Returns the highest-confidence ship detection, or None if no ship is found.

        Runs YOLOv8 on the raw BGR frame and filters for the COCO 'boat' class.
        """
        results = self._model.predict(frame.image, verbose=False)
        return self._extract_best_detection(results[0], frame.source_label)

    def _load_model(self, model_path: str):
        """Loads a YOLO model from the given path, downloading weights if needed."""
        from ultralytics import YOLO

        return YOLO(model_path)

    def _extract_best_detection(
        self, result, source_label: str
    ) -> ShipDetection | None:
        """Returns the highest-confidence boat box from a YOLO result, or None."""
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return None

        best_conf = -1.0
        best_box = None

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = boxes.conf[i].item()
            if cls_id != _COCO_BOAT_CLASS:
                continue
            if conf < self._confidence_threshold:
                continue
            if conf > best_conf:
                best_conf = conf
                x1, y1, x2, y2 = boxes.xyxy[i].tolist()
                best_box = (int(x1), int(y1), int(x2), int(y2))

        if best_box is None:
            return None

        return ShipDetection(
            box=best_box,
            confidence=best_conf,
            source_label=source_label,
        )

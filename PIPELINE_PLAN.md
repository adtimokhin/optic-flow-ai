# Pipeline Implementation Plan

## Overview

This document supersedes the original `IMPLEMENTATION_ORDER.md`. The approach has
changed from training custom models to combining pretrained zero-shot models with a
VLM API, eliminating all model training requirements while preserving or improving
accuracy at each stage.

---

## Architecture

```
Video frame (webcam / YouTube / file)
        │
        ▼
┌───────────────────┐
│  F1  Ingestion    │  FrameEmitter → raw BGR Frame
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  F2a  Preprocess  │  FramePreprocessor → normalised tensor  [BUILT]
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  F2b  Ship detect │  YOLOv8-nano (COCO pretrained) → bounding box
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  F2c  Hull seg    │  SAM (Meta, PyTorch) prompted with YOLO box → hull mask
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  F2d  Waterline   │  OpenCV contours on SAM mask → waterline y-coordinate (px)
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  F2e  Frame gate  │  TensorFlow/Keras binary classifier → accept / reject frame
└────────┬──────────┘  (filters blurry/obstructed frames before expensive VLM call)
         │
         ▼
┌───────────────────┐
│  F3   Draft read  │  VLM API (Claude) + cropped hull image → draft depth estimate
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  F4   Geometry    │  Draft depth + vessel dims + block coefficient → displaced vol
└────────┬──────────┘  → estimated cargo weight (tonnes)
         │
         ▼
┌───────────────────┐
│  F5   AIS lookup  │  VesselFinder / AISHub free tier → vessel ID, origin, dest,
└────────┬──────────┘  commodity type
         │
         ▼
┌───────────────────┐
│  F6   Output      │  JSON record: vessel_id, load_pct, draft_m, origin, dest,
└───────────────────┘  commodity, timestamp
```

---

## Framework split (satisfies NF2)

| Framework   | Used for                                                     |
|-------------|--------------------------------------------------------------|
| PyTorch     | SAM (Segment Anything Model) hull segmentation               |
| TensorFlow  | Frame quality gate — lightweight binary classifier that      |
|             | accepts or rejects a frame before the VLM API is called.    |
|             | Saves cost by filtering blurry, occluded, or dark frames.   |

---

## What is already built

| Component                        | Location                              | Status        |
|----------------------------------|---------------------------------------|---------------|
| File / webcam / YouTube ingest   | `src/ingestion/`                      | Complete      |
| Frame preprocessor               | `src/detection/waterline.py`          | Complete      |
| Segmentation stub                | `src/detection/waterline.py`          | Stub only     |

---

## Implementation stages

### Stage 1 — Ship detection (F2b)
**Goal:** Given a raw frame, return a bounding box `[x1, y1, x2, y2]` around the ship.

- Load `yolov8n.pt` (COCO pretrained, no training needed) via Ultralytics
- Run inference on the raw frame
- Filter detections to the `boat` class (COCO class 8)
- Return the highest-confidence box; return `None` if no ship found
- **New file:** `src/detection/ship_detector.py` — `ShipDetector` class
- **Config:** `detection.ship.confidence_threshold` in `settings.yaml`
- **New dependency:** `ultralytics` (already installed)

### Stage 2 — Hull segmentation + waterline extraction (F2c + F2d)
**Goal:** Given the bounding box from Stage 1, produce a precise hull mask and extract
the waterline y-coordinate in pixels.

- Load SAM (`sam_vit_b` checkpoint — smallest, ~375MB) via `segment-anything`
- Prompt SAM with the YOLO bounding box as a box prompt
- SAM returns a binary mask covering the ship hull
- Extract the waterline:
  - Find the largest contour in the mask
  - The waterline is the lowest horizontal boundary of the hull contour **above** the
    water region — found by scanning from the bottom of the contour upward until the
    mask transitions from hull to non-hull
  - Return the y-coordinate in original frame pixel space
- **New file:** `src/detection/hull_segmentor.py` — `HullSegmentor` class
- **New file:** update `src/detection/waterline.py` — replace stub with
  `WaterlineExtractor` that wraps SAM + contour logic
- **New dependency:** `segment-anything` (Meta, PyTorch-based)
- **Model weights:** `sam_vit_b_01ec64.pth` saved to `models/sam/`

### Stage 3 — Frame quality gate (F2e)
**Goal:** Before making a VLM API call, classify whether the frame is usable —
ship is clearly visible, waterline not obscured by glare, fog, or occlusion.

- Simple MobileNetV2-based binary classifier (accept / reject) built in Keras
- Input: cropped hull region from the SAM mask bounding box, resized to 224×224
- Output: probability of "good frame"
- Initially trained on a small bootstrap dataset (good frames from the pipeline vs.
  deliberately degraded versions — blur, brightness shifts, crops)
- If probability < threshold, frame is skipped; no VLM call is made
- **New file:** `src/detection/frame_quality.py` — `FrameQualityGate` class
- **Model weights:** `models/frame_quality/weights.keras`
- **Note:** This is the TensorFlow component satisfying NF2

### Stage 4 — Draft reading via VLM (F3)
**Goal:** Given the cropped hull image, ask a VLM to read visible draft markings or
Plimsoll line and return an estimated draft depth in metres.

- Crop the hull region from the original full-resolution frame using the SAM mask bbox
- Send to Claude API (claude-sonnet-4-6) as a vision request with a structured prompt
- Parse the response to extract `draft_metres` and `confidence`
- If no markings are visible the VLM returns `null` and the frame is skipped
- **New file:** `src/detection/draft_reader.py` — `DraftReader` class
- **New dependency:** `anthropic` SDK
- **Secret:** `ANTHROPIC_API_KEY` in `.env`

### Stage 5 — Depth geometry (F4)
**Goal:** Convert draft depth in metres to estimated cargo weight in tonnes.

- Inputs: `draft_metres`, vessel `length`, `beam`, `block_coefficient` (from AIS or
  vessel registry lookup)
- Formula: displaced volume = length × beam × draft × block_coefficient
  → displaced weight (tonnes) = displaced volume × seawater density (1.025 t/m³)
  → cargo weight = displaced weight − lightship weight (from vessel specs)
- Pure Python math, no ML
- **New file:** `src/detection/depth_estimation.py` — `DraftGeometry` class
  (file already exists as a stub)

### Stage 6 — AIS integration (F5)
**Goal:** Given a vessel identifier visible in the frame or from an AIS stream,
pull vessel metadata.

- Connect to `aisstream.io` WebSocket to receive live AIS position messages
- Match vessel by MMSI or call sign
- Pull: vessel name, ship type, length, beam, origin port, destination port,
  declared commodity (if available)
- **Existing file:** `src/ais/client.py` (stub)
- **Secret:** `AISSTREAM_API_KEY` in `.env`

### Stage 7 — Structured output (F6)
**Goal:** Assemble and emit a single JSON record per processed vessel.

```json
{
  "vessel_id": "IMO1234567",
  "mmsi": "123456789",
  "timestamp_utc": "2026-04-16T10:00:00Z",
  "draft_metres": 8.4,
  "estimated_cargo_tonnes": 42000,
  "load_pct": 78,
  "origin": "Rotterdam",
  "destination": "Shanghai",
  "commodity": "iron ore",
  "frame_source": "webcam:0"
}
```

- **Existing file:** `src/output/emitter.py` (stub)

---

## New dependencies to add to requirements.txt

```
ultralytics          # YOLOv8 ship detection
segment-anything     # Meta SAM hull segmentation
anthropic            # VLM API for draft reading
```

---

## New secrets to add to .env.example

```
ANTHROPIC_API_KEY=
AISSTREAM_API_KEY=
```

---

## Implementation order

1. **Stage 1** — Ship detector (fast win, unblocks everything visual)
2. **Stage 2** — SAM hull segmentation + waterline extraction (core CV)
3. **Stage 4** — VLM draft reader (can be tested with static images immediately)
4. **Stage 5** — Depth geometry (pure math, no dependencies)
5. **Stage 3** — Frame quality gate (TF classifier, bootstrapped from pipeline output)
6. **Stage 6** — AIS integration (independent data track)
7. **Stage 7** — Structured output (ties everything together)

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| SAM fires on water not hull | Medium | Use YOLO box as hard crop before prompting SAM |
| VLM can't read worn draft markings | Medium | Prompt engineering + fallback to geometry-only estimate |
| COCO YOLO misses ships at unusual angles | Low | Confirmed working on side-on port footage in testing |
| AIS MMSI not matchable from frame alone | High | Treat AIS as optional enrichment, not a hard dependency |
| Frame quality gate bootstrap data too small | Low | Augment with synthetic degradations (blur, brightness) |

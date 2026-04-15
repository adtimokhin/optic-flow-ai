# F1 — Video Ingestion

Captures video frames from a **local webcam** or a **YouTube stream** and emits them as a uniform stream of `Frame` objects for downstream processing.

---

## Module structure

```
src/ingestion/
├── frames.py           # Frame dataclass
├── webcam_capture.py   # WebcamCapture — OpenCV VideoCapture wrapper
├── youtube_capture.py  # YouTubeCapture — yt-dlp + OpenCV wrapper
└── frame_emitter.py    # FrameEmitter — unified iterator, factory methods
```

---

## The Frame dataclass

Every frame emitted by this module is a `Frame` instance defined in `frames.py`:

```python
@dataclass
class Frame:
    image: np.ndarray   # BGR pixel data, shape (H, W, 3)
    timestamp_utc: datetime
    source_label: str   # e.g. "webcam:0" or "youtube:<url>"
```

---

## Quick start

### From a webcam

```python
from src.ingestion import FrameEmitter

emitter = FrameEmitter.from_webcam(device_index=0)
for frame in emitter:
    process(frame.image)   # numpy BGR array
```

### From a YouTube URL

```python
from src.ingestion import FrameEmitter

emitter = FrameEmitter.from_youtube("https://www.youtube.com/watch?v=<id>")
for frame in emitter:
    process(frame.image)
```

### From `config/settings.yaml`

```python
from src.ingestion import FrameEmitter

emitter = FrameEmitter.from_config("config/settings.yaml")
for frame in emitter:
    process(frame.image)
```

Set `ingestion.source` to `"webcam"` or `"youtube"` in the config to switch sources without changing code.

---

## Configuration (`config/settings.yaml`)

```yaml
ingestion:
  source: "webcam"       # "webcam" or "youtube"
  path: ""               # YouTube URL when source is "youtube"
  webcam:
    device_index: 0      # cv2.VideoCapture device index
    frame_width: 1280
    frame_height: 720
    frame_rate: 30
  youtube:
    format: "best[ext=mp4]/best"   # yt-dlp format selector
    frame_rate: 30
```

All thresholds and defaults live here — nothing is hardcoded in the source modules.

---

## Class reference

### `WebcamCapture`

| Method | Description |
|---|---|
| `__init__(device_index, frame_width, frame_height, frame_rate)` | Construct with explicit parameters |
| `from_config(cfg: dict)` | Construct from a parsed `settings.yaml` dict |
| `__enter__` / `__exit__` | Opens/releases the `cv2.VideoCapture` device |
| `__iter__` | Yields `Frame` objects until the device stops returning frames |

Must be used as a context manager before iterating.

### `YouTubeCapture`

| Method | Description |
|---|---|
| `__init__(url, format_spec, frame_rate)` | Construct with explicit parameters |
| `from_config(cfg: dict)` | Construct from a parsed `settings.yaml` dict |
| `__enter__` / `__exit__` | Resolves stream URL via `yt-dlp`, opens/releases `cv2.VideoCapture` |
| `__iter__` | Yields `Frame` objects until the stream ends |
| `_resolve_stream_url(url, format_spec)` | Calls `yt-dlp --get-url` as a subprocess; returns the direct CDN URL |

Must be used as a context manager before iterating.

### `FrameEmitter`

| Method | Description |
|---|---|
| `from_webcam(device_index, frame_width, frame_height, frame_rate)` | Factory — returns emitter over a local webcam |
| `from_youtube(url, format_spec, frame_rate)` | Factory — returns emitter over a YouTube stream |
| `from_config(config_path)` | Factory — reads source type and parameters from `settings.yaml` |
| `__iter__` | Opens the capture, yields all `Frame` objects, then closes the capture |

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | `cv2.VideoCapture` for frame decoding |
| `yt-dlp` | Resolves YouTube URLs to direct CDN stream URLs |
| `numpy` | Frame pixel data (`np.ndarray`) |
| `pyyaml` | Parsing `settings.yaml` in `from_config` |

---

## Tests

Tests live in `tests/` and mirror this module's structure:

```
tests/
├── test_webcam_capture.py    # 6 tests — iteration, shape, release, guards, config
├── test_youtube_capture.py   # 6 tests — iteration, yt-dlp args, release, guards, config
└── test_frame_emitter.py     # 8 tests — factory methods, from_config, field preservation
```

Run all ingestion tests:

```bash
pytest tests/test_webcam_capture.py tests/test_youtube_capture.py tests/test_frame_emitter.py -v
```

Tests use minimal dummy `np.ndarray` tensors and subprocess stubs — no real webcam or network required.

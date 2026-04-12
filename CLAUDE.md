# Claude Code — Project Conventions

## Stack
Python 3.11 · PyTorch (waterline segmentation) · TensorFlow/Keras (Plimsoll classifier) · OpenCV · yt-dlp · aisstream.io WebSocket

## Formatting (automated)
`black` (line-length 88) + `isort` (profile=black) run automatically after every file edit via a PostToolUse hook.  
Config lives in `pyproject.toml`. Do not override formatting manually — let the hook handle it.

## Naming conventions

### Files & modules
- All lowercase, underscores: `video_capture.py`, `depth_estimation.py`
- One clear responsibility per module (matches the folder it lives in)

### Classes
- `PascalCase`: `WaterlineDetector`, `PlimsollClassifier`, `AISClient`, `FrameEmitter`
- Each class owns a single pipeline stage

### Functions & methods
- `snake_case`: `detect_waterline()`, `estimate_draft_depth()`, `fetch_vessel_data()`
- Public methods have a one-line docstring describing return value and side effects
- Private helpers prefixed with `_`: `_preprocess_frame()`, `_parse_ais_message()`

### Variables
- `snake_case` throughout
- Avoid single-letter names outside of tight loops (`i`, `j` in `for` loops are fine)
- Prefer descriptive names: `frame_bgr`, `waterline_y_px`, `draft_meters`

### Constants
- `UPPER_SNAKE_CASE` at module level: `DEFAULT_CONF_THRESHOLD = 0.5`

### Type hints
- All public function signatures must have type hints (parameters + return type)
- Use `from __future__ import annotations` at the top of every module

## Module structure (top-to-bottom order)
```
from __future__ import annotations

# stdlib
# third-party
# internal (src.*)

CONSTANTS = ...

class Foo:
    """One-line class docstring."""

    def public_method(self, ...) -> ...:
        """One-line docstring."""
        ...

    def _private_helper(self, ...) -> ...:
        ...
```

## Folder responsibilities
| Path | Responsibility |
|---|---|
| `src/ingestion/` | Frame capture from webcam / YouTube via `yt-dlp` + OpenCV |
| `src/detection/` | Waterline (PyTorch) and Plimsoll (TF) models + depth geometry |
| `src/ais/` | AIS WebSocket client and data normalization |
| `src/output/` | JSON record assembly and emission |
| `models/` | Saved model weights only — no Python source |
| `config/` | YAML config only — no secrets |
| `tests/` | Unit tests mirroring `src/` structure; filename `test_<module>.py` |

## Tests
- One test file per module: `tests/test_depth_estimation.py` mirrors `src/detection/depth_estimation.py`
- Test function names: `test_<function_under_test>_<scenario>()`
- No mocking of PyTorch/TF models in unit tests — use minimal dummy tensors instead

## Environment / secrets
- All secrets (API keys, URLs) go in `.env` (gitignored); read via `python-dotenv`
- `.env.example` must stay updated with every new key added

## Do not
- Do not commit model weights to git
- Do not hardcode thresholds — put them in `config/settings.yaml`
- Do not mix PyTorch and TensorFlow in the same module

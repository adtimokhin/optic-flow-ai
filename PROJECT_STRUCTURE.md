# Project Structure

## Stack

- **Language:** Python 3.11
- **CV / ML:** PyTorch (waterline segmentation model), TensorFlow/Keras (Plimsoll line classifier), OpenCV (frame I/O and classical CV primitives)
- **AIS:** `aisstream.io` WebSocket client or `requests` against MarineTraffic free tier
- **Video ingestion:** OpenCV `VideoCapture` + `yt-dlp` for YouTube streams
- **Reasoning:** Python is the natural home for both PyTorch and TensorFlow. The two frameworks own distinct model components — segmentation (PyTorch) vs. classification (TensorFlow) — which satisfies NF2 cleanly. OpenCV handles frame I/O without adding a third ML framework.

## Folder Tree

```
optic-flow-ai/
├── src/
│   ├── ingestion/
│   │   └── video_capture.py       # Frame capture from webcam or video file/stream
│   ├── detection/
│   │   ├── waterline.py           # Waterline detection model (PyTorch)
│   │   ├── plimsoll.py            # Plimsoll line detection model (TensorFlow)
│   │   └── depth_estimation.py    # Draft depth geometry from waterline + Plimsoll
│   ├── ais/
│   │   └── client.py              # AIS data fetcher (aisstream.io or MarineTraffic)
│   └── output/
│       └── emitter.py             # Structured JSON record assembly and output
├── models/
│   ├── waterline/                 # PyTorch model weights and config
│   └── plimsoll/                  # TensorFlow model weights and config
├── config/
│   └── settings.yaml              # Thresholds, API keys (refs only), zone config
├── data/
│   └── samples/                   # Sample port footage for development and demo
├── notebooks/
│   └── exploration.ipynb          # Prototyping and model experimentation
├── tests/
│   └── test_depth_estimation.py   # Unit tests for the geometry layer
├── .env.example                   # Required environment variables
├── requirements.txt
└── README.md
```

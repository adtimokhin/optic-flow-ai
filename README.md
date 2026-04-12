# Maritime Ship Draft Estimator

A computer vision pipeline that estimates bulk cargo load on vessels using waterline detection against hull markings, cross-referenced with live AIS transponder data.

## What It Does

- Detects the waterline and Plimsoll line markings on ship hulls from port camera footage
- Estimates draft depth and cargo load percentage from their relative positions
- Cross-references visual estimates with live AIS vessel data (identity, origin, destination, commodity)
- Emits structured records: `vessel_id → estimated load % → origin/destination → commodity type`

## Stack

- **PyTorch** — waterline segmentation model
- **TensorFlow** — Plimsoll line detection model
- **OpenCV** — frame I/O and classical CV primitives
- **yt-dlp** — YouTube/webcam stream ingestion

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your AIS API key to .env
```

## Usage

```bash
python -m src.ingestion.video_capture
```

## Data Sources

- Port camera footage: YouTube, public port webcams
- AIS vessel data: [aisstream.io](https://aisstream.io) (free tier)

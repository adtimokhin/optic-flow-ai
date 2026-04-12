# Features

## Functional Requirements

- **F1: Video ingestion** — Accepts port webcam feeds or YouTube footage of ships at port as input
- **F2: Waterline detection** — Detects the waterline against the ship hull using edge detection + segmentation
- **F3: Plimsoll line detection** — Identifies Plimsoll/draft markings painted on the hull to establish reference points for depth estimation
- **F4: Draft depth estimation** — Computes estimated draft depth (and thus cargo load %) from waterline position relative to Plimsoll markings
- **F5: AIS data integration** — Pulls live vessel data from `aisstream.io` or MarineTraffic API (free tier); provides vessel ID, origin, destination, commodity type
- **F6: Structured output** — Emits records in the form `vessel_id → estimated load % → origin/destination → commodity type`

## Non-Functional Requirements

- **NF1: Free data sources only** — Port camera footage from YouTube/public webcams; AIS from free-tier APIs
- **NF2: Must use PyTorch and TensorFlow** — Both frameworks must be used somewhere in the pipeline (not just one)

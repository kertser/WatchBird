# WatchBird - Drone Biometric Identification System

Real-time, fully on-device multi-modal biometric identification system for drone camera streams running on Raspberry Pi 4 or Jetson-class embedded hardware.

## Overview

WatchBird implements a production-oriented face recognition system that classifies detected persons as:
- **Friendly**: Identified as enrolled identity with high confidence
- **Enemy**: Not identified after configured timeout (unknown persons)

The system is designed for **headless operation** with no cloud dependencies, running entirely on-device.

## Features

- ✅ **Multi-modal architecture** (MVP: face-only, expandable to ReID + Gait)
- ✅ **Real-time tracking** with stable track IDs
- ✅ **Quality gating** for blur, size, and detection confidence
- ✅ **Temporal aggregation** for robust multi-frame decisions
- ✅ **Headless-friendly** with MJPEG streaming for remote debugging
- ✅ **Configurable** via YAML
- ✅ **Type-safe** Python with comprehensive type hints

## Architecture

```
Camera → Detection → Tracking → Embedding → FAISS Search → 
Fusion → Temporal Aggregation → State Machine → Events
```

## Installation

### Prerequisites

- Python 3.11+
- Raspberry Pi OS (Debian) or compatible Linux
- Raspberry Pi 4 or Jetson Nano

### Install Dependencies

```bash
# Install system dependencies (RPi)
sudo apt-get update
sudo apt-get install -y python3-opencv python3-numpy

# Clone repository
git clone <repo-url>
cd WatchBird

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install package
pip install -e .
```

### Download Models

The system requires the following ONNX models:

1. **Face Detector**: YuNet (built-in with OpenCV, or download ONNX)
2. **Person Detector**: YOLOv8n or similar lightweight model
3. **Face Embedder**: MobileFaceNet or ArcFace

Place models in the `models/` directory. See [Model Setup](docs/models.md) for download links.

## Quick Start

### 1. Prepare Enrollment Data

Organize friendly identity images:

```
friendly/
├── alice/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── bob/
│   ├── img_001.jpg
│   └── ...
└── ...
```

### 2. Run Enrollment

```bash
python tools/enroll.py --data-dir friendly --config config.yaml
```

This creates:
- `data/index/face.index` (FAISS index)
- `data/index/meta.jsonl` (metadata)

### 3. Run Runtime Recognition

**With Picamera2 (on Raspberry Pi):**
```bash
python tools/run_runtime.py --backend picamera2 --config config.yaml
```

**With video file (for testing):**
```bash
python tools/run_runtime.py --backend video_file --video test_video.mp4 --config config.yaml
```

### 4. View Debug Stream

Open browser and navigate to:
```
http://<raspberry-pi-ip>:8080/stream
```

## Configuration

Edit `config.yaml` to customize:

- Camera settings (resolution, FPS)
- Detection thresholds
- Tracking parameters
- Quality gating thresholds
- Fusion and decision thresholds
- Model paths

## Project Structure

```
.
├── config.yaml              # Configuration file
├── src/WatchBird/            # Main package
│   ├── camera/              # Camera backends (Picamera2, video file)
│   ├── detect/              # Person & face detectors
│   ├── track/               # Object tracker
│   ├── embed/               # Face/ReID embedders
│   ├── index/               # FAISS index wrapper
│   ├── fusion/              # Multi-modal fusion
│   ├── runtime/             # State machine & pipeline
│   ├── stream/              # MJPEG server
│   └── utils/               # Utilities
├── tools/                   # CLI tools
│   ├── enroll.py            # Enrollment tool
│   ├── run_runtime.py       # Runtime recognition
│   └── calibrate_thresholds.py  # Threshold calibration
├── data/index/              # FAISS indexes
├── models/                  # ONNX models
└── friendly/                # Enrollment images

```

## Development

### Run Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Format code
black src/

# Type checking
mypy src/

# Linting
ruff check src/
```

## Performance

**Raspberry Pi 4 Targets:**
- FPS: 5-10 fps
- Latency: < 2 seconds detection → classification
- Memory: < 1GB RAM

## Roadmap

### MVP (Current)
- [x] Core infrastructure
- [x] Camera backends
- [x] Detection & tracking
- [ ] Face embedding
- [ ] FAISS integration
- [ ] State machine
- [ ] MJPEG streaming
- [ ] Enrollment tool
- [ ] Runtime tool

### Post-MVP
- [ ] Body re-identification (ReID)
- [ ] Gait/motion signatures
- [ ] Multi-modal fusion
- [ ] Threshold calibration tool
- [ ] GPU acceleration (Jetson)
- [ ] UDP event output

## License

[Specify License]

## Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Acknowledgments

- YuNet face detector from OpenCV
- SORT tracker algorithm
- FAISS vector search library


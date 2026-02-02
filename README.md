# WatchBird

Real-time face recognition and person classification for embedded devices.

```
┌─────────────────────────────────────────────────────────────────┐
│                         WatchBird                               │
│                                                                 │
│   Camera → Detect → Track → Recognize → FRIENDLY/ENEMY          │
│                     ↓                                           │
│              Body Detection → Segmentation → Classification     │
│                                              (IDF/ARMED/CIV)    │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- **Face Recognition**: SCRFD detection + ArcFace embeddings + PLDA scoring
- **Body Detection**: YOLOv8 human detection with colored contours
- **Person Classification**: CLIP-based soldier/armed/civilian detection
- **GPU Accelerated**: DirectML (AMD/Intel) and CUDA (NVIDIA) support
- **Real-time Streaming**: MJPEG stream at `http://localhost:8080/stream`

## Quick Start

```bash
# Install
git clone <repo> && cd WatchBird
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -e .
python tools/download_models.py

# Export CLIP to ONNX (optional, for person classification) or use quantized model
python tools/export_clip_onnx.py

# Enroll a person
python tools/auto_enroll.py --person mike --auto-enroll

# Run
python tools/run_runtime.py
```

View stream: `http://localhost:8080/stream`

## How It Works

```
┌──────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────┐
│  Camera  │───>│ Detector │───>│   Tracker     │───>│ Embedder │
│  Frame   │    │  SCRFD   │    │  SORT-based   │    │ ArcFace  │
└────┬─────┘    └──────────┘    └───────────────┘    └────┬─────┘
     │              │ GPU                                 │
     │              ▼                                     ▼
     │         ┌──────────┐    ┌───────────────┐    ┌──────────┐
     │         │  Body    │    │  Aggregator   │<───│  Scorer  │
     │         │ YOLOv8   │    │  Multi-frame  │    │FAISS+PLDA│
     │         └────┬─────┘    └───────┬───────┘    └──────────┘
     │              │                  │
     ▼              ▼                  ▼
┌──────────┐   ┌───────────┐    ┌───────────────┐
│  CLIP    │<──│Segmenter  │    │ State Machine │
│Classifier│   │PP-HumanSeg│    │SUSPECT→FRIEND │
└────┬─────┘   └────┬──────┘    └───────┬───────┘
     │              │                   │
     └──────────────┴───────────────────┘
                    │
                    ▼
             ┌────────────┐
             │  Annotated │
             │   Stream   │
             └────────────┘
```

### Pipeline Components

| Component | Model | Size | Backend |
|-----------|-------|------|---------|
| Face Detector | SCRFD 2.5G | 3.1 MB | DirectML GPU |
| Face Embedder | ArcFace R100 | 249 MB | DirectML GPU |
| Body Detector | YOLOv8n | 12 MB | DirectML GPU |
| Human Segmenter | PP-HumanSeg | 168 MB | DirectML GPU |
| Person Classifier | CLIP ViT-B/32 (INT8) | 85 MB | DirectML GPU |

## Classification States

```
         ┌─────────────────────────────────┐
         │                                 │
         ▼                                 │
    ┌─────────┐   confidence >= 0.90   ┌───┴─────┐   cumulative   ┌───────────┐
───>│ SUSPECT │───────────────────────>│FRIENDLY │───────────────>│ CONFIRMED │
    └────┬────┘   + consistency >= 10  └─────────┘   >= 0.95      └───────────┘
         │                                                         (tracking only)
         │  timeout (15s)
         ▼
    ┌─────────┐
    │  ENEMY  │
    └─────────┘
```

| State | Color | Meaning |
|-------|-------|---------|
| SUSPECT | 🟡 Yellow | Unknown, collecting data |
| FRIENDLY | 🟢 Light Green | Matched enrolled person |
| CONFIRMED | 🟢 Bright Green | High confidence, tracking only |
| ENEMY | 🔴 Red | Unknown person (timeout) |

## Person Classification (CLIP)

Classify detected persons as soldiers, armed civilians, or unarmed civilians:

| Classification | Label | Color | Description |
|----------------|-------|-------|-------------|
| Soldier | IDF | 🟢 Green | Military uniform (OD green) |
| Armed Civilian | ARMED | 🔴 Red | Civilian with visible weapon |
| Unarmed Civilian | CIV | 🔵 Cyan | Regular civilian |

### CLIP Model Options

| Version | Size | Startup | Export Command |
|---------|------|---------|----------------|
| **ONNX INT8** (default) | 85 MB | Fast (~1s) | `python tools/export_clip_onnx.py` |
| PyTorch | 600 MB | Slow (~10s) | Auto-download from HuggingFace |

```yaml
detection:
  person_classification: true
  clip_onnx: true              # Use ONNX (recommended) or PyTorch
  armed_threshold: 0.6         # Min confidence for ARMED (higher = stricter)
  soldier_threshold: 0.5       # Min confidence for IDF
```

## Configuration

Key settings in `config.yaml`:

```yaml
detection:
  detector_type: scrfd            # GPU-accelerated face detection
  face_conf_threshold: 0.5
  body_detection: true            # Enable body detection
  segmentation: true              # Enable body contours
  person_classification: true     # Enable CLIP classification
  clip_onnx: true                 # Use ONNX model (85MB vs 600MB)

thresholds:
  t_accept: 0.90                  # Min score for FRIENDLY
  t_margin: 0.005                 # Min margin between candidates
  t_timeout: 15.0                 # Seconds before ENEMY
  confirm_threshold: 0.95         # Cumulative confidence for CONFIRMED

models:
  face_detector: models/scrfd_2.5g.onnx
  face_embedder: models/arcface_r100.onnx
  body_detector: models/yolov8n.onnx
  human_segmenter: models/human_seg.onnx
  clip_vision: models/clip_vision_int8.onnx
  clip_embeddings: models/clip_text_embeddings.npy
```

## Performance

Typical FPS at 640x480 resolution (slow webcam):

| Configuration | FPS |
|---------------|-----|
| Face only | ~15-20 |
| Face + Body | ~12-15 |
| Face + Body + Segmentation | ~10-13 |
| Full pipeline (+ CLIP) | ~10-13 |

### Optimization Tips

```yaml
# For higher FPS
camera:
  resolution: [640, 480]        # Lower resolution

detection:
  classification_interval: 10   # Classify less frequently
  segmentation: false           # Disable contours
```

## Tools

| Tool | Purpose |
|------|---------|
| `run_runtime.py` | Main application |
| `auto_enroll.py` | Capture & enroll faces |
| `enroll.py` | Enroll from photos |
| `export_clip_onnx.py` | Export CLIP to ONNX (85MB) |
| `download_models.py` | Download required models |
| `calibrate_thresholds.py` | Find optimal thresholds |

## Troubleshooting

**CLIP not loading?**
```bash
# Export ONNX model
python tools/export_clip_onnx.py
```

**Low FPS?**
- Reduce resolution to 640x480
- Increase `classification_interval` to 10+
- Disable segmentation

**False positives in face recognition?**
- Increase `t_accept` threshold (try 0.92-0.95)
- Ensure good lighting during enrollment
- Enroll more varied poses

**False ARMED classification?**
- Increase `armed_threshold` (try 0.7-0.8)

## Documentation

- [Configuration Reference](docs/Configuration.md)
- [Body Detection & Classification](docs/BodyDetection.md)
- [Architecture Overview](docs/Architecture.md)
- [Deployment Guide](docs/Deployment.md)

## License

MIT

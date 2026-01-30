# WatchBird

Real-time face recognition for embedded devices.

```
┌─────────────────────────────────────────────────────────────────┐
│                         WatchBird                               │
│                                                                 │
│   Camera → Detect → Track → Recognize → FRIENDLY/ENEMY          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install
git clone <repo> && cd WatchBird
python -m venv .venv && .venv\Scripts\activate  # Windows
pip install -e .
python tools/download_models.py

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
│  Frame   │    │  YuNet   │    │  SORT-based   │    │ ArcFace  │
└──────────┘    └──────────┘    └───────────────┘    └────┬─────┘
                                                          │
                                                          ▼
┌──────────┐    ┌──────────┐    ┌───────────────┐    ┌──────────┐
│  Output  │<───│  State   │<───│  Aggregator   │<───│  Scorer  │
│ FRIENDLY │    │ Machine  │    │  Multi-frame  │    │FAISS+PLDA│
└──────────┘    └──────────┘    └───────────────┘    └──────────┘
```

### Key Concepts

1. **Detection**: Find faces in each frame (YuNet)
2. **Tracking**: Assign consistent IDs across frames (SORT)
3. **Embedding**: Extract 512-dim face vector (MobileFaceNet)
4. **Aggregation**: Collect multiple embeddings, compute quality-weighted centroid
5. **Scoring**: Match against enrolled faces (FAISS + PLDA)
6. **State Machine**: SUSPECT → FRIENDLY/ENEMY based on confidence

## Embedding Aggregation

Instead of matching each frame individually (noisy), we aggregate multiple embeddings:

```
Frame 1 ──▶ Embedding 1 ─┐
Frame 2 ──▶ Embedding 2 ─┼──> Quality-Weighted ──> Match vs
Frame 3 ──▶ Embedding 3 ─┤    Centroid            Database
...                      │Frame N ──▶ Embedding N ─┘
                         │
                   Outliers filtered
                   Low-quality rejected
```

Quality factors:
- Detection confidence (40%)
- Blur score (25%) - sharper is better
- Brightness (15%) - optimal ~0.5
- Face size (20%) - larger is better

## Classification States

```
         ┌─────────────────────────────────┐
         │                                 │
         ▼                                 │
    ┌─────────┐   confidence >= 0.78   ┌───┴─────┐
───>│ SUSPECT │───────────────────────>│FRIENDLY │
    └────┬────┘   + margin >= 0.20     └─────────┘
         │        + consistency >= 12
         │
         │  timeout (15s)
         ▼
    ┌─────────┐
    │  ENEMY  │
    └─────────┘
```

| State    | Color  | Meaning                  |
|----------|--------|--------------------------|
| SUSPECT  | Yellow | Unknown, collecting data |
| FRIENDLY | Green  | Matched enrolled person  |
| ENEMY    | Red    | Unknown person (timeout) |

## Performance Tuning

### Auto-Resolution (Recommended)

Automatically find the highest resolution that achieves your target FPS:

```bash
# Test auto-resolution tuning
python tools/test_resolution.py --target-fps 10.0

# Enable in config.yaml
camera:
  auto_resolution: true
  target_fps: 10.0
  min_fps: 8.0
```

### Manual Optimization

For high-resolution cameras (1080p+):

```yaml
detection:
  max_detection_size: 640  # Downscale to 640px for detection (3-5x faster)

camera:
  resolution: [1280, 720]  # Manual resolution setting
```

**Expected FPS by resolution:**
- 1920x1080 with `max_detection_size: 640` → ~8-10 FPS
- 1280x720 with `max_detection_size: 640` → ~12-15 FPS  
- 640x480 (no downscaling) → ~20-25 FPS

## Configuration

````

```yaml
thresholds:
  t_accept: 0.78       # Min score for FRIENDLY
  t_margin: 0.20       # Min margin between candidates
  t_timeout: 15.0      # Seconds before ENEMY

fusion:
  embedding_window: 20      # Embeddings to collect
  embedding_min: 6          # Min before matching
  consistency_count: 12     # Consistent frames needed

models:
  face_embedder: models/mobilefacenet.onnx
```

## Tools

| Tool                      | Purpose                  |
|---------------------------|--------------------------|
| `run_runtime.py`          | Main application         |
| `auto_enroll.py`          | Capture & enroll faces   |
| `enroll.py`               | Enroll from photos       |
| `calibrate_thresholds.py` | Find optimal thresholds  |
| `evaluate_model.py`       | Test model quality       |
| `check_photos.py`         | Verify enrollment photos |

## Troubleshooting

**Low recognition accuracy?**
- Run `python tools/calibrate_thresholds.py --test-all-models`
- Ensure good lighting during enrollment
- Capture varied poses/expressions

**False positives?**
- Increase `t_accept` threshold
- Increase `t_margin` for better separation
- Check embedding variance with `evaluate_model.py`

**Slow recognition?**
- Reduce `embedding_min` (faster but less accurate)
- Increase `embedding_sample_interval`

## License

MIT

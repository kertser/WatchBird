# WatchBird

Real-time face recognition system for embedded devices (Raspberry Pi 4 / Jetson Nano).

## Features

- **On-device processing** - No cloud dependencies
- **Real-time tracking** - Stable track IDs with re-identification
- **Quality gating** - Filters blur, size, and low confidence
- **Rotated face support** - Handles tilted heads via landmark alignment
- **MJPEG streaming** - Remote debugging via browser

## Quick Start

### 1. Install

```bash
git clone <repo-url> && cd WatchBird
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python tools/download_models.py
```

### 2. Enroll People

```bash
# Auto-enrollment (recommended) - captures until 85% confidence
python tools/auto_enroll.py --person mike --auto-enroll

# Or manual capture
python tools/capture_and_enroll.py --person mike --count 15 --auto-enroll
```

### 3. Run

```bash
# USB camera (parallel pipeline - recommended, faster FPS)
python tools/run_runtime_parallel.py --backend usb

# USB camera (sequential pipeline - simpler, lower FPS)
python tools/run_runtime.py --backend usb

# Raspberry Pi camera
python tools/run_runtime_parallel.py --backend picamera2
```

### 4. View Stream

Open `http://<device-ip>:8080/stream` in browser.

## Classification States

| State | Color | Meaning |
|-------|-------|---------|
| SUSPECT | Yellow | Analyzing (< 5 seconds) |
| FRIENDLY | Green | Identified as enrolled person |
| ENEMY | Red | Unknown after timeout |

## Project Structure

```
WatchBird/
├── config.yaml          # Configuration
├── models/              # ONNX models (yunet, mobilefacenet)
├── friendly/            # Enrollment photos by person
├── data/index/          # FAISS index + metadata
├── src/watchbird/       # Main package
└── tools/               # CLI tools
```

## Configuration

Key settings in `config.yaml`:

```yaml
thresholds:
  t_accept: 0.65      # Min similarity for FRIENDLY
  t_margin: 0.10      # Min margin between best/second match
  t_timeout: 5.0      # Seconds before ENEMY classification

fusion:
  embedding_sample_interval: 3  # Process every Nth frame (save compute)
```

## Tools

| Tool | Purpose |
|------|---------|
| `auto_enroll.py` | Smart enrollment with confidence testing |
| `capture_and_enroll.py` | Manual photo capture + enrollment |
| `enroll.py` | Build index from existing photos |
| `run_runtime_parallel.py` | **Main recognition pipeline (parallel, faster)** |
| `run_runtime.py` | Recognition pipeline (sequential, simpler) |
| `check_photos.py` | Verify enrollment photo quality |

## Performance Tips

### Parallel vs Sequential Pipeline

| Pipeline | FPS | Use Case |
|----------|-----|----------|
| `run_runtime_parallel.py` | **15-17 FPS** | Production (recommended) |
| `run_runtime.py` | 9-10 FPS | Debugging, simpler code |

The parallel pipeline overlaps CPU detection with GPU embedding for ~70% faster processing.

### Other optimizations:
- Use `embedding_sample_interval: 3-5` for low-power devices
- Lower resolution in `config.yaml` if needed
- Use `mobilefacenet.onnx` (faster) vs `arcface_r100.onnx` (more accurate)

## Troubleshooting

**Wrong person detected:**
- Re-enroll with more varied photos (angles, lighting, expressions)
- Use `python tools/check_photos.py --data-dir friendly` to verify quality

**Low FPS:**
- Increase `embedding_sample_interval`
- Reduce camera resolution

**Face not detected when tilted:**
- System auto-tries rotated detection
- Ensure good lighting

## License

MIT

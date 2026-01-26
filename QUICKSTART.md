# Quick Start

## 1. Install

```bash
# Clone and setup
git clone <repo-url> && cd WatchBird
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

# Download models
python tools/download_models.py

# Test installation
python tools/test_installation.py
```

### GPU Acceleration (Optional but Recommended)

For faster and more accurate inference, install GPU support:

```bash
# NVIDIA GPU (CUDA) - Best performance
pip install onnxruntime-gpu

# Windows with any GPU (AMD, Intel, or NVIDIA via DirectML)
pip install onnxruntime-directml
```

> **Note:** GPU acceleration is enabled by default in `config.yaml`. Set `use_gpu: false` under `inference` to disable it.

## 2. Enroll People

### Option A: Auto-Enrollment (Recommended)

Automatically captures photos until recognition confidence is high enough:

```bash
python tools/auto_enroll.py --person yourname --auto-enroll
```

During capture:
- Position face in camera view
- Slowly move head: left, right, up, down
- Change expressions: neutral, smiling
- System stops when confidence reaches 85%

### Option B: Manual Capture

```bash
python tools/capture_and_enroll.py --person yourname --count 15 --auto-enroll
```

Press SPACE to capture each photo.

## 3. Run Recognition

```bash
# USB camera
python tools/run_runtime.py --backend usb

# Raspberry Pi camera
python tools/run_runtime.py --backend picamera2

# Test with video file
python tools/run_runtime.py --backend video_file --video test.mp4
```

## 4. View Stream

Open in browser: `http://localhost:8080/stream`

## Verify Enrollment

```bash
# Check enrolled people
python tools/check_index.py

# Check photo quality
python tools/check_photos.py --data-dir friendly
```

## Re-Enroll

If recognition is poor:

```bash
# Delete old index
Remove-Item -Recurse data\index  # Windows
rm -rf data/index                 # Linux

# Re-enroll all people
python tools/enroll.py --data-dir friendly
```

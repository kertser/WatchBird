# Quick Start Guide

This guide will help you get WatchBird up and running quickly on your Raspberry Pi 4.

## Prerequisites

- Raspberry Pi 4 (4GB+ RAM recommended)
- Raspberry Pi OS (Debian-based)
- Python 3.11+
- CSI camera (optional, can use video files for testing)

## Installation

### 1. System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv git
sudo apt-get install -y libopencv-dev python3-opencv
```

### 2. Clone Repository

```bash
git clone <your-repo-url>
cd WatchBird
```

### 3. Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Python Packages

```bash
pip install --upgrade pip
pip install -e .
```

### 5. Test Installation

```bash
python tools/test_installation.py
```

You should see all tests pass. If any fail, check the error messages.

## Model Setup

### Download Face Embedder Model

The minimum required model is the face embedder:

```bash
# Create models directory
mkdir -p models

# Download a lightweight face recognition model
# Option 1: Use a pre-trained ArcFace model (recommended)
wget https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx \
     -O models/mobilefacenet.onnx

# OR Option 2: If you have your own model, copy it here
cp /path/to/your/model.onnx models/mobilefacenet.onnx
```

See `docs/models.md` for more model options.

## Enrollment

### Option 1: Interactive Photo Capture (Recommended) ⭐

Use the interactive capture tool to take photos:

```bash
# Step 1: Capture 10 photos of a person
python tools/capture_and_enroll.py --person mike --count 10

# Step 2: Enroll manually (recommended for first time)
python tools/enroll.py --data-dir friendly --config config.yaml
```

**Or capture and auto-enroll in one step:**
```bash
python tools/capture_and_enroll.py --person mike --count 10 --auto-enroll
```

**During capture:**
- Press **SPACE** to take a photo when ready
- Press **Q** to finish early
- Press **ESC** to cancel
- The tool will automatically select the largest face if multiple people are in frame
- Photos are ranked by quality and only the best 10 are saved

**Tips for best results:**
- Vary head position: straight, left, right, up, down
- Vary expressions: neutral, smiling, serious
- Ensure good lighting on face
- Keep only ONE person in frame for best results
- Stay at similar distance from camera

**With different camera backends:**
```bash
# USB camera (default, device 0)
python tools/capture_and_enroll.py --person alice --backend usb --device-id 0 --auto-enroll

# USB camera (device 1)
python tools/capture_and_enroll.py --person alice --backend usb --device-id 1 --auto-enroll

# Raspberry Pi camera
python tools/capture_and_enroll.py --person alice --backend picamera2 --auto-enroll

# Video file (for testing)
python tools/capture_and_enroll.py --person alice --backend video_file --video test.mp4
```

### Option 2: Manual Photo Preparation

If you already have photos, prepare them manually:

```bash
mkdir -p friendly/alice friendly/bob friendly/charlie

# Copy images to each person's folder
# Each person should have 10-15 face images from different angles
# IMPORTANT: Use photos with ONLY ONE PERSON per image!
cp /path/to/alice_photos/*.jpg friendly/alice/
cp /path/to/bob_photos/*.jpg friendly/bob/
```

**Check photo quality before enrollment:**
```bash
python tools/check_photos.py --data-dir friendly
```

This shows which photos have:
- ✅ Single face (good for enrollment)
- ⚠️ Multiple faces (will use first/largest - may be wrong person!)
- ✗ No faces (unusable)

### Run Enrollment

```bash
# After capturing photos or preparing them manually
python tools/enroll.py --data-dir friendly --config config.yaml
```

This will:
- Detect faces in all images
- Extract embeddings
- Build FAISS index in `data/index/face.index`
- Save metadata in `data/index/meta.jsonl`

Expected output:
```
INFO - Found 10 images for alice
INFO - Found 10 images for bob
INFO - Processing alice...
INFO - Enrolled 10 face embeddings for alice
...
INFO - Total: 20 embeddings from 2 identities
INFO - Saved FAISS index to data/index/face.index
INFO - Enrollment complete!
```

**⚠️ Important Notes:**
- Use photos with **ONLY ONE person** visible for best results
- Multiple faces in one photo will use the **first or largest** detected face (may not be the target person!)
- Aim for **10-15 high-quality photos** per person
- Photos with poor quality will be skipped
- Run `check_photos.py` to verify photo quality before enrollment

## Runtime Recognition

### Test with USB Camera (Default)

```bash
# Run with default USB camera (device 0)
python tools/run_runtime.py --backend usb --config config.yaml

# Or specify a different camera device ID
python tools/run_runtime.py --backend usb --device-id 1 --config config.yaml
```

### Test with Video File

```bash
# Download or prepare a test video with the enrolled people
python tools/run_runtime.py \
    --backend video_file \
    --video /path/to/test_video.mp4 \
    --config config.yaml
```

### Run with Raspberry Pi Camera

```bash
python tools/run_runtime.py \
    --backend picamera2 \
    --config config.yaml
```

### View Debug Stream

While the runtime is running, open a browser and navigate to:

```
http://<raspberry-pi-ip>:8080/stream
```

You should see:
- **Green boxes**: FRIENDLY persons (identified)
- **Yellow boxes**: SUSPECT persons (being evaluated)
- **Red boxes**: ENEMY persons (unknown after timeout)

### Monitor Events

The runtime outputs JSON events to stdout:

```json
{"ts": 1768600000.123, "track_id": 1, "state": "SUSPECT", "person_id": null, "confidence": 0.0, "modalities_used": {}}
{"ts": 1768600002.456, "track_id": 1, "state": "FRIENDLY", "person_id": "alice", "confidence": 0.83, "modalities_used": {"face": 0.7}}
```

You can redirect to a file:
```bash
python tools/run_runtime.py --backend video_file --video test.mp4 > events.jsonl
```

## Configuration

Edit `config.yaml` to adjust:

### Detection Thresholds
```yaml
detection:
  face_conf_threshold: 0.7  # Lower = more detections, more false positives
```

### Recognition Thresholds
```yaml
thresholds:
  t_accept: 0.50    # Minimum similarity for FRIENDLY
  t_margin: 0.03    # Margin between best and second-best (lower for small enrollment)
  t_timeout: 10.0   # Seconds before SUSPECT → ENEMY
```

**Note:** With only 2-3 people enrolled, use `t_margin: 0.03`. With 10+ people, you can increase to `0.07-0.10`.

### Quality Gating
```yaml
quality:
  min_face_quality: 0.4  # Minimum quality score (0-1)
  min_bbox_size: 50      # Minimum face size in pixels
```

### Temporal Aggregation
```yaml
fusion:
  window_size: 10        # Rolling window size (frames)
  consistency_count: 3   # Required consistency (frames) - lower for faster recognition
```

## Troubleshooting

### "Failed to load face detector"
- YuNet is built into OpenCV 4.9+. If using older version, download manually:
  ```bash
  wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
       -O models/yunet.onnx
  ```

### "Failed to load FAISS index"
- Make sure you ran enrollment first: `python tools/enroll.py`
- Check that `data/index/face.index` exists

### Low FPS on Raspberry Pi
- Reduce camera resolution in `config.yaml`:
  ```yaml
  camera:
    resolution: [320, 240]  # Smaller = faster
  ```
- Lower detection thresholds to skip more frames

### Picamera2 not available
- Install on Raspberry Pi:
  ```bash
  sudo apt-get install -y python3-picamera2
  ```
- Or use video file backend for testing

### False FRIENDLY detections (unknown person recognized as known)
- Increase `t_accept` threshold (e.g., 0.60)
- Increase `t_margin` (e.g., 0.10 - but only if you have 5+ people enrolled)
- Increase `consistency_count` (e.g., 5)

### Too many ENEMY (known person not recognized)
- Decrease `t_accept` threshold (e.g., 0.45)
- **Decrease `t_margin`** (e.g., 0.01-0.02 for 2-3 people enrolled) ⭐ MOST COMMON FIX
- Increase `t_timeout` to give more time (e.g., 15.0)
- Decrease `consistency_count` for faster recognition (e.g., 2-3)
- Add more enrollment images per person
- Run `python tools/check_index.py` to diagnose margin issues

## Next Steps

Once the MVP is working:

1. **Add more identities**: Enroll 20-200 people
2. **Test in real scenarios**: Test with drone camera footage
3. **Tune thresholds**: Use `tools/calibrate_thresholds.py` (future)
4. **Add ReID**: Implement body re-identification for multi-modal fusion
5. **Optimize performance**: Profile and optimize for target hardware

## Support

For issues and questions:
- Check logs for error messages
- Run `tools/test_installation.py` to verify setup
- See `README.md` for full documentation
- Check `docs/models.md` for model troubleshooting


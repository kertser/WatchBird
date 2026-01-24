# Models

## Required Models

Place these in the `models/` directory:

| Model | File | Purpose |
|-------|------|---------|
| YuNet | `yunet.onnx` | Face detection |
| MobileFaceNet | `mobilefacenet.onnx` | Face embedding (fast) |
| ArcFace R100 | `arcface_r100.onnx` | Face embedding (accurate) |

## Download

```bash
python tools/download_models.py
```

Or manually:

```bash
mkdir -p models

# Face detector (YuNet)
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -O models/yunet.onnx

# Face embedder (MobileFaceNet - faster, smaller)
wget https://github.com/onnx/models/raw/main/validated/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx \
  -O models/mobilefacenet.onnx
```

## Model Selection

In `config.yaml`:

```yaml
models:
  face_detector: "models/yunet.onnx"
  face_embedder: "models/mobilefacenet.onnx"  # Fast
  # face_embedder: "models/arcface_r100.onnx"  # More accurate
```

## Performance

| Model | Size | Speed (RPi4) | Accuracy |
|-------|------|--------------|----------|
| MobileFaceNet | ~5 MB | ~50ms | Good |
| ArcFace R100 | ~250 MB | ~500ms | Best |

Use MobileFaceNet for real-time on Raspberry Pi.

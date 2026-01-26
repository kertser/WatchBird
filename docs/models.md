# Models

## Required Models

| Model | File | Purpose |
|-------|------|---------|
| YuNet | `yunet.onnx` | Face detection |
| MobileFaceNet | `mobilefacenet.onnx` | Face embedding (fast) |
| ArcFace R100 | `arcface_r100.onnx` | Face embedding (accurate) |

## Download

```bash
python tools/download_models.py
```

## Model Selection

Configure in `config.yaml`:

```yaml
models:
  face_detector: models/yunet.onnx
  face_embedder: models/arcface_r100.onnx    # Accurate, slower
  # face_embedder: models/mobilefacenet.onnx  # Fast, good accuracy
```

## Performance Comparison

| Model | Size | Speed (RPi4) | Speed (GPU) | Accuracy |
|-------|------|--------------|-------------|----------|
| MobileFaceNet | ~5 MB | ~50ms | ~5ms | Good |
| ArcFace R100 | ~250 MB | ~500ms | ~15ms | Best |

**Recommendation:**
- Desktop/GPU: Use `arcface_r100.onnx` for best accuracy
- Raspberry Pi: Use `mobilefacenet.onnx` for real-time performance

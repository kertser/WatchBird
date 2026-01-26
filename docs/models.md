# Models

## Required Models

| Model | File | Purpose |
|-------|------|---------|
| YuNet | `yunet.onnx` | Face detection |
| MobileFaceNet | `mobilefacenet.onnx` | Face embedding (fast) |
| ArcFace R100 | `arcface_r100.onnx` | Face embedding (accurate) |

### Optional: AdaFace Models

AdaFace provides state-of-the-art recognition accuracy, especially for low-quality images.

| Model | File | Purpose |
|-------|------|---------|
| AdaFace-R100 | `adaface_r100.onnx` | Face embedding (SOTA accuracy) |
| AdaFace-ViT | `adaface_vit.onnx` | Face embedding (Vision Transformer) |

**Source:** https://github.com/jahongir7174/FaceID/tree/master/weights

**Note:** AdaFace is for **recognition only** (embedding extraction). You still need YuNet for face detection.

## Download

```bash
python tools/download_models.py
```

For AdaFace, manually download from the FaceID repo and place in `models/` folder.

## Model Selection

Configure in `config.yaml`:

```yaml
models:
  face_detector: models/yunet.onnx
  face_embedder: models/arcface_r100.onnx    # Accurate, slower
  # face_embedder: models/mobilefacenet.onnx  # Fast, good accuracy
  # face_embedder: models/adaface_r100.onnx   # SOTA accuracy (if available)
```

## Performance Comparison

| Model | Size | Speed (RPi4) | Speed (GPU) | Accuracy | Notes |
|-------|------|--------------|-------------|----------|-------|
| MobileFaceNet | ~5 MB | ~50ms | ~5ms | Good | Best for real-time |
| ArcFace R100 | ~250 MB | ~500ms | ~15ms | Very Good | Balanced choice |
| AdaFace R100 | ~250 MB | ~500ms | ~15ms | Best | SOTA, handles low-quality faces |
| AdaFace ViT | ~90 MB | ~600ms | ~20ms | Best | Vision Transformer architecture |

**Recommendation:**
- Desktop/GPU: Use `adaface_r100.onnx` or `arcface_r100.onnx` for best accuracy
- Raspberry Pi: Use `mobilefacenet.onnx` for real-time performance
- Low-quality/surveillance: AdaFace excels with challenging images (blur, low-res, occlusion)

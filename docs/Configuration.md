# Configuration Reference

All settings in `config.yaml`.

## Camera

```yaml
camera:
  backend: usb              # usb | picamera2 | video_file
  resolution: [640, 480]    # Width x Height
  fps: 30                   # Target frame rate
  video_path: null          # Path for video_file backend
```

## Detection

```yaml
detection:
  face_conf_threshold: 0.7  # Min confidence (0-1)
```

## Tracking

```yaml
tracking:
  max_age: 30               # Frames before track dies
  min_hits: 3               # Hits to confirm track
  iou_threshold: 0.3        # Matching threshold
```

## Quality Filtering

```yaml
quality:
  min_face_quality: 0.3     # Min for processing
  min_enroll_quality: 0.8   # Min for enrollment
  min_bbox_size: 40         # Min face size (pixels)
  blur_threshold: 100.0     # Laplacian variance threshold
```

## Embedding Aggregation

```yaml
fusion:
  embedding_window: 20           # Max embeddings per track
  embedding_min: 6               # Min before matching
  embedding_quality_threshold: 0.4    # Min combined quality
  embedding_outlier_threshold: 0.25   # Outlier distance
  embedding_sample_interval: 2   # Sample every N frames
```

### Quality Score Formula

```
combined_quality = 0.40 × detection_confidence
                 + 0.25 × blur_score
                 + 0.15 × brightness_score
                 + 0.20 × size_score
```

## Decision Thresholds

```yaml
thresholds:
  t_accept: 0.78            # Min score for FRIENDLY
  t_margin: 0.20            # Min margin (best - second)
  t_timeout: 15.0           # Seconds before ENEMY
  identity_switch_margin: 0.25  # Extra margin to switch ID
```

## State Machine

```yaml
fusion:
  window_size: 30           # Observation window
  consistency_count: 12     # Consistent frames needed
  confidence_decay_threshold: 25  # Frames before decay
```

## PLDA Scoring (Likelihood Ratio)

PLDA provides probabilistic scoring using log-likelihood ratios (LLR) for robust unknown rejection:
- **LLR > 0**: More likely same person
- **LLR < 0**: More likely different person (unknown)

```yaml
plda:
  enabled: true                   # Use PLDA second stage
  model_path: data/index/plda.npz
  
  # Dimensionality reduction
  latent_dim: 128                 # Reduced dimension for stability
  
  # Regularization (for noisy embeddings)
  between_class_reg: 0.1          # Between-class shrinkage
  within_class_reg: 0.3           # Within-class shrinkage (higher = more tolerant)
  
  # Decision thresholds
  faiss_k: 5                      # Top-K candidates from FAISS
  llr_threshold: 0.5              # Min LLR for acceptance
  margin_threshold: 0.3           # Min margin between candidates
  calibrate: true                 # Map scores to 0-1 range
```

**Note**: With few enrolled identities (<3), PLDA falls back to cosine-similarity based scoring.

## Models

```yaml
models:
  face_detector: models/yunet.onnx
  face_embedder: models/mobilefacenet.onnx  # Recommended
```

Available embedders:
| Model | Speed | Accuracy | Size |
|-------|-------|----------|------|
| `mobilefacenet.onnx` | Fast | Good | 4MB |
| `arcface_r100.onnx` | Slow | Best | 249MB |
| `adaface_r100.onnx` | Slow | Best | 249MB |

## Inference

```yaml
inference:
  use_gpu: true             # Use GPU if available
  gpu_device_id: 0          # GPU device index
```

## Streaming

```yaml
stream:
  enabled: true
  host: 0.0.0.0
  port: 8080
```

## Logging

```yaml
logging:
  level: INFO               # DEBUG | INFO | WARNING | ERROR
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
```

---

## Tuning Guide

### For Accuracy (fewer false positives)

```yaml
thresholds:
  t_accept: 0.85           # Higher threshold
  t_margin: 0.25           # Wider margin

fusion:
  embedding_min: 10        # More embeddings
  consistency_count: 15    # More consistent frames
```

### For Speed (faster recognition)

```yaml
thresholds:
  t_accept: 0.70           # Lower threshold
  t_margin: 0.15           # Narrower margin

fusion:
  embedding_min: 4         # Fewer embeddings
  embedding_sample_interval: 3  # Sample less often
```

### For Robustness (varying conditions)

```yaml
fusion:
  embedding_window: 30     # Larger window
  embedding_outlier_threshold: 0.30  # More tolerant
  embedding_quality_threshold: 0.35  # Accept more
```

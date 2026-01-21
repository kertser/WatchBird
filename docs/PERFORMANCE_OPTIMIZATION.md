# Performance Optimization Guide

## Overview

WatchBird's main computational bottleneck is **face embedding extraction** (ONNX model inference). This document explains how to optimize performance on resource-constrained devices like Raspberry Pi.

## The Bottleneck

At 20 FPS with 2 faces in frame:
- **Without optimization**: 40 embedding extractions/second
- **With `embedding_sample_interval: 3`**: ~13 embedding extractions/second (66% reduction)
- **With `embedding_sample_interval: 5`**: ~8 embedding extractions/second (80% reduction)

## Frame Sampling Strategy

### How It Works

1. **Detection & Tracking**: Runs on EVERY frame (lightweight operations)
2. **Embedding Extraction**: Only happens every Nth frame per track (expensive operation)
3. **Quality Checks**: Applied before embedding extraction to avoid wasted compute
4. **Per-Track Sampling**: Each tracked face has its own frame counter, ensuring consistent sampling

### Configuration

Edit `config.yaml`:

```yaml
fusion:
  embedding_sample_interval: 3  # Extract embeddings every N frames
```

### Performance Impact

| Interval | Effective FPS | Compute Reduction | Recognition Delay | Use Case |
|----------|--------------|-------------------|-------------------|----------|
| 1 | 20 fps | 0% (baseline) | Instant | High-end hardware, maximum accuracy |
| 2 | 10 fps | 50% | ~100ms | Balanced performance |
| 3 | 6.7 fps | 66% | ~150ms | **Recommended for RPi 4** |
| 5 | 4 fps | 80% | ~250ms | Very constrained devices |
| 10 | 2 fps | 90% | ~500ms | Ultra-low power mode |

### Trade-offs

**Benefits:**
- ✅ Dramatically reduced CPU usage
- ✅ Higher overall FPS (no frame stalls)
- ✅ Lower temperature on embedded devices
- ✅ More responsive UI/streaming

**Considerations:**
- ⚠️ Slightly delayed recognition (~150ms with interval=3)
- ⚠️ May miss very fast-moving faces (rare in typical use)
- ✅ No impact on tracking continuity (tracks persist between embedding samples)

## Other Optimization Options

### 1. Reduce Camera Resolution

```yaml
camera:
  resolution: [320, 240]  # Lower resolution = faster detection
```

**Impact:**
- Faster face detection (~2x speedup)
- Smaller images to process
- May reduce accuracy on distant faces

### 2. Reduce Camera FPS

```yaml
camera:
  fps: 10  # Half the frames = half the compute
```

**Impact:**
- Linear reduction in compute
- May miss fast movements
- Simpler than frame sampling

### 3. Increase Quality Thresholds

```yaml
quality:
  min_face_quality: 0.5  # Skip more low-quality frames
  min_bbox_size: 60      # Skip smaller faces
```

**Impact:**
- Fewer embeddings extracted
- Better quality inputs
- May miss some valid detections

### 4. Use Terminal States

The system already optimizes by skipping tracks in terminal states (FRIENDLY/ENEMY confirmed). This saves compute on already-classified faces.

## Recommended Settings

### Raspberry Pi 4 (4GB)

```yaml
camera:
  resolution: [640, 480]
  fps: 20

fusion:
  embedding_sample_interval: 3  # 66% compute reduction
  consistency_count: 3

quality:
  min_face_quality: 0.4
  min_bbox_size: 50
```

### Raspberry Pi 3 / Lower-end Devices

```yaml
camera:
  resolution: [480, 360]
  fps: 15

fusion:
  embedding_sample_interval: 5  # 80% compute reduction
  consistency_count: 3

quality:
  min_face_quality: 0.5
  min_bbox_size: 60
```

### Desktop / High-end Hardware

```yaml
camera:
  resolution: [1280, 720]
  fps: 30

fusion:
  embedding_sample_interval: 1  # No sampling, maximum accuracy
  consistency_count: 4

quality:
  min_face_quality: 0.3
  min_bbox_size: 40
```

## Monitoring Performance

Watch the terminal output for FPS metrics:

```
INFO - Processed 300 frames (19.8 FPS)  # Good performance
INFO - Processed 300 frames (8.2 FPS)   # Bottlenecked - increase embedding_sample_interval
```

If FPS is significantly below your camera's configured FPS, increase `embedding_sample_interval`.

## Implementation Details

### Per-Track Frame Counters

Each tracked face maintains its own frame counter:

```python
self.track_frame_counters[track_id] += 1

# Only extract embedding every N frames
if self.track_frame_counters[track_id] % embedding_sample_interval != 0:
    continue  # Skip expensive embedding extraction

embedding = self.face_embedder.extract(face_roi)  # Run every Nth frame
```

### Memory Management

Lost tracks are automatically cleaned up to prevent memory leaks:

```python
active_track_ids = {track.track_id for track in tracks}
lost_track_ids = set(self.track_frame_counters.keys()) - active_track_ids
for track_id in lost_track_ids:
    del self.track_frame_counters[track_id]
```

## Future Optimizations

1. **Model Quantization**: Use INT8 quantized ONNX models (2-4x speedup)
2. **Hardware Acceleration**: 
   - Use `TensorRTExecutionProvider` on Jetson devices
   - Use `OpenVINOExecutionProvider` on Intel devices
   - Use `CoreMLExecutionProvider` on Apple Silicon
3. **Batch Processing**: Extract embeddings for multiple faces in one inference call
4. **Adaptive Sampling**: Increase interval for confirmed tracks, decrease for new detections
5. **GPU Acceleration**: Use CUDA/OpenCL for ONNX inference

## Conclusion

The `embedding_sample_interval` parameter provides a simple, effective way to reduce compute by 66-80% with minimal impact on recognition accuracy. Start with `3` and adjust based on your hardware capabilities and accuracy requirements.

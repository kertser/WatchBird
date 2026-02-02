# Body Detection, Segmentation & Classification Guide

WatchBird supports optional human body detection, segmentation, and CLIP-based person classification for enhanced visualization and threat assessment.

## Features

### 1. Body Detection
- Detects human bodies using YOLOv8 ONNX models
- GPU-accelerated via DirectML/CUDA
- Matches detected bodies with faces for state association
- Supports multiple detection model sizes (nano, small)

### 2. Human Segmentation
- Extracts precise body silhouettes using PP-HumanSeg
- Draws colored contours around detected persons
- Contour color indicates recognition or classification state

### 3. Person Classification (CLIP)
- Zero-shot classification using OpenAI CLIP model
- Classifies persons as:
  - **IDF Soldier** (green) - Military uniform detected
  - **Armed Civilian** (red) - Civilian with visible weapon
  - **Unarmed Civilian** (cyan) - Regular civilian
- GPU-accelerated via DirectML/CUDA/PyTorch
- Classification label displayed near face indicator

### State Colors (Face Recognition)
- **Green** (CONFIRMED): High confidence, recognized friendly
- **Light Green/Teal** (FRIENDLY): Building confidence
- **Red** (ENEMY): Unknown person or timeout
- **Orange** (DETECTING): Initial detection phase
- **Yellow** (SUSPECT): Under evaluation

### Classification Colors (CLIP)
- **Green** (IDF): Military uniform/soldier
- **Red** (ARMED): Armed civilian (threat)
- **Cyan** (CIV): Unarmed civilian

## Setup

### 1. Download Models

```bash
# Download body detection and segmentation models
python tools/download_models.py
# Select option 'B' to download YOLOv8n + PP-HumanSeg
```

Or download individually:
```bash
python tools/download_models.py
# Select '7' for YOLOv8n body detector
# Select '9' for PP-HumanSeg segmenter
```

### 2. Enable in Configuration

Edit `config.yaml`:

```yaml
detection:
  # ... existing face detection config ...
  
  # Body detection and segmentation
  body_detection: true            # Enable human body detection
  body_conf_threshold: 0.6        # Min body detection confidence
  segmentation: true              # Enable human segmentation
  segmentation_threshold: 0.8     # Min segmentation probability
  
  # Contour visualization
  contour_thickness: 1            # Thickness of body contour lines
  contour_fill_alpha: 0.15        # Fill transparency (0-1)

  # Person classification (CLIP-based)
  person_classification: true     # Enable soldier/civilian classification
  classification_interval: 5      # Classify every N frames
  armed_threshold: 0.6            # Min confidence for ARMED (higher = stricter)
  soldier_threshold: 0.5          # Min confidence for IDF soldier

models:
  # ... existing models ...
  body_detector: models/yolov8n.onnx
  human_segmenter: models/human_seg.onnx
  clip_cache: models/clip_cache   # CLIP model cache (~600MB)
```

### Classification Thresholds

The `armed_threshold` and `soldier_threshold` control how strict the classification is:

| Threshold | Effect |
|-----------|--------|
| **Low (0.3-0.4)** | More sensitive, may produce false positives |
| **Medium (0.5-0.6)** | Balanced - recommended starting point |
| **High (0.7-0.8)** | Strict - requires high confidence, fewer false positives |

If the top classification doesn't meet its threshold, the system falls back to "CIV" (unarmed civilian).

**Example scenarios:**
- `armed_threshold: 0.7` - Only show "ARMED" when very confident (reduces false alarms)
- `soldier_threshold: 0.4` - More lenient soldier detection (good for partial uniform visibility)

### 3. Run

```bash
python tools/run_runtime.py --config config.yaml
```

The MJPEG stream at `http://localhost:8080/stream` will show colored body contours.

## Model Options

### Body Detectors

| Model | Size | Speed | Accuracy | Notes |
|-------|------|-------|----------|-------|
| **YOLOv8n** | 6.3 MB | Fast | Good | Recommended for real-time |
| YOLOv8s | 22.5 MB | Medium | Better | Higher accuracy |

### Segmentation Models

| Model | Size | Speed | Accuracy | Notes |
|-------|------|-------|----------|-------|
| **PP-HumanSeg Lite** | 2.1 MB | Fast | Good | Recommended |
| YOLOv8-seg | ~12 MB | Medium | Better | Combined detection+segmentation |

## Performance Considerations

### FPS Impact

Body detection, segmentation, and classification add computational overhead:

| Configuration | Typical FPS @ 640x480 |
|---------------|----------------------|
| Face-only (baseline) | ~15-20 FPS |
| Face + Body Detection | ~12-15 FPS |
| Face + Body + Segmentation | ~10-13 FPS |
| Face + Body + Seg + CLIP | ~8-12 FPS |

### Optimization Tips

1. **Lower resolution**: Use 640x480 instead of 1280x960
   ```yaml
   camera:
     resolution: [640, 480]
   ```

2. **ROI-based segmentation**: When body detection is enabled, segmentation only processes detected body regions (faster than full-frame)

3. **Disable segmentation for higher FPS**: Keep body detection but disable segmentation
   ```yaml
   detection:
     body_detection: true
     segmentation: false  # Disable contours for speed
   ```

4. **Use YOLOv8n**: Nano model is 3-4x faster than YOLOv8s with minimal accuracy loss

## Technical Details

### Body-to-Face Matching

The system matches detected faces with bodies using spatial heuristics:

1. **Spatial containment**: Face center must be within body bounding box (with 10% margin)
2. **Vertical position**: Face must be in upper 40% of body
3. **Horizontal alignment**: Face should be horizontally centered on body
4. **Scoring**: Combined horizontal offset + vertical position (lower = better match)

This allows the body contour to inherit the recognition state from the matched face.

### Segmentation Modes

Two modes are supported:

1. **ROI mode** (body detection enabled):
   - Segments only detected body regions
   - Faster, more efficient
   - Each body gets independent state coloring

2. **Full-frame mode** (no body detection):
   - Segments entire image
   - Slower but catches all persons
   - Uses primary face's state for coloring

### State Color Mapping

Body contours use the same color scheme as face bounding boxes:

```python
state_colors = {
    "CONFIRMED": (0, 255, 0),      # Bright green
    "FRIENDLY": (0, 200, 100),     # Light green/teal
    "ENEMY": (0, 0, 255),          # Red
    "DETECTING": (0, 165, 255),    # Orange
    "SUSPECT": (0, 255, 255),      # Yellow
}
```

Contours include:
- **Outline**: Thick colored line (configurable thickness)
- **Fill**: Semi-transparent colored overlay (configurable alpha)

## Use Cases

### 1. Enhanced Visual Feedback
See at a glance who has been recognized with full-body colored silhouettes.

### 2. Multiple Person Scenarios
When multiple people are in frame, body segmentation makes it easier to distinguish between tracked individuals.

### 3. Distance Recognition
Body contours remain visible even when faces are small or partially occluded.

### 4. Recording/Logging
Save annotated video with clear visual indicators of recognition states.

## Troubleshooting

### Models not loading

```
WARNING - Body detector model not found: models/yolov8n.onnx
INFO - Download YOLOv8n ONNX model with: python tools/download_models.py --body
```

**Solution**: Download models with `python tools/download_models.py` → option 'B'

### Low FPS

**Solution**: 
- Reduce camera resolution to 640x480
- Disable segmentation (keep body_detection: false)
- Use YOLOv8n instead of YOLOv8s
- Ensure GPU acceleration is working (check logs for "DirectML GPU" or "CUDA GPU")

### Body-face mismatch

If body contours show wrong recognition state:

**Solution**:
- Adjust `body_conf_threshold` (try 0.6-0.7 for stricter matching)
- Check face detection is working properly
- Ensure faces are visible in upper portion of body

### Contours too thick/thin

**Solution**: Adjust in config.yaml:
```yaml
detection:
  contour_thickness: 2      # Reduce for thinner lines
  contour_fill_alpha: 0.1   # Reduce for more transparent fill
```

## API Reference

### BodyDetector

```python
from watchbird.detect import BodyDetector

detector = BodyDetector(
    model_path="models/yolov8n.onnx",
    conf_threshold=0.5,
    use_gpu=True
)
detector.load()

# Detect bodies
bboxes, confidences = detector.detect(frame)

# Match face to body
body_idx = detector.match_face_to_body(face_bbox, body_bboxes)
```

### HumanSegmenter

```python
from watchbird.detect import HumanSegmenter, draw_body_contour_by_state

segmenter = HumanSegmenter(
    model_path="models/human_seg.onnx",
    threshold=0.5,
    use_gpu=True
)
segmenter.load()

# Segment full frame
mask = segmenter.segment(frame)

# Segment specific ROI (faster)
mask = segmenter.segment_roi(frame, body_bbox)

# Draw colored contour
annotated = draw_body_contour_by_state(
    frame, mask, state="FRIENDLY",
    contour_thickness=3, fill_alpha=0.15
)
```

### PersonClassifier (CLIP)

```python
from watchbird.detect.person_classifier import PersonClassifier

classifier = PersonClassifier(
    cache_dir="models/clip_cache"  # Local model cache
)
classifier.load()

# Classify a single person crop
person_type, confidence = classifier.classify(body_crop)
# Returns: ("soldier" | "armed_civilian" | "unarmed_civilian", 0.0-1.0)

# Batch classification (more efficient)
results = classifier.classify_batch([crop1, crop2, crop3])
# Returns: [("unarmed_civilian", 0.85), ("soldier", 0.72), ...]
```

#### Classification Categories

| Category | Description | Color |
|----------|-------------|-------|
| `soldier` | Military uniform (IDF OD green) | Green |
| `armed_civilian` | Civilian with visible weapon | Red |
| `unarmed_civilian` | Regular civilian, no weapons | Cyan |

#### CLIP Model Details

- **Model**: OpenAI CLIP ViT-B/32
- **Size**: ~600MB (cached locally)
- **Backend**: PyTorch with DirectML/CUDA GPU support
- **First Run**: Downloads from HuggingFace (one-time)

## Future Enhancements

Planned improvements:
- [x] CLIP-based person classification (soldier/civilian)
- [ ] Pose estimation for gesture recognition
- [ ] Multi-person segmentation with instance IDs
- [ ] Body re-identification (match bodies across frames)
- [ ] Skeleton tracking for activity recognition
- [ ] Gait analysis for identity verification
- [ ] Weapon detection model (dedicated YOLO)

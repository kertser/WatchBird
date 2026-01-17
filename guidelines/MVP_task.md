# MVP Development Plan - Face-Only Biometric Identification System

## Overview
This MVP focuses on implementing a face-only biometric identification system while maintaining the architecture for future multi-modal expansion. The system will classify detected persons as Friendly (enrolled) or Enemy (unknown after timeout) using face recognition with quality gating and temporal aggregation.

## Core MVP Features

### 1. Enrollment Application (`tools/enroll.py`)
**Goal:** Build a FAISS index of Friendly identities from face images.

**Inputs:**
- Directory structure: `friendly/<person_id>/*.jpg`
- 20-200 identities support

**Outputs:**
- `data/index/face.index` (FAISS IndexFlatIP)
- `data/index/meta.jsonl` (metadata per embedding)

**Implementation Steps:**
1. Load face images per identity from directory
2. Detect faces using YuNet/MediaPipe
3. Extract face embeddings using lightweight ArcFace ONNX model
4. Apply quality filtering (blur, bbox size, detection confidence)
5. L2-normalize embeddings
6. Build FAISS index and save metadata

**Key Modules:**
- `src/bioguard/detect/face_detector.py`
- `src/bioguard/embed/face_embedder.py`
- `src/bioguard/index/faiss_wrapper.py`
- `src/bioguard/utils/quality.py`

---

### 2. Runtime Recognition Application (`tools/run_runtime.py`)
**Goal:** Real-time classification of detected persons in camera stream.

**Pipeline Flow:**
```
Camera Input → Person Detection → Face Detection → Tracking → 
Face Embedding → FAISS Search → Quality Gating → 
Temporal Aggregation → State Machine → Output Events
```

**Implementation Steps:**

#### 2.1 Input Handling
- **Module:** `src/bioguard/camera/`
- Implement `picamera2` backend for Raspberry Pi CSI camera
- Implement `video_file` backend for testing
- Abstract interface: `CameraBackend.get_frame() → ndarray`

#### 2.2 Detection
- **Module:** `src/bioguard/detect/`
- Person detector: Lightweight YOLO ONNX or MobileNet SSD
- Face detector: OpenCV YuNet (lightweight, no dependency)
- Return bounding boxes with confidence scores

#### 2.3 Tracking
- **Module:** `src/bioguard/track/tracker.py`
- Implement SORT or centroid+IoU tracker
- Assign stable `track_id` to each person
- Maintain track state (active, lost grace period)
- Associate face detections with person tracks using spatial overlap

#### 2.4 Feature Extraction
- **Module:** `src/bioguard/embed/face_embedder.py`
- Load ArcFace/MobileFaceNet ONNX model
- Extract 128 or 512-dim face embeddings
- L2-normalize for cosine similarity

#### 2.5 Quality Gating
- **Module:** `src/bioguard/utils/quality.py`
- Compute face reliability score `r_face`:
  - Bbox size (larger = better)
  - Detection confidence
  - Blur metric (Laplacian variance)
  - Face angle/pose heuristic (optional for MVP)
- Reject embeddings below `MIN_FACE_QUALITY` threshold

#### 2.6 FAISS Search & Fusion
- **Module:** `src/bioguard/fusion/similarity.py`
- Search FAISS index for top-2 matches per embedding
- Compute cosine similarity scores
- For MVP (face-only): `S = s_face` (fusion simplified)
- Store results in per-track buffer

#### 2.7 Temporal Aggregation
- **Module:** `src/bioguard/fusion/aggregation.py`
- Maintain rolling window of M frames per track
- Aggregate using median or mean of similarity scores
- Track consistency: count occurrences of `best_id`

#### 2.8 State Machine
- **Module:** `src/bioguard/runtime/state_machine.py`

**States:**
- `SUSPECT`: initial state
- `FRIENDLY`: identified with confidence
- `ENEMY`: timeout expired without identification

**Transitions:**
- `SUSPECT → FRIENDLY` if:
  - Same `best_id` wins ≥ N of last M frames
  - `median(best_score) ≥ T_ACCEPT`
  - `median(best_score - second_score) ≥ T_MARGIN`
  - Quality-weighted frames used
  
- `SUSPECT → ENEMY` if:
  - `time_since_first_seen > T_TIMEOUT`

#### 2.9 Output Events
- **Module:** `src/bioguard/runtime/events.py`
- Emit JSON lines to stdout:
```json
{
  "ts": 1768600000.123,
  "track_id": 7,
  "state": "FRIENDLY",
  "person_id": "alice",
  "confidence": 0.83,
  "modalities_used": {"face": 0.7}
}
```

---

### 3. Configuration System
**Module:** `src/bioguard/config.py`

**Configuration Parameters (`config.yaml`):**
```yaml
camera:
  backend: picamera2  # or video_file
  resolution: [640, 480]
  fps: 15
  video_path: null  # for video_file backend

detection:
  person_conf_threshold: 0.5
  face_conf_threshold: 0.7

tracking:
  max_age: 30  # frames
  min_hits: 3
  iou_threshold: 0.3

quality:
  min_face_quality: 0.4
  min_bbox_size: 50  # pixels

fusion:
  window_size: 10  # M frames
  consistency_count: 6  # N frames

thresholds:
  t_accept: 0.65
  t_margin: 0.10
  t_timeout: 5.0  # seconds

index:
  face_index_path: data/index/face.index
  meta_path: data/index/meta.jsonl

models:
  face_detector: models/yunet.onnx
  person_detector: models/yolo_nano.onnx
  face_embedder: models/mobilefacenet.onnx
```

---

### 4. Debug Visualization (Headless-Friendly)
**Module:** `src/bioguard/stream/mjpeg_server.py`

**Features:**
- Lightweight Flask/aiohttp MJPEG server
- Route `/stream` serves annotated frames
- Draw bounding boxes with:
  - Green: FRIENDLY (+ person_id)
  - Yellow: SUSPECT (+ track_id)
  - Red: ENEMY (+ track_id)
- Optional: periodic JPEG dumps to `debug/` folder

---

### 5. Multi-Modal Placeholder Architecture
**Goal:** Keep system ready for ReID + Gait integration.

**Placeholder Modules:**
- `src/bioguard/embed/reid_embedder.py` (stub returning zeros)
- `src/bioguard/embed/gait_extractor.py` (stub returning zeros)
- `src/bioguard/fusion/similarity.py` designed for weighted multi-modal fusion

**Future Integration:**
- Add ReID index (`data/index/reid.index`)
- Implement body re-ID embedding extraction
- Update fusion to combine `r_face * s_face + r_reid * s_reid`

---

## Project Structure

```
bioguard/
├── README.md
├── pyproject.toml
├── config.yaml
├── data/
│   └── index/
│       ├── face.index
│       └── meta.jsonl
├── models/
│   ├── yunet.onnx
│   ├── yolo_nano.onnx
│   └── mobilefacenet.onnx
├── friendly/
│   ├── alice/
│   ├── bob/
│   └── ...
├── src/
│   └── bioguard/
│       ├── __init__.py
│       ├── config.py
│       ├── camera/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── picamera_backend.py
│       │   └── video_backend.py
│       ├── detect/
│       │   ├── __init__.py
│       │   ├── person_detector.py
│       │   └── face_detector.py
│       ├── track/
│       │   ├── __init__.py
│       │   └── tracker.py
│       ├── embed/
│       │   ├── __init__.py
│       │   ├── face_embedder.py
│       │   ├── reid_embedder.py (placeholder)
│       │   └── gait_extractor.py (placeholder)
│       ├── index/
│       │   ├── __init__.py
│       │   ├── faiss_wrapper.py
│       │   └── meta_store.py
│       ├── fusion/
│       │   ├── __init__.py
│       │   ├── similarity.py
│       │   └── aggregation.py
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── state_machine.py
│       │   ├── pipeline.py
│       │   └── events.py
│       ├── stream/
│       │   ├── __init__.py
│       │   └── mjpeg_server.py
│       └── utils/
│           ├── __init__.py
│           ├── quality.py
│           ├── image_ops.py
│           └── bbox_ops.py
└── tools/
    ├── enroll.py
    ├── run_runtime.py
    └── calibrate_thresholds.py (future)
```

---

## Development Phases

### Phase 1: Core Infrastructure (Week 1)
- [ ] Project setup: `pyproject.toml`, directory structure
- [ ] Configuration system: `config.py`, `config.yaml`
- [ ] Camera backends: Picamera2 + video file
- [ ] Logging setup
- [ ] Basic unit tests

### Phase 2: Detection & Tracking (Week 1-2)
- [ ] Person detector integration (ONNX)
- [ ] Face detector integration (YuNet)
- [ ] SORT tracker implementation
- [ ] Face-person association logic
- [ ] Quality metrics (blur, bbox size)

### Phase 3: Enrollment (Week 2)
- [ ] Face embedder integration (ArcFace ONNX)
- [ ] FAISS index builder
- [ ] Metadata store (`meta.jsonl`)
- [ ] `tools/enroll.py` CLI
- [ ] Test with sample dataset

### Phase 4: Runtime Recognition (Week 2-3)
- [ ] FAISS search integration
- [ ] Temporal aggregation buffer
- [ ] State machine implementation
- [ ] Event output (JSON lines)
- [ ] `tools/run_runtime.py` CLI

### Phase 5: Visualization & Testing (Week 3)
- [ ] MJPEG server for debug streaming
- [ ] Annotated frame rendering
- [ ] End-to-end testing with video files
- [ ] Performance profiling (FPS, latency)

### Phase 6: Documentation & Polish (Week 4)
- [ ] README with setup instructions
- [ ] Model download links
- [ ] Configuration guide
- [ ] Example datasets
- [ ] Deployment notes for Raspberry Pi

---

## Key Technical Decisions

### Models (Recommended Defaults)
1. **Face Detector:** OpenCV YuNet (built-in, no extra download)
2. **Person Detector:** YOLOv8n ONNX or MobileNet SSD
3. **Face Embedder:** MobileFaceNet ONNX (128-dim, ~1MB)

### Dependencies
```toml
[tool.poetry.dependencies]
python = "^3.11"
numpy = "^1.26"
opencv-python-headless = "^4.9"
onnxruntime = "^1.17"
faiss-cpu = "^1.8"
picamera2 = "^0.3"
pyyaml = "^6.0"
flask = "^3.0"  # for MJPEG server
```

### Performance Targets (Raspberry Pi 4)
- **FPS:** 5-10 fps (acceptable for drone use case)
- **Latency:** < 2 seconds from detection to classification
- **Memory:** < 1GB RAM usage

---

## Testing Strategy

### Unit Tests
- Quality metric calculations
- FAISS search correctness
- Temporal aggregation logic
- State machine transitions

### Integration Tests
- Enrollment pipeline with sample images
- Runtime pipeline with recorded video
- MJPEG server accessibility

### System Tests
- End-to-end: enroll → runtime → correct classifications
- Timeout behavior (Suspect → Enemy)
- Multi-person tracking stability

---

## Success Criteria for MVP

1. **Enrollment works:** Build FAISS index from 20+ identities
2. **Detection works:** Detect and track persons/faces in video
3. **Classification works:** 
   - Enrolled persons → FRIENDLY (>90% accuracy)
   - Unknown persons → ENEMY after timeout
4. **Headless operation:** Runs via SSH with MJPEG debug stream
5. **No false Friendlies:** Conservative thresholds prevent misidentification
6. **Modular code:** Easy to add ReID/Gait in next iteration

---

## Next Steps After MVP

1. Add body re-identification (ReID embedder + index)
2. Implement gait/motion signature extraction
3. Update fusion to multi-modal weighted scoring
4. Build calibration tool for threshold tuning
5. Optimize for Jetson Nano (GPU acceleration)
6. Add UDP event output for external consumers
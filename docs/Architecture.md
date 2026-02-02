# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              WatchBird Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────┐   ┌──────────┐   ┌─────────┐   ┌──────────┐   ┌───────────┐    │
│  │ Camera  │──>│Face Det. │──>│ Tracker │──>│ Embedder │──>│Aggregator │    │
│  └────┬────┘   └──────────┘   └─────────┘   └──────────┘   └─────┬─────┘    │
│       │                                                           │          │
│       │        ┌──────────┐   ┌──────────┐                       ▼          │
│       └───────>│Body Det. │──>│Segmenter │         ┌────────────────────┐    │
│                └──────────┘   └────┬─────┘         │  Unified Scorer    │    │
│                                    │               │ ┌──────┐  ┌─────┐  │    │
│                                    │               │ │FAISS │─>│PLDA │  │    │
│                                    │               │ └──────┘  └─────┘  │    │
│                                    │               └─────────┬──────────┘    │
│  ┌─────────┐   ┌──────────┐       │                         │               │
│  │ Output  │<──│  State   │<──────┴─────────────────────────┘               │
│  │ Stream  │   │ Machine  │  (colored contours by state)                    │
│  └─────────┘   └──────────┘                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

### 1. Camera Backend
```
USB Camera ──┐
             ├──▶ Frame (640x480 @ 30fps)
PiCamera2 ───┘
```

### 2. Face Detector (SCRFD/YuNet/UltraFace)
```
Frame ──▶ Detector ──▶ [bbox, confidence, landmarks]
                │              │
                │              ▼
                │    Filter: confidence > 0.5
                │
                ├── SCRFD (GPU, accurate landmarks) ← Recommended
                ├── UltraFace (GPU, fast, no landmarks)
                └── YuNet (CPU only, OpenCV)
```

### 2b. Body Detector (Optional - YOLOv8)
```
Frame ──▶ YOLOv8 ──▶ [body_bbox, confidence]
                │              │
                │              ▼
                │    Filter: class=person, confidence > 0.5
                │
                ├── YOLOv8n (Nano - 6.3MB, fast)
                └── YOLOv8s (Small - 22.5MB, accurate)
                
Body-to-Face Matching:
  - Spatial heuristics (face in upper 40% of body)
  - Horizontal alignment (face centered on body)
```

### 2c. Human Segmenter (Optional)
```
Frame/ROI ──▶ Segmenter ──▶ Binary Mask ──▶ Contours
                  │                            │
                  │                            ▼
                  │              ┌─────────────────────────┐
                  │              │ Colored by State:       │
                  │              │ • CONFIRMED → Green     │
                  │              │ • FRIENDLY → Light Green│
                  │              │ • ENEMY → Red           │
                  │              │ • DETECTING → Orange    │
                  │              │ • SUSPECT → Yellow      │
                  │              └─────────────────────────┘
                  │
                  ├── PP-HumanSeg (2.1MB, fast)
                  └── YOLOv8-seg (segmentation variant)
```

### 3. Tracker (SORT-based)
```
Detections ──▶ Hungarian Matching ──▶ Track IDs
                     │
                     ▼
              Track = {id, bbox, age, hits}
```

### 4. Face Embedder
```
Face ROI ──▶ Preprocess ──▶ Model ──▶ 512-dim vector
             (112x112)      │
                            ├── MobileFaceNet (fastest)
                            ├── ArcFace R100 (accurate)
                            └── AdaFace R100 (balanced)
```

### 5. Embedding Aggregator

```
┌────────────────────────────────────────────────────────────┐
│                   Per-Track Aggregator                     │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Embedding ──> Quality Check ──> Outlier Check ──> Buffer  │
│                     │                 │                    │
│                     ▼                 ▼                    │
│              Combined Score     Distance to                │
│              ┌─────────────┐    centroid < 0.25            │
│              │Quality = 0.4 × detection_conf               │
│              │        + 0.25 × blur_score                  │
│              │        + 0.15 × brightness_score            │
│              │        + 0.20 × size_score                  │
│              └─────────────┘                               │
│                                                            │
│  Buffer (20 max) ──> Quality² Weighted ──> Centroid        │
│                      Average                               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

### 6. Unified Scorer (FAISS + PLDA)

```
Centroid Embedding
        │
        ▼
┌───────────────┐
│    FAISS      │  Fast nearest neighbor search
│   (Stage 1)   │  Returns: top-K candidates
└───────┬───────┘
        │
        ▼
┌───────────────┐
│     PLDA      │  Probabilistic scoring
│   (Stage 2)   │  Returns: calibrated confidence
└───────┬───────┘
        │
        ▼
Score = blend(PLDA, FAISS)
        │
        ├── High FAISS (≥0.75): 70% PLDA + 30% FAISS
        ├── Med FAISS (0.6-0.75): 50% PLDA + 50% FAISS
        └── Low FAISS (<0.6): 30% PLDA + 70% FAISS
```

### 7. State Machine

```
                    ┌──────────────────────────┐
                    │                          │
                    ▼                          │
┌─────────┐   score ≥ 0.78               ┌─────┴─────┐
│ SUSPECT │───margin ≥ 0.20────────────> │ FRIENDLY  │
│ (new)   │   consistency ≥ 12           │ (matched) │
└────┬────┘                              └───────────┘
     │
     │ timeout (15s)
     │ no confident match
     ▼
┌─────────┐
│  ENEMY  │
│(unknown)│
└─────────┘
```

## Data Flow

```
Frame ─┬─▶ Detect Faces ─▶ For each detection:
       │                         │
       │                         ▼
       │                   Track Assignment
       │                         │
       │                         ▼
       │                   Extract Embedding
       │                         │
       │                         ▼
       │                   Add to Aggregator
       │                         │
       │                         ▼
       │                   Ready? (≥6 embeddings)
       │                         │
       │              ┌──────────┴──────────┐
       │              │                     │
       │             No                    Yes
       │              │                     │
       │              ▼                     ▼
       │         Continue            Get Centroid
       │         collecting               │
       │                                  ▼
       │                           Score vs Database
       │                                  │
       │                                  ▼
       │                           Update State Machine
       │                                  │
       └──────────────────────────────────┘
                                          │
                                          ▼
                                    Annotate Frame
                                          │
                                          ▼
                                    Stream Output
```

## File Structure

```
WatchBird/
├── config.yaml              # Main configuration
├── README.md                # This file
│
├── src/watchbird/
│   ├── camera/              # Camera backends
│   ├── detect/              # Face detection
│   │   ├── face_detector.py         # YuNet (CPU)
│   │   ├── ultraface_detector.py    # UltraFace (GPU)
│   │   └── scrfd_detector.py        # SCRFD (GPU) ← Recommended
│   ├── embed/               # Face embedding extraction
│   ├── track/               # Object tracking (SORT)
│   ├── fusion/
│   │   ├── embedding_aggregator.py  # Multi-frame aggregation
│   │   ├── plda_scorer.py           # PLDA scoring
│   │   └── unified_scorer.py        # FAISS + PLDA fusion
│   ├── index/               # FAISS index & metadata
│   ├── runtime/             # State machine & events
│   └── stream/              # MJPEG server
│
├── tools/                   # CLI utilities
├── models/                  # ONNX models
├── data/index/              # Enrolled face database
└── friendly/                # Enrollment photos
```

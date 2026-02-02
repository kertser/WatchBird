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
│       │        ┌──────────┐   ┌──────────┐   ┌──────────┐        ▼          │
│       └───────>│Body Det. │──>│Segmenter │──>│  CLIP    │  ┌───────────┐    │
│                └──────────┘   └────┬─────┘   │Classifier│  │  Scorer   │    │
│                                    │         └────┬─────┘  │FAISS+PLDA │    │
│                                    │              │        └─────┬─────┘    │
│  ┌─────────┐   ┌──────────┐       │              │              │          │
│  │ Output  │<──│  State   │<──────┴──────────────┴──────────────┘          │
│  │ Stream  │   │ Machine  │  (colored contours + classification labels)    │
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

### 2. Face Detector (SCRFD)
```
Frame ──▶ Detector ──▶ [bbox, confidence, landmarks]
                │              │
                │              ▼
                │    Filter: confidence > 0.5
                │
                └── SCRFD (DirectML GPU, accurate 5-point landmarks)
```

### 2b. Body Detector (YOLOv8)
```
Frame ──▶ YOLOv8 ──▶ [body_bbox, confidence]
                │              │
                │              ▼
                │    Filter: class=person, confidence > 0.5
                │
                └── YOLOv8n (Nano - 6.3MB, DirectML GPU)
                
Body-to-Face Matching:
  - Spatial heuristics (face in upper 40% of body)
  - Horizontal alignment (face centered on body)
```

### 2c. Human Segmenter (PP-HumanSeg)
```
Body ROI ──▶ Segmenter ──▶ Binary Mask ──▶ Contours
                │                              │
                │                              ▼
                │              ┌─────────────────────────┐
                │              │ Colored by:             │
                │              │ • Recognition State     │
                │              │ • CLIP Classification   │
                │              └─────────────────────────┘
                │
                └── PP-HumanSeg Lite (2.1MB, DirectML GPU)
```

### 2d. Person Classifier (CLIP)
```
Body Crop ──▶ CLIP ──▶ [soldier | armed_civilian | unarmed_civilian]
               │                    │
               │                    ▼
               │        ┌─────────────────────────┐
               │        │ Display Labels:          │
               │        │ • IDF 85%  → Green      │
               │        │ • ARMED 72% → Red       │
               │        │ • CIV 63%  → Cyan       │
               │        └─────────────────────────┘
               │
               └── OpenAI CLIP ViT-B/32 (PyTorch + DirectML)
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

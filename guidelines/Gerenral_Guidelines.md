# Bioguard Drone Biometric ID System — Project Guidelines

> **A production-oriented Python project implementing a real-time, fully on-device multi-modal biometric identification system for a drone camera stream.**

These are the guidelines for the project.

---

## 🎯 Mission (in general)

Implement a system that classifies each detected person into:

- **Friendly**: identified as one of the enrolled friendly identities with high confidence
- **Enemy**: not identified as **Friendly** within a configured timeout  
  *(Enemy = Unknown-after-timeout, no enemy gallery)*

There is **no cloud**, **no external compute**, **no LLM onboard**. The system runs on embedded hardware (Raspberry Pi 4 or Jetson-class). Everything must be **headless-friendly**.

---

## ⚠️ Hard Rules

- Only **Friendly** identities are enrolled (20–200 people). **No Enemy gallery.**
- Every person starts as **Suspect** and becomes:
  - **Friendly** if proven with strict multi-frame, multi-modal evidence
  - **Enemy** if still not Friendly after `T_TIMEOUT` seconds
- For the **end-product** face-only is forbidden. We must use additional biometrics (at minimum: **face + body re-ID**; optional: **gait/motion**).
- For the **development**, face-only is acceptable **as long as** the architecture supports multi-modal fusion.
- No single-frame decisions. Use tracking + evidence accumulation.
- Prioritize **no false Friendly**. Conservative thresholds, margin checks, quality gating.  
  *(False Enemy is acceptable by timeout logic.)*
- Headless operation: no `cv2.imshow()`. Provide **MJPEG streaming** and/or periodic **JPEG dumps** for debugging.

---

## 📦 Deliverables

Create a repository with:

### A) Enrollment App (separate tool)

Collect data for each Friendly person (face + body + optional short walking sequence) and build FAISS indexes and metadata.

**Output:**

- `data/index/face.index`
- `data/index/reid.index`
- `data/index/meta.jsonl`
- `data/index/prototypes.json` (optional)

**Must support input sources:**

- folder of images per identity
- optional live capture mode from Picamera2

---

### B) Runtime Recognition App (main drone runtime)

**Input:** live camera frames from Picamera2 (CSI camera) OR a video file for testing.

**Components:**

- Person detection (lightweight)
- Face detection (lightweight)
- Tracking (`track_id` assignment, stable per person)

**Feature extraction:**

- Face embedding model (ONNX)
- Re-ID embedding model (ONNX)
- Optional gait/motion signature (cheap engineered baseline if no model)

**Fusion:**

- quality-weighted fusion score per candidate identity
- multi-frame aggregation

**State machine per track:**

- `Suspect → Friendly` OR `Suspect → Enemy` after timeout

**Output:**

- live JSON events (stdout or UDP) containing `track_id`, label, `best_id` (if friendly), confidence, timestamps
- optional debug snapshots

---

### C) Calibration Tool (offline, not part of runtime; later stages)

**Script to compute similarity distributions:**

- friendly-vs-friendly (same person vs different people)
- friendly-vs-nonfriendly (if provided)
- Suggest threshold defaults (`T_ACCEPT`, `T_MARGIN`, quality thresholds)

---

## 🔧 Constraints and Non-Functional Requirements

- Python 3.11+ (works on Pi OS / Debian)
- Prefer ONNX Runtime for inference
- Use FAISS for vector search
- Must run in headless SSH environment
- Must be structured, modular, testable:
  - `src/` package
  - config via `config.yaml` or env vars
  - logging
  - deterministic behavior where possible
- Avoid heavy dependencies unless justified

---

## 📁 Enrollment Format

- `friendly/<person_id>/*.jpg`
- `person_id` is a short string. Support 20–200 identities.

---

## 📄 Metadata

`meta.jsonl` each line:

```json
{
  "vec_id": 123,
  "person_id": "alice",
  "modality": "face",
  "source": "friendly/alice/img_0001.jpg",
  "quality": 0.82,
  "ts": "2026-01-16T00:00:00Z"
}
````

---

## 🧮 Algorithms (must implement)

### 1) Tracking

Assign stable `track_id` per person. Use a standard lightweight tracker (SORT / centroid + IoU) and keep a track alive for a short grace window.

### 2) Quality gating

For each modality compute a reliability score `r`:

- **Face reliability** `r_face`: based on bbox size, detection conf, blur, occlusion heuristic
- **ReID reliability** `r_reid`: based on bbox size, truncation, blur
- **Gait reliability** `r_gait`: based on track duration, motion magnitude

Only accept embeddings into the buffer if reliability is above configured minima.

### 3) FAISS search

Use cosine similarity:

- L2-normalize embeddings
- FAISS index = `IndexFlatIP`
- Keep separate indexes per modality (`face`, `reid`)

### 4) Fusion and aggregation

For a candidate identity `id`, compute fused similarity:

```python
S = (r_face*s_face + r_reid*s_reid + r_gait*s_gait) / (r_face + r_reid + r_gait + eps)
```

Aggregate over time using median/mean over a window `M`.

### 5) Decision policy per track

Maintain per-track rolling window of decisions:

- know: `best_id`, `best_score`, `second_score`, reliabilities, timestamp

**Friendly acceptance requires:**

- **Consistency**: same `best_id` wins >= `N` of last `M`
- **Strength**: median(`best_score`) >= `T_ACCEPT`
- **Margin**: median(`best_score - second_score`) >= `T_MARGIN`
- **Multi-modal**: at least two modalities contributed meaningful reliability in the window

**Enemy after timeout:**

- If not Friendly after `T_TIMEOUT`, label Enemy and stop expensive inference for that track.

---

## 🛠️ Engineering Requirements

- Provide code with type hints
- Handle exceptions and camera errors gracefully
- Make it easy to switch hardware:
  - `--backend picamera2` or `--backend video_file`
- Include a minimal MJPEG server for remote viewing:
  - route `/stream` provides annotated frames (bounding boxes + track_id + label)

---

## 🤖 Model Choices

Implement model loading as **pluggable modules**. Provide default choices but keep the system functional even if models are replaced.

**Recommended models:**

- **Face detector**: OpenCV YuNet or MediaPipe Face Detection (if light enough)
- **Person detector**: lightweight YOLO variant ONNX or MobileNet SSD
- **Face embedder**: lightweight ArcFace/MobileFaceNet ONNX
- **ReID embedder**: lightweight OSNet/MobileNet-based reID ONNX
- **Optional gait**:
  - start with engineered descriptor (e.g., track-level motion + limb ratios from pose if available)
  - if too heavy, implement placeholder interface

**If models are not bundled:**

- include code to download them manually (document links)
- **do not** hardcode network download in runtime

---

## 📊 Output Format for Runtime Events

Emit JSON lines like:

```json
{
  "ts": 1768600000.123,
  "track_id": 7,
  "state": "FRIENDLY",
  "person_id": "alice",
  "confidence": 0.83,
  "modalities_used": {"face": 0.7, "reid": 0.4, "gait": 0.0}
}
```

---

## 📂 Project Structure Required

```
.
├── README.md
├── pyproject.toml
├── src/
│   └── bioguard/
│       ├── config.py
│       ├── camera/          # picamera2 backend, video file backend
│       ├── detect/          # person detector, face detector
│       ├── track/           # tracker
│       ├── embed/           # face embedder, reid embedder, gait interface
│       ├── index/           # faiss wrapper, meta store
│       ├── fusion/          # reliability, fusion, aggregation
│       ├── runtime/         # state machine, pipeline
│       ├── stream/          # mjpeg annotated streaming
│       └── utils/           # image ops, blur metric, bbox ops
└── tools/
    ├── enroll.py
    ├── run_runtime.py
    └── calibrate_thresholds.py
```

Include example configs and CLI arguments.

---

**Note:** This is a comprehensive guideline document. Implement incrementally, starting with core components and expanding to multi-modal fusion.

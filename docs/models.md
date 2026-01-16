# Model Setup Guide

This document provides instructions for downloading and setting up the required ONNX models for Bioguard.

## Required Models

### 1. Face Detector: YuNet

**Built-in with OpenCV** - No download needed if using OpenCV 4.9+

YuNet is included with modern OpenCV installations and will be used automatically.

**Alternative Download (optional):**
```bash
wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx -O models/yunet.onnx
```

### 2. Face Embedder: MobileFaceNet

Download a lightweight face recognition model:

**Option A: InsightFace MobileFaceNet**
```bash
# Download from InsightFace model zoo
wget https://github.com/onnx/models/raw/main/vision/body_analysis/arcface/model/arcfaceresnet100-8.onnx -O models/mobilefacenet.onnx
```

**Option B: Convert from PyTorch**
If you have a custom model, convert it to ONNX format:
```python
import torch
import torch.onnx

# Load your model
model = load_your_model()
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 3, 112, 112)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "models/mobilefacenet.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
)
```

### 3. Person Detector: YOLO (Optional for MVP)

For the full multi-modal system, download a lightweight person detector:

**YOLOv8n (Nano):**
```bash
# Install ultralytics
pip install ultralytics

# Export YOLOv8n to ONNX
python -c "from ultralytics import YOLO; model = YOLO('yolov8n.pt'); model.export(format='onnx')"

# Move to models directory
mv yolov8n.onnx models/yolo_nano.onnx
```

**Alternative: Download pre-converted:**
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx -O models/yolo_nano.onnx
```

## Model Directory Structure

After downloading, your `models/` directory should look like:

```
models/
├── yunet.onnx              # Face detector (optional, built-in)
├── mobilefacenet.onnx      # Face embedder (REQUIRED)
└── yolo_nano.onnx          # Person detector (optional for MVP)
```

## Testing Models

Test that models load correctly:

```python
import onnxruntime as ort

# Test face embedder
session = ort.InferenceSession("models/mobilefacenet.onnx")
print("Face embedder loaded successfully!")
print(f"Input shape: {session.get_inputs()[0].shape}")
print(f"Output shape: {session.get_outputs()[0].shape}")
```

## Model Specifications

### Face Embedder
- **Input:** RGB image, shape `[1, 3, 112, 112]`
- **Output:** Embedding vector, shape `[1, 128]` or `[1, 512]`
- **Preprocessing:** Normalize to [-1, 1] with mean=0.5, std=0.5

### Person Detector (YOLO)
- **Input:** RGB image, shape `[1, 3, 640, 640]`
- **Output:** Detections with bounding boxes and class scores
- **Preprocessing:** Normalize to [0, 1]

## Troubleshooting

### ONNX Runtime Issues

If you encounter ONNX Runtime errors:

```bash
# Ensure ONNX Runtime is installed
pip install onnxruntime --upgrade

# On Raspberry Pi, use CPU-only version
pip install onnxruntime
```

### Model Not Found Errors

Verify model paths in `config.yaml`:

```yaml
models:
  face_detector: models/yunet.onnx
  person_detector: models/yolo_nano.onnx
  face_embedder: models/mobilefacenet.onnx
```

### Performance Optimization

For Raspberry Pi 4:
- Use quantized models (INT8) for faster inference
- Reduce input resolution if needed
- Consider model pruning for size reduction

## License Notes

- **YuNet**: Apache 2.0
- **MobileFaceNet**: MIT (varies by implementation)
- **YOLOv8**: AGPL-3.0

Ensure compliance with model licenses for your use case.

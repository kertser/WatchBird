#!/usr/bin/env python3
"""Quick GPU test script."""
import sys
sys.path.insert(0, "src")

import cv2
import onnxruntime as ort

print("=" * 60)
print("GPU STATUS CHECK")
print("=" * 60)

# OpenCV
print(f"\n1. OpenCV: {cv2.__version__}")
try:
    cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
    print(f"   CUDA devices: {cuda_count}")
except Exception as e:
    print(f"   CUDA: Not available ({e})")

# ONNX Runtime
print(f"\n2. ONNX Runtime Providers:")
for p in ort.get_available_providers():
    print(f"   - {p}")

# Face Detector
print(f"\n3. Face Detector:")
from watchbird.detect.face_detector import FaceDetector
detector = FaceDetector("models/yunet.onnx", use_gpu=True)
detector.load()
print(f"   Backend: {detector.backend_used}")

# Face Embedder
print(f"\n4. Face Embedder:")
from watchbird.embed.face_embedder import FaceEmbedder
embedder = FaceEmbedder("models/arcface_r100.onnx", use_gpu=True)
embedder.load()
print(f"   Provider: {embedder.active_provider}")

print("\n" + "=" * 60)
print("SUMMARY:")
print("  - Face Detection: " + ("GPU (CUDA)" if "CUDA" in detector.backend_used else "CPU"))
print("  - Face Embedding: " + ("GPU (DirectML)" if "Dml" in (embedder.active_provider or "") else "CPU"))
print("=" * 60)


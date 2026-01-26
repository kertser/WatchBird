#!/usr/bin/env python3
"""Test GPU inference support."""

import sys
import onnxruntime as ort

print("=" * 50)
print("GPU Inference Test")
print("=" * 50)

# Check available providers
providers = ort.get_available_providers()
print(f"\n1. Available ONNX Runtime Providers:")
for p in providers:
    print(f"   - {p}")

# Check if GPU provider is available
has_gpu = 'DmlExecutionProvider' in providers or 'CUDAExecutionProvider' in providers
print(f"\n2. GPU Support: {'✓ Available' if has_gpu else '✗ Not available'}")

if 'DmlExecutionProvider' in providers:
    print("   Using: DirectML (Windows GPU)")
elif 'CUDAExecutionProvider' in providers:
    print("   Using: NVIDIA CUDA")

# Test loading model with GPU
print("\n3. Testing Face Embedder with GPU:")
try:
    from watchbird.config import Config
    from watchbird.embed.face_embedder import FaceEmbedder, get_available_providers

    config = Config("config.yaml")
    embedder = FaceEmbedder(
        model_path=config.models.get("face_embedder"),
        use_gpu=config.inference.get("use_gpu", True),
        gpu_device_id=config.inference.get("gpu_device_id", 0)
    )
    success = embedder.load()

    if success:
        print(f"   ✓ Model loaded successfully")
        print(f"   Active provider: {embedder.active_provider}")
        print(f"   Embedding size: {embedder.embedding_size}")
    else:
        print("   ✗ Failed to load model")

except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 50)


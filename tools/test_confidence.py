#!/usr/bin/env python3
"""Test confidence calculation on existing photos."""

import cv2
import numpy as np
from pathlib import Path

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.index.faiss_wrapper import FaissIndex
from watchbird.utils.image_ops import extract_roi

# Load config
config = Config("config.yaml")

# Initialize
detector = FaceDetector(
    model_path=config.models.get("face_detector"),
    conf_threshold=config.detection["face_conf_threshold"],
    use_gpu=config.inference.get("use_gpu", True),
    gpu_device_id=config.inference.get("gpu_device_id", 0)
)
detector.load()

embedder = FaceEmbedder(
    model_path=config.models.get("face_embedder"),
    use_gpu=config.inference.get("use_gpu", True),
    gpu_device_id=config.inference.get("gpu_device_id", 0)
)
embedder.load()

# Find mike's photos
mike_dir = Path("friendly/mike")
photos = sorted(mike_dir.glob("*.jpg"))[:10]  # Test with first 10

print(f"Testing with {len(photos)} photos from {mike_dir}")

# Extract embeddings
embeddings = []
for i, photo_path in enumerate(photos):
    img = cv2.imread(str(photo_path))
    if img is None:
        print(f"  {i+1}. {photo_path.name}: Failed to load")
        continue

    bboxes, confs = detector.detect(img)
    if len(bboxes) == 0:
        print(f"  {i+1}. {photo_path.name}: No face detected")
        continue

    face_roi = extract_roi(img, bboxes[0])
    embedding = embedder.extract(face_roi)

    if embedding is None:
        print(f"  {i+1}. {photo_path.name}: Failed to extract embedding")
        continue

    embeddings.append(embedding)
    print(f"  {i+1}. {photo_path.name}: ✓ Embedding extracted (dim={len(embedding)})")

print(f"\nTotal embeddings: {len(embeddings)}")

if len(embeddings) < 2:
    print("ERROR: Need at least 2 embeddings to test confidence")
    exit(1)

# Build index
index = FaissIndex()
index.build(np.array(embeddings))

# Test confidence
confidences = []
for i, embedding in enumerate(embeddings):
    k = min(3, len(embeddings))
    similarities, indices = index.search(embedding, k=k)

    if len(similarities) > 0 and len(similarities[0]) > 1:
        # Exclude self-match
        other_sims = similarities[0][1:]
        avg_conf = np.mean(other_sims)
        confidences.append(avg_conf)
        print(f"  Photo {i+1}: similarities={other_sims}, avg={avg_conf:.3f}")
    else:
        print(f"  Photo {i+1}: Not enough matches")

if confidences:
    final_conf = np.mean(confidences)
    print(f"\n✅ Final confidence: {final_conf:.3f}")
    print(f"   Min: {min(confidences):.3f}, Max: {max(confidences):.3f}")
else:
    print("\n❌ No confidences calculated!")

"""Diagnostic tool to evaluate face recognition model quality."""
import sys
sys.path.insert(0, "src")

import logging
from pathlib import Path
import cv2
import numpy as np
from itertools import combinations

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.utils.image_ops import extract_roi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def evaluate_model(data_dir: Path, config_path: str):
    """Evaluate face recognition model quality."""

    print("=" * 80)
    print("FACE RECOGNITION MODEL DIAGNOSTIC")
    print("=" * 80)

    # Load config
    config = Config(config_path)

    # Load face detector
    print("\n1. Loading face detector...")
    face_detector = FaceDetector(
        model_path=config.models.get("face_detector", "models/yunet.onnx"),
        conf_threshold=config.detection["face_conf_threshold"]
    )
    if not face_detector.load():
        print("✗ Failed to load face detector")
        return
    print("✓ Face detector loaded")

    # Load face embedder
    print("\n2. Loading face embedder...")
    face_embedder = FaceEmbedder(
        model_path=config.models.get("face_embedder", "models/mobilefacenet.onnx")
    )
    if not face_embedder.load():
        print("✗ Failed to load face embedder")
        return
    print(f"✓ Face embedder loaded")
    print(f"  Model: {face_embedder.model_path.name}")
    print(f"  Embedding size: {face_embedder.embedding_size} dimensions")

    # Collect face embeddings by person
    print("\n3. Processing enrollment images...")
    person_embeddings = {}
    total_images = 0
    total_faces = 0

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        image_paths = (
            list(person_dir.glob("*.[jJ][pP][gG]")) +
            list(person_dir.glob("*.[pP][nN][gG]"))
        )

        embeddings = []

        for img_path in image_paths:
            total_images += 1
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            # Detect faces
            bboxes, confs = face_detector.detect(image, try_rotations=True)
            if len(bboxes) == 0:
                print(f"  ⚠ No face in {person_id}/{img_path.name}")
                continue

            total_faces += 1

            # Extract embedding
            face_roi = extract_roi(image, bboxes[0])
            embedding = face_embedder.extract(face_roi)

            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            person_embeddings[person_id] = embeddings
            print(f"  ✓ {person_id}: {len(embeddings)} face embeddings")

    print(f"\nProcessed: {total_images} images, {total_faces} faces detected")

    # Compute statistics
    print("\n4. Computing similarity statistics...")
    print("-" * 80)

    intra_person_sims = []  # Same person
    inter_person_sims = []  # Different persons

    # Intra-person similarities (same person, different photos)
    for person_id, embeddings in person_embeddings.items():
        if len(embeddings) < 2:
            continue

        for emb1, emb2 in combinations(embeddings, 2):
            sim = cosine_similarity(emb1, emb2)
            intra_person_sims.append(sim)

    # Inter-person similarities (different persons)
    person_ids = list(person_embeddings.keys())
    for i, person1 in enumerate(person_ids):
        for person2 in person_ids[i+1:]:
            for emb1 in person_embeddings[person1]:
                for emb2 in person_embeddings[person2]:
                    sim = cosine_similarity(emb1, emb2)
                    inter_person_sims.append(sim)

    # Print results
    print("\n📊 SIMILARITY ANALYSIS")
    print("=" * 80)

    if intra_person_sims:
        print(f"\n🟢 SAME PERSON (should be HIGH):")
        print(f"   Count: {len(intra_person_sims)} pairs")
        print(f"   Mean:  {np.mean(intra_person_sims):.4f}")
        print(f"   Min:   {np.min(intra_person_sims):.4f}")
        print(f"   Max:   {np.max(intra_person_sims):.4f}")
        print(f"   Std:   {np.std(intra_person_sims):.4f}")

    if inter_person_sims:
        print(f"\n🔴 DIFFERENT PERSONS (should be LOW):")
        print(f"   Count: {len(inter_person_sims)} pairs")
        print(f"   Mean:  {np.mean(inter_person_sims):.4f}")
        print(f"   Min:   {np.min(inter_person_sims):.4f}")
        print(f"   Max:   {np.max(inter_person_sims):.4f}")
        print(f"   Std:   {np.std(inter_person_sims):.4f}")

    if intra_person_sims and inter_person_sims:
        separation = np.mean(intra_person_sims) - np.mean(inter_person_sims)
        print(f"\n⚡ SEPARATION MARGIN: {separation:.4f}")

        # Determine quality
        print("\n📈 MODEL QUALITY ASSESSMENT:")
        if separation > 0.35:
            quality = "EXCELLENT ⭐⭐⭐"
            recommendation = "Model is performing very well!"
        elif separation > 0.25:
            quality = "GOOD ⭐⭐"
            recommendation = "Model is adequate, but could be improved."
        elif separation > 0.15:
            quality = "FAIR ⭐"
            recommendation = "Consider upgrading to a better model (ArcFace)."
        else:
            quality = "POOR ❌"
            recommendation = "UPGRADE NEEDED! Model is not reliable."

        print(f"   Quality: {quality}")
        print(f"   Separation: {separation:.4f}")
        print(f"   Recommendation: {recommendation}")

        # Threshold recommendations
        print(f"\n🎯 RECOMMENDED THRESHOLDS:")

        # Conservative threshold (minimize false positives)
        if intra_person_sims:
            conservative_threshold = np.percentile(intra_person_sims, 25)  # 75% of same-person above this
            print(f"   Conservative (high security): {conservative_threshold:.3f}")

            # Balanced threshold
            optimal_threshold = np.mean(intra_person_sims) - 1.5 * np.std(intra_person_sims)
            print(f"   Balanced (recommended):       {max(0.5, optimal_threshold):.3f}")

            # Permissive threshold
            permissive_threshold = np.min(intra_person_sims) - 0.05
            print(f"   Permissive (catch more):      {max(0.4, permissive_threshold):.3f}")

        # Update config recommendation
        optimal_t_accept = max(0.5, np.mean(intra_person_sims) - 1.5 * np.std(intra_person_sims))
        print(f"\n💡 UPDATE config.yaml:")
        print(f"   thresholds:")
        print(f"     t_accept: {optimal_t_accept:.2f}  # Current: {config.thresholds.get('t_accept', 0.65)}")
        print(f"     t_margin: 0.10")

    print("\n" + "=" * 80)

    if not intra_person_sims and not inter_person_sims:
        print("\n⚠ WARNING: Not enough data to evaluate model")
        print("   Need: Multiple images per person, multiple persons")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate face recognition model")
    parser.add_argument("--data-dir", type=str, default="friendly", help="Directory with enrollment images")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")

    args = parser.parse_args()

    evaluate_model(Path(args.data_dir), args.config)

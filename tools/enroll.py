#!/usr/bin/env python3
"""Enrollment tool for building FAISS index from friendly identities."""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.index.faiss_wrapper import FaissIndex
from watchbird.index.meta_store import MetaStore
from watchbird.utils.image_ops import extract_roi
from watchbird.utils.quality import compute_face_quality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_identity_images(data_dir: Path) -> List[Tuple[str, List[Path]]]:
    """Load identity images from directory structure.

    Args:
        data_dir: Base directory containing friendly/<person_id>/*.jpg

    Returns:
        List of (person_id, image_paths) tuples
    """
    identities = []

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        # Case-insensitive image loading
        image_paths = (
            list(person_dir.glob("*.jpg")) +
            list(person_dir.glob("*.JPG")) +
            list(person_dir.glob("*.jpeg")) +
            list(person_dir.glob("*.JPEG")) +
            list(person_dir.glob("*.png")) +
            list(person_dir.glob("*.PNG"))
        )

        if image_paths:
            identities.append((person_id, image_paths))
            logger.info(f"Found {len(image_paths)} images for {person_id}")

    return identities


def process_enrollment(
    identities: List[Tuple[str, List[Path]]],
    face_detector: FaceDetector,
    face_embedder: FaceEmbedder,
    min_quality: float = 0.4
) -> Tuple[np.ndarray, MetaStore]:
    """Process enrollment images and extract embeddings.

    Args:
        identities: List of (person_id, image_paths)
        face_detector: Face detector instance
        face_embedder: Face embedder instance
        min_quality: Minimum quality threshold

    Returns:
        Tuple of (embeddings_array, meta_store)
    """
    embeddings = []
    meta_store = MetaStore("data/index/meta.jsonl")
    vec_id = 0

    for person_id, image_paths in identities:
        logger.info(f"Processing {person_id}...")

        person_embeddings = 0

        for img_path in image_paths:
            # Load image
            logger.info(f"  Processing: {img_path.name}")
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"  ✗ Failed to load {img_path}")
                continue

            logger.debug(f"  Image shape: {image.shape}")

            # Detect faces
            face_bboxes, face_confs, _ = face_detector.detect(image)

            if len(face_bboxes) == 0:
                logger.warning(f"  ✗ No face detected in {img_path.name}")
                continue

            logger.info(f"  ✓ Detected {len(face_bboxes)} face(s)")

            if len(face_bboxes) > 1:
                logger.warning(f"  ⚠ Multiple faces detected in {img_path.name}, using first")

            # Use first face
            face_bbox = face_bboxes[0]
            face_conf = face_confs[0]

            # Extract face ROI
            face_roi = extract_roi(image, face_bbox)

            # Compute quality
            quality, metrics = compute_face_quality(
                face_bbox,
                face_roi,
                face_conf
            )

            if quality < min_quality:
                logger.debug(f"Low quality face in {img_path}: {quality:.3f}")
                continue

            # Extract embedding
            embedding = face_embedder.extract(face_roi)

            if embedding is None:
                logger.warning(f"Failed to extract embedding from {img_path}")
                continue

            # Store embedding and metadata
            embeddings.append(embedding)
            meta_store.add(
                vec_id=vec_id,
                person_id=person_id,
                modality="face",
                source=str(img_path),
                quality=quality
            )

            vec_id += 1
            person_embeddings += 1

        logger.info(f"Enrolled {person_embeddings} face embeddings for {person_id}")

    if embeddings:
        embeddings_array = np.array(embeddings)
        logger.info(f"Total: {len(embeddings)} embeddings from {len(identities)} identities")
        return embeddings_array, meta_store
    else:
        raise ValueError("No embeddings extracted!")


def main() -> None:
    """Main enrollment function."""
    parser = argparse.ArgumentParser(description="Enroll friendly identities")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="friendly",
        help="Directory containing identity folders"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="Minimum face quality threshold (default: from config)"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Get min quality from args or config
    min_quality = args.min_quality
    if min_quality is None:
        min_quality = config.quality.get("min_enroll_quality", 0.5)

    # Initialize components
    logger.info("Loading models...")

    face_detector = FaceDetector(
        model_path=config.models.get("face_detector", ""),
        conf_threshold=config.detection["face_conf_threshold"]
    )

    if not face_detector.load():
        logger.error("Failed to load face detector")
        return

    face_embedder = FaceEmbedder(
        model_path=config.models.get("face_embedder", "models/mobilefacenet.onnx")
    )

    if not face_embedder.load():
        logger.error("Failed to load face embedder")
        return

    # Load identity images
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return

    identities = load_identity_images(data_dir)

    if not identities:
        logger.error(f"No identities found in {data_dir}")
        return

    # Process enrollment
    logger.info(f"Processing enrollment (min_quality={min_quality:.2f})...")
    embeddings, meta_store = process_enrollment(
        identities,
        face_detector,
        face_embedder,
        min_quality=min_quality
    )

    # Build FAISS index
    logger.info("Building FAISS index...")
    faiss_index = FaissIndex(embedding_dim=embeddings.shape[1])

    if not faiss_index.build(embeddings):
        logger.error("Failed to build FAISS index")
        return

    # Save index and metadata
    index_path = config.index["face_index_path"]
    meta_path = config.index["meta_path"]

    if faiss_index.save(index_path):
        logger.info(f"Saved FAISS index to {index_path}")
    else:
        logger.error("Failed to save FAISS index")
        return

    if meta_store.save():
        logger.info(f"Saved metadata to {meta_path}")
    else:
        logger.error("Failed to save metadata")
        return

    logger.info("Enrollment complete!")


if __name__ == "__main__":
    main()


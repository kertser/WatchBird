#!/usr/bin/env python3
"""Enrollment tool for building FAISS index from friendly identities."""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.detect.scrfd_detector import SCRFDDetector
from watchbird.detect.ultraface_detector import UltraFaceDetector
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.fusion.plda_scorer import PLDAScorer, PLDAConfig
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


def is_likely_face_crop(image: np.ndarray, max_size: int = 250) -> bool:
    """Check if image is likely a pre-cropped face ROI.

    Args:
        image: Input image
        max_size: Maximum dimension to consider as a face crop

    Returns:
        True if image appears to be a pre-cropped face
    """
    h, w = image.shape[:2]
    # Small images with roughly square aspect ratio are likely face crops
    max_dim = max(h, w)
    aspect_ratio = max(h, w) / min(h, w) if min(h, w) > 0 else 999
    return max_dim <= max_size and aspect_ratio < 2.0


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

            h, w = image.shape[:2]
            logger.debug(f"  Image shape: {image.shape}")

            # Check if this is a pre-cropped face image
            if is_likely_face_crop(image):
                # Use the entire image as the face ROI
                logger.info(f"  → Pre-cropped face ({w}x{h}), skipping detection")
                face_roi = image
                # Use a default quality based on filename if present, otherwise estimate
                quality = 0.9  # Default high quality for pre-cropped faces
                # Try to extract quality from filename (e.g., mike_123_01_q0.98.jpg)
                quality_match = re.search(r'_q(\d+\.\d+)', img_path.name)
                if quality_match:
                    quality = float(quality_match.group(1))
            else:
                # Full image - need face detection
                face_bboxes, face_confs, _ = face_detector.detect(image)

                if len(face_bboxes) == 0:
                    logger.warning(f"  ✗ No face detected in {img_path.name}")
                    continue

                if len(face_bboxes) > 1:
                    logger.warning(f"  ✗ Multiple faces ({len(face_bboxes)}) in {img_path.name}, skipping to avoid false enrollment")
                    continue

                logger.info(f"  ✓ Detected 1 face")

                # Use the single detected face
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
    parser.add_argument(
        "--train-plda",
        action="store_true",
        help="Train PLDA model for second-stage scoring (requires 2+ identities)"
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

    # Face detector - choose based on detector_type config
    detector_type = config.detection.get("detector_type", "yunet")

    if detector_type == "scrfd":
        # SCRFD - GPU accelerated with proper landmark detection
        face_detector = SCRFDDetector(
            model_path=config.models.get("face_detector"),
            conf_threshold=config.detection["face_conf_threshold"],
            use_gpu=config.inference.get("use_gpu", True),
            gpu_device_id=config.inference.get("gpu_device_id", 0),
            max_detection_size=config.detection.get("max_detection_size", 640)
        )
    elif detector_type == "ultraface":
        # UltraFace - GPU accelerated but no landmark detection
        face_detector = UltraFaceDetector(
            model_path=config.models.get("face_detector"),
            conf_threshold=config.detection["face_conf_threshold"],
            use_gpu=config.inference.get("use_gpu", True),
            gpu_device_id=config.inference.get("gpu_device_id", 0),
            max_detection_size=config.detection.get("max_detection_size", 640)
        )
    else:
        # Default to YuNet (CPU-only via OpenCV)
        face_detector = FaceDetector(
            model_path=config.models.get("face_detector"),
            conf_threshold=config.detection["face_conf_threshold"],
            use_gpu=config.inference.get("use_gpu", True),
            gpu_device_id=config.inference.get("gpu_device_id", 0),
            detection_scale=config.detection.get("detection_scale", 1.0),
            max_detection_size=config.detection.get("max_detection_size", 640)
        )

    if not face_detector.load():
        logger.error("Failed to load face detector")
        return

    face_embedder = FaceEmbedder(
        model_path=config.models.get("face_embedder"),
        use_gpu=config.inference.get("use_gpu", True),
        gpu_device_id=config.inference.get("gpu_device_id", 0)
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

    # Train PLDA model if requested
    if args.train_plda:
        logger.info("Training PLDA model...")

        # Build labels list from metadata
        labels = [meta_store.get_person_id(i) for i in range(len(embeddings))]

        # Check if we have enough identities for PLDA
        unique_identities = set(labels)
        if len(unique_identities) < 2:
            logger.warning(
                f"PLDA requires at least 2 identities, found {len(unique_identities)}. "
                "Skipping PLDA training."
            )
        else:
            # Get PLDA config from config file
            plda_config = PLDAConfig(
                embedding_dim=embeddings.shape[1],
                latent_dim=config.get('plda.latent_dim', 128),
                between_class_reg=config.get('plda.between_class_reg', 0.1),
                within_class_reg=config.get('plda.within_class_reg', 0.3),
                min_eigenvalue=1e-4,
                min_samples_per_class=3
            )

            plda_scorer = PLDAScorer(config=plda_config)

            if plda_scorer.train(embeddings, labels):
                plda_path = config.get('plda.model_path', 'data/index/plda.npz')
                if plda_scorer.save(plda_path):
                    logger.info(f"Saved PLDA model to {plda_path}")
                    logger.info(
                        f"PLDA model: {len(unique_identities)} identities, "
                        f"{len(embeddings)} total samples, "
                        f"latent_dim={plda_config.latent_dim}, "
                        f"between_reg={plda_config.between_class_reg}, "
                        f"within_reg={plda_config.within_class_reg}"
                    )
                else:
                    logger.error("Failed to save PLDA model")
            else:
                logger.error("PLDA training failed")

    logger.info("Enrollment complete!")


if __name__ == "__main__":
    main()


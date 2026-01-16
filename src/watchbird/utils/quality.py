"""Quality assessment utilities for biometric features."""

from typing import Tuple

import numpy as np

from watchbird.utils.bbox_ops import box_size
from watchbird.utils.image_ops import compute_blur_metric


def compute_face_quality(
    face_bbox: np.ndarray,
    face_roi: np.ndarray,
    detection_conf: float,
    min_bbox_size: float = 50.0,
    blur_threshold: float = 100.0
) -> Tuple[float, dict]:
    """Compute face quality/reliability score.

    Args:
        face_bbox: Face bounding box [x1, y1, x2, y2]
        face_roi: Face region of interest image
        detection_conf: Detection confidence score
        min_bbox_size: Minimum acceptable bbox size
        blur_threshold: Blur threshold (Laplacian variance)

    Returns:
        Tuple of (quality_score, quality_metrics)
    """
    metrics = {}

    # Bbox size score
    width, height = box_size(face_bbox)
    bbox_size = min(width, height)
    size_score = min(1.0, bbox_size / (min_bbox_size * 2))
    metrics["bbox_size"] = bbox_size
    metrics["size_score"] = size_score

    # Detection confidence
    conf_score = detection_conf
    metrics["conf_score"] = conf_score

    # Blur score
    blur_metric = compute_blur_metric(face_roi)
    blur_score = min(1.0, blur_metric / (blur_threshold * 2))
    metrics["blur_metric"] = blur_metric
    metrics["blur_score"] = blur_score

    # Combined quality score (weighted average)
    quality = (
        0.3 * size_score +
        0.3 * conf_score +
        0.4 * blur_score
    )

    metrics["quality"] = quality

    return float(quality), metrics


def compute_reid_quality(
    person_bbox: np.ndarray,
    person_roi: np.ndarray,
    detection_conf: float,
    min_bbox_size: float = 100.0
) -> Tuple[float, dict]:
    """Compute ReID quality/reliability score.

    Args:
        person_bbox: Person bounding box [x1, y1, x2, y2]
        person_roi: Person region of interest image
        detection_conf: Detection confidence score
        min_bbox_size: Minimum acceptable bbox size

    Returns:
        Tuple of (quality_score, quality_metrics)
    """
    metrics = {}

    # Bbox size score
    width, height = box_size(person_bbox)
    bbox_size = min(width, height)
    size_score = min(1.0, bbox_size / (min_bbox_size * 2))
    metrics["bbox_size"] = bbox_size
    metrics["size_score"] = size_score

    # Detection confidence
    conf_score = detection_conf
    metrics["conf_score"] = conf_score

    # Truncation check (simple heuristic based on aspect ratio)
    aspect_ratio = width / max(height, 1)
    # Typical person aspect ratio is 0.4-0.6
    truncation_score = 1.0 - abs(aspect_ratio - 0.5) / 0.5
    truncation_score = max(0.0, min(1.0, truncation_score))
    metrics["aspect_ratio"] = aspect_ratio
    metrics["truncation_score"] = truncation_score

    # Combined quality score
    quality = (
        0.4 * size_score +
        0.3 * conf_score +
        0.3 * truncation_score
    )

    metrics["quality"] = quality

    return float(quality), metrics


def compute_gait_quality(
    track_duration: float,
    motion_magnitude: float,
    min_duration: float = 1.0
) -> Tuple[float, dict]:
    """Compute gait/motion quality score (placeholder for MVP).

    Args:
        track_duration: Track duration in seconds
        motion_magnitude: Total motion magnitude
        min_duration: Minimum track duration for reliable gait

    Returns:
        Tuple of (quality_score, quality_metrics)
    """
    metrics = {}

    # Duration score
    duration_score = min(1.0, track_duration / (min_duration * 2))
    metrics["duration"] = track_duration
    metrics["duration_score"] = duration_score

    # Motion score
    motion_score = min(1.0, motion_magnitude / 100.0)
    metrics["motion_magnitude"] = motion_magnitude
    metrics["motion_score"] = motion_score

    # Combined quality
    quality = (
        0.6 * duration_score +
        0.4 * motion_score
    )

    metrics["quality"] = quality

    return float(quality), metrics


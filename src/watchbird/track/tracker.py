"""Simple object tracker using centroid and IoU matching."""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from watchbird.utils.bbox_ops import compute_iou

logger = logging.getLogger(__name__)


class Track:
    """Represents a single tracked object."""

    def __init__(self, track_id: int, bbox: np.ndarray, confidence: float):
        """Initialize track.

        Args:
            track_id: Unique track identifier
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
        """
        self.track_id = track_id
        self.bbox = bbox
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.start_time = time.time()

        # Compute centroid
        self.centroid = self._compute_centroid(bbox)

        # Face association
        self.associated_face_bbox: Optional[np.ndarray] = None
        self.associated_face_conf: float = 0.0

    @staticmethod
    def _compute_centroid(bbox: np.ndarray) -> np.ndarray:
        """Compute centroid of bounding box.

        Args:
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Centroid [cx, cy]
        """
        return np.array([
            (bbox[0] + bbox[2]) / 2,
            (bbox[1] + bbox[3]) / 2
        ])

    def update(self, bbox: np.ndarray, confidence: float) -> None:
        """Update track with new detection.

        Args:
            bbox: New bounding box
            confidence: Detection confidence
        """
        self.bbox = bbox
        self.confidence = confidence
        self.centroid = self._compute_centroid(bbox)
        self.hits += 1
        self.time_since_update = 0

    def predict(self) -> np.ndarray:
        """Predict next position (simple: return current bbox).

        Returns:
            Predicted bounding box
        """
        return self.bbox

    @property
    def duration(self) -> float:
        """Get track duration in seconds."""
        return time.time() - self.start_time


class Tracker:
    """Simple tracker using centroid and IoU matching."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3
    ):
        """Initialize tracker.

        Args:
            max_age: Maximum frames to keep alive without update
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

    def update(
        self,
        detections: List[np.ndarray],
        confidences: List[float]
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detection bboxes [x1, y1, x2, y2]
            confidences: List of detection confidences

        Returns:
            List of active confirmed tracks
        """
        # Predict current positions
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1

        # Match detections to tracks
        matched_tracks, unmatched_detections = self._match_detections(
            detections,
            confidences
        )

        # Update matched tracks
        for track_id, (bbox, conf) in matched_tracks.items():
            self.tracks[track_id].update(bbox, conf)

        # Create new tracks for unmatched detections
        for bbox, conf in unmatched_detections:
            track = Track(self.next_id, bbox, conf)
            self.tracks[self.next_id] = track
            self.next_id += 1

        # Remove dead tracks
        self._remove_dead_tracks()

        # Return confirmed tracks
        return [
            track for track in self.tracks.values()
            if track.hits >= self.min_hits
        ]

    def _match_detections(
        self,
        detections: List[np.ndarray],
        confidences: List[float]
    ) -> Tuple[Dict[int, Tuple[np.ndarray, float]], List[Tuple[np.ndarray, float]]]:
        """Match detections to existing tracks.

        Args:
            detections: List of detection bboxes
            confidences: List of confidences

        Returns:
            Tuple of (matched_tracks, unmatched_detections)
        """
        if len(detections) == 0:
            return {}, []

        if len(self.tracks) == 0:
            unmatched = list(zip(detections, confidences))
            return {}, unmatched

        # Compute IoU matrix
        track_ids = list(self.tracks.keys())
        iou_matrix = np.zeros((len(detections), len(track_ids)))

        for i, det_bbox in enumerate(detections):
            for j, track_id in enumerate(track_ids):
                track_bbox = self.tracks[track_id].predict()
                iou_matrix[i, j] = compute_iou(det_bbox, track_bbox)

        # Greedy matching
        matched_tracks = {}
        matched_det_indices = set()

        # Sort by IoU (highest first)
        for _ in range(min(len(detections), len(track_ids))):
            max_iou = iou_matrix.max()

            if max_iou < self.iou_threshold:
                break

            det_idx, track_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)

            track_id = track_ids[track_idx]
            matched_tracks[track_id] = (detections[det_idx], confidences[det_idx])
            matched_det_indices.add(det_idx)

            # Invalidate matched row and column
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1

        # Unmatched detections
        unmatched_detections = [
            (detections[i], confidences[i])
            for i in range(len(detections))
            if i not in matched_det_indices
        ]

        return matched_tracks, unmatched_detections

    def _remove_dead_tracks(self) -> None:
        """Remove tracks that have been inactive too long."""
        dead_track_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.time_since_update > self.max_age
        ]

        for track_id in dead_track_ids:
            logger.debug(f"Removing dead track {track_id}")
            del self.tracks[track_id]

    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID.

        Args:
            track_id: Track identifier

        Returns:
            Track object or None if not found
        """
        return self.tracks.get(track_id)

    def associate_faces(
        self,
        face_bboxes: List[np.ndarray],
        face_confidences: List[float]
    ) -> Dict[int, Tuple[np.ndarray, float]]:
        """Associate face detections with person tracks.

        Args:
            face_bboxes: List of face bboxes
            face_confidences: List of face confidences

        Returns:
            Dict mapping track_id to (face_bbox, face_conf)
        """
        associations = {}

        for track_id, track in self.tracks.items():
            if track.hits < self.min_hits:
                continue

            # Find face bbox with highest overlap
            best_iou = 0.0
            best_face_idx = -1

            for i, face_bbox in enumerate(face_bboxes):
                iou = compute_iou(track.bbox, face_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_face_idx = i

            # Associate if overlap is sufficient
            if best_iou > 0.1:  # Loose threshold since face is inside person
                track.associated_face_bbox = face_bboxes[best_face_idx]
                track.associated_face_conf = face_confidences[best_face_idx]
                associations[track_id] = (
                    face_bboxes[best_face_idx],
                    face_confidences[best_face_idx]
                )

        return associations


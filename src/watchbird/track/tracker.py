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
        self.landmarks: Optional[np.ndarray] = None  # 5x2 facial landmarks
        self.head_tilt: float = 0.0  # Head tilt angle in degrees

        # Identity association (for track re-identification)
        self.identified_person_id: Optional[str] = None
        self.identification_confidence: float = 0.0
        self.identification_count: int = 0  # How many times this person was identified

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

    def set_identity(self, person_id: str, confidence: float) -> None:
        """Set identified person for this track.

        Args:
            person_id: Identified person ID
            confidence: Identification confidence
        """
        if person_id == self.identified_person_id:
            self.identification_count += 1
            # Update confidence with exponential moving average
            self.identification_confidence = (
                0.7 * self.identification_confidence + 0.3 * confidence
            )
        else:
            self.identified_person_id = person_id
            self.identification_confidence = confidence
            self.identification_count = 1

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
        iou_threshold: float = 0.3,
        enable_reid: bool = True,
        reid_timeout: float = 30.0
    ):
        """Initialize tracker.

        Args:
            max_age: Maximum frames to keep alive without update
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
            enable_reid: Enable track re-identification (same person = same track)
            reid_timeout: How long to keep dead tracks for re-identification (seconds)
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.enable_reid = enable_reid
        self.reid_timeout = reid_timeout

        self.tracks: Dict[int, Track] = {}
        self.next_id = 1

        # Store recently dead tracks for re-identification
        # Maps person_id -> (track_id, death_time, last_state_machine_data)
        self.dead_track_pool: Dict[str, Dict] = {}

    def update(
        self,
        detections: List[np.ndarray],
        confidences: List[float],
        landmarks_list: Optional[List[np.ndarray]] = None
    ) -> List[Track]:
        """Update tracker with new detections.

        Args:
            detections: List of detection bboxes [x1, y1, x2, y2]
            confidences: List of detection confidences
            landmarks_list: Optional list of 5x2 landmark arrays

        Returns:
            List of active confirmed tracks
        """
        # Predict current positions
        for track in self.tracks.values():
            track.age += 1
            track.time_since_update += 1

        # Build landmarks dict for easy lookup (skip None landmarks)
        landmarks_dict = {}
        if landmarks_list is not None:
            for i, landmarks in enumerate(landmarks_list):
                if landmarks is not None:
                    landmarks_dict[i] = landmarks

        # Match detections to tracks
        matched_tracks, unmatched_detections = self._match_detections(
            detections,
            confidences
        )

        # Update matched tracks with landmarks
        for track_id, (bbox, conf, det_idx) in matched_tracks.items():
            self.tracks[track_id].update(bbox, conf)
            # Store landmarks and compute head tilt
            if det_idx in landmarks_dict:
                lm = landmarks_dict[det_idx]
                self.tracks[track_id].landmarks = lm
                self.tracks[track_id].head_tilt = self._compute_head_tilt(lm)

        # Create new tracks for unmatched detections
        for bbox, conf, det_idx in unmatched_detections:
            track = Track(self.next_id, bbox, conf)
            # Store landmarks for new track
            if det_idx in landmarks_dict:
                lm = landmarks_dict[det_idx]
                track.landmarks = lm
                track.head_tilt = self._compute_head_tilt(lm)
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
    ) -> Tuple[Dict[int, Tuple[np.ndarray, float, int]], List[Tuple[np.ndarray, float, int]]]:
        """Match detections to existing tracks.

        Args:
            detections: List of detection bboxes
            confidences: List of confidences

        Returns:
            Tuple of (matched_tracks, unmatched_detections)
            - matched_tracks: Dict mapping track_id to (bbox, conf, det_idx)
            - unmatched_detections: List of (bbox, conf, det_idx)
        """
        if len(detections) == 0:
            return {}, []

        if len(self.tracks) == 0:
            unmatched = [(detections[i], confidences[i], i) for i in range(len(detections))]
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
            matched_tracks[track_id] = (detections[det_idx], confidences[det_idx], int(det_idx))
            matched_det_indices.add(det_idx)

            # Invalidate matched row and column
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1

        # Unmatched detections
        unmatched_detections = [
            (detections[i], confidences[i], i)
            for i in range(len(detections))
            if i not in matched_det_indices
        ]

        return matched_tracks, unmatched_detections

    @staticmethod
    def _compute_head_tilt(landmarks: np.ndarray) -> float:
        """Compute head tilt angle from eye landmarks.

        Args:
            landmarks: 5x2 array of landmarks (right_eye, left_eye, nose, right_mouth, left_mouth)

        Returns:
            Head tilt angle in degrees (positive = clockwise tilt)
        """
        if landmarks is None or len(landmarks) < 2:
            return 0.0

        right_eye = landmarks[0]
        left_eye = landmarks[1]

        # Calculate angle between eyes
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))
        return float(angle)

    def _remove_dead_tracks(self) -> None:
        """Remove tracks that have been inactive too long."""
        current_time = time.time()
        dead_track_ids = [
            track_id for track_id, track in self.tracks.items()
            if track.time_since_update > self.max_age
        ]

        for track_id in dead_track_ids:
            track = self.tracks[track_id]

            # If track was identified, store for re-identification
            if self.enable_reid and track.identified_person_id is not None:
                person_id = track.identified_person_id
                self.dead_track_pool[person_id] = {
                    'original_track_id': track_id,
                    'death_time': current_time,
                    'confidence': track.identification_confidence,
                    'identification_count': track.identification_count,
                    'last_bbox': track.bbox.copy(),
                    'last_centroid': track.centroid.copy()
                }
                logger.debug(
                    f"Track {track_id} ({person_id}) added to re-id pool "
                    f"(conf={track.identification_confidence:.3f})"
                )

            logger.debug(f"Removing dead track {track_id}")
            del self.tracks[track_id]

        # Clean up old entries in dead track pool
        if self.enable_reid:
            expired_persons = [
                person_id for person_id, info in self.dead_track_pool.items()
                if current_time - info['death_time'] > self.reid_timeout
            ]
            for person_id in expired_persons:
                logger.debug(f"Removing expired re-id entry for {person_id}")
                del self.dead_track_pool[person_id]

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

    def check_reid(self, track_id: int, person_id: str, confidence: float) -> Optional[Dict]:
        """Check if a newly identified track matches a recently lost track.

        Call this when a track gets identified. If the person_id matches
        a recently dead track, this returns info for linking them.

        Args:
            track_id: Current track ID
            person_id: Identified person ID
            confidence: Identification confidence

        Returns:
            Dict with re-id info if match found, None otherwise
        """
        if not self.enable_reid:
            return None

        # Update track's identity
        if track_id in self.tracks:
            self.tracks[track_id].set_identity(person_id, confidence)

        # Check dead track pool for matching person
        if person_id in self.dead_track_pool:
            info = self.dead_track_pool[person_id]

            # Remove from pool since we're linking
            del self.dead_track_pool[person_id]

            logger.info(
                f"Re-identification: Track {track_id} → same as previous "
                f"Track {info['original_track_id']} ({person_id})"
            )

            return {
                'original_track_id': info['original_track_id'],
                'person_id': person_id,
                'previous_confidence': info['confidence'],
                'time_gap': time.time() - info['death_time']
            }

        return None

    def get_active_person_track(self, person_id: str) -> Optional[int]:
        """Get the active track ID for a given person.

        Used to check if a person is already being tracked.

        Args:
            person_id: Person identifier

        Returns:
            Track ID if person is currently tracked, None otherwise
        """
        for track_id, track in self.tracks.items():
            if track.identified_person_id == person_id:
                return track_id
        return None

    def merge_tracks(self, old_track_id: int, new_track_id: int) -> bool:
        """Merge new track into old track (keep old track ID).

        This is used when re-identification determines two tracks
        are the same person.

        Args:
            old_track_id: Original track ID to keep
            new_track_id: New track ID to merge into old

        Returns:
            True if merge successful, False otherwise
        """
        if new_track_id not in self.tracks:
            return False

        new_track = self.tracks[new_track_id]

        # If old track is still active, update it with new track's position
        if old_track_id in self.tracks:
            old_track = self.tracks[old_track_id]
            old_track.update(new_track.bbox, new_track.confidence)
            old_track.identified_person_id = new_track.identified_person_id
            old_track.identification_confidence = max(
                old_track.identification_confidence,
                new_track.identification_confidence
            )
            del self.tracks[new_track_id]
            logger.debug(f"Merged track {new_track_id} into active track {old_track_id}")
            return True

        # Old track is dead - can't merge, but we've already linked via re-id
        return False


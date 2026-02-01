#!/usr/bin/env python3
"""Runtime recognition application."""

import argparse
import logging
import time
from typing import Dict, Optional

import numpy as np

from watchbird.camera.usb_backend import USBCameraBackend
from watchbird.camera.video_backend import VideoBackend
try:
    from watchbird.camera.picamera_backend import Picamera2Backend
except ImportError:
    Picamera2Backend = None  # type: ignore
from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.detect.ultraface_detector import UltraFaceDetector
from watchbird.detect.scrfd_detector import SCRFDDetector
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.fusion.embedding_aggregator import TrackEmbeddingManager
from watchbird.fusion.similarity import SimilarityFusion
from watchbird.fusion.unified_scorer import create_scorer_from_config
from watchbird.index.faiss_wrapper import FaissIndex
from watchbird.index.meta_store import MetaStore
from watchbird.runtime.events import EventEmitter
from watchbird.runtime.state_machine import TrackStateMachine
from watchbird.stream.mjpeg_server import MJPEGServer, draw_detection_boxes
from watchbird.track.tracker import Tracker
from watchbird.utils.image_ops import extract_roi
from watchbird.utils.quality import compute_face_quality
from watchbird.utils.resolution_tuner import find_optimal_resolution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_scrfd_landmarks_to_roi(landmarks: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Convert SCRFD landmarks from frame coordinates to ROI coordinates for embedder.

    SCRFD and the embedder use different landmark ordering conventions:
    - SCRFD: landmarks are labeled left/right from the subject's perspective
    - Embedder (ArcFace reference): landmarks are labeled left/right from image perspective

    This function:
    1. Shifts landmarks from frame coords to ROI coords (subtracts bbox origin)
    2. Reorders landmarks to match embedder expectation (swaps 0↔1 and 3↔4)

    Args:
        landmarks: SCRFD landmarks in frame coordinates, shape (5, 2)
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Landmarks in ROI coordinates, reordered for embedder, shape (5, 2)
    """
    x1, y1 = int(bbox[0]), int(bbox[1])
    lm = landmarks.copy()
    lm[:, 0] -= x1  # Shift X to ROI coords
    lm[:, 1] -= y1  # Shift Y to ROI coords

    # Reorder: swap indices 0↔1 (eyes) and 3↔4 (mouth corners)
    return np.array([
        lm[1],  # SCRFD[1] "right eye" → embedder[0]
        lm[0],  # SCRFD[0] "left eye" → embedder[1]
        lm[2],  # nose (unchanged)
        lm[4],  # SCRFD[4] "right mouth" → embedder[3]
        lm[3],  # SCRFD[3] "left mouth" → embedder[4]
    ], dtype=np.float32)


class RecognitionPipeline:
    """Main recognition pipeline."""

    def __init__(self, config: Config):
        """Initialize recognition pipeline.

        Args:
            config: Configuration object
        """
        self.config = config

        # Components
        self.camera = None
        self.face_detector = None
        self.face_embedder = None
        self.faiss_index = None
        self.meta_store = None
        self.unified_scorer = None  # New: unified FAISS + PLDA scorer
        self.tracker = None
        self.fusion = None
        self.event_emitter = None
        self.mjpeg_server = None
        self.embedding_manager = None  # New: embedding aggregation

        # Track state machines
        self.track_states: Dict[int, TrackStateMachine] = {}

        # Frame counters per track for sampling embeddings
        self.track_frame_counters: Dict[int, int] = {}

        # Track recovery: store embeddings of recently lost CONFIRMED/FRIENDLY tracks
        # Format: {lost_time: (person_id, embedding, cumulative_confidence, bbox_center)}
        self.lost_tracks: Dict[float, tuple] = {}
        self.track_recovery_timeout = 10.0  # Will be updated from config in initialize()
        self.track_recovery_threshold = 0.65  # Will be updated from config in initialize()

        # Store last known bounding box for each track (for recovery spatial matching)
        self.track_last_bbox: Dict[int, np.ndarray] = {}

    def initialize(self, backend: str, video_path: Optional[str] = None, device_id: int = 0) -> bool:
        """Initialize all components.

        Args:
            backend: Camera backend (usb, picamera2, or video_file)
            video_path: Path to video file (for video_file backend)
            device_id: Camera device ID (for usb backend, default 0)

        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing components...")

        # Camera
        if backend == "usb":
            self.camera = USBCameraBackend(
                device_id=device_id,
                resolution=tuple(self.config.camera["resolution"]),
                fps=self.config.camera["fps"]
            )
        elif backend == "picamera2":
            if Picamera2Backend is None:
                logger.error("Picamera2 not available on this platform")
                return False
            self.camera = Picamera2Backend(
                resolution=tuple(self.config.camera["resolution"]),
                fps=self.config.camera["fps"]
            )
        elif backend == "video_file":
            if not video_path:
                logger.error("Video path required for video_file backend")
                return False
            self.camera = VideoBackend(
                video_path=video_path,
                resolution=tuple(self.config.camera["resolution"]),
                fps=self.config.camera["fps"]
            )
        else:
            logger.error(f"Unknown backend: {backend}")
            return False

        # For USB camera with auto-resolution, determine optimal resolution BEFORE opening
        camera_resolution = tuple(self.config.camera["resolution"])

        if backend == "usb" and self.config.camera.get("auto_resolution", False):
            target_fps = self.config.camera.get("target_fps", 10.0)
            min_fps = self.config.camera.get("min_fps", 8.0)

            # Load face detector first (needed for FPS measurement)
            temp_detector = FaceDetector(
                model_path=self.config.models.get("face_detector"),
                conf_threshold=self.config.detection["face_conf_threshold"],
                use_gpu=self.config.inference.get("use_gpu", True),
                gpu_device_id=self.config.inference.get("gpu_device_id", 0),
                detection_scale=self.config.detection.get("detection_scale", 1.0),
                max_detection_size=self.config.detection.get("max_detection_size", 640)
            )

            if temp_detector.load():
                logger.info(f"Auto-tuning resolution for target {target_fps} FPS...")
                camera_resolution = find_optimal_resolution(
                    device_id=device_id,
                    detector=temp_detector,
                    target_fps=target_fps,
                    min_fps=min_fps,
                    test_frames=30
                )
                logger.info(f"Selected resolution: {camera_resolution[0]}x{camera_resolution[1]}")
                # Wait for camera to fully release (Windows MSMF needs this)
                time.sleep(1.0)
            else:
                logger.warning("Could not load detector for auto-tuning, using config resolution")

            # Update camera object with optimal resolution
            self.camera.resolution = camera_resolution

        if not self.camera.open():
            logger.error("Failed to open camera")
            return False

        # Face detector - choose based on detector_type config
        detector_type = self.config.detection.get("detector_type", "yunet")

        if detector_type == "scrfd":
            # SCRFD - GPU accelerated with proper landmark detection
            self.face_detector = SCRFDDetector(
                model_path=self.config.models.get("face_detector"),
                conf_threshold=self.config.detection["face_conf_threshold"],
                use_gpu=self.config.inference.get("use_gpu", True),
                gpu_device_id=self.config.inference.get("gpu_device_id", 0),
                max_detection_size=self.config.detection.get("max_detection_size", 640)
            )
        elif detector_type == "ultraface":
            # UltraFace - GPU accelerated but no landmark detection
            self.face_detector = UltraFaceDetector(
                model_path=self.config.models.get("face_detector"),
                conf_threshold=self.config.detection["face_conf_threshold"],
                use_gpu=self.config.inference.get("use_gpu", True),
                gpu_device_id=self.config.inference.get("gpu_device_id", 0),
                max_detection_size=self.config.detection.get("max_detection_size", 640)
            )
        else:
            # Default to YuNet (CPU-only via OpenCV)
            self.face_detector = FaceDetector(
                model_path=self.config.models.get("face_detector"),
                conf_threshold=self.config.detection["face_conf_threshold"],
                use_gpu=self.config.inference.get("use_gpu", True),
                gpu_device_id=self.config.inference.get("gpu_device_id", 0),
                detection_scale=self.config.detection.get("detection_scale", 1.0),
                max_detection_size=self.config.detection.get("max_detection_size", 640)
            )

        if not self.face_detector.load():
            logger.error("Failed to load face detector")
            return False


        # Face embedder
        self.face_embedder = FaceEmbedder(
            model_path=self.config.models.get("face_embedder"),
            use_gpu=self.config.inference.get("use_gpu", True),
            gpu_device_id=self.config.inference.get("gpu_device_id", 0)
        )

        if not self.face_embedder.load():
            logger.error("Failed to load face embedder")
            return False

        # FAISS index
        self.faiss_index = FaissIndex()
        index_path = self.config.index["face_index_path"]

        if not self.faiss_index.load(index_path):
            logger.error("Failed to load FAISS index")
            return False

        # Metadata
        self.meta_store = MetaStore(self.config.index["meta_path"])

        if not self.meta_store.load():
            logger.error("Failed to load metadata")
            return False

        # Unified scorer (FAISS + optional PLDA)
        self.unified_scorer = create_scorer_from_config(
            faiss_index=self.faiss_index,
            meta_store=self.meta_store,
            config=self.config
        )

        scoring_info = self.unified_scorer.get_scoring_info()
        logger.info(f"Scoring backend: {scoring_info['backend']}")
        if scoring_info['plda_available']:
            logger.info(
                f"PLDA enabled: {scoring_info['plda_identities']} identities, "
                f"LLR threshold={scoring_info['plda_llr_threshold']:.2f}"
            )

        # Tracker
        # Note: enable_reid=False because with low separation margin between identities,
        # re-identification tends to propagate wrong identity assignments
        self.tracker = Tracker(
            max_age=self.config.tracking["max_age"],
            min_hits=self.config.tracking["min_hits"],
            iou_threshold=self.config.tracking["iou_threshold"],
            enable_reid=False
        )

        # Embedding aggregation manager
        # This collects multiple embeddings per track and compares the centroid
        # against the database for more robust recognition
        embedding_window = self.config.fusion.get("embedding_window", 20)
        embedding_min = self.config.fusion.get("embedding_min", 6)
        embedding_quality = self.config.fusion.get("embedding_quality_threshold", 0.4)
        embedding_outlier = self.config.fusion.get("embedding_outlier_threshold", 0.25)
        self.embedding_manager = TrackEmbeddingManager(
            window_size=embedding_window,
            min_embeddings=embedding_min,
            quality_threshold=embedding_quality,
            outlier_threshold=embedding_outlier
        )
        logger.info(
            f"Embedding aggregation: window={embedding_window}, min={embedding_min}, "
            f"quality_thresh={embedding_quality}, outlier_thresh={embedding_outlier}"
        )

        # Fusion
        self.fusion = SimilarityFusion()

        # Event emitter
        self.event_emitter = EventEmitter()

        # MJPEG server
        if self.config.stream.get("enabled", True):
            self.mjpeg_server = MJPEGServer(
                host=self.config.stream.get("host", "0.0.0.0"),
                port=self.config.stream.get("port", 8080)
            )
            self.mjpeg_server.start()

        # Load track recovery config
        self.track_recovery_timeout = self.config.thresholds.get("track_recovery_timeout", 10.0)
        self.track_recovery_threshold = self.config.thresholds.get("track_recovery_threshold", 0.65)

        logger.info("Initialization complete!")
        return True

    def _try_match_active_track(
        self,
        embedding: np.ndarray,
        bbox: np.ndarray,
        exclude_track_id: int
    ) -> Optional[tuple]:
        """Try to match a new track against active FRIENDLY/CONFIRMED tracks.

        This handles the case where a face briefly disappears and reappears
        as a new track, while the old track is still "active" in the tracker.

        Args:
            embedding: New track's embedding
            bbox: New track's bounding box
            exclude_track_id: Track ID to exclude (the new track itself)

        Returns:
            (person_id, cumulative_confidence) if matched, None otherwise
        """
        bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        best_match = None
        best_score = 0.0

        for track_id, state_machine in self.track_states.items():
            if track_id == exclude_track_id:
                continue

            # Only consider FRIENDLY or CONFIRMED tracks with good confidence
            if state_machine.state.value not in ("FRIENDLY", "CONFIRMED"):
                continue
            if state_machine.person_id is None:
                continue
            if state_machine.cumulative_confidence < 0.3:
                continue

            # Check spatial proximity first
            last_bbox = self.track_last_bbox.get(track_id)
            if last_bbox is None:
                continue

            old_center = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)
            spatial_dist = np.sqrt(
                (bbox_center[0] - old_center[0])**2 +
                (bbox_center[1] - old_center[1])**2
            )

            # Only consider if spatially close (within 300 pixels)
            if spatial_dist > 300:
                continue

            # Get embedding from the active track
            active_embedding = self.embedding_manager.get_centroid(track_id)
            if active_embedding is None:
                active_embedding = self.embedding_manager.get_last_embedding(track_id)
            if active_embedding is None:
                continue

            # Compute cosine similarity
            similarity = np.dot(embedding, active_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(active_embedding) + 1e-6
            )

            # Boost for spatial proximity
            proximity_boost = max(0, (300 - spatial_dist) / 300) * 0.1
            combined_score = similarity + proximity_boost

            logger.debug(
                f"Active track match candidate: track={track_id}, person={state_machine.person_id}, "
                f"sim={similarity:.3f}, dist={spatial_dist:.0f}px, score={combined_score:.3f}"
            )

            if combined_score > best_score and similarity > self.track_recovery_threshold:
                best_score = combined_score
                best_match = (state_machine.person_id, state_machine.cumulative_confidence, track_id)

        if best_match:
            person_id, cumulative_conf, matched_track_id = best_match
            logger.info(
                f"Active track match: {person_id} from track {matched_track_id} "
                f"(score={best_score:.3f}, conf={cumulative_conf:.2f})"
            )
            return (person_id, cumulative_conf)

        return None

    def _store_lost_track(
        self,
        person_id: str,
        embedding: np.ndarray,
        cumulative_confidence: float,
        bbox: np.ndarray
    ) -> None:
        """Store a lost CONFIRMED/FRIENDLY track for potential recovery.

        Args:
            person_id: The identified person
            embedding: Last known embedding
            cumulative_confidence: Cumulative confidence at time of loss
            bbox: Last known bounding box
        """
        # Calculate bbox center for spatial matching
        bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        # Store with current time as key
        self.lost_tracks[time.time()] = (person_id, embedding, cumulative_confidence, bbox_center)

        # Clean up old entries
        current_time = time.time()
        old_count = len(self.lost_tracks)
        self.lost_tracks = {
            t: v for t, v in self.lost_tracks.items()
            if current_time - t < self.track_recovery_timeout
        }
        removed = old_count - len(self.lost_tracks)

        logger.info(
            f"Stored lost track for recovery: {person_id} "
            f"(conf={cumulative_confidence:.2f}, center={bbox_center}). "
            f"Total: {len(self.lost_tracks)} lost tracks"
        )

    def _try_track_recovery(
        self,
        embedding: np.ndarray,
        bbox: np.ndarray
    ) -> Optional[tuple]:
        """Try to recover a lost track by matching embedding.

        Args:
            embedding: New track's embedding
            bbox: New track's bounding box

        Returns:
            (person_id, cumulative_confidence) if recovered, None otherwise
        """
        if not self.lost_tracks:
            return None

        current_time = time.time()
        bbox_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

        best_match = None
        best_similarity = self.track_recovery_threshold

        for lost_time, (person_id, lost_embedding, cumulative_conf, lost_center) in list(self.lost_tracks.items()):
            # Skip if too old
            age = current_time - lost_time
            if age > self.track_recovery_timeout:
                continue

            # Compute cosine similarity
            similarity = np.dot(embedding, lost_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(lost_embedding) + 1e-6
            )

            # Boost similarity if spatially close (within 200 pixels)
            spatial_dist = np.sqrt(
                (bbox_center[0] - lost_center[0])**2 +
                (bbox_center[1] - lost_center[1])**2
            )
            if spatial_dist < 200:
                similarity += 0.05  # Small boost for spatial proximity

            logger.debug(
                f"Recovery candidate: {person_id}, sim={similarity:.3f}, "
                f"threshold={self.track_recovery_threshold}, age={age:.1f}s, dist={spatial_dist:.0f}px"
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = (person_id, cumulative_conf, lost_time)

        if best_match:
            person_id, cumulative_conf, lost_time = best_match
            # Don't remove from lost tracks immediately - it might be needed for other new tracks
            # in the same frame. The entry will expire naturally after track_recovery_timeout.
            # Only remove if the similarity is very high (definite match)
            if best_similarity > 0.7:
                del self.lost_tracks[lost_time]

            logger.info(
                f"Track recovery: {person_id} (similarity={best_similarity:.3f}, "
                f"conf={cumulative_conf:.2f}, age={current_time - lost_time:.1f}s)"
            )
            return (person_id, cumulative_conf)

        # Log if we had candidates but none matched
        if self.lost_tracks:
            # Find the best similarity we found (even if below threshold)
            best_sim_found = 0.0
            best_person_found = None
            for lost_time, (person_id, lost_embedding, cumulative_conf, lost_center) in self.lost_tracks.items():
                age = current_time - lost_time
                if age <= self.track_recovery_timeout:
                    similarity = np.dot(embedding, lost_embedding) / (
                        np.linalg.norm(embedding) * np.linalg.norm(lost_embedding) + 1e-6
                    )
                    if similarity > best_sim_found:
                        best_sim_found = similarity
                        best_person_found = person_id

            if best_person_found:
                logger.info(
                    f"Track recovery failed: best candidate {best_person_found} "
                    f"sim={best_sim_found:.3f} < threshold={self.track_recovery_threshold}"
                )

        return None

    def _cleanup_lost_tracks(self, active_track_ids: set, tracks: list) -> None:
        """Cleanup tracks that are no longer active and store for recovery.

        IMPORTANT: This should be called BEFORE processing new tracks,
        so that lost track embeddings are available for recovery.

        Args:
            active_track_ids: Set of currently active track IDs
            tracks: List of current tracks (for bbox info)
        """
        # Only delete tracks that are NOT in the current tracks list at all
        lost_track_ids = set(self.track_frame_counters.keys()) - active_track_ids

        for track_id in lost_track_ids:
            # Before deleting, store CONFIRMED/FRIENDLY tracks for potential recovery
            if track_id in self.track_states:
                state_machine = self.track_states[track_id]
                if (state_machine.state.value in ("CONFIRMED", "FRIENDLY") and
                    state_machine.person_id is not None and
                    state_machine.cumulative_confidence > 0.3):

                    # Get the centroid embedding for this track (more robust than last embedding)
                    # Fall back to last embedding if centroid not available
                    recovery_embedding = self.embedding_manager.get_centroid(track_id)
                    if recovery_embedding is None:
                        recovery_embedding = self.embedding_manager.get_last_embedding(track_id)

                    if recovery_embedding is not None:
                        # Use stored last known bbox if available
                        last_bbox = self.track_last_bbox.get(track_id)

                        if last_bbox is not None:
                            self._store_lost_track(
                                state_machine.person_id,
                                recovery_embedding,
                                state_machine.cumulative_confidence,
                                last_bbox
                            )
                        else:
                            logger.debug(f"Track {track_id}: No last bbox available for recovery storage")

                del self.track_states[track_id]

            if track_id in self.track_frame_counters:
                del self.track_frame_counters[track_id]
            if track_id in self.track_last_bbox:
                del self.track_last_bbox[track_id]

        # Also cleanup embedding manager
        self.embedding_manager.cleanup_stale_tracks(active_track_ids)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame.

        Args:
            frame: Input frame

        Returns:
            Annotated frame
        """
        # Detect faces (for MVP, we'll use faces as "persons")
        face_bboxes, face_confs, face_landmarks = self.face_detector.detect(frame)

        # Update tracker with face detections (treating faces as persons for MVP)
        tracks = self.tracker.update(face_bboxes, face_confs, face_landmarks)

        # IMPORTANT: Cleanup lost tracks BEFORE processing new tracks
        # This ensures lost track embeddings are available for recovery
        active_track_ids = {track.track_id for track in tracks}
        self._cleanup_lost_tracks(active_track_ids, tracks)

        # Get embedding sample interval from config
        embedding_sample_interval = self.config.fusion.get("embedding_sample_interval", 1)

        # Process each confirmed track
        for track in tracks:
            track_id = track.track_id

            # Store last known bbox for recovery spatial matching
            self.track_last_bbox[track_id] = track.bbox.copy() if hasattr(track.bbox, 'copy') else np.array(track.bbox)

            # Create state machine if new track
            if track_id not in self.track_states:
                logger.info(f"New track {track_id} created, lost_tracks available: {len(self.lost_tracks)}")
                self.track_states[track_id] = TrackStateMachine(
                    track_id=track_id,
                    t_accept=self.config.thresholds["t_accept"],
                    t_margin=self.config.thresholds["t_margin"],
                    t_timeout=self.config.thresholds["t_timeout"],
                    consistency_count=self.config.fusion["consistency_count"],
                    window_size=self.config.fusion["window_size"],
                    confidence_decay_threshold=self.config.fusion.get("confidence_decay_threshold", 3),
                    identity_switch_margin=self.config.thresholds.get("identity_switch_margin", 0.10),
                    # Cumulative confidence parameters
                    confirm_threshold=self.config.thresholds.get("confirm_threshold", 0.95),
                    confidence_gain_rate=self.config.thresholds.get("confidence_gain_rate", 0.05),
                    confidence_decay_rate=self.config.thresholds.get("confidence_decay_rate", 0.02),
                    track_lost_timeout=self.config.thresholds.get("track_lost_timeout", 3.0)
                )
                # Initialize frame counter for this track
                self.track_frame_counters[track_id] = 0

                # Try track recovery for new tracks
                # Extract one embedding to check against lost tracks AND active tracks
                face_roi = extract_roi(frame, track.bbox)
                recovery_done = False

                # Convert landmarks for embedder
                recovery_landmarks = None
                if track.landmarks is not None:
                    recovery_landmarks = convert_scrfd_landmarks_to_roi(track.landmarks, track.bbox)

                if face_roi is not None and face_roi.size > 0:
                    recovery_embedding = self.face_embedder.extract(face_roi, landmarks=recovery_landmarks)
                    if recovery_embedding is not None:
                        # First try lost tracks
                        if self.lost_tracks:
                            recovery_result = self._try_track_recovery(recovery_embedding, track.bbox)
                            if recovery_result:
                                person_id, cumulative_conf = recovery_result
                                self.track_states[track_id].fast_recover(person_id, cumulative_conf)
                                # Store the recovery embedding
                                self.embedding_manager.add_embedding(
                                    track_id, recovery_embedding, quality=0.8,
                                    blur_score=0.8, brightness=0.5, face_size=50
                                )
                                recovery_done = True

                        # If no lost track match, try matching against active FRIENDLY/CONFIRMED tracks
                        # This handles the case where tracker still has the old track active
                        # Can be disabled via config if causing false positives
                        if not recovery_done and self.config.thresholds.get("active_track_matching", True):
                            active_match = self._try_match_active_track(recovery_embedding, track.bbox, track_id)
                            if active_match:
                                person_id, cumulative_conf = active_match
                                self.track_states[track_id].fast_recover(person_id, cumulative_conf)
                                self.embedding_manager.add_embedding(
                                    track_id, recovery_embedding, quality=0.8,
                                    blur_score=0.8, brightness=0.5, face_size=50
                                )
                                recovery_done = True

                        if recovery_done:
                            continue  # Skip normal processing for this frame
                    else:
                        logger.debug(f"Track {track_id}: Recovery skipped (embedding extraction failed)")

            state_machine = self.track_states[track_id]

            # For CONFIRMED tracks: only track, skip recognition
            if state_machine.should_skip_recognition():
                # Notify that track is still visible (keeps CONFIRMED state)
                state_machine.notify_track_seen()
                # Still check for state changes (e.g., track lost timeout)
                state_machine._check_transition()

                # Periodically extract embedding to keep buffer fresh for recovery
                # Only every 30 frames (about once per second at 30fps)
                if self.track_frame_counters.get(track_id, 0) % 30 == 0:
                    face_roi = extract_roi(frame, track.bbox)
                    if face_roi is not None and face_roi.size > 0:
                        # Convert landmarks for embedder
                        refresh_landmarks = None
                        if track.landmarks is not None:
                            refresh_landmarks = convert_scrfd_landmarks_to_roi(track.landmarks, track.bbox)

                        embedding = self.face_embedder.extract(face_roi, landmarks=refresh_landmarks)
                        if embedding is not None:
                            self.embedding_manager.add_embedding(
                                track_id, embedding, quality=0.7,
                                blur_score=0.7, brightness=0.5, face_size=50
                            )

                self.track_frame_counters[track_id] = self.track_frame_counters.get(track_id, 0) + 1
                continue

            # Skip if already in terminal state (shouldn't happen with new logic)
            if state_machine.is_terminal():
                continue

            # Increment frame counter for this track
            self.track_frame_counters[track_id] += 1

            # Apply frame sampling - only extract embeddings every N frames
            if self.track_frame_counters[track_id] % embedding_sample_interval != 0:
                # Skip embedding extraction on this frame, but still track the face
                continue

            # Extract face ROI
            face_roi = extract_roi(frame, track.bbox)

            # Convert SCRFD landmarks to ROI coordinates for embedder
            roi_landmarks = None
            if track.landmarks is not None:
                roi_landmarks = convert_scrfd_landmarks_to_roi(track.landmarks, track.bbox)

            # Compute face quality
            quality, quality_details = compute_face_quality(
                track.bbox,
                face_roi,
                track.confidence,
                min_bbox_size=self.config.quality["min_bbox_size"],
                blur_threshold=self.config.quality.get("blur_threshold", 150.0)
            )

            # Skip if quality too low (includes blur-based rejection)
            if quality < self.config.quality["min_face_quality"]:
                # Log blur rejection for debugging
                blur_metric = quality_details.get("blur_metric", 0)
                blur_threshold = self.config.quality.get("blur_threshold", 150.0)
                if blur_metric < blur_threshold * 0.3:
                    logger.debug(
                        f"Track {track_id}: Rejected blurry frame "
                        f"(blur={blur_metric:.1f} < {blur_threshold * 0.3:.1f})"
                    )
                continue

            # Extract embedding using SCRFD landmarks (faster than re-detecting)
            embedding = self.face_embedder.extract(face_roi, landmarks=roi_landmarks)

            if embedding is None:
                continue

            # Use actual quality metrics from compute_face_quality
            blur_score = quality_details.get("blur_score", 0.5)

            # Simple brightness from detection confidence
            brightness = 0.5  # Assume optimal, skip gray conversion

            # Face size from quality computation
            face_size = int(quality_details.get("bbox_size", 50))

            # Add embedding to aggregator for this track with quality metrics
            self.embedding_manager.add_embedding(
                track_id,
                embedding,
                quality=quality,
                blur_score=blur_score,
                brightness=brightness,
                face_size=face_size
            )

            # Check if we have enough embeddings for reliable matching
            if not self.embedding_manager.is_ready(track_id):
                # Not enough embeddings yet, skip scoring this frame
                emb_count = self.embedding_manager.get_embedding_count(track_id)
                logger.debug(
                    f"Track {track_id}: collecting embeddings ({emb_count}/"
                    f"{self.config.fusion.get('embedding_min', 5)})"
                )
                continue

            # Get centroid embedding for this track
            centroid_embedding = self.embedding_manager.get_centroid(track_id)

            if centroid_embedding is None:
                continue

            # Check embedding variance - high variance indicates unstable/confused track
            variance = self.embedding_manager.get_variance(track_id)
            if variance > 0.15:  # High variance threshold
                logger.debug(
                    f"Track {track_id}: high embedding variance ({variance:.3f}), "
                    f"recognition may be unreliable"
                )

            # Score using unified scorer with AGGREGATED centroid embedding
            # This is more robust than scoring individual frames
            best_person_id, best_score, margin, all_scores = self.unified_scorer.score(centroid_embedding)

            if best_person_id is None:
                continue

            second_score = best_score - margin

            # Check for track re-identification (same person from lost track)
            if best_person_id is not None and best_score > 0.5:
                reid_info = self.tracker.check_reid(track_id, best_person_id, best_score)
                if reid_info is not None:
                    # This track is the same person as a previously lost track
                    # Transfer state machine from new track to maintain continuity
                    original_track_id = reid_info['original_track_id']

                    # If we still have state for the original track, use it
                    if original_track_id in self.track_states:
                        # Keep the original state machine, just update with current track
                        old_state = self.track_states[original_track_id]
                        self.track_states[track_id] = old_state
                        old_state.track_id = track_id  # Update track ID reference
                        del self.track_states[original_track_id]
                        logger.info(
                            f"Re-linked track {track_id} to previous track {original_track_id} "
                            f"({best_person_id}, gap={reid_info['time_gap']:.1f}s)"
                        )

            # Update state machine
            state_changed = state_machine.update(
                best_person_id=best_person_id,
                best_score=best_score,
                second_score=second_score,
                reliability=quality,
                modality="face"
            )

            # Emit event if state changed
            if state_changed:
                event = state_machine.get_event()
                self.event_emitter.emit(event)


        # Draw annotations
        annotated_frame = frame.copy()

        for track in tracks:
            track_id = track.track_id

            # Skip drawing stale tracks (not updated in recent frames)
            # This prevents empty bounding boxes from lingering when face is lost
            if track.time_since_update > 0:
                continue

            # Get state machine info if exists
            if track_id in self.track_states:
                state_machine = self.track_states[track_id]

                annotated_frame = draw_detection_boxes(
                    annotated_frame,
                    track_id=track_id,
                    bbox=track.bbox,
                    state=state_machine.state.value,
                    person_id=state_machine.person_id,
                    confidence=state_machine.confidence,
                    cumulative_confidence=state_machine.cumulative_confidence,
                    head_tilt=track.head_tilt,
                    landmarks=track.landmarks
                )
            else:
                # Draw basic detection box for untracked faces
                annotated_frame = draw_detection_boxes(
                    annotated_frame,
                    track_id=track_id,
                    bbox=track.bbox,
                    state="DETECTING",
                    person_id=None,
                    confidence=track.confidence,
                    cumulative_confidence=0.0,
                    head_tilt=track.head_tilt,
                    landmarks=track.landmarks
                )

        return annotated_frame

    def run(self) -> None:
        """Run recognition pipeline."""
        logger.info("Starting recognition pipeline...")

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                # Get frame
                frame = self.camera.get_frame()

                if frame is None:
                    logger.info("No more frames")
                    break

                # Process frame
                annotated_frame = self.process_frame(frame)

                # Update MJPEG server
                if self.mjpeg_server:
                    self.mjpeg_server.update_frame(annotated_frame)

                # Update stats
                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    logger.info(f"Processed {frame_count} frames ({fps:.1f} FPS)")

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up...")

        if self.camera:
            self.camera.release()

        if self.mjpeg_server:
            self.mjpeg_server.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run runtime recognition")
    parser.add_argument(
        "--backend",
        type=str,
        choices=["usb", "picamera2", "video_file"],
        default="usb",
        help="Camera backend (usb=USB webcam, picamera2=RPi camera, video_file=video file)"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Camera device ID for USB backend (default: 0)"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file (for video_file backend)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Create pipeline
    pipeline = RecognitionPipeline(config)

    # Initialize
    if not pipeline.initialize(args.backend, args.video, args.device_id):
        logger.error("Failed to initialize pipeline")
        return

    # Run
    pipeline.run()


if __name__ == "__main__":
    main()


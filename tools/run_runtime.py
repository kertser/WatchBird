#!/usr/bin/env python3
"""Runtime recognition application."""

import argparse
import logging
import time
from typing import Dict, Optional

import cv2
import numpy as np

from watchbird.camera.usb_backend import USBCameraBackend
from watchbird.camera.video_backend import VideoBackend
try:
    from watchbird.camera.picamera_backend import Picamera2Backend
except ImportError:
    Picamera2Backend = None  # type: ignore
from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
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

        # Face detector
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

        logger.info("Initialization complete!")
        return True

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

        # Get embedding sample interval from config
        embedding_sample_interval = self.config.fusion.get("embedding_sample_interval", 1)

        # Process each confirmed track
        for track in tracks:
            track_id = track.track_id

            # Create state machine if new track
            if track_id not in self.track_states:
                self.track_states[track_id] = TrackStateMachine(
                    track_id=track_id,
                    t_accept=self.config.thresholds["t_accept"],
                    t_margin=self.config.thresholds["t_margin"],
                    t_timeout=self.config.thresholds["t_timeout"],
                    consistency_count=self.config.fusion["consistency_count"],
                    window_size=self.config.fusion["window_size"],
                    confidence_decay_threshold=self.config.fusion.get("confidence_decay_threshold", 3),
                    identity_switch_margin=self.config.thresholds.get("identity_switch_margin", 0.10)
                )
                # Initialize frame counter for this track
                self.track_frame_counters[track_id] = 0

            state_machine = self.track_states[track_id]

            # Skip if already in terminal state
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

            # Compute face quality
            quality, quality_details = compute_face_quality(
                track.bbox,
                face_roi,
                track.confidence,
                min_bbox_size=self.config.quality["min_bbox_size"]
            )

            # Skip if quality too low
            if quality < self.config.quality["min_face_quality"]:
                continue

            # Extract embedding (let embedder handle alignment internally for accuracy)
            # Note: Passing landmarks was causing misalignment issues
            embedding = self.face_embedder.extract(face_roi)

            if embedding is None:
                continue

            # Compute simplified quality metrics for embedding aggregation
            # Skip expensive Laplacian blur computation for FPS
            bbox = track.bbox
            face_size = int(((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) ** 0.5)

            # Simple blur estimate from face size (larger faces = usually sharper)
            blur_score = min(1.0, face_size / 100.0)

            # Simple brightness from detection confidence
            brightness = 0.5  # Assume optimal, skip gray conversion

            # Face size: area of bounding box
            bbox = track.bbox
            face_size = int((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) ** 0.5)  # sqrt of area

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

        # Cleanup lost tracks to prevent memory leaks
        active_track_ids = {track.track_id for track in tracks}
        # Also consider tracks stale if they haven't been updated recently
        stale_track_ids = {
            track.track_id for track in tracks
            if track.time_since_update > 5  # Not updated in last 5 frames
        }
        lost_track_ids = (set(self.track_frame_counters.keys()) - active_track_ids) | stale_track_ids

        for track_id in lost_track_ids:
            if track_id in self.track_frame_counters:
                del self.track_frame_counters[track_id]
            if track_id in self.track_states:
                del self.track_states[track_id]

        # Also cleanup embedding manager
        self.embedding_manager.cleanup_stale_tracks(active_track_ids)

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


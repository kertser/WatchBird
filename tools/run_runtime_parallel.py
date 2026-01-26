#!/usr/bin/env python3
"""Parallel runtime recognition application with improved FPS.

This version uses a multi-threaded pipeline that overlaps:
- Frame capture
- Face detection (CPU)
- Face embedding (GPU)
- Tracking & recognition (CPU)

This allows CPU and GPU to work in parallel, achieving 2-3x higher FPS.
"""

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
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.fusion.similarity import SimilarityFusion
from watchbird.index.faiss_wrapper import FaissIndex
from watchbird.index.meta_store import MetaStore
from watchbird.runtime.events import EventEmitter
from watchbird.runtime.parallel_pipeline import ParallelPipeline, FrameData
from watchbird.runtime.state_machine import TrackStateMachine
from watchbird.stream.mjpeg_server import MJPEGServer, draw_detection_boxes
from watchbird.track.tracker import Tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ParallelRecognitionPipeline:
    """Main recognition pipeline with parallel processing."""

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
        self.tracker = None
        self.fusion = None
        self.event_emitter = None
        self.mjpeg_server = None
        self.parallel_pipeline = None

        # Track state machines
        self.track_states: Dict[int, TrackStateMachine] = {}

    def initialize(self, backend: str, video_path: Optional[str] = None, device_id: int = 0) -> bool:
        """Initialize all components.

        Args:
            backend: Camera backend (usb, picamera2, or video_file)
            video_path: Path to video file (for video_file backend)
            device_id: Camera device ID (for usb backend, default 0)

        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing parallel pipeline components...")

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

        if not self.camera.open():
            logger.error("Failed to open camera")
            return False

        # Face detector
        self.face_detector = FaceDetector(
            model_path=self.config.models.get("face_detector"),
            conf_threshold=self.config.detection["face_conf_threshold"],
            use_gpu=self.config.inference.get("use_gpu", True),
            gpu_device_id=self.config.inference.get("gpu_device_id", 0)
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

        # Tracker
        self.tracker = Tracker(
            max_age=self.config.tracking["max_age"],
            min_hits=self.config.tracking["min_hits"],
            iou_threshold=self.config.tracking["iou_threshold"]
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

        # Create parallel pipeline
        self.parallel_pipeline = ParallelPipeline(
            camera=self.camera,
            face_detector=self.face_detector,
            face_embedder=self.face_embedder,
            tracker=self.tracker,
            config=self.config,
            queue_size=2  # Small queue = low latency
        )

        logger.info("Parallel initialization complete!")
        return True

    def process_recognition(self, data: FrameData) -> np.ndarray:
        """Process recognition for a frame with pre-computed embeddings.

        Args:
            data: FrameData with detection and embedding results

        Returns:
            Annotated frame
        """
        frame = data.frame
        embeddings = data.embeddings or []
        embedding_indices = data.embedding_indices or []
        qualities = data.qualities or []

        # First, update tracker with all detections (this is fast, done in main thread)
        tracks = []
        if data.face_bboxes is not None and len(data.face_bboxes) > 0:
            tracks = self.tracker.update(data.face_bboxes, data.face_confs, data.face_landmarks)

        # Create a mapping from detection index to track
        detection_to_track = {}
        for track in tracks:
            # Find which detection this track corresponds to (by IoU)
            if data.face_bboxes is not None:
                best_iou = 0
                best_idx = -1
                for i, bbox in enumerate(data.face_bboxes):
                    iou = self._compute_iou(track.bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                if best_iou > 0.5:
                    detection_to_track[best_idx] = track

        # Process each embedding (matched to tracks via detection index)
        for embedding, det_idx, quality in zip(embeddings, embedding_indices, qualities):
            # Find the track for this detection
            if det_idx not in detection_to_track:
                continue
            
            track = detection_to_track[det_idx]
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
                    confidence_decay_threshold=self.config.fusion.get("confidence_decay_threshold", 3)
                )

            state_machine = self.track_states[track_id]

            # Skip if already in terminal state
            if state_machine.is_terminal():
                continue

            # Search FAISS index
            similarities, indices = self.faiss_index.search(embedding, k=2)

            if len(similarities) == 0:
                continue

            # Get person IDs from metadata
            person_ids = [
                self.meta_store.get_person_id(int(idx))
                for idx in indices[0]
            ]

            # Get top matches
            matches = list(zip(similarities[0], person_ids))

            # Aggregate best candidate
            best_person_id, best_score, margin = self.fusion.aggregate_top_candidate(matches)
            second_score = best_score - margin

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

        # Cleanup old track states
        active_track_ids = {track.track_id for track in tracks}
        stale_ids = [tid for tid in self.track_states if tid not in active_track_ids]
        for tid in stale_ids[-10:]:  # Keep some history, remove oldest
            if tid in self.track_states:
                del self.track_states[tid]

        # Draw annotations
        annotated_frame = frame.copy()

        for track in tracks:
            track_id = track.track_id

            # Skip drawing stale tracks
            if track.time_since_update > 0:
                continue

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

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x, y, w, h]."""
        x1, y1, w1, h1 = box1[:4]
        x2, y2, w2, h2 = box2[:4]
        
        # Convert to x1, y1, x2, y2 format
        box1_x2, box1_y2 = x1 + w1, y1 + h1
        box2_x2, box2_y2 = x2 + w2, y2 + h2
        
        # Intersection
        inter_x1 = max(x1, x2)
        inter_y1 = max(y1, y2)
        inter_x2 = min(box1_x2, box2_x2)
        inter_y2 = min(box1_y2, box2_y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # Union
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area

    def run(self) -> None:
        """Run parallel recognition pipeline."""
        logger.info("Starting PARALLEL recognition pipeline...")
        logger.info("(Should achieve 2-3x higher FPS than sequential version)")

        # Start parallel pipeline
        self.parallel_pipeline.start()

        frame_count = 0
        start_time = time.time()

        try:
            while True:
                # Get processed frame from pipeline (non-blocking with timeout)
                data = self.parallel_pipeline.get_processed_frame(timeout=0.1)

                if data is None:
                    # Check if capture stage stopped (e.g., end of video)
                    stats = self.parallel_pipeline.get_stats()
                    if stats["capture"]["processed"] > 0 and stats["embedding"]["processed"] == stats["capture"]["processed"]:
                        logger.info("All frames processed")
                        break
                    continue

                # Process recognition (FAISS search + state machine - fast operations)
                annotated_frame = self.process_recognition(data)

                # Update MJPEG server
                if self.mjpeg_server:
                    self.mjpeg_server.update_frame(annotated_frame)

                # Update stats
                frame_count += 1

                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = frame_count / elapsed
                    stats = self.parallel_pipeline.get_stats()
                    logger.info(
                        f"Processed {frame_count} frames ({fps:.1f} FPS) | "
                        f"Captured: {stats['capture']['processed']} | "
                        f"Dropped: {stats['capture']['dropped']}"
                    )

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("Cleaning up...")

        if self.parallel_pipeline:
            self.parallel_pipeline.stop()

        if self.camera:
            self.camera.release()

        if self.mjpeg_server:
            self.mjpeg_server.stop()


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run PARALLEL runtime recognition (improved FPS)")
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
    pipeline = ParallelRecognitionPipeline(config)

    # Initialize
    if not pipeline.initialize(args.backend, args.video, args.device_id):
        logger.error("Failed to initialize pipeline")
        return

    # Run
    pipeline.run()


if __name__ == "__main__":
    main()


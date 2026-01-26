"""Parallel recognition pipeline for improved FPS.

This module implements a multi-threaded pipeline that overlaps:
- Frame capture (camera thread)
- Face detection (detection thread)
- Face embedding (embedding thread - GPU)
- Tracking & display (main thread)

This allows the GPU and CPU to work in parallel, significantly improving FPS.
"""

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameData:
    """Data container for a single frame in the pipeline."""
    frame_id: int
    frame: np.ndarray
    timestamp: float

    # Filled by detection stage
    face_bboxes: Optional[np.ndarray] = None
    face_confs: Optional[np.ndarray] = None
    face_landmarks: Optional[np.ndarray] = None

    # Filled by embedding stage (per-detection, not per-track)
    embeddings: Optional[List[np.ndarray]] = None
    embedding_indices: Optional[List[int]] = None  # Index into face_bboxes
    qualities: Optional[List[float]] = None  # Quality score for each embedding


class PipelineStage:
    """Base class for pipeline stages."""

    def __init__(self, name: str, input_queue: queue.Queue, output_queue: queue.Queue):
        self.name = name
        self.input_queue = input_queue
        self.output_queue = output_queue
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.stats = {"processed": 0, "dropped": 0}

    def start(self):
        """Start the processing thread."""
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, name=f"Pipeline-{self.name}")
        self._thread.daemon = True
        self._thread.start()
        logger.info(f"Started pipeline stage: {self.name}")

    def stop(self):
        """Stop the processing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            logger.info(f"Stopped pipeline stage: {self.name}")

    def _run_loop(self):
        """Main processing loop."""
        while self._running:
            try:
                data = self.input_queue.get(timeout=0.1)
                if data is None:  # Poison pill
                    break

                result = self.process(data)
                if result is not None:
                    # Non-blocking put - drop if output queue is full
                    try:
                        self.output_queue.put_nowait(result)
                        self.stats["processed"] += 1
                    except queue.Full:
                        self.stats["dropped"] += 1

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in {self.name} stage: {e}")

    def process(self, data: FrameData) -> Optional[FrameData]:
        """Process data. Override in subclass."""
        raise NotImplementedError


class CaptureStage(PipelineStage):
    """Stage that captures frames from camera."""

    def __init__(self, camera, output_queue: queue.Queue, target_fps: int = 30):
        # No input queue for capture stage
        super().__init__("Capture", queue.Queue(), output_queue)
        self.camera = camera
        self.target_fps = target_fps
        self.frame_id = 0

    def _run_loop(self):
        """Override to capture frames instead of processing from queue."""
        frame_interval = 1.0 / self.target_fps

        while self._running:
            start_time = time.time()

            frame = self.camera.get_frame()
            if frame is None:
                logger.info("Camera returned None, stopping capture")
                break

            data = FrameData(
                frame_id=self.frame_id,
                frame=frame,
                timestamp=time.time()
            )
            self.frame_id += 1

            # Non-blocking put - drop frame if queue is full
            try:
                self.output_queue.put_nowait(data)
                self.stats["processed"] += 1
            except queue.Full:
                self.stats["dropped"] += 1

            # Maintain target FPS
            elapsed = time.time() - start_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)


class DetectionStage(PipelineStage):
    """Stage that runs face detection."""

    def __init__(self, face_detector, input_queue: queue.Queue, output_queue: queue.Queue):
        super().__init__("Detection", input_queue, output_queue)
        self.face_detector = face_detector

    def process(self, data: FrameData) -> Optional[FrameData]:
        """Run face detection on frame."""
        face_bboxes, face_confs, face_landmarks = self.face_detector.detect(data.frame)

        data.face_bboxes = face_bboxes
        data.face_confs = face_confs
        data.face_landmarks = face_landmarks

        return data


class EmbeddingStage(PipelineStage):
    """Stage that extracts face embeddings (GPU-accelerated)."""

    def __init__(self, face_embedder, tracker, config,
                 input_queue: queue.Queue, output_queue: queue.Queue):
        super().__init__("Embedding", input_queue, output_queue)
        self.face_embedder = face_embedder
        self.tracker = tracker  # Not used in parallel mode, kept for compatibility
        self.config = config
        self.frame_counter = 0

    def process(self, data: FrameData) -> Optional[FrameData]:
        """Extract embeddings for detected faces."""
        from watchbird.utils.image_ops import extract_roi
        from watchbird.utils.quality import compute_face_quality

        self.frame_counter += 1
        embedding_sample_interval = self.config.fusion.get("embedding_sample_interval", 1)

        # Always pass through frames, but only extract embeddings every N frames
        if data.face_bboxes is None or len(data.face_bboxes) == 0:
            data.embeddings = []
            data.embedding_indices = []
            data.qualities = []
            return data

        # Skip embedding extraction on non-sample frames (but still pass the frame)
        if self.frame_counter % embedding_sample_interval != 0:
            data.embeddings = []
            data.embedding_indices = []
            data.qualities = []
            return data

        min_face_quality = self.config.quality["min_face_quality"]
        min_bbox_size = self.config.quality["min_bbox_size"]

        embeddings = []
        indices = []
        qualities = []

        # Process ALL detected faces (sampling is done per-track in main thread)
        for i, bbox in enumerate(data.face_bboxes):
            # Extract face ROI
            face_roi = extract_roi(data.frame, bbox)

            # Compute face quality
            quality, _ = compute_face_quality(
                bbox,
                face_roi,
                data.face_confs[i],
                min_bbox_size=min_bbox_size
            )

            if quality < min_face_quality:
                continue

            # Extract embedding (GPU operation)
            embedding = self.face_embedder.extract(face_roi)

            if embedding is not None:
                embeddings.append(embedding)
                indices.append(i)
                qualities.append(quality)

        data.embeddings = embeddings
        data.embedding_indices = indices
        data.qualities = qualities

        return data


class ParallelPipeline:
    """Multi-threaded recognition pipeline."""

    def __init__(self, camera, face_detector, face_embedder, tracker, config,
                 queue_size: int = 3):
        """Initialize parallel pipeline.

        Args:
            camera: Camera backend
            face_detector: Face detector
            face_embedder: Face embedder
            tracker: Object tracker
            config: Configuration object
            queue_size: Size of inter-stage queues (smaller = lower latency)
        """
        self.config = config
        self.tracker = tracker

        # Create queues
        self.capture_to_detection = queue.Queue(maxsize=queue_size)
        self.detection_to_embedding = queue.Queue(maxsize=queue_size)
        self.embedding_to_output = queue.Queue(maxsize=queue_size)

        # Create stages
        self.capture_stage = CaptureStage(
            camera,
            self.capture_to_detection,
            target_fps=config.camera["fps"]
        )

        self.detection_stage = DetectionStage(
            face_detector,
            self.capture_to_detection,
            self.detection_to_embedding
        )

        self.embedding_stage = EmbeddingStage(
            face_embedder,
            tracker,
            config,
            self.detection_to_embedding,
            self.embedding_to_output
        )

        self._running = False

    def start(self):
        """Start all pipeline stages."""
        self._running = True
        self.capture_stage.start()
        self.detection_stage.start()
        self.embedding_stage.start()
        logger.info("Parallel pipeline started")

    def stop(self):
        """Stop all pipeline stages."""
        self._running = False
        self.capture_stage.stop()
        self.detection_stage.stop()
        self.embedding_stage.stop()
        logger.info("Parallel pipeline stopped")

    def get_processed_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        """Get the next processed frame.

        Args:
            timeout: Timeout in seconds

        Returns:
            FrameData with detection and embedding results, or None
        """
        try:
            return self.embedding_to_output.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "capture": self.capture_stage.stats.copy(),
            "detection": self.detection_stage.stats.copy(),
            "embedding": self.embedding_stage.stats.copy(),
        }


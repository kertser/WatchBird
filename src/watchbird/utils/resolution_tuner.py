"""Auto-resolution tuning for optimal FPS performance.

This module provides utilities to automatically detect the optimal camera
resolution that achieves a target FPS with the current pipeline.
"""

import logging
import time
from typing import List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)

# Common resolutions to test, ordered from highest to lowest
COMMON_RESOLUTIONS = [
    (1920, 1080),  # 1080p
    (1600, 1200),  # UXGA
    (1280, 960),   # 960p
    (1280, 720),   # 720p
    (1024, 768),   # XGA
    (800, 600),    # SVGA
    (640, 480),    # VGA
]


def measure_pipeline_fps(
    cap: cv2.VideoCapture,
    detector,
    num_frames: int = 30,
    warmup_frames: int = 5
) -> float:
    """Measure actual pipeline FPS with detection.

    Args:
        cap: OpenCV VideoCapture object
        detector: Face detector with detect() method
        num_frames: Number of frames to measure
        warmup_frames: Warmup frames to skip

    Returns:
        Measured FPS
    """
    # Warmup
    for _ in range(warmup_frames):
        ret, frame = cap.read()
        if ret and frame is not None:
            detector.detect(frame)

    # Measure
    start_time = time.perf_counter()
    frames_processed = 0

    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        detector.detect(frame)
        frames_processed += 1

    elapsed = time.perf_counter() - start_time

    if elapsed > 0 and frames_processed > 0:
        return frames_processed / elapsed
    return 0.0


def find_optimal_resolution(
    device_id: int,
    detector,
    target_fps: float = 10.0,
    min_fps: float = 8.0,
    resolutions: Optional[List[Tuple[int, int]]] = None,
    test_frames: int = 30
) -> Tuple[int, int]:
    """Find the highest resolution that achieves target FPS.

    Iterates through resolutions from highest to lowest, measuring actual
    pipeline FPS with detection, and returns the first resolution that
    meets the target FPS.

    Args:
        device_id: Camera device ID
        detector: Face detector with detect() method
        target_fps: Target FPS to achieve
        min_fps: Minimum acceptable FPS (fallback)
        resolutions: List of resolutions to test (default: COMMON_RESOLUTIONS)
        test_frames: Number of frames to test per resolution

    Returns:
        Optimal (width, height) resolution tuple
    """
    if resolutions is None:
        resolutions = COMMON_RESOLUTIONS

    logger.info(f"Auto-tuning resolution for target FPS >= {target_fps}...")

    best_resolution = resolutions[-1]  # Default to lowest
    best_fps = 0.0

    # On Windows, use DirectShow backend
    import platform
    if platform.system() == 'Windows':
        cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        logger.error(f"Cannot open camera {device_id} for resolution tuning")
        return best_resolution

    try:
        for width, height in resolutions:
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

            # Check actual resolution
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Skip if camera doesn't support this resolution
            if (actual_w, actual_h) != (width, height):
                logger.debug(f"  {width}x{height} → actual {actual_w}x{actual_h}, skipping")
                continue

            # Measure FPS
            fps = measure_pipeline_fps(cap, detector, num_frames=test_frames)
            logger.info(f"  {width}x{height}: {fps:.1f} FPS")

            # Check if this meets target
            if fps >= target_fps:
                logger.info(f"✓ Selected {width}x{height} @ {fps:.1f} FPS (meets target)")
                return (width, height)

            # Track best so far (in case nothing meets target)
            if fps > best_fps:
                best_fps = fps
                best_resolution = (width, height)

    finally:
        cap.release()
        # Give Windows time to fully release the camera (MSMF issue)
        time.sleep(0.5)

    # If nothing met target, use the resolution with best FPS
    if best_fps >= min_fps:
        logger.warning(
            f"No resolution met target {target_fps} FPS. "
            f"Using {best_resolution[0]}x{best_resolution[1]} @ {best_fps:.1f} FPS"
        )
    else:
        logger.warning(
            f"Performance below minimum! Best: {best_resolution[0]}x{best_resolution[1]} @ {best_fps:.1f} FPS"
        )

    return best_resolution


class ResolutionAutoTuner:
    """Auto-tuner that can be integrated into the pipeline."""

    def __init__(
        self,
        target_fps: float = 10.0,
        min_fps: float = 8.0,
        test_frames: int = 30,
        resolutions: Optional[List[Tuple[int, int]]] = None
    ):
        """Initialize resolution auto-tuner.

        Args:
            target_fps: Target FPS to achieve
            min_fps: Minimum acceptable FPS
            test_frames: Number of frames to test per resolution
            resolutions: Custom list of resolutions to test
        """
        self.target_fps = target_fps
        self.min_fps = min_fps
        self.test_frames = test_frames
        self.resolutions = resolutions or COMMON_RESOLUTIONS
        self.selected_resolution: Optional[Tuple[int, int]] = None
        self.measured_fps: float = 0.0

    def tune(self, device_id: int, detector) -> Tuple[int, int]:
        """Run auto-tuning and return optimal resolution.

        Args:
            device_id: Camera device ID
            detector: Face detector instance

        Returns:
            Optimal (width, height) resolution
        """
        self.selected_resolution = find_optimal_resolution(
            device_id=device_id,
            detector=detector,
            target_fps=self.target_fps,
            min_fps=self.min_fps,
            resolutions=self.resolutions,
            test_frames=self.test_frames
        )
        return self.selected_resolution

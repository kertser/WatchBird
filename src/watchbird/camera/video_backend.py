"""Video file backend for testing and development."""

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from watchbird.camera.base import CameraBackend

logger = logging.getLogger(__name__)


class VideoBackend(CameraBackend):
    """Video file input backend."""

    def __init__(
        self,
        video_path: str,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 15,
        loop: bool = True
    ):
        """Initialize video backend.

        Args:
            video_path: Path to video file
            resolution: Target resolution (width, height)
            fps: Target FPS (used for frame skipping)
            loop: Whether to loop video when it ends
        """
        super().__init__(resolution, fps)
        self.video_path = Path(video_path)
        self.loop = loop
        self.cap: Optional[cv2.VideoCapture] = None
        self.source_fps: float = 0.0
        self.frame_skip: int = 1

    def open(self) -> bool:
        """Open video file.

        Returns:
            True if successful, False otherwise
        """
        if not self.video_path.exists():
            logger.error(f"Video file not found: {self.video_path}")
            return False

        self.cap = cv2.VideoCapture(str(self.video_path))

        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return False

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps > 0:
            self.frame_skip = max(1, int(self.source_fps / self.fps))

        self.is_opened = True
        logger.info(f"Opened video: {self.video_path} (FPS: {self.source_fps}, skip: {self.frame_skip})")

        return True

    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from video.

        Returns:
            Frame as numpy array or None if end of video
        """
        if not self.is_opened or self.cap is None:
            return None

        # Skip frames to match target FPS
        for _ in range(self.frame_skip - 1):
            self.cap.read()

        ret, frame = self.cap.read()

        if not ret:
            if self.loop:
                logger.info("End of video, looping...")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if not ret:
                    return None
            else:
                logger.info("End of video")
                return None

        # Resize if needed
        if frame.shape[1] != self.resolution[0] or frame.shape[0] != self.resolution[1]:
            frame = cv2.resize(frame, self.resolution)

        return frame

    def release(self) -> None:
        """Release video capture."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_opened = False
        logger.info("Released video backend")


"""USB camera backend using OpenCV VideoCapture."""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from watchbird.camera.base import CameraBackend

logger = logging.getLogger(__name__)


class USBCameraBackend(CameraBackend):
    """USB camera backend using cv2.VideoCapture."""

    def __init__(
        self,
        device_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 15
    ):
        """Initialize USB camera backend.

        Args:
            device_id: Camera device ID (usually 0 for first camera)
            resolution: Camera resolution (width, height)
            fps: Target frames per second
        """
        super().__init__(resolution, fps)
        self.device_id = device_id
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        """Open USB camera connection.

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Opening USB camera {self.device_id}...")

            # On Windows, use DirectShow (DSHOW) backend instead of MSMF
            # MSMF has issues with some cameras (frame grab errors)
            import platform
            if platform.system() == 'Windows':
                self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_DSHOW)
                logger.debug("Using DirectShow backend (Windows)")
            else:
                self.cap = cv2.VideoCapture(self.device_id)

            if not self.cap.isOpened():
                logger.error(f"Failed to open camera device {self.device_id}")
                return False

            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            # Set buffer size to 1 to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Verify actual resolution
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))

            logger.info(
                f"USB camera opened: {actual_width}x{actual_height} @ {actual_fps}fps "
                f"(requested: {self.resolution[0]}x{self.resolution[1]} @ {self.fps}fps)"
            )

            # Read a test frame to make sure camera is working
            ret, test_frame = self.cap.read()
            if not ret or test_frame is None:
                logger.error("Camera opened but failed to read test frame")
                self.cap.release()
                return False

            logger.debug(f"Test frame captured: {test_frame.shape}")

            self.is_opened = True
            return True

        except Exception as e:
            logger.error(f"Error opening USB camera: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from USB camera.

        Returns:
            Frame as numpy array (H, W, 3) in BGR format, or None if failed
        """
        if not self.is_opened or self.cap is None:
            logger.warning("Camera not opened")
            return None

        ret, frame = self.cap.read()

        if not ret or frame is None:
            logger.warning("Failed to read frame from camera")
            return None

        return frame

    def release(self) -> None:
        """Release USB camera resources."""
        if self.cap is not None:
            logger.info("Releasing USB camera")
            self.cap.release()
            self.cap = None
            self.is_opened = False

    def set_resolution(self, width: int, height: int) -> bool:
        """Change camera resolution.

        Args:
            width: New width
            height: New height

        Returns:
            True if successful, False otherwise
        """
        if self.cap is None or not self.is_opened:
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if (actual_w, actual_h) == (width, height):
            self.resolution = (width, height)
            logger.info(f"Resolution changed to {width}x{height}")
            return True

        logger.warning(f"Resolution {width}x{height} not supported, got {actual_w}x{actual_h}")
        return False

    def get_actual_resolution(self) -> Tuple[int, int]:
        """Get the actual camera resolution.

        Returns:
            (width, height) tuple
        """
        if self.cap is None:
            return self.resolution

        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )


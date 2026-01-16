"""Picamera2 backend for Raspberry Pi CSI camera."""

import logging
from typing import Optional, Tuple

import numpy as np

from watchbird.camera.base import CameraBackend

logger = logging.getLogger(__name__)


class Picamera2Backend(CameraBackend):
    """Picamera2 input backend for Raspberry Pi."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 15):
        """Initialize Picamera2 backend.

        Args:
            resolution: Camera resolution (width, height)
            fps: Target frames per second
        """
        super().__init__(resolution, fps)
        self.picam2 = None

    def open(self) -> bool:
        """Open Picamera2 connection.

        Returns:
            True if successful, False otherwise
        """
        try:
            from picamera2 import Picamera2

            self.picam2 = Picamera2()

            # Configure camera
            config = self.picam2.create_preview_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.fps}
            )

            self.picam2.configure(config)
            self.picam2.start()

            self.is_opened = True
            logger.info(f"Opened Picamera2: {self.resolution} @ {self.fps} FPS")

            return True

        except ImportError:
            logger.error("Picamera2 not available. Install with: pip install picamera2")
            return False
        except Exception as e:
            logger.error(f"Failed to open Picamera2: {e}")
            return False

    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from camera.

        Returns:
            Frame as numpy array in BGR format (for OpenCV compatibility)
        """
        if not self.is_opened or self.picam2 is None:
            return None

        try:
            # Capture frame (RGB format)
            frame = self.picam2.capture_array()

            # Convert RGB to BGR for OpenCV compatibility
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return frame

        except Exception as e:
            logger.error(f"Failed to capture frame: {e}")
            return None

    def release(self) -> None:
        """Release camera resources."""
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception as e:
                logger.warning(f"Error releasing Picamera2: {e}")
            finally:
                self.picam2 = None

        self.is_opened = False
        logger.info("Released Picamera2 backend")


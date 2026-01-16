"""Base camera backend interface."""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class CameraBackend(ABC):
    """Abstract base class for camera backends."""

    def __init__(self, resolution: Tuple[int, int] = (640, 480), fps: int = 15):
        """Initialize camera backend.

        Args:
            resolution: Camera resolution (width, height)
            fps: Target frames per second
        """
        self.resolution = resolution
        self.fps = fps
        self.is_opened = False

    @abstractmethod
    def open(self) -> bool:
        """Open camera connection.

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    def get_frame(self) -> Optional[np.ndarray]:
        """Get next frame from camera.

        Returns:
            Frame as numpy array (H, W, 3) in BGR format, or None if failed
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass

    def __enter__(self) -> "CameraBackend":
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # type: ignore
        """Context manager exit."""
        self.release()


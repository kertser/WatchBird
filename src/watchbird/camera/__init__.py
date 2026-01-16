"""Camera input backends."""

from watchbird.camera.base import CameraBackend
from watchbird.camera.usb_backend import USBCameraBackend
from watchbird.camera.video_backend import VideoBackend

try:
    from watchbird.camera.picamera_backend import Picamera2Backend
except ImportError:
    # Picamera2 only available on Raspberry Pi
    Picamera2Backend = None  # type: ignore

__all__ = [
    "CameraBackend",
    "USBCameraBackend",
    "VideoBackend",
    "Picamera2Backend",
]



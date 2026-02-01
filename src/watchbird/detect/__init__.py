"""Face detection modules.

Available detectors:
    - FaceDetector: YuNet-based detector (CPU only, via OpenCV)
    - UltraFaceDetector: UltraFace ONNX (GPU via DirectML/CUDA, no landmarks)
    - SCRFDDetector: SCRFD ONNX (GPU via DirectML/CUDA, with 5-point landmarks) ← Recommended
"""

from watchbird.detect.face_detector import FaceDetector
from watchbird.detect.ultraface_detector import UltraFaceDetector
from watchbird.detect.scrfd_detector import SCRFDDetector

__all__ = ["FaceDetector", "UltraFaceDetector", "SCRFDDetector"]

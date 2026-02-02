"""Face and body detection modules.

Available detectors:
    - FaceDetector: YuNet-based detector (CPU only, via OpenCV)
    - UltraFaceDetector: UltraFace ONNX (GPU via DirectML/CUDA, no landmarks)
    - SCRFDDetector: SCRFD ONNX (GPU via DirectML/CUDA, with 5-point landmarks) ← Recommended
    - BodyDetector: YOLOv8-based human body detector (GPU via DirectML/CUDA)
    - HumanSegmenter: Human segmentation for body contours (GPU via DirectML/CUDA)
"""

from watchbird.detect.face_detector import FaceDetector
from watchbird.detect.ultraface_detector import UltraFaceDetector
from watchbird.detect.scrfd_detector import SCRFDDetector
from watchbird.detect.body_detector import BodyDetector
from watchbird.detect.human_segmenter import HumanSegmenter, draw_body_contour_by_state

__all__ = [
    "FaceDetector",
    "UltraFaceDetector",
    "SCRFDDetector",
    "BodyDetector",
    "HumanSegmenter",
    "draw_body_contour_by_state",
]

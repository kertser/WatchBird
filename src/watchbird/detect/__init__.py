"""Face and body detection modules.

Available detectors:
    - SCRFDDetector: SCRFD ONNX (GPU via DirectML/CUDA, with 5-point landmarks) ← Recommended
    - BodyDetector: YOLOv8-based human body detector (GPU via DirectML/CUDA)
    - HumanSegmenter: Human segmentation for body contours (GPU via DirectML/CUDA)
    - PersonClassifier: CLIP-based soldier/civilian classification (PyTorch + GPU)

Legacy (kept for compatibility):
    - FaceDetector: YuNet-based detector (CPU only, via OpenCV)
    - UltraFaceDetector: UltraFace ONNX (GPU via DirectML/CUDA, no landmarks)
"""

from watchbird.detect.scrfd_detector import SCRFDDetector
from watchbird.detect.body_detector import BodyDetector
from watchbird.detect.human_segmenter import HumanSegmenter, draw_body_contour_by_state
from watchbird.detect.person_classifier import PersonClassifier

# Legacy imports (for compatibility)
from watchbird.detect.face_detector import FaceDetector
from watchbird.detect.ultraface_detector import UltraFaceDetector

__all__ = [
    "SCRFDDetector",
    "BodyDetector",
    "HumanSegmenter",
    "draw_body_contour_by_state",
    "PersonClassifier",
    # Legacy
    "FaceDetector",
    "UltraFaceDetector",
]

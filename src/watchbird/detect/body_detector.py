"""Human body detection using YOLOv8 Nano ONNX model with GPU support.

Provides detection of human bodies to pair with face detection for
full-body tracking and visualization.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class BodyDetector:
    """Human body detector using YOLOv8 Nano ONNX model.

    Detects persons (class 0 in COCO) and returns bounding boxes.
    Can be combined with face detection to associate faces with bodies.
    """

    # COCO class index for person
    PERSON_CLASS = 0

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """Initialize body detector.

        Args:
            model_path: Path to YOLOv8 ONNX model
            conf_threshold: Minimum confidence threshold
            nms_threshold: NMS IoU threshold
            use_gpu: Whether to use GPU acceleration
            gpu_device_id: GPU device ID
            input_size: Model input size (width, height)
        """
        self.model_path = model_path or "models/yolov8n.onnx"
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.input_size = input_size

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None
        self._loaded = False

    def load(self) -> bool:
        """Load the ONNX model.

        Returns:
            True if successful, False otherwise
        """
        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.warning(f"Body detector model not found: {model_path}")
            logger.info("Download YOLOv8n ONNX model with: python tools/download_models.py --body")
            return False

        logger.info(f"Loading body detector from {model_path}")

        try:
            # Configure providers
            providers = []

            if self.use_gpu:
                # Try DirectML first (Windows AMD/Intel/NVIDIA)
                if "DmlExecutionProvider" in ort.get_available_providers():
                    providers.append(("DmlExecutionProvider", {"device_id": self.gpu_device_id}))
                # Try CUDA (NVIDIA on Linux/Windows with CUDA toolkit)
                elif "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.append(("CUDAExecutionProvider", {"device_id": self.gpu_device_id}))

            # Always add CPU as fallback
            providers.append("CPUExecutionProvider")

            # Suppress ONNX Runtime warnings about initializers
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3  # Only errors

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            # Determine which provider is being used
            actual_provider = self.session.get_providers()[0]
            if "Dml" in actual_provider:
                backend_info = f"DirectML GPU (device {self.gpu_device_id})"
            elif "CUDA" in actual_provider:
                backend_info = f"CUDA GPU (device {self.gpu_device_id})"
            else:
                backend_info = "CPU"

            logger.info(f"Loaded body detector - backend: {backend_info}")
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load body detector: {e}")
            return False

    @property
    def loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Preprocess image for YOLO inference.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (preprocessed_blob, scale_factor, padding_offset)
        """
        h, w = image.shape[:2]
        target_w, target_h = self.input_size

        # Calculate scale to fit in input size while maintaining aspect ratio
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image (letterboxing)
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert to blob (NCHW format, normalized)
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

        return blob, scale, (pad_w, pad_h)

    def _postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        padding: Tuple[int, int],
        img_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess YOLO outputs.

        Args:
            outputs: Raw model outputs
            scale: Preprocessing scale factor
            padding: Preprocessing padding (pad_w, pad_h)
            img_shape: Original image shape (height, width)

        Returns:
            Tuple of (bboxes, confidences)
        """
        # YOLOv8 output shape: (1, 84, 8400) - 84 = 4 bbox + 80 classes
        predictions = outputs[0]  # Shape: (84, 8400) or (1, 84, 8400)

        if predictions.ndim == 3:
            predictions = predictions[0]  # Shape: (84, 8400)

        # Transpose to (8400, 84)
        predictions = predictions.T

        # Extract class confidences (indices 4:)
        class_scores = predictions[:, 4:]

        # Get person class confidence (class 0)
        person_scores = class_scores[:, self.PERSON_CLASS]

        # Filter by confidence
        mask = person_scores > self.conf_threshold
        filtered_predictions = predictions[mask]
        filtered_scores = person_scores[mask]

        if len(filtered_predictions) == 0:
            return np.array([]).reshape(0, 4), np.array([])

        # Extract bboxes (xywh format - normalized 0-1)
        boxes_xywh = filtered_predictions[:, :4]

        # Scale from normalized to input size (640x640)
        target_w, target_h = self.input_size
        boxes_xywh = boxes_xywh.copy()
        boxes_xywh[:, 0] *= target_w  # cx
        boxes_xywh[:, 1] *= target_h  # cy
        boxes_xywh[:, 2] *= target_w  # w
        boxes_xywh[:, 3] *= target_h  # h

        # Convert to xyxy
        boxes = np.zeros_like(boxes_xywh)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2  # x2
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2  # y2

        # Remove padding and scale back to original image coordinates
        pad_w, pad_h = padding
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / scale

        # Clip to image bounds
        img_h, img_w = img_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_h)

        # Apply NMS
        if len(boxes) > 0:
            indices = cv2.dnn.NMSBoxes(
                boxes.tolist(),
                filtered_scores.tolist(),
                self.conf_threshold,
                self.nms_threshold
            )

            if len(indices) > 0:
                # Handle both old (1D array) and new (2D array) OpenCV NMSBoxes return format
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                elif isinstance(indices, (list, tuple)):
                    indices = np.array(indices).flatten()
                boxes = boxes[indices]
                filtered_scores = filtered_scores[indices]

        return boxes, filtered_scores

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Detect human bodies in image.

        Args:
            image: BGR image

        Returns:
            Tuple of:
                - bboxes: (N, 4) array of bounding boxes [x1, y1, x2, y2]
                - confidences: (N,) array of confidence scores
        """
        if not self._loaded:
            return np.array([]).reshape(0, 4), np.array([])

        # Preprocess
        blob, scale, padding = self._preprocess(image)
        img_shape = image.shape[:2]

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Postprocess
        bboxes, confidences = self._postprocess(outputs[0], scale, padding, img_shape)

        return bboxes, confidences

    def match_face_to_body(
        self,
        face_bbox: np.ndarray,
        body_bboxes: np.ndarray
    ) -> Optional[int]:
        """Match a face to the most likely body.

        Uses spatial heuristics: face should be in upper portion of body,
        horizontally centered.

        Args:
            face_bbox: Face bounding box [x1, y1, x2, y2]
            body_bboxes: Body bounding boxes (N, 4)

        Returns:
            Index of matching body, or None if no match
        """
        if len(body_bboxes) == 0:
            return None

        face_cx = (face_bbox[0] + face_bbox[2]) / 2
        face_cy = (face_bbox[1] + face_bbox[3]) / 2

        best_score = float("inf")
        best_idx = None

        for idx, body in enumerate(body_bboxes):
            bx1, by1, bx2, by2 = body
            body_cx = (bx1 + bx2) / 2
            body_w = bx2 - bx1
            body_h = by2 - by1

            # Check if face center is within body bounding box (with margin)
            margin = 0.1
            if not (bx1 - body_w * margin <= face_cx <= bx2 + body_w * margin):
                continue
            if not (by1 - body_h * margin <= face_cy <= by2 + body_h * margin):
                continue

            # Face should be in upper 40% of body
            upper_region_y = by1 + body_h * 0.4
            if face_cy > upper_region_y:
                continue

            # Score based on horizontal alignment and vertical position
            h_offset = abs(face_cx - body_cx) / (body_w + 1e-6)
            v_position = (face_cy - by1) / (body_h + 1e-6)  # 0 = top, 1 = bottom

            # Prefer faces closer to horizontal center and in upper portion
            score = h_offset + v_position

            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

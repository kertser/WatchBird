"""Face detection using UltraFace ONNX model with GPU support via DirectML/CUDA.

UltraFace is a lightweight, fast face detector optimized for edge devices.
It supports GPU acceleration via ONNX Runtime providers (CUDA, DirectML).
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def get_optimal_providers(use_gpu: bool = True, gpu_device_id: int = 0) -> List[Tuple[str, dict]]:
    """Get optimal execution providers based on availability and preference."""
    available = ort.get_available_providers()
    providers = []

    if use_gpu:
        # CUDA (NVIDIA)
        if 'CUDAExecutionProvider' in available:
            providers.append(('CUDAExecutionProvider', {
                'device_id': gpu_device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }))

        # DirectML (Windows - AMD/Intel/NVIDIA)
        if 'DmlExecutionProvider' in available:
            providers.append(('DmlExecutionProvider', {
                'device_id': gpu_device_id,
            }))

    # CPU fallback
    providers.append(('CPUExecutionProvider', {}))
    return providers


class UltraFaceDetector:
    """UltraFace detector with GPU support via ONNX Runtime.

    UltraFace is a lightweight face detector (~1MB) that's very fast
    on both CPU and GPU. Two variants available:
        - version-RFB-320.onnx: 320x240 input, faster
        - version-RFB-640.onnx: 640x480 input, more accurate
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.7,
        nms_threshold: float = 0.3,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        max_detection_size: int = 640
    ):
        """Initialize UltraFace detector.

        Args:
            model_path: Path to UltraFace ONNX model
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            use_gpu: Whether to use GPU acceleration
            gpu_device_id: GPU device ID
            max_detection_size: Maximum dimension for detection
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.max_detection_size = max_detection_size

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.input_shape: Tuple[int, int] = (320, 240)  # Default, will be updated
        self.backend_used: str = "Not loaded"

    def load(self) -> bool:
        """Load UltraFace model.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"UltraFace model not found at {self.model_path}")
                return False

            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            # Suppress verbose warnings about initializers
            sess_options.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal

            # Get providers
            providers = get_optimal_providers(self.use_gpu, self.gpu_device_id)
            provider_names = [p[0] for p in providers]

            logger.info(f"Loading UltraFace from {self.model_path}")
            logger.debug(f"Trying providers: {provider_names}")

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers
            )

            # Get actual provider used
            actual_providers = self.session.get_providers()
            if 'CUDAExecutionProvider' in actual_providers:
                self.backend_used = f"CUDA GPU (device {self.gpu_device_id})"
            elif 'DmlExecutionProvider' in actual_providers:
                self.backend_used = f"DirectML GPU (device {self.gpu_device_id})"
            else:
                self.backend_used = "CPU"

            # Get input info
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name

            # Input shape is NCHW: [1, 3, H, W]
            shape = input_info.shape
            if len(shape) == 4:
                self.input_shape = (shape[3], shape[2])  # (W, H)

            logger.info(f"Loaded UltraFace face detector - backend: {self.backend_used}")
            logger.info(f"Input size: {self.input_shape[0]}x{self.input_shape[1]}")

            return True

        except Exception as e:
            logger.error(f"Failed to load UltraFace: {e}")
            return False

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Preprocess image for UltraFace.

        Args:
            image: Input BGR image

        Returns:
            (preprocessed_blob, scale_x, scale_y)
        """
        h, w = image.shape[:2]
        input_w, input_h = self.input_shape

        # Resize to model input size
        resized = cv2.resize(image, (input_w, input_h))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1] (UltraFace uses (img - 127) / 128)
        normalized = (rgb.astype(np.float32) - 127.0) / 128.0

        # HWC to NCHW
        blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        scale_x = w / input_w
        scale_y = h / input_h

        return blob, scale_x, scale_y

    def _decode_outputs(
        self,
        confidences: np.ndarray,
        boxes: np.ndarray,
        scale_x: float,
        scale_y: float,
        orig_size: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Decode UltraFace outputs to bboxes and scores.

        Args:
            confidences: Confidence scores [1, N, 2] (bg, fg)
            boxes: Bounding boxes [1, N, 4] (x1, y1, x2, y2 normalized)
            scale_x: X scale factor
            scale_y: Y scale factor
            orig_size: Original image size (w, h)

        Returns:
            (bboxes, scores)
        """
        orig_w, orig_h = orig_size
        input_w, input_h = self.input_shape

        # Get foreground confidence
        scores = confidences[0, :, 1]

        # Filter by confidence
        mask = scores >= self.conf_threshold
        scores = scores[mask]
        boxes = boxes[0, mask, :]

        if len(scores) == 0:
            return np.array([]), np.array([])

        # Convert normalized coordinates to pixel coordinates
        # UltraFace outputs normalized coords [0, 1] relative to input size
        bboxes = np.zeros_like(boxes)
        bboxes[:, 0] = boxes[:, 0] * input_w * scale_x  # x1
        bboxes[:, 1] = boxes[:, 1] * input_h * scale_y  # y1
        bboxes[:, 2] = boxes[:, 2] * input_w * scale_x  # x2
        bboxes[:, 3] = boxes[:, 3] * input_h * scale_y  # y2

        # Clip to image bounds
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, orig_w)
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, orig_h)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, orig_w)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, orig_h)

        return bboxes, scores

    def _nms(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply non-maximum suppression.

        Args:
            bboxes: Detection bboxes (N, 4)
            scores: Detection scores (N,)

        Returns:
            Filtered (bboxes, scores)
        """
        if len(bboxes) == 0:
            return bboxes, scores

        # Use OpenCV NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )

        if len(indices) == 0:
            return np.array([]), np.array([])

        indices = indices.flatten()

        return bboxes[indices], scores[indices]

    def detect(
        self,
        image: np.ndarray,
        try_rotations: bool = False
    ) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """Detect faces in image.

        Args:
            image: Input BGR image
            try_rotations: Whether to try rotated detection (not implemented)

        Returns:
            (bboxes, confidences, landmarks) where:
                - bboxes: List of [x1, y1, x2, y2] arrays
                - confidences: List of confidence scores
                - landmarks: List of (5, 2) landmark arrays (estimated for UltraFace)
        """
        if self.session is None:
            logger.error("UltraFace not loaded, call load() first")
            return [], [], []

        h, w = image.shape[:2]

        # Preprocess
        blob, scale_x, scale_y = self._preprocess(image)

        # Run inference
        # UltraFace outputs: [confidences, boxes]
        outputs = self.session.run(None, {self.input_name: blob})
        confidences, boxes = outputs[0], outputs[1]

        # Decode outputs
        bboxes, scores = self._decode_outputs(confidences, boxes, scale_x, scale_y, (w, h))

        # Apply NMS
        bboxes, scores = self._nms(bboxes, scores)

        # Convert to lists
        bbox_list = [bbox for bbox in bboxes]
        score_list = scores.tolist() if len(scores) > 0 else []

        # UltraFace doesn't output landmarks - return None for each detection
        # The face embedder will detect landmarks internally using YuNet for alignment
        # This is more accurate than estimating from bbox
        landmark_list = [None for _ in bbox_list]

        return bbox_list, score_list, landmark_list

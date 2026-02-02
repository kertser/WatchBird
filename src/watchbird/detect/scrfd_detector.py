"""Face detection using SCRFD ONNX model with GPU support via DirectML/CUDA.

Based on: https://github.com/yakhyo/facial-analysis
Paper: "Sample and Computation Redistribution for Efficient Face Detection"
       https://arxiv.org/abs/2105.04714

SCRFD outputs accurate 5-point facial landmarks which are essential for
proper face alignment before embedding extraction.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode distance prediction to bounding box.

    Args:
        points: Shape (n, 2), anchor centers [x, y]
        distance: Distance from point to 4 boundaries (left, top, right, bottom)

    Returns:
        Decoded bboxes with shape (n, 4) as [x1, y1, x2, y2]
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    return np.stack([x1, y1, x2, y2], axis=-1)


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode distance prediction to keypoints.

    Args:
        points: Shape (n, 2), anchor centers [x, y]
        distance: Distance predictions for keypoints

    Returns:
        Decoded keypoints with shape (n, num_kps*2)
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFDDetector:
    """SCRFD face detector with GPU support via ONNX Runtime.

    SCRFD (Sample and Computation Redistribution for Face Detection) is a
    fast and accurate face detector that outputs 5-point facial landmarks.

    Model variants from https://github.com/yakhyo/facial-analysis:
        - det_500m.onnx: Fastest (~2.4MB)
        - det_2.5g.onnx: Balanced (~3.1MB)
        - det_10g.onnx: Most accurate (~16MB)
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        input_size: Tuple[int, int] = (640, 640),
        max_detection_size: int = 640
    ):
        """Initialize SCRFD detector.

        Args:
            model_path: Path to SCRFD ONNX model
            conf_threshold: Confidence threshold for detections
            nms_threshold: NMS IoU threshold
            use_gpu: Whether to use GPU acceleration
            gpu_device_id: GPU device ID
            input_size: Model input size (width, height)
            max_detection_size: Maximum dimension for detection
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.input_size = input_size
        self.max_detection_size = max_detection_size

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = ""
        self.output_names: List[str] = []
        self.backend_used: str = "Not loaded"

        # SCRFD model params
        self.fmc = 3  # Feature map count
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        # Normalization
        self.mean = 127.5
        self.std = 128.0

        # Cache for anchor centers
        self.center_cache = {}

    def _get_providers(self) -> List:
        """Get ONNX Runtime execution providers."""
        available = ort.get_available_providers()
        providers = []

        if self.use_gpu:
            if 'CUDAExecutionProvider' in available:
                providers.append('CUDAExecutionProvider')
            if 'DmlExecutionProvider' in available:
                providers.append('DmlExecutionProvider')

        providers.append('CPUExecutionProvider')
        return providers

    def load(self) -> bool:
        """Load SCRFD model.

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.model_path.exists():
                logger.error(f"SCRFD model not found at {self.model_path}")
                logger.info("Download from: https://github.com/yakhyo/facial-analysis/releases")
                return False

            # Create session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.log_severity_level = 3  # Suppress warnings

            providers = self._get_providers()

            logger.info(f"Loading SCRFD from {self.model_path}")

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

            # Get input/output info
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [o.name for o in self.session.get_outputs()]

            logger.info(f"Loaded SCRFD face detector - backend: {self.backend_used}")

            return True

        except Exception as e:
            logger.error(f"Failed to load SCRFD: {e}")
            return False

    def _forward(self, image: np.ndarray) -> Tuple[List, List, List]:
        """Run SCRFD forward pass.

        Args:
            image: Preprocessed input image

        Returns:
            (scores_list, bboxes_list, kpss_list) per stride level
        """
        scores_list = []
        bboxes_list = []
        kpss_list = []

        input_size = tuple(image.shape[0:2][::-1])

        # Create blob
        blob = cv2.dnn.blobFromImage(
            image,
            1.0 / self.std,
            input_size,
            (self.mean, self.mean, self.mean),
            swapRB=True
        )

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        input_height = blob.shape[2]
        input_width = blob.shape[3]

        fmc = self.fmc
        for idx, stride in enumerate(self._feat_stride_fpn):
            scores = outputs[idx]
            bbox_preds = outputs[idx + fmc] * stride

            if self.use_kps:
                kps_preds = outputs[idx + fmc * 2] * stride

            height = input_height // stride
            width = input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(
                    np.mgrid[:height, :width][::-1], axis=-1
                ).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))

                if self._num_anchors > 1:
                    anchor_centers = np.stack(
                        [anchor_centers] * self._num_anchors, axis=1
                    ).reshape((-1, 2))

                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= self.conf_threshold)[0]
            bboxes = _distance2bbox(anchor_centers, bbox_preds)
            pos_scores = scores[pos_inds]
            pos_bboxes = bboxes[pos_inds]

            scores_list.append(pos_scores)
            bboxes_list.append(pos_bboxes)

            if self.use_kps:
                kpss = _distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

        return scores_list, bboxes_list, kpss_list

    def _nms(self, dets: np.ndarray) -> List[int]:
        """Apply non-maximum suppression.

        Args:
            dets: Detections with shape (N, 5) as [x1, y1, x2, y2, score]

        Returns:
            List of indices to keep
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            indices = np.where(ovr <= self.nms_threshold)[0]
            order = order[indices + 1]

        return keep

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
                - landmarks: List of (5, 2) landmark arrays
        """
        if self.session is None:
            logger.error("SCRFD not loaded, call load() first")
            return [], [], []

        h, w = image.shape[:2]
        width, height = self.input_size

        # Calculate resize scale maintaining aspect ratio
        im_ratio = float(h) / w
        model_ratio = height / width

        if im_ratio > model_ratio:
            new_height = height
            new_width = int(new_height / im_ratio)
        else:
            new_width = width
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / h
        resized_image = cv2.resize(image, (new_width, new_height))

        # Create padded image
        det_image = np.zeros((height, width, 3), dtype=np.uint8)
        det_image[:new_height, :new_width, :] = resized_image

        # Forward pass
        scores_list, bboxes_list, kpss_list = self._forward(det_image)

        # Concatenate results from all strides
        if len(scores_list) == 0 or all(len(s) == 0 for s in scores_list):
            return [], [], []

        scores = np.vstack(scores_list)
        scores_ravel = scores.ravel()
        order = scores_ravel.argsort()[::-1]

        bboxes = np.vstack(bboxes_list) / det_scale

        if self.use_kps and len(kpss_list) > 0:
            kpss = np.vstack(kpss_list) / det_scale
        else:
            kpss = None

        # Prepare for NMS
        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        pre_det = pre_det[order, :]

        keep = self._nms(pre_det)
        det = pre_det[keep, :]

        if kpss is not None:
            kpss = kpss[order, :, :]
            kpss = kpss[keep, :, :]

        # Convert to output format
        bbox_list = []
        score_list = []
        landmark_list = []

        for i in range(det.shape[0]):
            bbox = det[i, :4]
            score = float(det[i, 4])

            # Clip bbox to image bounds
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(w, bbox[2])
            bbox[3] = min(h, bbox[3])

            bbox_list.append(bbox)
            score_list.append(score)

            if kpss is not None:
                # Clip landmarks to image bounds
                kps = kpss[i].copy()
                kps[:, 0] = np.clip(kps[:, 0], 0, w)
                kps[:, 1] = np.clip(kps[:, 1], 0, h)
                landmark_list.append(kps)
            else:
                landmark_list.append(None)

        return bbox_list, score_list, landmark_list

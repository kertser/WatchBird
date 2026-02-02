"""Human segmentation for body contour extraction.

Uses a lightweight segmentation model to extract human silhouettes
for visualization overlay on the video stream.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Sequence, Any

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class HumanSegmenter:
    """Human segmentation using a lightweight ONNX model.

    Produces per-pixel masks for detected humans, which can be used
    to draw colored contours around the body silhouette.

    Supports multiple backends:
    - SelfieSegmentation-style models (MediaPipe)
    - YOLOv8-seg models
    - PP-HumanSeg models
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        threshold: float = 0.5
    ):
        """Initialize human segmenter.

        Args:
            model_path: Path to segmentation ONNX model
            use_gpu: Whether to use GPU acceleration
            gpu_device_id: GPU device ID
            threshold: Segmentation confidence threshold
        """
        self.model_path = model_path or "models/human_seg.onnx"
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.threshold = threshold

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.input_shape: Optional[Tuple[int, int]] = None
        self.output_names: Optional[List[str]] = None
        self._loaded = False

    def load(self) -> bool:
        """Load the ONNX model.

        Returns:
            True if successful, False otherwise
        """
        model_path = Path(self.model_path)
        if not model_path.exists():
            logger.warning(f"Human segmentation model not found: {model_path}")
            logger.info("Download model with: python tools/download_models.py --segment")
            return False

        logger.info(f"Loading human segmenter from {model_path}")

        try:
            # Configure providers
            providers = []

            if self.use_gpu:
                if "DmlExecutionProvider" in ort.get_available_providers():
                    providers.append(("DmlExecutionProvider", {"device_id": self.gpu_device_id}))
                elif "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.append(("CUDAExecutionProvider", {"device_id": self.gpu_device_id}))

            providers.append("CPUExecutionProvider")

            # Suppress ONNX Runtime warnings
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3

            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=sess_options,
                providers=providers
            )

            # Get input/output info
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name

            # Get input shape (typically NCHW: 1, 3, H, W)
            shape = input_info.shape
            if isinstance(shape[2], int) and isinstance(shape[3], int):
                self.input_shape = (shape[3], shape[2])  # (W, H)
            else:
                # Dynamic shape - use default
                self.input_shape = (256, 256)

            self.output_names = [o.name for o in self.session.get_outputs()]

            # Determine backend
            actual_provider = self.session.get_providers()[0]
            if "Dml" in actual_provider:
                backend_info = f"DirectML GPU (device {self.gpu_device_id})"
            elif "CUDA" in actual_provider:
                backend_info = f"CUDA GPU (device {self.gpu_device_id})"
            else:
                backend_info = "CPU"

            logger.info(f"Loaded human segmenter - backend: {backend_info}")
            logger.info(f"Input shape: {self.input_shape}")
            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load human segmenter: {e}")
            return False

    @property
    def loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for segmentation.

        Args:
            image: Input BGR image

        Returns:
            Preprocessed blob
        """
        # Resize to model input size
        resized = cv2.resize(image, self.input_shape, interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to NCHW format
        blob = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        return blob

    def segment(self, image: np.ndarray) -> np.ndarray:
        """Segment humans in image.

        Args:
            image: BGR image

        Returns:
            Binary mask of human regions (same size as input)
        """
        if not self._loaded:
            return np.zeros(image.shape[:2], dtype=np.uint8)

        h, w = image.shape[:2]

        # Preprocess
        blob = self._preprocess(image)

        # Run inference
        outputs = self.session.run(self.output_names, {self.input_name: blob})

        # Process output - format depends on model type
        mask = self._process_output(outputs)

        # Resize to original size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        # Threshold to binary
        binary_mask = (mask > self.threshold).astype(np.uint8) * 255

        return binary_mask

    def _process_output(self, outputs: Sequence[Any]) -> np.ndarray:
        """Process model output to segmentation mask.

        Args:
            outputs: Model outputs

        Returns:
            Segmentation probability mask (H, W) with values in [0, 1]
        """
        output = outputs[0]

        # Handle different output formats
        if output.ndim == 4:
            # NCHW format - typically (1, 1, H, W) or (1, 2, H, W)
            if output.shape[1] == 1:
                # Single channel output
                mask = output[0, 0]
            elif output.shape[1] == 2:
                # Two classes (background, human) - use softmax
                mask = np.exp(output[0, 1]) / (np.exp(output[0, 0]) + np.exp(output[0, 1]))
            else:
                # Multi-class - use person class
                mask = output[0, 15] if output.shape[1] > 15 else output[0, 1]
        elif output.ndim == 3:
            # NHW format
            mask = output[0]
        else:
            # HW format
            mask = output

        # Apply sigmoid if values are outside [0, 1]
        if mask.min() < 0 or mask.max() > 1:
            mask = 1 / (1 + np.exp(-mask))

        return mask.astype(np.float32)

    def segment_roi(
        self,
        image: np.ndarray,
        bbox: np.ndarray
    ) -> np.ndarray:
        """Segment a specific region of interest.

        More efficient for single-person segmentation when body is detected.

        Args:
            image: Full BGR image
            bbox: Bounding box [x1, y1, x2, y2]

        Returns:
            Binary mask for the ROI region (full image size, zeros outside ROI)
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]

        # Clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros((h, w), dtype=np.uint8)

        # Extract ROI
        roi = image[y1:y2, x1:x2]

        # Segment ROI
        roi_mask = self.segment(roi)

        # Place in full mask
        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = roi_mask

        return full_mask


def extract_contours(
    mask: np.ndarray,
    min_area: int = 500
) -> List[np.ndarray]:
    """Extract contours from segmentation mask.

    Args:
        mask: Binary mask (uint8)
        min_area: Minimum contour area to keep

    Returns:
        List of contour arrays
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter by area
    filtered = [c for c in contours if cv2.contourArea(c) >= min_area]

    return filtered


def draw_segmentation_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    contour_thickness: int = 2,
    fill_alpha: float = 0.1,
    draw_contour: bool = True,
    draw_fill: bool = True
) -> np.ndarray:
    """Draw segmentation overlay on frame.

    Args:
        frame: Input BGR frame
        mask: Binary segmentation mask
        color: BGR color for overlay
        contour_thickness: Thickness of contour lines
        fill_alpha: Alpha for fill (0 = transparent, 1 = opaque)
        draw_contour: Whether to draw contour lines
        draw_fill: Whether to fill the segmented region

    Returns:
        Frame with overlay
    """
    result = frame.copy()

    # Extract contours
    contours = extract_contours(mask)

    if not contours:
        return result

    # Draw filled region with transparency
    if draw_fill and fill_alpha > 0:
        overlay = result.copy()
        cv2.drawContours(overlay, contours, -1, color, -1)
        result = cv2.addWeighted(overlay, fill_alpha, result, 1 - fill_alpha, 0)

    # Draw contour outline
    if draw_contour:
        cv2.drawContours(result, contours, -1, color, contour_thickness)

    return result


def draw_body_contour_by_state(
    frame: np.ndarray,
    mask: np.ndarray,
    state: str,
    contour_thickness: int = 3,
    fill_alpha: float = 0.15
) -> np.ndarray:
    """Draw body contour colored by recognition state.

    Args:
        frame: Input BGR frame
        mask: Binary segmentation mask
        state: Recognition state (DETECTING, SUSPECT, FRIENDLY, CONFIRMED, ENEMY)
        contour_thickness: Contour line thickness
        fill_alpha: Fill transparency

    Returns:
        Frame with colored body overlay
    """
    # Color by state (BGR)
    state_colors = {
        "CONFIRMED": (0, 255, 0),      # Bright green
        "FRIENDLY": (0, 200, 100),     # Light green/teal
        "ENEMY": (0, 0, 255),          # Red
        "DETECTING": (0, 165, 255),    # Orange
        "SUSPECT": (0, 255, 255),      # Yellow
    }

    color = state_colors.get(state, (128, 128, 128))  # Gray default

    return draw_segmentation_overlay(
        frame,
        mask,
        color=color,
        contour_thickness=contour_thickness,
        fill_alpha=fill_alpha,
        draw_contour=True,
        draw_fill=True
    )

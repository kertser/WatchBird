"""Lightweight ONNX-based person classifier using quantized CLIP.

This is a faster, smaller alternative to the PyTorch-based PersonClassifier.
Uses a quantized ONNX model (~150MB vs ~600MB) with DirectML/CUDA GPU support.

Usage:
    # First, export the ONNX model:
    python tools/export_clip_onnx.py

    # Then use in code:
    from watchbird.detect.person_classifier_onnx import PersonClassifierONNX

    classifier = PersonClassifierONNX()
    classifier.load()
    person_type, confidence = classifier.classify(body_crop)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Check for ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("onnxruntime not available. Install with: pip install onnxruntime-directml")


class PersonClassifierONNX:
    """Person classifier using quantized CLIP ONNX model.

    Much smaller than PyTorch version:
    - ONNX INT8: ~150MB
    - PyTorch: ~600MB

    Supports DirectML/CUDA GPU acceleration via ONNX Runtime.
    """

    # Default category labels (must match export order)
    LABELS = ["soldier", "armed_civilian", "unarmed_civilian"]

    # CLIP image preprocessing constants
    CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    CLIP_SIZE = 224

    def __init__(
        self,
        model_path: str = "models/clip_vision_int8.onnx",
        text_embeddings_path: str = "models/clip_text_embeddings.npy",
        labels_path: str = "models/clip_labels.txt",
        use_gpu: bool = True,
        gpu_device_id: int = 0,
        armed_threshold: float = 0.5,
        soldier_threshold: float = 0.5,
        default_category: str = "unarmed_civilian"
    ):
        """Initialize ONNX-based classifier.

        Args:
            model_path: Path to CLIP vision ONNX model
            text_embeddings_path: Path to pre-computed text embeddings
            labels_path: Path to category labels file
            use_gpu: Whether to use GPU acceleration
            gpu_device_id: GPU device index
            armed_threshold: Min confidence for armed_civilian
            soldier_threshold: Min confidence for soldier
            default_category: Fallback when thresholds not met
        """
        self.model_path = Path(model_path)
        self.text_embeddings_path = Path(text_embeddings_path)
        self.labels_path = Path(labels_path)
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.armed_threshold = armed_threshold
        self.soldier_threshold = soldier_threshold
        self.default_category = default_category

        self.session: Optional[ort.InferenceSession] = None
        self.text_embeddings: Optional[np.ndarray] = None
        self.labels: List[str] = self.LABELS.copy()
        self._loaded = False
        self.backend_used = "Not loaded"

    def load(self) -> bool:
        """Load ONNX model and text embeddings.

        Returns:
            True if successful, False otherwise
        """
        if not ONNX_AVAILABLE:
            logger.error("onnxruntime not installed")
            return False

        # Check for model file
        if not self.model_path.exists():
            # Try fallback to non-quantized version
            fallback_path = self.model_path.parent / "clip_vision.onnx"
            if fallback_path.exists():
                self.model_path = fallback_path
                logger.info(f"Using non-quantized model: {fallback_path}")
            else:
                logger.error(f"CLIP ONNX model not found: {self.model_path}")
                logger.error("Run: python tools/export_clip_onnx.py")
                return False

        # Check for text embeddings
        if not self.text_embeddings_path.exists():
            logger.error(f"Text embeddings not found: {self.text_embeddings_path}")
            logger.error("Run: python tools/export_clip_onnx.py")
            return False

        try:
            # Setup ONNX Runtime providers
            providers = []
            provider_options = []

            if self.use_gpu:
                available = ort.get_available_providers()

                if "DmlExecutionProvider" in available:
                    providers.append("DmlExecutionProvider")
                    provider_options.append({"device_id": self.gpu_device_id})
                    self.backend_used = f"DirectML GPU (device {self.gpu_device_id})"
                elif "CUDAExecutionProvider" in available:
                    providers.append("CUDAExecutionProvider")
                    provider_options.append({"device_id": self.gpu_device_id})
                    self.backend_used = f"CUDA GPU (device {self.gpu_device_id})"

            providers.append("CPUExecutionProvider")
            provider_options.append({})

            if self.backend_used == "Not loaded":
                self.backend_used = "CPU"

            # Load ONNX model
            logger.info(f"Loading CLIP ONNX: {self.model_path}")

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=providers,
                provider_options=provider_options if len(provider_options) == len(providers) else None
            )

            # Load text embeddings
            self.text_embeddings = np.load(str(self.text_embeddings_path))

            # Load labels if available
            if self.labels_path.exists():
                with open(self.labels_path, "r") as f:
                    self.labels = [line.strip() for line in f if line.strip()]

            model_size = self.model_path.stat().st_size / 1024 / 1024
            logger.info(f"CLIP ONNX loaded ({model_size:.1f} MB) - backend: {self.backend_used}")
            logger.info(f"Categories: {', '.join(self.labels)}")

            self._loaded = True
            return True

        except Exception as e:
            logger.error(f"Failed to load CLIP ONNX: {e}")
            return False

    @property
    def loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for CLIP.

        Args:
            image: BGR image (any size)

        Returns:
            Preprocessed tensor [1, 3, 224, 224]
        """
        # Convert BGR to RGB
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        resized = cv2.resize(rgb, (self.CLIP_SIZE, self.CLIP_SIZE), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img = resized.astype(np.float32) / 255.0

        # Apply CLIP normalization
        img = (img - self.CLIP_MEAN) / self.CLIP_STD

        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        # Add batch dimension
        return img[np.newaxis, ...].astype(np.float32)

    def _apply_thresholds(
        self,
        probs: np.ndarray
    ) -> Tuple[str, float]:
        """Apply classification thresholds.

        Args:
            probs: Probability array for each category

        Returns:
            (category, confidence) tuple
        """
        best_idx = int(np.argmax(probs))
        best_category = self.labels[best_idx]
        best_score = float(probs[best_idx])

        # Apply thresholds
        if best_category == "armed_civilian" and best_score < self.armed_threshold:
            soldier_idx = self.labels.index("soldier")
            soldier_score = float(probs[soldier_idx])
            if soldier_score >= self.soldier_threshold:
                return "soldier", soldier_score
            else:
                default_idx = self.labels.index(self.default_category)
                return self.default_category, float(probs[default_idx])

        elif best_category == "soldier" and best_score < self.soldier_threshold:
            armed_idx = self.labels.index("armed_civilian")
            armed_score = float(probs[armed_idx])
            if armed_score >= self.armed_threshold:
                return "armed_civilian", armed_score
            else:
                default_idx = self.labels.index(self.default_category)
                return self.default_category, float(probs[default_idx])

        return best_category, best_score

    def classify(self, image: np.ndarray) -> Tuple[str, float]:
        """Classify a person crop.

        Args:
            image: BGR image of cropped person

        Returns:
            (category, confidence) tuple
        """
        if not self._loaded or self.session is None:
            return "unarmed_civilian", 0.0

        if image is None or image.size == 0:
            return "unarmed_civilian", 0.0

        # Minimum size check
        if image.shape[0] < 32 or image.shape[1] < 32:
            return "unarmed_civilian", 0.0

        try:
            # Preprocess
            pixel_values = self.preprocess(image)

            # Get image features from ONNX model
            outputs = self.session.run(
                ["image_features"],
                {"pixel_values": pixel_values}
            )
            image_features = outputs[0]  # [1, 512]

            # Compute similarity with text embeddings
            similarities = (image_features @ self.text_embeddings.T)[0]  # [3]

            # Softmax with temperature scaling
            exp_sim = np.exp(similarities * 100)
            probs = exp_sim / exp_sim.sum()

            return self._apply_thresholds(probs)

        except Exception as e:
            logger.warning(f"Classification failed: {e}")
            return "unarmed_civilian", 0.0

    def classify_batch(
        self,
        images: List[np.ndarray]
    ) -> List[Tuple[str, float]]:
        """Classify multiple person crops in a batch.

        Args:
            images: List of BGR images

        Returns:
            List of (category, confidence) tuples
        """
        if not self._loaded or not images:
            return [("unarmed_civilian", 0.0)] * len(images)

        try:
            # Filter valid images and preprocess
            valid_inputs = []
            valid_indices = []

            for i, img in enumerate(images):
                if img is not None and img.size > 0:
                    if img.shape[0] >= 32 and img.shape[1] >= 32:
                        valid_inputs.append(self.preprocess(img))
                        valid_indices.append(i)

            if not valid_inputs:
                return [("unarmed_civilian", 0.0)] * len(images)

            # Batch inference
            batch = np.vstack(valid_inputs)
            outputs = self.session.run(
                ["image_features"],
                {"pixel_values": batch}
            )
            image_features = outputs[0]  # [N, 512]

            # Compute similarities
            similarities = image_features @ self.text_embeddings.T  # [N, 3]

            # Softmax
            exp_sim = np.exp(similarities * 100)
            probs = exp_sim / exp_sim.sum(axis=1, keepdims=True)

            # Build results with thresholds
            results = [("unarmed_civilian", 0.0)] * len(images)
            for batch_idx, orig_idx in enumerate(valid_indices):
                results[orig_idx] = self._apply_thresholds(probs[batch_idx])

            return results

        except Exception as e:
            logger.warning(f"Batch classification failed: {e}")
            return [("unarmed_civilian", 0.0)] * len(images)


# Utility functions for visualization (same as person_classifier.py)
def get_person_type_color(person_type: str) -> Tuple[int, int, int]:
    """Get visualization color for person type (BGR)."""
    colors = {
        "soldier": (0, 180, 0),         # Green
        "armed_civilian": (0, 0, 255),   # Red
        "unarmed_civilian": (255, 180, 0),  # Cyan
    }
    return colors.get(person_type, (128, 128, 128))


def get_person_type_label(person_type: str, confidence: float) -> str:
    """Get display label for person type."""
    labels = {
        "soldier": "IDF",
        "armed_civilian": "ARMED",
        "unarmed_civilian": "CIV",
    }
    label = labels.get(person_type, "?")
    return f"{label} {confidence:.0%}"

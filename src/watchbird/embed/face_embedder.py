"""Face embedding extractor using ONNX models."""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from watchbird.utils.image_ops import normalize_image

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Face embedding extractor using ONNX model (MobileFaceNet/ArcFace)."""

    def __init__(self, model_path: str, embedding_size: int = 128):
        """Initialize face embedder.

        Args:
            model_path: Path to ONNX model
            embedding_size: Size of face embedding vector
        """
        self.model_path = Path(model_path)
        self.embedding_size = embedding_size
        self.session = None
        self.input_name = None
        self.input_shape = None

    def load(self) -> bool:
        """Load face embedding model.

        Returns:
            True if successful, False otherwise
        """
        if not self.model_path.exists():
            logger.error(f"Model not found: {self.model_path}")
            return False

        try:
            # Create ONNX Runtime session
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=['CPUExecutionProvider']
            )

            # Get input details
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape

            # Get actual embedding size from model output
            output_shape = self.session.get_outputs()[0].shape
            if len(output_shape) >= 2:
                self.embedding_size = output_shape[-1]

            logger.info(f"Loaded face embedder: {self.model_path.name}")
            logger.info(f"Input shape: {self.input_shape}, embedding size: {self.embedding_size}")

            return True

        except Exception as e:
            logger.error(f"Failed to load face embedder: {e}")
            return False

    def _preprocess(self, face_image: np.ndarray) -> np.ndarray:
        """Preprocess face image for model input.

        Args:
            face_image: Face ROI in BGR format

        Returns:
            Preprocessed image tensor
        """
        # Get target size from model input shape
        if len(self.input_shape) == 4:
            # Format: [batch, channels, height, width]
            target_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 112
            target_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 112
        else:
            target_h = target_w = 112  # Default for face models

        # Resize face
        resized = cv2.resize(face_image, (target_w, target_h))

        # Normalize (model-specific, using common values)
        # Convert BGR to RGB and normalize to [-1, 1]
        normalized = normalize_image(resized, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # Add batch dimension
        tensor = np.expand_dims(normalized, axis=0)

        return tensor.astype(np.float32)

    def extract(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Extract face embedding from image.

        Args:
            face_image: Face ROI in BGR format

        Returns:
            L2-normalized embedding vector or None if failed
        """
        if self.session is None:
            logger.error("Model not loaded")
            return None

        try:
            # Preprocess
            input_tensor = self._preprocess(face_image)

            # Run inference
            outputs = self.session.run(None, {self.input_name: input_tensor})

            # Get embedding (first output)
            embedding = outputs[0][0]  # Remove batch dimension

            # L2 normalize for cosine similarity
            embedding = self._l2_normalize(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Face embedding extraction failed: {e}")
            return None

    @staticmethod
    def _l2_normalize(embedding: np.ndarray) -> np.ndarray:
        """L2 normalize embedding vector.

        Args:
            embedding: Embedding vector

        Returns:
            L2-normalized embedding
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm


"""Face embedding extractor using ONNX models."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from watchbird.utils.image_ops import normalize_image

logger = logging.getLogger(__name__)

# Reference facial points for 112x112 alignment (standard for face recognition)
REFERENCE_POINTS_112 = np.array([
    [38.2946, 51.6963],   # left eye
    [73.5318, 51.5014],   # right eye
    [56.0252, 71.7366],   # nose
    [41.5493, 92.3655],   # left mouth
    [70.7299, 92.2041]    # right mouth
], dtype=np.float32)


def get_available_providers() -> List[str]:
    """Get list of available ONNX Runtime execution providers.

    Returns:
        List of available provider names
    """
    return ort.get_available_providers()


def get_optimal_providers(use_gpu: bool = True, gpu_device_id: int = 0) -> List[Tuple[str, dict]]:
    """Get optimal execution providers based on availability and preference.

    Args:
        use_gpu: Whether to try GPU providers first
        gpu_device_id: GPU device ID to use

    Returns:
        List of (provider_name, options) tuples in priority order
    """
    available = get_available_providers()
    providers = []

    if use_gpu:
        # CUDA (NVIDIA) - best performance on NVIDIA GPUs
        if 'CUDAExecutionProvider' in available:
            providers.append(('CUDAExecutionProvider', {
                'device_id': gpu_device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 2GB limit
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }))
            logger.info(f"CUDA GPU provider available (device {gpu_device_id})")

        # DirectML (Windows - works with AMD, Intel, NVIDIA)
        if 'DmlExecutionProvider' in available:
            providers.append(('DmlExecutionProvider', {
                'device_id': gpu_device_id,
            }))
            logger.info(f"DirectML GPU provider available (device {gpu_device_id})")

        # TensorRT (NVIDIA optimized)
        if 'TensorrtExecutionProvider' in available:
            providers.append(('TensorrtExecutionProvider', {
                'device_id': gpu_device_id,
            }))
            logger.info(f"TensorRT provider available (device {gpu_device_id})")

    # Always include CPU as fallback
    providers.append(('CPUExecutionProvider', {}))

    return providers


class FaceEmbedder:
    """Face embedding extractor using ONNX model (MobileFaceNet/ArcFace)."""

    def __init__(self, model_path: str, embedding_size: int = 128,
                 use_gpu: bool = True, gpu_device_id: int = 0):
        """Initialize face embedder.

        Args:
            model_path: Path to ONNX model
            embedding_size: Size of face embedding vector
            use_gpu: Whether to use GPU acceleration if available
            gpu_device_id: GPU device ID to use
        """
        self.model_path = Path(model_path)
        self.embedding_size = embedding_size
        self.use_gpu = use_gpu
        self.gpu_device_id = gpu_device_id
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.active_provider = None

        # Landmark detector for face alignment (lazy loaded)
        self._landmark_detector = None

    def load(self) -> bool:
        """Load face embedding model.

        Returns:
            True if successful, False otherwise
        """
        if not self.model_path.exists():
            logger.error(f"Model not found: {self.model_path}")
            return False

        try:
            # Get optimal providers based on GPU preference
            providers = get_optimal_providers(self.use_gpu, self.gpu_device_id)
            provider_names = [p[0] for p in providers]
            provider_options = [p[1] for p in providers]

            logger.info(f"Attempting to load model with providers: {provider_names}")

            # Create session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Create ONNX Runtime session with GPU support
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=list(zip(provider_names, provider_options))
            )

            # Log which provider is actually being used
            self.active_provider = self.session.get_providers()[0]
            logger.info(f"Face embedder using: {self.active_provider}")

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

    def _get_landmark_detector(self):
        """Lazy-load landmark detector for face alignment."""
        if self._landmark_detector is None:
            try:
                # Use OpenCV's built-in face mesh or cascade-based landmark detector
                # For simplicity, use the same YuNet detector on the cropped face
                yunet_path = self.model_path.parent / "yunet.onnx"
                if yunet_path.exists():
                    self._landmark_detector = cv2.FaceDetectorYN.create(
                        str(yunet_path), "", (112, 112), 0.5
                    )
                    logger.debug("Loaded YuNet for face alignment")
            except Exception as e:
                logger.debug(f"Could not load landmark detector: {e}")
        return self._landmark_detector

    def _detect_landmarks_on_roi(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Detect 5-point landmarks on a cropped face ROI.

        Args:
            face_roi: Cropped face image

        Returns:
            5x2 array of landmarks or None if detection fails
        """
        detector = self._get_landmark_detector()
        if detector is None:
            return None

        try:
            h, w = face_roi.shape[:2]
            detector.setInputSize((w, h))
            _, faces = detector.detect(face_roi)

            if faces is None or len(faces) == 0:
                return None

            # Extract landmarks from first face
            # YuNet format: [x, y, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, conf]
            face = faces[0]
            landmarks = np.array([
                [face[6], face[7]],   # left eye
                [face[4], face[5]],   # right eye
                [face[8], face[9]],   # nose
                [face[12], face[13]], # left mouth
                [face[10], face[11]]  # right mouth
            ], dtype=np.float32)

            return landmarks

        except Exception as e:
            logger.debug(f"Landmark detection failed: {e}")
            return None

    def _align_face(self, face_roi: np.ndarray, target_size: Tuple[int, int] = (112, 112),
                    landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Align face using detected landmarks.

        Args:
            face_roi: Cropped face image
            target_size: Output size (width, height)
            landmarks: Optional pre-computed 5x2 landmarks in ROI coordinates.
                      If None, will attempt to detect landmarks (slower).

        Returns:
            Aligned face image (or resized original if alignment fails)
        """
        # Use pre-computed landmarks if provided, otherwise detect
        if landmarks is None:
            landmarks = self._detect_landmarks_on_roi(face_roi)

        if landmarks is None:
            # Fallback: just resize without alignment
            return cv2.resize(face_roi, target_size)

        try:
            # Scale reference points to target size
            scale = target_size[0] / 112.0
            ref_points = REFERENCE_POINTS_112 * scale

            # Estimate similarity transform (rotation + scale + translation)
            tform, _ = cv2.estimateAffinePartial2D(landmarks, ref_points)

            if tform is None:
                return cv2.resize(face_roi, target_size)

            # Warp face to canonical pose
            aligned = cv2.warpAffine(
                face_roi, tform, target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )

            logger.debug("Face aligned using landmarks")
            return aligned

        except Exception as e:
            logger.debug(f"Face alignment failed: {e}")
            return cv2.resize(face_roi, target_size)

    def _preprocess(self, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> np.ndarray:
        """Preprocess face image for model input.

        Args:
            face_image: Face ROI in BGR format
            landmarks: Optional pre-computed 5x2 landmarks in ROI coordinates

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

        # Align and resize face (uses pre-computed landmarks if provided)
        aligned = self._align_face(face_image, (target_w, target_h), landmarks=landmarks)

        # Normalize (model-specific, using common values)
        # Convert BGR to RGB and normalize to [-1, 1]
        normalized = normalize_image(aligned, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

        # Add batch dimension
        tensor = np.expand_dims(normalized, axis=0)

        return tensor.astype(np.float32)

    def extract(self, face_image: np.ndarray, landmarks: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Extract face embedding from image.

        Args:
            face_image: Face ROI in BGR format
            landmarks: Optional pre-computed 5x2 landmarks array. If provided, skips
                      re-detection (faster). Landmarks should be in ROI coordinates.

        Returns:
            L2-normalized embedding vector or None if failed
        """
        if self.session is None:
            logger.error("Model not loaded")
            return None

        try:
            # Preprocess (includes alignment)
            input_tensor = self._preprocess(face_image, landmarks=landmarks)

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


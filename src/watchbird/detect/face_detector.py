"""Face detection using OpenCV YuNet."""

import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class FaceDetector:
    """Face detector using OpenCV YuNet."""

    def __init__(self, model_path: str, conf_threshold: float = 0.7):
        """Initialize face detector.

        Args:
            model_path: Path to YuNet ONNX model
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.detector = None
        self.input_size = (320, 320)

    def load(self) -> bool:
        """Load face detection model.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to use YuNet from OpenCV
            if not self.model_path.exists():
                logger.warning(f"YuNet model not found at {self.model_path}, using built-in detector")
                # Create detector with default parameters
                self.detector = cv2.FaceDetectorYN.create(
                    "",  # Empty string uses built-in model
                    "",
                    self.input_size,
                    self.conf_threshold
                )
            else:
                self.detector = cv2.FaceDetectorYN.create(
                    str(self.model_path),
                    "",
                    self.input_size,
                    self.conf_threshold
                )

            logger.info("Loaded face detector (YuNet)")
            return True

        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
            return False

    def detect(self, image: np.ndarray, try_rotations: bool = True) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """Detect faces in image.

        Args:
            image: Input image in BGR format
            try_rotations: If True, try detecting faces in rotated images

        Returns:
            Tuple of (bboxes, confidences, landmarks)
            - bboxes: List of [x1, y1, x2, y2] arrays
            - confidences: List of confidence scores
            - landmarks: List of 5x2 landmark arrays (right_eye, left_eye, nose, right_mouth, left_mouth)
        """
        if self.detector is None:
            return [], [], []

        h, w = image.shape[:2]

        # Update input size to match image
        self.detector.setInputSize((w, h))

        # Try detecting in original orientation
        bboxes, confidences, landmarks = self._detect_single(image)

        # If no faces found and rotation is enabled, try rotated images
        if len(bboxes) == 0 and try_rotations:
            for angle in [90, 180, 270]:
                rotated = self._rotate_image(image, angle)
                rot_bboxes, rot_confs, rot_landmarks = self._detect_single(rotated)

                if len(rot_bboxes) > 0:
                    # Transform bboxes back to original orientation
                    bboxes = self._transform_bboxes_back(rot_bboxes, angle, w, h)
                    # Transform landmarks back
                    landmarks = self._transform_landmarks_back(rot_landmarks, angle, w, h)
                    confidences = rot_confs
                    logger.debug(f"Found {len(bboxes)} face(s) at {angle}° rotation")
                    break

        return bboxes, confidences, landmarks

    def _detect_single(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[np.ndarray]]:
        """Detect faces in a single image without rotation.

        Args:
            image: Input image in BGR format

        Returns:
            Tuple of (bboxes, confidences, landmarks)
        """
        h, w = image.shape[:2]
        self.detector.setInputSize((w, h))

        try:
            # Detect faces
            _, faces = self.detector.detect(image)

            if faces is None or len(faces) == 0:
                return [], [], []

            bboxes = []
            confidences = []
            landmarks_list = []

            for face in faces:
                # YuNet returns: [x, y, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, conf]
                x, y, fw, fh = face[:4]
                conf = face[-1]

                if conf >= self.conf_threshold:
                    # Convert to [x1, y1, x2, y2]
                    bbox = np.array([x, y, x + fw, y + fh])
                    bboxes.append(bbox)
                    confidences.append(float(conf))

                    # Extract 5-point landmarks
                    # YuNet order: right_eye, left_eye, nose, right_mouth, left_mouth
                    landmarks = np.array([
                        [face[4], face[5]],   # right eye
                        [face[6], face[7]],   # left eye
                        [face[8], face[9]],   # nose
                        [face[10], face[11]], # right mouth
                        [face[12], face[13]]  # left mouth
                    ], dtype=np.float32)
                    landmarks_list.append(landmarks)

            return bboxes, confidences, landmarks_list

        except Exception as e:
            logger.error(f"Error during face detection: {e}")
            return [], [], []

    def _rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle.

        Args:
            image: Input image
            angle: Rotation angle in degrees (90, 180, 270)

        Returns:
            Rotated image
        """
        if angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return image

    def _transform_bboxes_back(
        self, bboxes: List[np.ndarray], angle: int, orig_w: int, orig_h: int
    ) -> List[np.ndarray]:
        """Transform bounding boxes back to original image orientation.

        Args:
            bboxes: Bounding boxes in rotated image coordinates
            angle: Rotation angle that was applied
            orig_w: Original image width
            orig_h: Original image height

        Returns:
            Bounding boxes in original image coordinates
        """
        transformed = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            if angle == 90:
                # After 90° CW rotation: new_x = orig_h - orig_y, new_y = orig_x
                # Reverse: orig_x = new_y, orig_y = orig_h - new_x
                new_x1 = y1
                new_y1 = orig_h - x2
                new_x2 = y2
                new_y2 = orig_h - x1
            elif angle == 180:
                # After 180° rotation: new_x = orig_w - orig_x, new_y = orig_h - orig_y
                # Reverse: same transformation
                new_x1 = orig_w - x2
                new_y1 = orig_h - y2
                new_x2 = orig_w - x1
                new_y2 = orig_h - y1
            elif angle == 270:
                # After 270° CW (90° CCW): new_x = orig_y, new_y = orig_w - orig_x
                # Reverse: orig_x = orig_w - new_y, orig_y = new_x
                new_x1 = orig_w - y2
                new_y1 = x1
                new_x2 = orig_w - y1
                new_y2 = x2
            else:
                new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2

            # Ensure x1 < x2 and y1 < y2
            new_bbox = np.array([
                min(new_x1, new_x2),
                min(new_y1, new_y2),
                max(new_x1, new_x2),
                max(new_y1, new_y2)
            ])
            transformed.append(new_bbox)

        return transformed

    def _transform_landmarks_back(
        self, landmarks_list: List[np.ndarray], angle: int, orig_w: int, orig_h: int
    ) -> List[np.ndarray]:
        """Transform landmarks back to original image orientation.

        Args:
            landmarks_list: List of 5x2 landmark arrays in rotated image coordinates
            angle: Rotation angle that was applied
            orig_w: Original image width
            orig_h: Original image height

        Returns:
            Landmarks in original image coordinates
        """
        transformed = []

        for landmarks in landmarks_list:
            new_landmarks = np.zeros_like(landmarks)

            for i, (x, y) in enumerate(landmarks):
                if angle == 90:
                    new_x = y
                    new_y = orig_h - x
                elif angle == 180:
                    new_x = orig_w - x
                    new_y = orig_h - y
                elif angle == 270:
                    new_x = orig_w - y
                    new_y = x
                else:
                    new_x, new_y = x, y

                new_landmarks[i] = [new_x, new_y]

            transformed.append(new_landmarks)

        return transformed

    @staticmethod
    def compute_head_tilt(landmarks: np.ndarray) -> float:
        """Compute head tilt angle from eye landmarks.

        Args:
            landmarks: 5x2 array of landmarks (right_eye, left_eye, nose, right_mouth, left_mouth)

        Returns:
            Head tilt angle in degrees (positive = clockwise tilt)
        """
        if landmarks is None or len(landmarks) < 2:
            return 0.0

        right_eye = landmarks[0]
        left_eye = landmarks[1]

        # Calculate angle between eyes
        dx = left_eye[0] - right_eye[0]
        dy = left_eye[1] - right_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))
        return float(angle)


"""Image processing operations and utilities."""

from typing import Tuple

import cv2
import numpy as np


def compute_blur_metric(image: np.ndarray) -> float:
    """Compute blur metric using Laplacian variance.

    Lower values indicate more blur.

    Args:
        image: Input image (grayscale or BGR)

    Returns:
        Laplacian variance (higher = sharper)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()

    return float(variance)


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int],
    pad: bool = True
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """Resize image maintaining aspect ratio.

    Args:
        image: Input image
        target_size: Target size (width, height)
        pad: Whether to pad to exact target size

    Returns:
        Tuple of (resized_image, scale_factors, padding)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size

    # Compute scale to fit within target
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if pad:
        # Create padded image
        pad_w = target_w - new_w
        pad_h = target_h - new_h
        pad_left = pad_w // 2
        pad_top = pad_h // 2

        if len(image.shape) == 3:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
        else:
            padded = np.zeros((target_h, target_w), dtype=image.dtype)

        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        return padded, (scale, scale), (pad_left, pad_top)

    return resized, (scale, scale), (0, 0)


def extract_roi(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Extract region of interest from image.

    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Cropped ROI
    """
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]

    # Clip to image boundaries
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    return image[y1:y2, x1:x2]


def normalize_image(image: np.ndarray, mean: Tuple[float, ...] = (0.5, 0.5, 0.5),
                    std: Tuple[float, ...] = (0.5, 0.5, 0.5)) -> np.ndarray:
    """Normalize image for model input.

    Args:
        image: Input image in BGR format [H, W, 3]
        mean: Mean values for normalization
        std: Std values for normalization

    Returns:
        Normalized image in CHW format [3, H, W]
    """
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert to float and normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Apply mean and std
    image = (image - np.array(mean)) / np.array(std)

    # Convert to CHW format
    image = np.transpose(image, (2, 0, 1))

    return image


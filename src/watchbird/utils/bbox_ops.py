"""Bounding box operations and utilities."""

from typing import List, Tuple

import numpy as np


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: Bounding box [x1, y1, x2, y2]
        box2: Bounding box [x1, y1, x2, y2]

    Returns:
        IoU score between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def box_area(box: np.ndarray) -> float:
    """Compute area of bounding box.

    Args:
        box: Bounding box [x1, y1, x2, y2]

    Returns:
        Area in pixels
    """
    return (box[2] - box[0]) * (box[3] - box[1])


def box_size(box: np.ndarray) -> Tuple[float, float]:
    """Compute width and height of bounding box.

    Args:
        box: Bounding box [x1, y1, x2, y2]

    Returns:
        Tuple of (width, height)
    """
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (width, height)


def clip_box(box: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
    """Clip bounding box to image boundaries.

    Args:
        box: Bounding box [x1, y1, x2, y2]
        image_shape: Image shape (height, width)

    Returns:
        Clipped bounding box
    """
    h, w = image_shape
    box = box.copy()
    box[0] = max(0, min(box[0], w - 1))
    box[1] = max(0, min(box[1], h - 1))
    box[2] = max(0, min(box[2], w - 1))
    box[3] = max(0, min(box[3], h - 1))
    return box


def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert bounding box from [x, y, w, h] to [x1, y1, x2, y2].

    Args:
        box: Bounding box [x, y, w, h]

    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    return np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    """Convert bounding box from [x1, y1, x2, y2] to [x, y, w, h].

    Args:
        box: Bounding box [x1, y1, x2, y2]

    Returns:
        Bounding box [x, y, w, h]
    """
    return np.array([box[0], box[1], box[2] - box[0], box[3] - box[1]])


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> List[int]:
    """Non-Maximum Suppression for overlapping boxes.

    Args:
        boxes: Array of bounding boxes [N, 4] in [x1, y1, x2, y2] format
        scores: Array of confidence scores [N]
        iou_threshold: IoU threshold for suppression

    Returns:
        List of indices to keep
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


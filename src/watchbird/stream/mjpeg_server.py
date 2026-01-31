"""MJPEG streaming server for debug visualization."""

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response

logger = logging.getLogger(__name__)


class MJPEGServer:
    """MJPEG streaming server for headless debugging."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """Initialize MJPEG server.

        Args:
            host: Server host address
            port: Server port
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.server_thread: Optional[threading.Thread] = None

        # Setup routes
        self.app.add_url_rule("/", "index", self._index)
        self.app.add_url_rule("/stream", "stream", self._stream)

    def update_frame(self, frame: np.ndarray) -> None:
        """Update current frame for streaming.

        Args:
            frame: Frame to stream (BGR format)
        """
        with self.frame_lock:
            self.current_frame = frame.copy()

    def _index(self) -> str:
        """Serve index page."""
        return """
        <html>
        <head><title>watchbird Debug Stream</title></head>
        <body>
        <h1>watchbird Debug Stream</h1>
        <img src="/stream" width="640" height="480" />
        </body>
        </html>
        """

    def _generate_frames(self) -> bytes:
        """Generate MJPEG frames."""
        while self.running:
            with self.frame_lock:
                frame = self.current_frame

            if frame is not None:
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

                if ret:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
                    )

            time.sleep(0.033)  # ~30 FPS max

    def _stream(self) -> Response:
        """Stream MJPEG frames."""
        return Response(
            self._generate_frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    def start(self) -> None:
        """Start MJPEG server in background thread."""
        if self.running:
            logger.warning("MJPEG server already running")
            return

        self.running = True

        def run_server() -> None:
            # Disable Flask logging
            log = logging.getLogger("werkzeug")
            log.setLevel(logging.ERROR)

            self.app.run(host=self.host, port=self.port, threaded=True, debug=False)

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()

        logger.info(f"MJPEG server started at http://{self.host}:{self.port}/stream")

    def stop(self) -> None:
        """Stop MJPEG server."""
        self.running = False
        logger.info("MJPEG server stopped")


def draw_detection_boxes(
    frame: np.ndarray,
    track_id: int,
    bbox: np.ndarray,
    state: str,
    person_id: Optional[str] = None,
    confidence: float = 0.0,
    cumulative_confidence: float = 0.0,
    head_tilt: float = 0.0,
    landmarks: Optional[np.ndarray] = None
) -> np.ndarray:
    """Draw detection box on frame.

    Args:
        frame: Input frame
        track_id: Track identifier
        bbox: Bounding box [x1, y1, x2, y2]
        state: Track state (DETECTING, SUSPECT, FRIENDLY, CONFIRMED, ENEMY)
        person_id: Person identifier (for FRIENDLY/CONFIRMED)
        confidence: Confidence score
        cumulative_confidence: Cumulative confidence (for FRIENDLY→CONFIRMED)
        head_tilt: Head tilt angle in degrees (for rotated box)
        landmarks: Optional 5x2 array of facial landmarks

    Returns:
        Annotated frame
    """
    # Color by state
    if state == "CONFIRMED":
        color = (0, 255, 0)  # Bright Green - high confidence, tracking only
        label = f"CONFIRMED: {person_id} ({cumulative_confidence:.0%})"
    elif state == "FRIENDLY":
        color = (0, 200, 100)  # Light green/teal - still building confidence
        label = f"FRIENDLY: {person_id} ({confidence:.2f}|{cumulative_confidence:.0%})"
    elif state == "ENEMY":
        color = (0, 0, 255)  # Red
        label = f"ENEMY: Track {track_id}"
    elif state == "DETECTING":
        color = (0, 165, 255)  # Orange (BGR)
        label = f"DETECTING: Track {track_id}"
    else:  # SUSPECT
        color = (0, 255, 255)  # Yellow
        label = f"SUSPECT: Track {track_id}"

    x1, y1, x2, y2 = map(int, bbox)

    # Draw rotated box if head tilt is significant (>5 degrees)
    if abs(head_tilt) > 5.0:
        # Calculate center and size
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1

        # Get rotated rectangle corners
        rect = ((cx, cy), (width, height), head_tilt)
        box_points = cv2.boxPoints(rect)
        box_points = np.int32(box_points)

        # Draw rotated box
        cv2.drawContours(frame, [box_points], 0, color, 2)

        # Draw tilt indicator line along eye axis
        if landmarks is not None and len(landmarks) >= 2:
            right_eye = tuple(map(int, landmarks[0]))
            left_eye = tuple(map(int, landmarks[1]))
            cv2.line(frame, right_eye, left_eye, (255, 255, 0), 1)  # Cyan line between eyes
    else:
        # Draw regular axis-aligned box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw facial landmarks if available
    if landmarks is not None:
        for i, (lx, ly) in enumerate(landmarks):
            # Different colors for different landmarks
            if i < 2:  # Eyes
                lm_color = (255, 255, 0)  # Cyan
            elif i == 2:  # Nose
                lm_color = (0, 255, 255)  # Yellow
            else:  # Mouth
                lm_color = (255, 0, 255)  # Magenta
            cv2.circle(frame, (int(lx), int(ly)), 2, lm_color, -1)

    # Draw label - rotated if head tilt is significant
    label_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

    if abs(head_tilt) > 5.0:
        # Create rotated label for tilted boxes
        cx = (x1 + x2) / 2
        cy = y1 - label_size[1] - 5  # Position above the box

        # Create a small image for the label with background
        label_width = label_size[0] + 10
        label_height = label_size[1] + 10
        label_img = np.zeros((label_height, label_width, 3), dtype=np.uint8)
        label_img[:] = color  # Fill with state color

        # Put text on label image
        cv2.putText(
            label_img,
            label,
            (5, label_height - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

        # Rotate the label image (negate angle to match bounding box rotation direction)
        rotation_matrix = cv2.getRotationMatrix2D(
            (label_width / 2, label_height / 2),
            -head_tilt,
            1.0
        )

        # Calculate new bounding box size after rotation
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int(label_height * sin_val + label_width * cos_val)
        new_height = int(label_height * cos_val + label_width * sin_val)

        # Adjust rotation matrix for the new size
        rotation_matrix[0, 2] += (new_width - label_width) / 2
        rotation_matrix[1, 2] += (new_height - label_height) / 2

        rotated_label = cv2.warpAffine(
            label_img,
            rotation_matrix,
            (new_width, new_height),
            borderValue=(0, 0, 0)
        )

        # Calculate position to place rotated label (above the rotated box)
        # Offset based on tilt angle (negated to match rotation direction)
        offset_x = int(np.sin(np.radians(-head_tilt)) * (label_size[1] + 10))
        place_x = int(cx - new_width / 2) - offset_x
        place_y = int(cy - new_height / 2)

        # Blend rotated label onto frame
        for i in range(new_height):
            for j in range(new_width):
                py = place_y + i
                px = place_x + j
                if 0 <= py < frame.shape[0] and 0 <= px < frame.shape[1]:
                    # Only draw non-black pixels
                    if np.any(rotated_label[i, j] > 10):
                        frame[py, px] = rotated_label[i, j]
    else:
        # Draw regular horizontal label
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            color,
            -1
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    return frame


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
    confidence: float = 0.0
) -> np.ndarray:
    """Draw detection box on frame.

    Args:
        frame: Input frame
        track_id: Track identifier
        bbox: Bounding box [x1, y1, x2, y2]
        state: Track state (DETECTING, SUSPECT, FRIENDLY, ENEMY)
        person_id: Person identifier (for FRIENDLY)
        confidence: Confidence score

    Returns:
        Annotated frame
    """
    # Color by state
    if state == "FRIENDLY":
        color = (0, 255, 0)  # Green
        label = f"FRIENDLY: {person_id} ({confidence:.2f})"
    elif state == "ENEMY":
        color = (0, 0, 255)  # Red
        label = f"ENEMY: Track {track_id}"
    elif state == "DETECTING":
        color = (255, 128, 0)  # Blue
        label = f"DETECTING: Track {track_id}"
    else:  # SUSPECT
        color = (0, 255, 255)  # Yellow
        label = f"SUSPECT: Track {track_id}"

    # Draw box
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Draw label background
    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        frame,
        (x1, y1 - label_size[1] - 10),
        (x1 + label_size[0], y1),
        color,
        -1
    )

    # Draw label text
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


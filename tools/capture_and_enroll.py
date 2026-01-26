#!/usr/bin/env python3
"""Interactive photo capture and enrollment tool."""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.camera.usb_backend import USBCameraBackend
from watchbird.camera.video_backend import VideoBackend
from watchbird.stream.mjpeg_server import MJPEGServer
try:
    from watchbird.camera.picamera_backend import Picamera2Backend
except ImportError:
    Picamera2Backend = None
from watchbird.utils.image_ops import extract_roi
from watchbird.utils.quality import compute_face_quality

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_gui_available() -> bool:
    """Check if OpenCV GUI (highgui) is available.

    Returns:
        True if GUI is available, False otherwise
    """
    try:
        # Try to create and destroy a test window
        cv2.namedWindow("__test__", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("__test__")
        return True
    except cv2.error:
        return False


class PhotoCaptureSession:
    """Interactive photo capture session for enrollment."""

    def __init__(self, person_name: str, output_dir: Path, face_detector: FaceDetector,
                 camera, config: Config, target_count: int = 10, headless: bool = False,
                 stream_port: int = 8080):
        """Initialize capture session.

        Args:
            person_name: Name of the person being enrolled
            output_dir: Directory to save photos
            face_detector: Face detector instance
            camera: Camera backend instance
            config: Configuration
            target_count: Number of photos to capture
            headless: Run in headless mode with MJPEG streaming
            stream_port: Port for MJPEG stream in headless mode
        """
        self.person_name = person_name
        self.output_dir = output_dir
        self.face_detector = face_detector
        self.camera = camera
        self.config = config
        self.target_count = target_count
        self.headless = headless
        self.stream_port = stream_port

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.captured_photos: List[Tuple[np.ndarray, float, np.ndarray, float]] = []  # (image, quality, bbox, conf)

        # For headless mode
        self.mjpeg_server: Optional[MJPEGServer] = None
        self.capture_requested = False
        self.quit_requested = False
        self.cancel_requested = False

    def _keyboard_listener(self):
        """Listen for keyboard input in headless mode."""
        logger.info("Keyboard listener started. Commands: ENTER=capture, q=finish, ESC/Ctrl+C=cancel")
        try:
            while not self.quit_requested and not self.cancel_requested:
                try:
                    user_input = input()
                    if user_input.lower() == 'q':
                        self.quit_requested = True
                        logger.info("Quit requested...")
                    elif user_input.lower() == 'esc' or user_input == '\x1b':
                        self.cancel_requested = True
                        logger.info("Cancel requested...")
                    else:
                        # Any other input (including empty ENTER) triggers capture
                        self.capture_requested = True
                except EOFError:
                    break
        except Exception as e:
            logger.debug(f"Keyboard listener ended: {e}")

    def run(self) -> bool:
        """Run interactive capture session.

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting photo capture for: {self.person_name}")
        logger.info(f"Target: {self.target_count} photos")
        logger.info("")

        if self.headless:
            # Start MJPEG server for preview
            self.mjpeg_server = MJPEGServer(host="0.0.0.0", port=self.stream_port)
            self.mjpeg_server.start()
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"📺 VIEW PREVIEW: http://localhost:{self.stream_port}/stream")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Instructions (headless mode):")
            logger.info("  - Press ENTER to capture a photo")
            logger.info("  - Type 'q' + ENTER to finish early")
            logger.info("  - Press Ctrl+C to cancel")

            # Start keyboard listener thread
            keyboard_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
            keyboard_thread.start()
        else:
            logger.info("Instructions:")
            logger.info("  - Press SPACE to capture a photo")
            logger.info("  - Press 'q' to finish early")
            logger.info("  - Press ESC to cancel")

        logger.info("")
        logger.info("Tips for best results:")
        logger.info("  - Vary your head position (straight, left, right, up, down)")
        logger.info("  - Vary expressions (neutral, smiling)")
        logger.info("  - Ensure good lighting on your face")
        logger.info("  - Stay at similar distance from camera")
        logger.info("")

        capture_count = 0
        frame_count = 0

        try:
            while capture_count < self.target_count:
                # Check for quit/cancel in headless mode
                if self.headless:
                    if self.cancel_requested:
                        logger.info("Capture cancelled by user")
                        return False
                    if self.quit_requested:
                        logger.info(f"Finishing early with {capture_count} photos")
                        break

                # Get frame
                frame = self.camera.get_frame()
                if frame is None:
                    logger.error("Failed to get frame from camera")
                    break

                frame_count += 1

                # Detect faces
                bboxes, confs, _ = self.face_detector.detect(frame, try_rotations=False)

                # Draw UI
                display_frame = frame.copy()

                # Status text
                status_text = f"Captured: {capture_count}/{self.target_count}"
                cv2.putText(display_frame, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Instructions
                if self.headless:
                    cv2.putText(display_frame, "ENTER: Capture | Q: Finish",
                               (10, display_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "SPACE: Capture | Q: Finish | ESC: Cancel",
                               (10, display_frame.shape[0] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Draw face boxes
                best_face_idx = -1
                if bboxes:
                    if len(bboxes) == 1:
                        best_face_idx = 0
                        color = (0, 255, 0)  # Green for single face
                        status = "Ready to capture"
                    else:
                        # Multiple faces - find largest
                        areas = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) for bbox in bboxes]
                        best_face_idx = np.argmax(areas)
                        color = (0, 255, 255)  # Yellow for multiple faces
                        status = f"{len(bboxes)} faces - will use largest"

                    cv2.putText(display_frame, status, (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Draw all faces
                    for idx, (bbox, conf) in enumerate(zip(bboxes, confs)):
                        x1, y1, x2, y2 = map(int, bbox)
                        if idx == best_face_idx:
                            # Best face - thick green box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                            cv2.putText(display_frame, f"Main ({conf:.2f})", (x1, y1-10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        else:
                            # Other faces - thin gray box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (128, 128, 128), 1)
                else:
                    cv2.putText(display_frame, "No face detected", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Handle capture request (headless mode)
                should_capture = False
                if self.headless:
                    # Update MJPEG stream
                    self.mjpeg_server.update_frame(display_frame)

                    if self.capture_requested:
                        self.capture_requested = False
                        should_capture = True

                    # Small delay to prevent CPU spinning
                    time.sleep(0.033)  # ~30 FPS
                else:
                    # Show frame (GUI mode)
                    cv2.imshow(f"Capture: {self.person_name}", display_frame)

                    # Handle keys
                    key = cv2.waitKey(1) & 0xFF

                    if key == 27:  # ESC
                        logger.info("Capture cancelled by user")
                        cv2.destroyAllWindows()
                        return False

                    elif key == ord('q'):  # Q
                        logger.info(f"Finishing early with {capture_count} photos")
                        break

                    elif key == ord(' '):  # SPACE
                        should_capture = True

                # Process capture
                if should_capture:
                    if best_face_idx >= 0:
                        # Capture photo
                        bbox = bboxes[best_face_idx]
                        conf = confs[best_face_idx]

                        # Extract face ROI
                        face_roi = extract_roi(frame, bbox)

                        # Compute quality
                        quality, _ = compute_face_quality(bbox, face_roi, conf)

                        # Store
                        self.captured_photos.append((frame.copy(), quality, bbox, conf))
                        capture_count += 1

                        logger.info(f"  ✓ Captured photo {capture_count}/{self.target_count} "
                                   f"(quality={quality:.3f}, conf={conf:.2f})")

                        if self.headless:
                            # Visual feedback on stream
                            feedback_frame = display_frame.copy()
                            cv2.putText(feedback_frame, "CAPTURED!",
                                       (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                            self.mjpeg_server.update_frame(feedback_frame)
                            time.sleep(0.5)  # Show for 500ms
                        else:
                            # Visual feedback (GUI mode)
                            feedback_frame = display_frame.copy()
                            cv2.putText(feedback_frame, "CAPTURED!",
                                       (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                            cv2.imshow(f"Capture: {self.person_name}", feedback_frame)
                            cv2.waitKey(500)  # Show for 500ms
                    else:
                        logger.warning("  ✗ No face detected - cannot capture")

        except KeyboardInterrupt:
            logger.info("\n  Interrupted by user")
            if self.headless:
                return False

        finally:
            if self.headless:
                if self.mjpeg_server:
                    self.mjpeg_server.stop()
            else:
                cv2.destroyAllWindows()

        if capture_count == 0:
            logger.error("No photos captured")
            return False

        # Save best photos
        logger.info(f"\nSaving {min(capture_count, self.target_count)} best photos...")
        return self._save_best_photos()

    def _save_best_photos(self) -> bool:
        """Save the best captured photos.

        Returns:
            True if successful, False otherwise
        """
        if not self.captured_photos:
            return False

        # Sort by quality (descending)
        sorted_photos = sorted(self.captured_photos, key=lambda x: x[1], reverse=True)

        # Take top N
        best_photos = sorted_photos[:self.target_count]

        # Save
        timestamp = int(time.time())
        for idx, (image, quality, bbox, conf) in enumerate(best_photos):
            filename = f"{self.person_name}_{timestamp}_{idx+1:02d}_q{quality:.2f}.jpg"
            filepath = self.output_dir / filename

            cv2.imwrite(str(filepath), image)
            logger.info(f"  ✓ Saved: {filename} (quality={quality:.3f}, conf={conf:.2f})")

        logger.info(f"\n✅ Saved {len(best_photos)} photos to: {self.output_dir}")
        return True


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Interactive photo capture and enrollment tool"
    )
    parser.add_argument(
        "--person",
        type=str,
        required=True,
        help="Name of person to enroll"
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["usb", "picamera2", "video_file"],
        default="usb",
        help="Camera backend"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Camera device ID (for usb backend)"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Video file path (for video_file backend)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="friendly",
        help="Base directory for enrollment photos"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of photos to capture"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--auto-enroll",
        action="store_true",
        help="Automatically run enrollment after capture"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode with MJPEG streaming (auto-enabled if no GUI)"
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        help="Port for MJPEG stream in headless mode (default: 8080)"
    )

    args = parser.parse_args()

    # Check if GUI is available, fall back to headless mode if not
    headless = args.headless
    if not headless and not check_gui_available():
        logger.warning("OpenCV GUI (highgui) not available - running in headless mode")
        logger.info("View the camera stream at http://localhost:%d/stream", args.stream_port)
        headless = True

    # Validate person name
    person_name = args.person.strip().lower()
    if not person_name or '/' in person_name or '\\' in person_name:
        logger.error("Invalid person name")
        return

    # Load configuration
    config = Config(args.config)

    # Setup output directory
    output_dir = Path(args.output_dir) / person_name

    # Initialize camera
    camera = None
    try:
        if args.backend == "usb":
            camera = USBCameraBackend(
                device_id=args.device_id,
                resolution=tuple(config.camera["resolution"]),
                fps=config.camera["fps"]
            )
        elif args.backend == "picamera2":
            if Picamera2Backend is None:
                logger.error("Picamera2 not available on this platform")
                return
            camera = Picamera2Backend(
                resolution=tuple(config.camera["resolution"]),
                fps=config.camera["fps"]
            )
        elif args.backend == "video_file":
            if not args.video:
                logger.error("Video path required for video_file backend")
                return
            camera = VideoBackend(
                video_path=args.video,
                resolution=tuple(config.camera["resolution"]),
                fps=config.camera["fps"]
            )

        if not camera.open():
            logger.error("Failed to open camera")
            return

        # Initialize face detector
        face_detector = FaceDetector(
            model_path=config.models.get("face_detector"),
            conf_threshold=config.detection["face_conf_threshold"],
            use_gpu=config.inference.get("use_gpu", True),
            gpu_device_id=config.inference.get("gpu_device_id", 0)
        )

        if not face_detector.load():
            logger.error("Failed to load face detector")
            return

        # Run capture session
        session = PhotoCaptureSession(
            person_name=person_name,
            output_dir=output_dir,
            face_detector=face_detector,
            camera=camera,
            config=config,
            target_count=args.count,
            headless=headless,
            stream_port=args.stream_port
        )

        if session.run():
            logger.info("\n" + "="*80)
            logger.info("Photo capture complete!")
            logger.info("="*80)

            if args.auto_enroll:
                logger.info("\nRunning enrollment...")
                import subprocess
                import sys
                result = subprocess.run([
                    sys.executable, "tools/enroll.py",  # Use current Python interpreter
                    "--data-dir", args.output_dir,
                    "--config", args.config
                ])

                if result.returncode == 0:
                    logger.info("\n✅ Enrollment complete!")
                else:
                    logger.error("\n❌ Enrollment failed")
            else:
                logger.info("\nTo enroll these photos, run:")
                logger.info(f"  python tools/enroll.py --data-dir {args.output_dir} --config {args.config}")

    finally:
        if camera:
            camera.release()


if __name__ == "__main__":
    main()

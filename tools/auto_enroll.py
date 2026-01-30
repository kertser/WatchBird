#!/usr/bin/env python3
"""Automatic enrollment with quality-driven capture.

Continuously captures frames and tests recognition confidence until
a high confidence threshold is reached.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from watchbird.camera.usb_backend import USBCameraBackend
try:
    from watchbird.camera.picamera_backend import Picamera2Backend
except ImportError:
    Picamera2Backend = None

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.index.faiss_wrapper import FaissIndex
from watchbird.index.meta_store import MetaStore
from watchbird.utils.image_ops import extract_roi
from watchbird.utils.quality import compute_face_quality

logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for detailed diagnostics
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


class AutoEnrollmentSession:
    """Automatic enrollment session with confidence-based capture."""

    def __init__(
        self,
        person_id: str,
        config: Config,
        target_confidence: float = 0.85,
        min_photos: int = 5,
        max_photos: int = 30,
        test_interval: int = 3,
        headless: bool = False
    ):
        """Initialize auto-enrollment session.

        Args:
            person_id: Person identifier
            config: Configuration object
            target_confidence: Target recognition confidence (0-1)
            min_photos: Minimum photos before testing
            max_photos: Maximum photos to capture
            test_interval: Test recognition every N photos
            headless: Run without GUI (no preview window)
        """
        self.person_id = person_id
        self.config = config
        self.target_confidence = target_confidence
        self.min_photos = min_photos
        self.max_photos = max_photos
        self.test_interval = test_interval
        self.headless = headless

        # Output directory
        self.output_dir = Path("friendly") / person_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Components
        self.camera = None
        self.face_detector = None
        self.face_embedder = None
        self.faiss_index = None
        self.meta_store = None

        # Captured photos
        self.photos: List[Tuple[np.ndarray, float, str]] = []  # (image, quality, filepath)
        self.timestamp = int(time.time())

    def initialize(self, backend: str = "usb", device_id: int = 0) -> bool:
        """Initialize components.

        Args:
            backend: Camera backend
            device_id: Camera device ID

        Returns:
            True if successful
        """
        logger.info("Initializing auto-enrollment system...")

        # Camera
        if backend == "usb":
            self.camera = USBCameraBackend(
                device_id=device_id,
                resolution=tuple(self.config.camera["resolution"]),
                fps=self.config.camera["fps"]
            )
        elif backend == "picamera2":
            if Picamera2Backend is None:
                logger.error("Picamera2 not available")
                return False
            self.camera = Picamera2Backend(
                resolution=tuple(self.config.camera["resolution"]),
                fps=self.config.camera["fps"]
            )
        else:
            logger.error(f"Unknown backend: {backend}")
            return False

        if not self.camera.open():
            logger.error("Failed to open camera")
            return False

        # Face detector
        self.face_detector = FaceDetector(
            model_path=self.config.models.get("face_detector"),
            conf_threshold=self.config.detection["face_conf_threshold"],
            use_gpu=self.config.inference.get("use_gpu", True),
            gpu_device_id=self.config.inference.get("gpu_device_id", 0),
            detection_scale=self.config.detection.get("detection_scale", 1.0),
            max_detection_size=self.config.detection.get("max_detection_size", 640)
        )

        if not self.face_detector.load():
            logger.error("Failed to load face detector")
            return False

        # Face embedder
        self.face_embedder = FaceEmbedder(
            model_path=self.config.models.get("face_embedder"),
            use_gpu=self.config.inference.get("use_gpu", True),
            gpu_device_id=self.config.inference.get("gpu_device_id", 0)
        )

        if not self.face_embedder.load():
            logger.error("Failed to load face embedder")
            return False

        # Load existing index if available (to test against other people)
        index_path = self.config.index["face_index_path"]
        if Path(index_path).exists():
            logger.info("Loading existing index for testing...")
            self.faiss_index = FaissIndex()
            self.faiss_index.load(index_path)
            self.meta_store = MetaStore(self.config.index["meta_path"])
            self.meta_store.load()
        else:
            logger.info("No existing index - will create new one")
            self.faiss_index = None
            self.meta_store = None

        logger.info("Initialization complete!")
        return True

    def capture_photo(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
        """Capture and validate a photo.

        Args:
            frame: Camera frame

        Returns:
            Tuple of (face_image, quality, bbox) or None if no good face
        """
        # Detect faces (without landmarks since alignment is disabled)
        face_bboxes, face_confs, _ = self.face_detector.detect(frame)

        if len(face_bboxes) == 0:
            logger.debug("No faces detected in frame")
            return None

        # Use first/largest face
        face_bbox = face_bboxes[0]
        face_conf = face_confs[0]

        # Extract ROI
        face_roi = extract_roi(frame, face_bbox)

        # Compute quality
        quality, _ = compute_face_quality(
            face_bbox,
            face_roi,
            face_conf,
            min_bbox_size=self.config.quality["min_bbox_size"]
        )

        logger.debug(f"Face detected: conf={face_conf:.3f}, quality={quality:.3f}, min_threshold={self.config.quality['min_face_quality']}")

        # Require minimum quality
        if quality < self.config.quality["min_face_quality"]:
            logger.debug(f"Quality too low: {quality:.3f} < {self.config.quality['min_face_quality']}")
            return None

        return face_roi, quality, face_bbox

    def test_recognition(self) -> Tuple[float, Optional[str]]:
        """Test recognition confidence with current photos.

        Returns:
            Tuple of (confidence, matched_person_id)
            - confidence: Recognition confidence (0-1)
            - matched_person_id: ID of matched person (None if no match)
        """
        if len(self.photos) < self.min_photos:
            return 0.0, None

        # Build temporary index with current photos
        embeddings = []
        for img_path, quality in [(p[2], p[1]) for p in self.photos]:
            img = cv2.imread(img_path)
            if img is None:
                continue

            bboxes, confs, _ = self.face_detector.detect(img)
            if len(bboxes) == 0:
                continue

            face_roi = extract_roi(img, bboxes[0])
            embedding = self.face_embedder.extract(face_roi)

            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) == 0:
            logger.warning("No embeddings extracted from captured photos!")
            return 0.0, None

        logger.debug(f"Testing {len(embeddings)} embeddings for recognition confidence")

        # Create temporary index
        temp_index = FaissIndex()
        temp_index.build(np.array(embeddings))

        # Test each photo against the temporary index
        confidences = []
        for i, embedding in enumerate(embeddings):
            # Search for top k matches (including self)
            k = min(3, len(embeddings))
            similarities, indices = temp_index.search(embedding, k=k)

            if len(similarities) > 0 and len(similarities[0]) > 0:
                # Exclude self-match (first result is always self with similarity ~1.0)
                if len(similarities[0]) > 1:
                    # Use second and third matches (excluding self)
                    other_sims = similarities[0][1:]
                    if len(other_sims) > 0:
                        avg_confidence = np.mean(other_sims)
                        confidences.append(avg_confidence)
                        logger.debug(f"  Photo {i+1}: similarities={other_sims}, avg={avg_confidence:.3f}")

        if not confidences:
            logger.warning("Could not calculate confidence - need at least 2 photos")
            return 0.0, None

        # Return average confidence
        avg_conf = np.mean(confidences)
        logger.debug(f"Average confidence across {len(confidences)} photos: {avg_conf:.3f}")

        # Also test against existing index if available
        matched_person = None
        if self.faiss_index is not None and self.meta_store is not None:
            # Test if we're being confused with another person
            test_embedding = embeddings[0]
            similarities, indices = self.faiss_index.search(test_embedding, k=1)

            if len(similarities) > 0 and similarities[0][0] > 0.7:
                matched_person = self.meta_store.get_person_id(int(indices[0][0]))

        return float(avg_conf), matched_person

    def run(self) -> bool:
        """Run auto-enrollment session.

        Returns:
            True if successful
        """
        logger.info("=" * 80)
        logger.info(f"AUTO-ENROLLMENT SESSION: {self.person_id}")
        logger.info("=" * 80)
        logger.info(f"Target confidence: {self.target_confidence:.2f}")
        logger.info(f"Min photos: {self.min_photos}, Max photos: {self.max_photos}")
        logger.info("")
        logger.info("Instructions:")
        logger.info("  - Position your face in the camera view")
        logger.info("  - Move your head slowly: left, right, up, down")
        logger.info("  - Change expressions: neutral, smiling")
        logger.info("  - System will auto-capture when face quality is good")
        if not self.headless:
            logger.info("  - Press 'q' to stop early")
        else:
            logger.info("  - Press Ctrl+C to stop early (headless mode)")
        logger.info("=" * 80)
        logger.info("")

        photo_count = 0
        last_capture_time = 0
        capture_cooldown = 0.5  # Seconds between captures

        try:
            while photo_count < self.max_photos:
                # Get frame
                frame = self.camera.get_frame()
                if frame is None:
                    logger.warning("Failed to get frame")
                    continue

                # Try to capture
                current_time = time.time()
                if current_time - last_capture_time > capture_cooldown:
                    result = self.capture_photo(frame)

                    if result is not None:
                        face_roi, quality, bbox = result

                        # Save photo
                        photo_count += 1
                        filename = f"{self.person_id}_{self.timestamp}_{photo_count:02d}_q{quality:.2f}.jpg"
                        filepath = str(self.output_dir / filename)

                        cv2.imwrite(filepath, face_roi)
                        self.photos.append((face_roi, quality, filepath))

                        logger.info(f"  ✓ Captured photo {photo_count}/{self.max_photos} (quality={quality:.2f})")

                        last_capture_time = current_time

                        # Test recognition every N photos
                        if photo_count >= self.min_photos and photo_count % self.test_interval == 0:
                            logger.info(f"\n  Testing recognition confidence...")
                            confidence, matched_person = self.test_recognition()

                            logger.info(f"  Current confidence: {confidence:.3f} (target: {self.target_confidence:.3f})")

                            if matched_person and matched_person != self.person_id:
                                logger.warning(f"  ⚠️  Being confused with '{matched_person}' - need more distinctive photos!")

                            if confidence >= self.target_confidence:
                                logger.info(f"\n  🎉 Target confidence reached: {confidence:.3f}")
                                logger.info(f"  ✅ Auto-enrollment complete with {photo_count} photos!\n")
                                break

                # GUI preview (only if not headless)
                if not self.headless:
                    display_frame = frame.copy()

                    # Draw captured face box if we just captured
                    if result is not None:
                        x1, y1, x2, y2 = bbox.astype(int)
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, "CAPTURED", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw status
                    status_text = f"Photos: {photo_count}/{self.max_photos}"
                    cv2.putText(display_frame, status_text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    if photo_count >= self.min_photos:
                        conf, _ = self.test_recognition()
                        conf_text = f"Confidence: {conf:.2f}/{self.target_confidence:.2f}"
                        cv2.putText(display_frame, conf_text, (10, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

                    # Show preview
                    cv2.imshow(f"Auto-Enrollment: {self.person_id}", display_frame)

                    # Check for quit
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        logger.info("\n  Stopped by user")
                        break
                else:
                    # In headless mode, add a small delay to prevent CPU spinning
                    time.sleep(0.01)

        except KeyboardInterrupt:
            logger.info("\n  Interrupted by user")

        finally:
            if not self.headless:
                cv2.destroyAllWindows()
            if self.camera:
                self.camera.release()

        # Final test
        if len(self.photos) >= self.min_photos:
            logger.info("\n" + "=" * 80)
            logger.info("FINAL RESULTS")
            logger.info("=" * 80)
            confidence, matched_person = self.test_recognition()
            logger.info(f"Photos captured: {len(self.photos)}")
            logger.info(f"Final confidence: {confidence:.3f}")

            if confidence >= self.target_confidence:
                logger.info(f"✅ SUCCESS - Target confidence reached!")
            else:
                logger.warning(f"⚠️  Target confidence not reached (need {self.target_confidence:.3f})")
                logger.warning(f"   Consider capturing more photos or lowering target")

            if matched_person and matched_person != self.person_id:
                logger.warning(f"⚠️  Photos similar to existing person '{matched_person}'")
                logger.warning(f"   Make sure correct person was photographed!")

            logger.info("=" * 80 + "\n")

            return confidence >= self.target_confidence

        else:
            logger.error(f"❌ Not enough photos captured (need at least {self.min_photos})")
            return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Auto-enrollment with confidence testing")
    parser.add_argument("--person", type=str, required=True, help="Person ID")
    parser.add_argument("--backend", type=str, default="usb", choices=["usb", "picamera2"],
                        help="Camera backend")
    parser.add_argument("--device-id", type=int, default=0, help="Camera device ID")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")
    parser.add_argument("--target-confidence", type=float, default=0.85,
                        help="Target recognition confidence (0-1)")
    parser.add_argument("--min-photos", type=int, default=5,
                        help="Minimum photos before testing")
    parser.add_argument("--max-photos", type=int, default=30,
                        help="Maximum photos to capture")
    parser.add_argument("--test-interval", type=int, default=3,
                        help="Test recognition every N photos")
    parser.add_argument("--auto-enroll", action="store_true",
                        help="Automatically run enrollment after capture")
    parser.add_argument("--headless", action="store_true",
                        help="Run without GUI preview (auto-detected if GUI unavailable)")

    args = parser.parse_args()

    # Check if GUI is available, fall back to headless if not
    headless = args.headless
    if not headless and not check_gui_available():
        logger.warning("OpenCV GUI (highgui) not available - running in headless mode")
        logger.info("To see preview, install opencv-python with GUI support or use MJPEG stream")
        headless = True

    # Load config
    config = Config(args.config)

    # Create session
    session = AutoEnrollmentSession(
        person_id=args.person,
        config=config,
        target_confidence=args.target_confidence,
        min_photos=args.min_photos,
        max_photos=args.max_photos,
        test_interval=args.test_interval,
        headless=headless
    )

    # Initialize
    if not session.initialize(args.backend, args.device_id):
        logger.error("Failed to initialize auto-enrollment")
        return

    # Run
    success = session.run()

    if success and args.auto_enroll:
        logger.info("\nRunning enrollment...")
        import subprocess
        import sys
        result = subprocess.run([
            sys.executable, "tools/enroll.py",  # Use same Python interpreter
            "--data-dir", "friendly",
            "--config", args.config
        ])

        if result.returncode == 0:
            logger.info("✅ Enrollment complete!")
        else:
            logger.error("❌ Enrollment failed")


if __name__ == "__main__":
    main()

"""Check enrollment photos for multiple faces and quality issues."""
import sys
sys.path.insert(0, "src")

import logging
from pathlib import Path
import cv2

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_enrollment_photos(data_dir: Path, config_path: str = "config.yaml"):
    """Check enrollment photos for quality issues."""

    print("=" * 80)
    print("ENROLLMENT PHOTO QUALITY CHECK")
    print("=" * 80)

    # Load config
    config = Config(config_path)

    # Load face detector
    face_detector = FaceDetector(
        model_path=config.models.get("face_detector", "models/yunet.onnx"),
        conf_threshold=config.detection["face_conf_threshold"]
    )
    if not face_detector.load():
        print("✗ Failed to load face detector")
        return

    # Check each person's directory
    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        print(f"\n{'='*80}")
        print(f"📁 {person_id.upper()}")
        print(f"{'='*80}")

        # Load all images
        image_paths = (
            list(person_dir.glob("*.[jJ][pP][gG]")) +
            list(person_dir.glob("*.[pP][nN][gG]"))
        )

        good_photos = []
        multi_face_photos = []
        no_face_photos = []

        for img_path in sorted(image_paths):
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ✗ {img_path.name} - Failed to load")
                continue

            # Detect faces
            bboxes, confs, _ = face_detector.detect(image, try_rotations=True)

            if len(bboxes) == 0:
                print(f"  ✗ {img_path.name} - NO FACES DETECTED")
                no_face_photos.append(img_path.name)
            elif len(bboxes) == 1:
                print(f"  ✅ {img_path.name} - 1 face (conf={confs[0]:.2f})")
                good_photos.append(img_path.name)
            else:
                print(f"  ⚠️  {img_path.name} - {len(bboxes)} FACES! ❌ NOT RECOMMENDED")
                multi_face_photos.append((img_path.name, len(bboxes)))

        # Summary
        print(f"\n📊 SUMMARY FOR {person_id.upper()}:")
        print(f"   Good photos (1 face):        {len(good_photos)}")
        print(f"   Multiple faces:              {len(multi_face_photos)}")
        print(f"   No faces detected:           {len(no_face_photos)}")

        if good_photos:
            print(f"\n   ✅ RECOMMENDED PHOTOS FOR ENROLLMENT:")
            for photo in good_photos:
                print(f"      • {photo}")

        if multi_face_photos:
            print(f"\n   ⚠️  PHOTOS WITH MULTIPLE FACES (will use first face - may be wrong person!):")
            for photo, count in multi_face_photos:
                print(f"      • {photo} - {count} faces")
            print(f"\n   🔧 RECOMMENDATION: Remove these or crop to show only {person_id}")

        if no_face_photos:
            print(f"\n   ✗ PHOTOS WITH NO FACES (unusable):")
            for photo in no_face_photos:
                print(f"      • {photo}")

    print(f"\n{'='*80}")
    print("💡 RECOMMENDATIONS:")
    print("   1. Use only photos with 1 face for best results")
    print("   2. Remove or crop photos with multiple faces")
    print("   3. Aim for 10-15 good quality photos per person")
    print("   4. Vary angles, lighting, and expressions")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check enrollment photo quality")
    parser.add_argument("--data-dir", type=str, default="friendly", help="Directory with enrollment photos")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file")

    args = parser.parse_args()

    check_enrollment_photos(Path(args.data_dir), args.config)

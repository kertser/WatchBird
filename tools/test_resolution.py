#!/usr/bin/env python3
"""Test auto-resolution tuning to find optimal camera settings."""

import argparse
import logging

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.utils.resolution_tuner import find_optimal_resolution, COMMON_RESOLUTIONS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Test resolution auto-tuning."""
    parser = argparse.ArgumentParser(description="Test auto-resolution tuning")
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=10.0,
        help="Target FPS (default: 10.0)"
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=8.0,
        help="Minimum acceptable FPS (default: 8.0)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-frames",
        type=int,
        default=30,
        help="Number of frames to test per resolution (default: 30)"
    )

    args = parser.parse_args()

    # Load configuration
    config = Config(args.config)

    # Load face detector
    logger.info("Loading face detector...")
    face_detector = FaceDetector(
        model_path=config.models.get("face_detector"),
        conf_threshold=config.detection["face_conf_threshold"],
        use_gpu=config.inference.get("use_gpu", True),
        gpu_device_id=config.inference.get("gpu_device_id", 0),
        detection_scale=config.detection.get("detection_scale", 1.0),
        max_detection_size=config.detection.get("max_detection_size", 640)
    )

    if not face_detector.load():
        logger.error("Failed to load face detector")
        return

    # Test resolutions
    print("\n" + "=" * 80)
    print("RESOLUTION AUTO-TUNING TEST")
    print("=" * 80)
    print(f"Camera device: {args.device_id}")
    print(f"Target FPS: {args.target_fps}")
    print(f"Min FPS: {args.min_fps}")
    print(f"Test frames: {args.test_frames}")
    print(f"Resolutions to test: {len(COMMON_RESOLUTIONS)}")
    print("=" * 80)

    optimal = find_optimal_resolution(
        device_id=args.device_id,
        detector=face_detector,
        target_fps=args.target_fps,
        min_fps=args.min_fps,
        test_frames=args.test_frames
    )

    print("\n" + "=" * 80)
    print(f"RECOMMENDED RESOLUTION: {optimal[0]}x{optimal[1]}")
    print("=" * 80)
    print("\nUpdate your config.yaml:")
    print(f"  camera:")
    print(f"    resolution: [{optimal[0]}, {optimal[1]}]")
    print()


if __name__ == "__main__":
    main()

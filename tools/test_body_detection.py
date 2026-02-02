#!/usr/bin/env python3
"""Test body detection and segmentation setup."""

import sys
from pathlib import Path

def test_imports():
    """Test if modules can be imported."""
    print("Testing imports...")
    try:
        from watchbird.detect import BodyDetector, HumanSegmenter
        print("✓ BodyDetector and HumanSegmenter imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_models():
    """Test if models are available."""
    print("\nChecking models...")
    models_dir = Path("models")

    models = {
        "yolov8n.onnx": "Body detector (YOLOv8n)",
        "human_seg.onnx": "Human segmenter",
    }

    all_exist = True
    for model_file, desc in models.items():
        path = models_dir / model_file
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"✓ {desc}: {path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {desc}: {path} NOT FOUND")
            all_exist = False

    if not all_exist:
        print("\nTo download missing models:")
        print("  python tools/download_models.py")
        print("  Select option 'B' for body detection + segmentation")

    return all_exist

def test_loading():
    """Test if models can be loaded."""
    print("\nTesting model loading...")

    try:
        from watchbird.detect import BodyDetector, HumanSegmenter

        # Test body detector
        print("Loading body detector...")
        body_detector = BodyDetector(
            model_path="models/yolov8n.onnx",
            use_gpu=False  # Use CPU for test
        )
        if body_detector.load():
            print("✓ Body detector loaded successfully")
        else:
            print("✗ Body detector failed to load")
            return False

        # Test segmenter
        print("Loading human segmenter...")
        segmenter = HumanSegmenter(
            model_path="models/human_seg.onnx",
            use_gpu=False  # Use CPU for test
        )
        if segmenter.load():
            print("✓ Human segmenter loaded successfully")
        else:
            print("✗ Human segmenter failed to load")
            return False

        return True

    except Exception as e:
        print(f"✗ Loading failed: {e}")
        return False

def test_inference():
    """Test basic inference."""
    print("\nTesting inference...")

    try:
        import numpy as np
        from watchbird.detect import BodyDetector, HumanSegmenter

        # Create dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test body detection
        body_detector = BodyDetector(model_path="models/yolov8n.onnx", use_gpu=False)
        if body_detector.load():
            bboxes, confs = body_detector.detect(dummy_frame)
            print(f"✓ Body detection inference OK (detected {len(bboxes)} bodies)")

        # Test segmentation
        segmenter = HumanSegmenter(model_path="models/human_seg.onnx", use_gpu=False)
        if segmenter.load():
            mask = segmenter.segment(dummy_frame)
            print(f"✓ Segmentation inference OK (mask shape: {mask.shape})")

        return True

    except Exception as e:
        print(f"✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 70)
    print("WatchBird - Body Detection & Segmentation Test")
    print("=" * 70)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test models
    results.append(("Models", test_models()))

    # Test loading (only if models exist)
    if results[-1][1]:
        results.append(("Loading", test_loading()))

        # Test inference (only if loading succeeded)
        if results[-1][1]:
            results.append(("Inference", test_inference()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ All tests passed! Body detection ready to use.")
        print("\nTo enable in runtime:")
        print("  1. Edit config.yaml:")
        print("     detection:")
        print("       body_detection: true")
        print("       segmentation: true")
        print("  2. Run: python tools/run_runtime.py --config config.yaml")
        return 0
    else:
        print("\n✗ Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

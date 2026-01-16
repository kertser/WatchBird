"""Simple test to verify installation and imports."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")

    try:
        from watchbird import __version__
        print(f"✓ watchbird version: {__version__}")

        from watchbird.config import Config
        print("✓ watchbird.config")

        from watchbird.camera.base import CameraBackend
        from watchbird.camera.video_backend import VideoBackend
        print("✓ watchbird.camera")

        from watchbird.detect.face_detector import FaceDetector
        print("✓ watchbird.detect")

        from watchbird.track.tracker import Tracker
        print("✓ watchbird.track")

        from watchbird.embed.face_embedder import FaceEmbedder
        print("✓ watchbird.embed")

        from watchbird.index.faiss_wrapper import FaissIndex
        from watchbird.index.meta_store import MetaStore
        print("✓ watchbird.index")

        from watchbird.fusion.similarity import SimilarityFusion
        from watchbird.fusion.aggregation import TemporalAggregator
        print("✓ watchbird.fusion")

        from watchbird.runtime.state_machine import TrackStateMachine
        from watchbird.runtime.events import EventEmitter
        print("✓ watchbird.runtime")

        from watchbird.stream.mjpeg_server import MJPEGServer
        print("✓ watchbird.stream")

        from watchbird.utils.bbox_ops import compute_iou
        from watchbird.utils.image_ops import compute_blur_metric
        from watchbird.utils.quality import compute_face_quality
        print("✓ watchbird.utils")

        print("\n✅ All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_dependencies():
    """Test that required dependencies are installed."""
    print("\nTesting dependencies...")

    deps = {
        "numpy": "numpy",
        "opencv": "cv2",
        "onnxruntime": "onnxruntime",
        "faiss": "faiss",
        "yaml": "yaml",
        "flask": "flask"
    }

    all_ok = True

    for name, module in deps.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            all_ok = False

    if all_ok:
        print("\n✅ All dependencies installed!")
    else:
        print("\n❌ Some dependencies missing. Run: pip install -e .")

    return all_ok


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")

    config_path = Path(__file__).parent.parent / "config.yaml"

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    try:
        from watchbird.config import Config
        config = Config(str(config_path))

        print(f"✓ Config loaded")
        print(f"  Camera backend: {config.camera['backend']}")
        print(f"  Resolution: {config.camera['resolution']}")
        print(f"  T_ACCEPT: {config.thresholds['t_accept']}")

        print("\n✅ Configuration OK!")
        return True

    except Exception as e:
        print(f"❌ Config error: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("watchbird Installation Test")
    print("=" * 60)

    results = []

    results.append(("Dependencies", test_dependencies()))
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print("\n🎉 All tests passed! System ready.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


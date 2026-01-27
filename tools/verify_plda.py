#!/usr/bin/env python3
"""Verify PLDA installation and configuration."""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_files():
    """Check if required files exist."""
    logger.info("Checking file structure...")

    files_to_check = {
        "PLDA Scorer": "src/watchbird/fusion/plda_scorer.py",
        "Unified Scorer": "src/watchbird/fusion/unified_scorer.py",
        "Config": "config.yaml",
        "FAISS Index": "data/index/face.index",
        "Metadata": "data/index/meta.jsonl",
    }

    all_exist = True
    for name, path in files_to_check.items():
        exists = Path(path).exists()
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {name}: {path}")
        if not exists:
            all_exist = False

    return all_exist


def check_plda_model():
    """Check if PLDA model exists."""
    logger.info("\nChecking PLDA model...")

    plda_npz = Path("data/index/plda.npz")
    plda_json = Path("data/index/plda.models.json")

    if plda_npz.exists() and plda_json.exists():
        logger.info(f"  ✓ PLDA model found")
        logger.info(f"    - {plda_npz} ({plda_npz.stat().st_size / 1024:.1f} KB)")
        logger.info(f"    - {plda_json} ({plda_json.stat().st_size / 1024:.1f} KB)")
        return True
    elif plda_npz.exists():
        logger.warning(f"  ⚠ PLDA parameters found but identity models missing")
        logger.warning(f"    Found: {plda_npz}")
        logger.warning(f"    Missing: {plda_json}")
        return False
    else:
        logger.warning("  ✗ PLDA model not found")
        logger.warning("    Run: python tools/enroll.py --train-plda")
        return False


def check_config():
    """Check PLDA configuration."""
    logger.info("\nChecking configuration...")

    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.error("  ✗ config.yaml not found")
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        if "plda" not in config:
            logger.warning("  ⚠ PLDA section missing in config.yaml")
            logger.warning("    Add PLDA configuration section")
            return False

        plda_config = config["plda"]
        enabled = plda_config.get("enabled", False)

        logger.info(f"  PLDA Enabled: {enabled}")
        logger.info(f"  Model Path: {plda_config.get('model_path', 'N/A')}")
        logger.info(f"  FAISS K: {plda_config.get('faiss_k', 'N/A')}")
        logger.info(f"  LLR Threshold: {plda_config.get('llr_threshold', 'N/A')}")
        logger.info(f"  Calibrate: {plda_config.get('calibrate', 'N/A')}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to read config: {e}")
        return False


def test_import():
    """Test importing PLDA modules."""
    logger.info("\nTesting module imports...")

    try:
        from watchbird.fusion.plda_scorer import PLDAScorer
        logger.info("  ✓ PLDAScorer imported successfully")
    except ImportError as e:
        logger.error(f"  ✗ Failed to import PLDAScorer: {e}")
        return False

    try:
        from watchbird.fusion.unified_scorer import UnifiedScorer
        logger.info("  ✓ UnifiedScorer imported successfully")
    except ImportError as e:
        logger.error(f"  ✗ Failed to import UnifiedScorer: {e}")
        return False

    return True


def test_plda_load():
    """Test loading PLDA model."""
    logger.info("\nTesting PLDA model loading...")

    plda_path = Path("data/index/plda.npz")
    if not plda_path.exists():
        logger.info("  ⊘ Skipping (PLDA model not found)")
        return True

    try:
        from watchbird.fusion.plda_scorer import PLDAScorer

        plda = PLDAScorer()
        success = plda.load(str(plda_path))

        if success:
            logger.info(f"  ✓ PLDA model loaded successfully")
            logger.info(f"    - Trained: {plda.is_trained}")
            logger.info(f"    - Embedding dim: {plda.embedding_dim}")
            logger.info(f"    - PLDA dim: {plda.plda_dim}")
            logger.info(f"    - Identities: {len(plda.identity_models)}")

            if plda.identity_models:
                logger.info(f"    - Identity names: {', '.join(plda.identity_models.keys())}")

                # Show sample counts
                for person_id, count in plda.identity_counts.items():
                    logger.info(f"      - {person_id}: {count} samples")

            return True
        else:
            logger.error("  ✗ PLDA model failed to load")
            return False

    except Exception as e:
        logger.error(f"  ✗ Failed to load PLDA model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unified_scorer():
    """Test creating UnifiedScorer."""
    logger.info("\nTesting UnifiedScorer creation...")

    try:
        from watchbird.config import Config
        from watchbird.index.faiss_wrapper import FaissIndex
        from watchbird.index.meta_store import MetaStore
        from watchbird.fusion.unified_scorer import create_scorer_from_config

        # Load config
        config = Config("config.yaml")

        # Load FAISS and metadata
        faiss_index = FaissIndex()
        if not faiss_index.load(config.index["face_index_path"]):
            logger.warning("  ⚠ FAISS index not found, skipping scorer test")
            return True

        meta_store = MetaStore(config.index["meta_path"])
        if not meta_store.load():
            logger.warning("  ⚠ Metadata not found, skipping scorer test")
            return True

        # Create scorer
        scorer = create_scorer_from_config(faiss_index, meta_store, config)

        logger.info("  ✓ UnifiedScorer created successfully")

        # Get info
        info = scorer.get_scoring_info()
        logger.info(f"    - Backend: {info['backend']}")
        logger.info(f"    - FAISS vectors: {info['faiss_vectors']}")
        logger.info(f"    - PLDA enabled: {info['plda_enabled']}")
        logger.info(f"    - PLDA available: {info['plda_available']}")

        if info['plda_available']:
            logger.info(f"    - PLDA identities: {info['plda_identities']}")

        return True

    except Exception as e:
        logger.error(f"  ✗ Failed to create UnifiedScorer: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main verification function."""
    parser = argparse.ArgumentParser(description="Verify PLDA installation")
    parser.add_argument("--full", action="store_true", help="Run full tests including scorer creation")
    args = parser.parse_args()

    print("=" * 70)
    print("PLDA Installation Verification")
    print("=" * 70)
    print()

    checks = []

    # Basic checks
    checks.append(("File Structure", check_files()))
    checks.append(("PLDA Model", check_plda_model()))
    checks.append(("Configuration", check_config()))
    checks.append(("Module Imports", test_import()))

    # Optional full tests
    if args.full:
        checks.append(("PLDA Load", test_plda_load()))
        checks.append(("UnifiedScorer", test_unified_scorer()))

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    passed = sum(1 for _, result in checks if result)
    total = len(checks)

    for name, result in checks:
        status = "PASS" if result else "FAIL"
        logger.info(f"  {status}: {name}")

    print()
    if passed == total:
        logger.info(f"✓ All checks passed ({passed}/{total})")
        print()
        print("PLDA is properly installed and configured!")
        print()
        print("Next steps:")
        print("  1. Enroll identities: python tools/enroll.py --train-plda")
        print("  2. Enable in config: plda.enabled = true")
        print("  3. Run recognition: python tools/run_runtime.py --backend usb")
        return 0
    else:
        logger.warning(f"⚠ Some checks failed ({passed}/{total} passed)")
        print()
        print("Please fix the issues above before using PLDA.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

"""Comprehensive threshold calibration tool for WatchBird.

This tool:
1. Evaluates model performance on enrolled data
2. Computes optimal thresholds for best precision/recall
3. Tests different models if available
4. Optionally updates config.yaml with recommended values
"""
import sys
sys.path.insert(0, "src")

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import numpy as np
from itertools import combinations
import yaml

from watchbird.config import Config
from watchbird.detect.face_detector import FaceDetector
from watchbird.embed.face_embedder import FaceEmbedder
from watchbird.utils.image_ops import extract_roi

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def compute_eer_threshold(intra_sims: List[float], inter_sims: List[float]) -> Tuple[float, float]:
    """Compute Equal Error Rate (EER) threshold.

    The EER threshold is where False Accept Rate = False Reject Rate.

    Returns:
        Tuple of (eer_threshold, eer_value)
    """
    thresholds = np.linspace(0.0, 1.0, 1000)

    best_threshold = 0.5
    min_diff = float('inf')
    best_eer = 0.5

    for thresh in thresholds:
        # False Accept Rate: proportion of different-person pairs above threshold
        far = sum(1 for s in inter_sims if s >= thresh) / max(len(inter_sims), 1)

        # False Reject Rate: proportion of same-person pairs below threshold
        frr = sum(1 for s in intra_sims if s < thresh) / max(len(intra_sims), 1)

        diff = abs(far - frr)
        if diff < min_diff:
            min_diff = diff
            best_threshold = thresh
            best_eer = (far + frr) / 2

    return best_threshold, best_eer


def compute_optimal_threshold(
    intra_sims: List[float],
    inter_sims: List[float],
    target_far: float = 0.01
) -> float:
    """Compute threshold for target False Accept Rate.

    Args:
        intra_sims: Same-person similarities
        inter_sims: Different-person similarities
        target_far: Target false accept rate (default 1%)

    Returns:
        Threshold value
    """
    # Sort inter-person similarities descending
    sorted_inter = sorted(inter_sims, reverse=True)

    # Find threshold where FAR = target_far
    idx = int(len(sorted_inter) * target_far)
    idx = max(0, min(idx, len(sorted_inter) - 1))

    return sorted_inter[idx] if sorted_inter else 0.5


def evaluate_at_threshold(
    intra_sims: List[float],
    inter_sims: List[float],
    threshold: float
) -> Dict[str, float]:
    """Evaluate performance at a given threshold.

    Returns:
        Dict with FAR, FRR, precision, recall, F1
    """
    # True positives: same-person pairs above threshold
    tp = sum(1 for s in intra_sims if s >= threshold)
    # False negatives: same-person pairs below threshold
    fn = sum(1 for s in intra_sims if s < threshold)
    # False positives: different-person pairs above threshold
    fp = sum(1 for s in inter_sims if s >= threshold)
    # True negatives: different-person pairs below threshold
    tn = sum(1 for s in inter_sims if s < threshold)

    far = fp / max(fp + tn, 1)  # False Accept Rate
    frr = fn / max(fn + tp, 1)  # False Reject Rate

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-6)

    return {
        'threshold': threshold,
        'far': far,
        'frr': frr,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fn': fn,
        'fp': fp,
        'tn': tn
    }


def load_embeddings(
    data_dir: Path,
    face_detector: FaceDetector,
    face_embedder: FaceEmbedder,
    max_per_person: int = 20
) -> Dict[str, List[np.ndarray]]:
    """Load and compute embeddings for all persons.

    Args:
        data_dir: Directory containing person subdirectories
        face_detector: Face detector instance
        face_embedder: Face embedder instance
        max_per_person: Maximum embeddings per person (to balance dataset)

    Returns:
        Dict mapping person_id to list of embeddings
    """
    person_embeddings = {}

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_id = person_dir.name
        image_paths = list(person_dir.glob("*.[jJ][pP][gG]")) + \
                     list(person_dir.glob("*.[pP][nN][gG]"))

        # Sort by name and limit
        image_paths = sorted(image_paths)[:max_per_person * 2]  # Get more, filter later

        embeddings = []

        for img_path in image_paths:
            if len(embeddings) >= max_per_person:
                break

            image = cv2.imread(str(img_path))
            if image is None:
                continue

            bboxes, confs, _ = face_detector.detect(image, try_rotations=False)
            if len(bboxes) == 0:
                continue

            # Take highest confidence face
            best_idx = np.argmax(confs)
            face_roi = extract_roi(image, bboxes[best_idx])
            embedding = face_embedder.extract(face_roi)

            if embedding is not None:
                embeddings.append(embedding)

        if embeddings:
            person_embeddings[person_id] = embeddings
            print(f"  {person_id}: {len(embeddings)} embeddings")

    return person_embeddings


def compute_similarity_stats(
    person_embeddings: Dict[str, List[np.ndarray]]
) -> Tuple[List[float], List[float], Dict[str, Dict[str, float]]]:
    """Compute similarity statistics.

    Returns:
        Tuple of (intra_sims, inter_sims, per_person_stats)
    """
    intra_sims = []  # Same person
    inter_sims = []  # Different persons
    per_person_stats = {}

    # Intra-person similarities
    for person_id, embeddings in person_embeddings.items():
        person_sims = []
        if len(embeddings) >= 2:
            for emb1, emb2 in combinations(embeddings, 2):
                sim = cosine_similarity(emb1, emb2)
                intra_sims.append(sim)
                person_sims.append(sim)

        if person_sims:
            per_person_stats[person_id] = {
                'mean': np.mean(person_sims),
                'min': np.min(person_sims),
                'max': np.max(person_sims),
                'std': np.std(person_sims),
                'count': len(person_sims)
            }

    # Inter-person similarities
    person_ids = list(person_embeddings.keys())
    for i, person1 in enumerate(person_ids):
        for person2 in person_ids[i+1:]:
            for emb1 in person_embeddings[person1]:
                for emb2 in person_embeddings[person2]:
                    sim = cosine_similarity(emb1, emb2)
                    inter_sims.append(sim)

    return intra_sims, inter_sims, per_person_stats


def test_model(
    model_path: str,
    data_dir: Path,
    config: Config,
    face_detector: FaceDetector
) -> Optional[Dict]:
    """Test a specific model and return performance metrics.

    Returns:
        Dict with model results or None if failed
    """
    model_name = Path(model_path).name
    print(f"\n{'='*60}")
    print(f"Testing: {model_name}")
    print(f"{'='*60}")

    try:
        embedder = FaceEmbedder(
            model_path=model_path,
            use_gpu=config.inference.get("use_gpu", True),
            gpu_device_id=config.inference.get("gpu_device_id", 0)
        )
        if not embedder.load():
            print(f"  ✗ Failed to load {model_name}")
            return None

        print(f"  Embedding size: {embedder.embedding_size}")

        # Load embeddings
        person_embeddings = load_embeddings(data_dir, face_detector, embedder)

        if len(person_embeddings) < 2:
            print(f"  ✗ Need at least 2 persons, found {len(person_embeddings)}")
            return None

        # Compute similarities
        intra_sims, inter_sims, per_person = compute_similarity_stats(person_embeddings)

        if not intra_sims or not inter_sims:
            print(f"  ✗ Not enough similarity pairs")
            return None

        # Compute metrics
        separation = np.mean(intra_sims) - np.mean(inter_sims)
        eer_threshold, eer = compute_eer_threshold(intra_sims, inter_sims)

        # Compute thresholds for different FAR targets
        thresh_001 = compute_optimal_threshold(intra_sims, inter_sims, 0.01)  # 1% FAR
        thresh_005 = compute_optimal_threshold(intra_sims, inter_sims, 0.05)  # 5% FAR
        thresh_01 = compute_optimal_threshold(intra_sims, inter_sims, 0.10)   # 10% FAR

        # Evaluate at different thresholds
        metrics_eer = evaluate_at_threshold(intra_sims, inter_sims, eer_threshold)
        metrics_001 = evaluate_at_threshold(intra_sims, inter_sims, thresh_001)

        results = {
            'model': model_name,
            'embedding_size': embedder.embedding_size,
            'persons': len(person_embeddings),
            'intra_mean': np.mean(intra_sims),
            'intra_std': np.std(intra_sims),
            'intra_min': np.min(intra_sims),
            'inter_mean': np.mean(inter_sims),
            'inter_std': np.std(inter_sims),
            'inter_max': np.max(inter_sims),
            'separation': separation,
            'eer': eer,
            'eer_threshold': eer_threshold,
            'thresh_far_001': thresh_001,
            'thresh_far_005': thresh_005,
            'thresh_far_01': thresh_01,
            'metrics_eer': metrics_eer,
            'metrics_far001': metrics_001,
            'per_person': per_person
        }

        # Print summary
        print(f"\n  📊 Results:")
        print(f"     Same-person mean:      {results['intra_mean']:.4f} ± {results['intra_std']:.4f}")
        print(f"     Different-person mean: {results['inter_mean']:.4f} ± {results['inter_std']:.4f}")
        print(f"     Separation:            {results['separation']:.4f}")
        print(f"     EER:                   {results['eer']:.2%} @ threshold {results['eer_threshold']:.3f}")
        print(f"     F1 @ EER threshold:    {metrics_eer['f1']:.4f}")

        return results

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return None


def generate_recommendations(results: Dict) -> Dict:
    """Generate configuration recommendations from test results.

    Returns:
        Dict with recommended config values
    """
    # Use EER threshold as base, with some margin for safety
    base_threshold = results['eer_threshold']

    # Adjust based on separation quality
    separation = results['separation']

    if separation > 0.25:
        # Good separation - can use aggressive threshold
        t_accept = base_threshold - 0.02
        t_margin = 0.05
        quality = "EXCELLENT"
    elif separation > 0.15:
        # Moderate separation
        t_accept = base_threshold
        t_margin = 0.08
        quality = "GOOD"
    elif separation > 0.08:
        # Poor separation - need conservative settings
        t_accept = base_threshold + 0.05
        t_margin = 0.10
        quality = "FAIR"
    else:
        # Very poor separation - be very conservative
        t_accept = max(base_threshold + 0.10, results['thresh_far_001'])
        t_margin = 0.15
        quality = "POOR"

    # Ensure t_accept is reasonable
    t_accept = max(0.45, min(0.90, t_accept))

    # PLDA recommendations
    if separation > 0.15:
        plda_enabled = True
        plda_llr = 1.5
    else:
        plda_enabled = True  # Use PLDA to help with poor separation
        plda_llr = 2.5  # Higher threshold for poor models

    return {
        'quality': quality,
        'thresholds': {
            't_accept': round(t_accept, 2),
            't_margin': round(t_margin, 2),
        },
        'plda': {
            'enabled': plda_enabled,
            'llr_threshold': plda_llr,
        },
        'fusion': {
            'consistency_count': 5 if separation > 0.15 else 7,
            'confidence_decay_threshold': 10 if separation > 0.15 else 15,
        }
    }


def update_config(config_path: str, recommendations: Dict) -> bool:
    """Update config.yaml with recommendations.

    Returns:
        True if successful
    """
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Helper to convert numpy types to native Python types
        def to_native(val):
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val

        # Update thresholds
        if 'thresholds' not in config_data:
            config_data['thresholds'] = {}
        config_data['thresholds']['t_accept'] = float(to_native(recommendations['thresholds']['t_accept']))
        config_data['thresholds']['t_margin'] = float(to_native(recommendations['thresholds']['t_margin']))

        # Update PLDA
        if 'plda' not in config_data:
            config_data['plda'] = {}
        config_data['plda']['enabled'] = bool(recommendations['plda']['enabled'])
        config_data['plda']['llr_threshold'] = float(to_native(recommendations['plda']['llr_threshold']))

        # Update fusion
        if 'fusion' not in config_data:
            config_data['fusion'] = {}
        config_data['fusion']['consistency_count'] = int(to_native(recommendations['fusion']['consistency_count']))
        config_data['fusion']['confidence_decay_threshold'] = int(to_native(recommendations['fusion']['confidence_decay_threshold']))

        with open(config_path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

        return True
    except Exception as e:
        print(f"Failed to update config: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate WatchBird thresholds")
    parser.add_argument("--data-dir", type=str, default="friendly",
                       help="Directory with enrollment images")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Config file path")
    parser.add_argument("--test-all-models", action="store_true",
                       help="Test all available models in models/ directory")
    parser.add_argument("--apply", action="store_true",
                       help="Apply recommended settings to config.yaml")

    args = parser.parse_args()

    print("="*70)
    print("WATCHBIRD THRESHOLD CALIBRATION")
    print("="*70)

    # Load config
    config = Config(args.config)
    data_dir = Path(args.data_dir)

    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return

    # Load face detector (shared across models)
    print("\n📷 Loading face detector...")
    face_detector = FaceDetector(
        model_path=config.models.get("face_detector", "models/yunet.onnx"),
        conf_threshold=config.detection["face_conf_threshold"],
        use_gpu=config.inference.get("use_gpu", True),
        gpu_device_id=config.inference.get("gpu_device_id", 0)
    )
    if not face_detector.load():
        print("✗ Failed to load face detector")
        return
    print("✓ Face detector loaded")

    # Find models to test
    models_dir = Path("models")
    if args.test_all_models:
        model_paths = list(models_dir.glob("*.onnx"))
        # Filter out detector models
        model_paths = [p for p in model_paths if "yunet" not in p.name.lower()]
    else:
        current_model = config.models.get("face_embedder", "models/arcface_r100.onnx")
        model_paths = [Path(current_model)]

    print(f"\n🔍 Models to test: {[p.name for p in model_paths]}")

    # Test each model
    all_results = []
    for model_path in model_paths:
        result = test_model(str(model_path), data_dir, config, face_detector)
        if result:
            all_results.append(result)

    if not all_results:
        print("\n✗ No models could be evaluated")
        return

    # Find best model
    print("\n" + "="*70)
    print("📊 MODEL COMPARISON")
    print("="*70)

    # Sort by separation (higher is better)
    all_results.sort(key=lambda x: x['separation'], reverse=True)

    print(f"\n{'Model':<25} {'Separation':>12} {'EER':>10} {'EER Thresh':>12} {'F1':>10}")
    print("-"*70)
    for r in all_results:
        print(f"{r['model']:<25} {r['separation']:>12.4f} {r['eer']:>10.2%} {r['eer_threshold']:>12.3f} {r['metrics_eer']['f1']:>10.4f}")

    best = all_results[0]
    print(f"\n🏆 Best model: {best['model']} (separation: {best['separation']:.4f})")

    # Generate recommendations
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS")
    print("="*70)

    recommendations = generate_recommendations(best)

    print(f"\nModel Quality: {recommendations['quality']}")
    print(f"\nThresholds:")
    print(f"  t_accept: {recommendations['thresholds']['t_accept']}")
    print(f"  t_margin: {recommendations['thresholds']['t_margin']}")
    print(f"\nPLDA:")
    print(f"  enabled: {recommendations['plda']['enabled']}")
    print(f"  llr_threshold: {recommendations['plda']['llr_threshold']}")
    print(f"\nFusion:")
    print(f"  consistency_count: {recommendations['fusion']['consistency_count']}")
    print(f"  confidence_decay_threshold: {recommendations['fusion']['confidence_decay_threshold']}")

    # Per-person analysis
    print("\n" + "="*70)
    print("👤 PER-PERSON ANALYSIS")
    print("="*70)

    for person_id, stats in best['per_person'].items():
        print(f"\n  {person_id}:")
        print(f"    Intra-similarity: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")

        if stats['min'] < recommendations['thresholds']['t_accept']:
            print(f"    ⚠ WARNING: Some photos may not match (min < t_accept)")

    # Apply if requested
    if args.apply:
        print("\n" + "="*70)
        print("📝 APPLYING RECOMMENDATIONS")
        print("="*70)

        if update_config(args.config, recommendations):
            print(f"✓ Updated {args.config}")
            print("\n⚠ NOTE: You should re-enroll faces after changing models or thresholds:")
            print("  python tools/auto_enroll.py --source friendly --rebuild")
        else:
            print("✗ Failed to update config")
    else:
        print(f"\n💡 To apply these settings, run:")
        print(f"   python tools/calibrate_thresholds.py --apply")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()

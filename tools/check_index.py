"""Check FAISS index quality and diagnose recognition issues."""
import sys
sys.path.insert(0, "src")

import logging
from pathlib import Path
import numpy as np

from watchbird.config import Config
from watchbird.index.faiss_wrapper import FaissIndex
from watchbird.index.meta_store import MetaStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_index_quality(config_path: str = "config.yaml"):
    """Check FAISS index quality."""

    print("=" * 80)
    print("FAISS INDEX QUALITY CHECK")
    print("=" * 80)

    # Load config
    config = Config(config_path)

    # Load index
    face_index_path = config.index.get("face_index_path", "data/index/face.index")
    meta_path = config.index.get("meta_path", "data/index/meta.jsonl")

    if not Path(face_index_path).exists():
        print(f"\n❌ ERROR: FAISS index not found at {face_index_path}")
        print("   Run enrollment first: python tools/enroll.py --data-dir friendly --config config.yaml")
        return

    # Load index and metadata
    faiss_index = FaissIndex(dimension=512)
    if not faiss_index.load(face_index_path):
        print(f"\n❌ ERROR: Failed to load FAISS index")
        return

    meta_store = MetaStore(meta_path)
    meta_store.load()

    print(f"\n📊 INDEX STATISTICS:")
    print(f"   Total vectors: {faiss_index.index.ntotal}")
    print(f"   Dimension: {faiss_index.dimension}")

    # Get unique persons
    persons = {}
    for vec_id in range(faiss_index.index.ntotal):
        person_id = meta_store.get_person_id(vec_id)
        if person_id:
            persons[person_id] = persons.get(person_id, 0) + 1

    print(f"   Unique persons: {len(persons)}")
    print(f"\n📋 ENROLLED PERSONS:")
    for person_id, count in sorted(persons.items()):
        print(f"   • {person_id}: {count} face embeddings")

    # Check for margin issues
    print(f"\n🔍 MARGIN ANALYSIS:")
    print("   Testing how well persons are separated...")

    # Sample some vectors and check their nearest neighbors
    total_low_margin = 0
    total_tests = 0

    for test_vec_id in range(min(20, faiss_index.index.ntotal)):
        # Get this vector's embedding
        vector = faiss_index.index.reconstruct(int(test_vec_id))
        vector = np.expand_dims(vector, axis=0)

        # Search for top 2 matches
        similarities, indices = faiss_index.search(vector, k=2)

        if len(similarities[0]) >= 2:
            top1_sim = similarities[0][0]
            top2_sim = similarities[0][1]
            margin = top1_sim - top2_sim

            top1_person = meta_store.get_person_id(int(indices[0][0]))
            top2_person = meta_store.get_person_id(int(indices[0][1]))

            total_tests += 1
            if margin < 0.07:
                total_low_margin += 1
                print(f"   ⚠ Vector {test_vec_id} ({top1_person}): margin={margin:.3f} "
                      f"(1st: {top1_sim:.3f}, 2nd: {top2_sim:.3f} to {top2_person})")

    if total_tests > 0:
        low_margin_pct = (total_low_margin / total_tests) * 100
        print(f"\n   Low margin rate: {total_low_margin}/{total_tests} ({low_margin_pct:.1f}%)")

        if low_margin_pct > 50:
            print(f"\n   ❌ HIGH RISK: {low_margin_pct:.1f}% of embeddings have margin < 0.07")
            print(f"      This explains why tracks fail to become FRIENDLY!")
            print(f"\n      SOLUTIONS:")
            print(f"      1. Lower t_margin in config.yaml to 0.05 or 0.03")
            print(f"      2. Improve enrollment: add more varied photos per person")
            print(f"      3. Upgrade face recognition model (better separation)")
        elif low_margin_pct > 20:
            print(f"\n   ⚠ MODERATE RISK: {low_margin_pct:.1f}% have low margin")
            print(f"      Consider lowering t_margin to 0.05")
        else:
            print(f"\n   ✅ GOOD: Only {low_margin_pct:.1f}% have low margin")

    # Check current thresholds
    print(f"\n⚙️  CURRENT THRESHOLDS:")
    t_accept = config.thresholds.get('t_accept', 0.65)
    t_margin = config.thresholds.get('t_margin', 0.10)
    t_timeout = config.thresholds.get('t_timeout', 5.0)
    consistency_count = config.fusion.get('consistency_count', 6)

    print(f"   t_accept: {t_accept}")
    print(f"   t_margin: {t_margin}")
    print(f"   t_timeout: {t_timeout}s")
    print(f"   consistency_count: {consistency_count}")

    # Recommendations based on observed face scores
    print(f"\n💡 RECOMMENDATIONS:")
    print(f"   Based on your logs showing scores of 0.64-0.69:")
    print(f"   ")
    print(f"   1. Margin is likely the problem (not score)")
    print(f"      → Lower t_margin from {t_margin} to 0.03")
    print(f"   ")
    print(f"   2. Or reduce consistency requirement")
    print(f"      → Lower consistency_count from {consistency_count} to 3")
    print(f"   ")
    print(f"   3. Update config.yaml:")
    print(f"      thresholds:")
    print(f"        t_margin: 0.03  # Was: {t_margin}")
    print(f"      fusion:")
    print(f"        consistency_count: 3  # Was: {consistency_count}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    check_index_quality()

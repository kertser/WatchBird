#!/usr/bin/env python3
"""Example: Using PLDA scorer programmatically.

This script demonstrates how to:
1. Train a PLDA model from scratch
2. Score embeddings using PLDA
3. Compare PLDA vs cosine similarity
4. Save and load PLDA models
"""

import numpy as np
from pathlib import Path

from watchbird.fusion.plda_scorer import PLDAScorer
from watchbird.index.faiss_wrapper import FaissIndex


def example_1_basic_training():
    """Example 1: Basic PLDA training."""
    print("=" * 70)
    print("Example 1: Basic PLDA Training")
    print("=" * 70)

    # Create synthetic embeddings (normally these come from face embedder)
    # 2 identities, 5 samples each, 128-dim embeddings
    np.random.seed(42)

    alice_mean = np.random.randn(128)
    bob_mean = np.random.randn(128)

    alice_samples = [alice_mean + 0.1 * np.random.randn(128) for _ in range(5)]
    bob_samples = [bob_mean + 0.1 * np.random.randn(128) for _ in range(5)]

    embeddings = np.array(alice_samples + bob_samples)
    labels = ["alice"] * 5 + ["bob"] * 5

    print(f"Training data: {len(embeddings)} embeddings, {len(set(labels))} identities")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Create and train PLDA
    plda = PLDAScorer(embedding_dim=128, plda_dim=64)

    print("\nTraining PLDA...")
    success = plda.train(embeddings, labels)

    if success:
        print(f"✓ Training successful")
        print(f"  - Identity models: {len(plda.identity_models)}")
        print(f"  - Identities: {list(plda.identity_models.keys())}")
    else:
        print("✗ Training failed")
        return

    # Test scoring
    print("\nTesting scoring...")
    probe = alice_samples[0]  # Use Alice's first sample as probe

    llr_alice = plda.score(probe, "alice")
    llr_bob = plda.score(probe, "bob")

    print(f"  LLR(probe, alice) = {llr_alice:.3f}")
    print(f"  LLR(probe, bob) = {llr_bob:.3f}")
    print(f"  Decision: {'alice' if llr_alice > llr_bob else 'bob'}")

    # Calibrated scores
    score_alice = plda.calibrate_score(llr_alice)
    score_bob = plda.calibrate_score(llr_bob)

    print(f"\nCalibrated scores [0, 1]:")
    print(f"  score(probe, alice) = {score_alice:.3f}")
    print(f"  score(probe, bob) = {score_bob:.3f}")

    print()


def example_2_batch_scoring():
    """Example 2: Batch scoring multiple candidates."""
    print("=" * 70)
    print("Example 2: Batch Scoring")
    print("=" * 70)

    # Setup (reuse from example 1)
    np.random.seed(42)
    alice_mean = np.random.randn(128)
    bob_mean = np.random.randn(128)
    charlie_mean = np.random.randn(128)

    embeddings = np.vstack([
        [alice_mean + 0.1 * np.random.randn(128) for _ in range(5)],
        [bob_mean + 0.1 * np.random.randn(128) for _ in range(5)],
        [charlie_mean + 0.1 * np.random.randn(128) for _ in range(5)]
    ])
    labels = ["alice"] * 5 + ["bob"] * 5 + ["charlie"] * 5

    plda = PLDAScorer(embedding_dim=128, plda_dim=64)
    plda.train(embeddings, labels)

    # Batch scoring
    probe = alice_mean + 0.05 * np.random.randn(128)
    candidates = ["alice", "bob", "charlie"]

    print(f"Scoring probe against {len(candidates)} candidates...")
    scores = plda.score_batch(probe, candidates)

    print("\nRaw LLR scores:")
    for person_id, llr in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        print(f"  {person_id}: {llr:.3f}")

    # Get decision
    best_id, best_llr, margin = plda.get_decision(
        probe,
        candidates,
        llr_threshold=0.0,
        margin_threshold=0.5
    )

    print(f"\nDecision:")
    print(f"  Best match: {best_id}")
    print(f"  LLR: {best_llr:.3f}")
    print(f"  Margin: {margin:.3f}")

    print()


def example_3_save_load():
    """Example 3: Save and load PLDA model."""
    print("=" * 70)
    print("Example 3: Save and Load PLDA Model")
    print("=" * 70)

    # Train model
    np.random.seed(42)
    embeddings = np.random.randn(10, 128)
    labels = ["person_a"] * 5 + ["person_b"] * 5

    plda = PLDAScorer(embedding_dim=128, plda_dim=64)
    plda.train(embeddings, labels)

    print("Original PLDA:")
    print(f"  {plda}")

    # Save
    model_path = "data/index/example_plda.npz"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving to {model_path}...")
    success = plda.save(model_path)

    if success:
        print("✓ Save successful")

        # Check file sizes
        npz_path = Path(model_path)
        json_path = npz_path.with_suffix('.models.json')

        print(f"  - {npz_path.name}: {npz_path.stat().st_size / 1024:.1f} KB")
        print(f"  - {json_path.name}: {json_path.stat().st_size / 1024:.1f} KB")
    else:
        print("✗ Save failed")
        return

    # Load into new instance
    print(f"\nLoading from {model_path}...")
    plda_loaded = PLDAScorer()
    success = plda_loaded.load(model_path)

    if success:
        print("✓ Load successful")
        print(f"  {plda_loaded}")
    else:
        print("✗ Load failed")
        return

    # Verify they produce same scores
    probe = np.random.randn(128)

    llr_original = plda.score(probe, "person_a")
    llr_loaded = plda_loaded.score(probe, "person_a")

    print(f"\nScore comparison:")
    print(f"  Original: {llr_original:.6f}")
    print(f"  Loaded: {llr_loaded:.6f}")
    print(f"  Match: {np.isclose(llr_original, llr_loaded)}")

    print()


def example_4_plda_vs_cosine():
    """Example 4: Compare PLDA vs cosine similarity."""
    print("=" * 70)
    print("Example 4: PLDA vs Cosine Similarity")
    print("=" * 70)

    # Create scenario: 2 identities, one unknown
    np.random.seed(42)

    alice_mean = np.array([1.0, 0.0] + [0.0] * 126)
    bob_mean = np.array([0.0, 1.0] + [0.0] * 126)

    # Training data
    alice_samples = [alice_mean + 0.1 * np.random.randn(128) for _ in range(5)]
    bob_samples = [bob_mean + 0.1 * np.random.randn(128) for _ in range(5)]

    embeddings = np.array(alice_samples + bob_samples)
    labels = ["alice"] * 5 + ["bob"] * 5

    # Train PLDA
    plda = PLDAScorer(embedding_dim=128, plda_dim=64)
    plda.train(embeddings, labels)

    # Test probes
    alice_probe = alice_mean + 0.05 * np.random.randn(128)
    bob_probe = bob_mean + 0.05 * np.random.randn(128)
    unknown_probe = np.random.randn(128)  # Random unknown person

    # Normalize for cosine similarity
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    alice_model = plda.identity_models["alice"]
    bob_model = plda.identity_models["bob"]

    # Compare on three probes
    test_cases = [
        ("Alice probe", alice_probe),
        ("Bob probe", bob_probe),
        ("Unknown probe", unknown_probe)
    ]

    print("\nComparison:")
    print(f"{'Probe':<15} {'Cosine (Alice)':<15} {'Cosine (Bob)':<15} {'PLDA (Alice)':<15} {'PLDA (Bob)'}")
    print("-" * 70)

    for name, probe in test_cases:
        # Cosine similarities
        cos_alice = cosine_sim(probe, alice_model)
        cos_bob = cosine_sim(probe, bob_model)

        # PLDA LLRs
        llr_alice = plda.score(probe, "alice")
        llr_bob = plda.score(probe, "bob")

        print(f"{name:<15} {cos_alice:>14.3f} {cos_bob:>14.3f} {llr_alice:>14.3f} {llr_bob:>14.3f}")

    print("\nObservations:")
    print("  - PLDA provides calibrated log-likelihood ratios")
    print("  - Unknown probes get negative LLRs (good rejection)")
    print("  - Cosine similarity doesn't calibrate for open-set")

    print()


def example_5_unified_scorer():
    """Example 5: Using UnifiedScorer (FAISS + PLDA)."""
    print("=" * 70)
    print("Example 5: UnifiedScorer (FAISS + PLDA)")
    print("=" * 70)

    try:
        from watchbird.config import Config
        from watchbird.index.meta_store import MetaStore
        from watchbird.fusion.unified_scorer import create_scorer_from_config

        # Load real index if available
        config = Config("config.yaml")

        faiss_index = FaissIndex()
        if not faiss_index.load(config.index["face_index_path"]):
            print("✗ FAISS index not found, skipping example")
            print("  Run: python tools/enroll.py --train-plda")
            print()
            return

        meta_store = MetaStore(config.index["meta_path"])
        if not meta_store.load():
            print("✗ Metadata not found, skipping example")
            print()
            return

        # Create scorer
        scorer = create_scorer_from_config(faiss_index, meta_store, config)

        info = scorer.get_scoring_info()
        print(f"Scorer info:")
        print(f"  Backend: {info['backend']}")
        print(f"  FAISS vectors: {info['faiss_vectors']}")
        print(f"  PLDA available: {info['plda_available']}")

        if info['plda_available']:
            print(f"  PLDA identities: {info['plda_identities']}")

        # Test with random probe (normally from face embedder)
        probe = np.random.randn(faiss_index.embedding_dim)

        print(f"\nScoring random probe...")
        best_id, best_score, margin, all_scores = scorer.score(probe)

        print(f"  Best match: {best_id}")
        print(f"  Score: {best_score:.3f}")
        print(f"  Margin: {margin:.3f}")

        print(f"\nAll scores:")
        for person_id, score in sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {person_id}: {score:.3f}")

    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Make sure WatchBird is properly set up")

    print()


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "PLDA Scorer - Usage Examples" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    examples = [
        example_1_basic_training,
        example_2_batch_scoring,
        example_3_save_load,
        example_4_plda_vs_cosine,
        example_5_unified_scorer,
    ]

    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example {i} failed: {e}\n")
            import traceback
            traceback.print_exc()

        if i < len(examples):
            input("Press Enter to continue to next example...")
            print("\n")

    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()

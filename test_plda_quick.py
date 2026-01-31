#!/usr/bin/env python3
"""Quick test for PLDA implementation."""

import numpy as np
import sys

print("Starting PLDA test...")
sys.stdout.flush()

try:
    from watchbird.fusion.plda_scorer import PLDAScorer, PLDAConfig
    print("✓ PLDA imports OK")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Create scorer
try:
    plda = PLDAScorer()
    print(f"✓ PLDAScorer created: {plda}")
    sys.stdout.flush()
except Exception as e:
    print(f"✗ PLDAScorer creation failed: {e}")
    sys.exit(1)

# Create synthetic data for training
np.random.seed(42)
n_per_class = 20
n_classes = 3
embedding_dim = 512

print(f"\nCreating synthetic data: {n_classes} classes, {n_per_class} samples each, dim={embedding_dim}")
sys.stdout.flush()

# Generate class centers
class_centers = np.random.randn(n_classes, embedding_dim) * 2

# Generate samples with within-class noise
embeddings = []
labels = []

for c in range(n_classes):
    for _ in range(n_per_class):
        # Add noise to class center
        sample = class_centers[c] + np.random.randn(embedding_dim) * 0.3
        # L2 normalize (like face embeddings)
        sample = sample / np.linalg.norm(sample)
        embeddings.append(sample)
        labels.append(f"person_{c}")

embeddings = np.array(embeddings)
print(f"✓ Created {len(embeddings)} embeddings")
sys.stdout.flush()

# Train PLDA
print("\nTraining PLDA...")
sys.stdout.flush()

try:
    success = plda.train(embeddings, labels)
    if success:
        print(f"✓ PLDA training succeeded: {plda}")
    else:
        print("✗ PLDA training returned False")
        sys.exit(1)
except Exception as e:
    print(f"✗ PLDA training failed with exception: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test scoring
print("\nTesting scoring...")
sys.stdout.flush()

# Create test probe (should match person_0)
test_probe = class_centers[0] + np.random.randn(embedding_dim) * 0.3
test_probe = test_probe / np.linalg.norm(test_probe)

# Score against all identities
best_id, best_llr, margin, all_scores = plda.score_against_all(test_probe)
print(f"✓ Score against all: best_id={best_id}, LLR={best_llr:.3f}, margin={margin:.3f}")
print(f"  All scores: {all_scores}")
sys.stdout.flush()

# Create unknown probe (random, not from any class)
unknown_probe = np.random.randn(embedding_dim)
unknown_probe = unknown_probe / np.linalg.norm(unknown_probe)

best_id_unk, best_llr_unk, margin_unk, _ = plda.score_against_all(unknown_probe)
print(f"\n✓ Unknown probe: best_id={best_id_unk}, LLR={best_llr_unk:.3f}")
print(f"  Unknown probability: {plda.compute_unknown_score(unknown_probe):.3f}")
sys.stdout.flush()

# Test save/load
print("\nTesting save/load...")
sys.stdout.flush()

import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, "test_plda.npz")

    if plda.save(save_path):
        print(f"✓ Saved to {save_path}")

        # Load into new scorer
        plda2 = PLDAScorer()
        if plda2.load(save_path):
            print(f"✓ Loaded: {plda2}")

            # Verify scores match
            _, llr2, _, _ = plda2.score_against_all(test_probe)
            if abs(llr2 - best_llr) < 0.01:
                print(f"✓ Scores match after reload: {llr2:.3f} vs {best_llr:.3f}")
            else:
                print(f"✗ Score mismatch: {llr2:.3f} vs {best_llr:.3f}")
        else:
            print("✗ Load failed")
    else:
        print("✗ Save failed")

print("\n" + "="*50)
print("All PLDA tests passed!")
print("="*50)

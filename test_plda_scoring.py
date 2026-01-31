#!/usr/bin/env python3
"""Test PLDA scoring with real enrolled data."""

import sys
import numpy as np

print("="*60)
print("PLDA Likelihood Ratio Scoring Test")
print("="*60)

# Load the trained PLDA model
from watchbird.fusion.plda_scorer import PLDAScorer

plda = PLDAScorer()
if not plda.load("data/index/plda.npz"):
    print("Failed to load PLDA model")
    sys.exit(1)

print(f"\nLoaded: {plda}")
print(f"Identities: {list(plda.identity_models.keys())}")
print(f"Samples per identity: {plda.identity_counts}")

# Get centroids for testing
ira_centroid = plda.model.identity_centroids['ira']
mike_centroid = plda.model.identity_centroids['mike']

print(f"\nCentroid norms: ira={np.linalg.norm(ira_centroid):.3f}, mike={np.linalg.norm(mike_centroid):.3f}")
print(f"Cosine similarity between centroids: {np.dot(ira_centroid, mike_centroid) / (np.linalg.norm(ira_centroid) * np.linalg.norm(mike_centroid)):.3f}")

print("\n" + "="*60)
print("Testing same-identity scoring (should have high LLR)")
print("="*60)

# Score ira samples against ira identity
ira_embeddings = plda.model.identity_embeddings['ira']
print(f"\nScoring {len(ira_embeddings)} ira embeddings against 'ira' identity:")

ira_llrs = []
for i, emb in enumerate(ira_embeddings[:5]):  # First 5
    llr = plda.score(emb, 'ira')
    llr_mike = plda.score(emb, 'mike')
    ira_llrs.append(llr)
    print(f"  Sample {i}: LLR(ira)={llr:+.3f}, LLR(mike)={llr_mike:+.3f}, margin={llr - llr_mike:+.3f}")

print(f"\nMean LLR for ira vs ira: {np.mean(ira_llrs):.3f}")

# Score mike samples against mike identity
mike_embeddings = plda.model.identity_embeddings['mike']
print(f"\nScoring {len(mike_embeddings)} mike embeddings against 'mike' identity:")

mike_llrs = []
for i, emb in enumerate(mike_embeddings[:5]):  # First 5
    llr = plda.score(emb, 'mike')
    llr_ira = plda.score(emb, 'ira')
    mike_llrs.append(llr)
    print(f"  Sample {i}: LLR(mike)={llr:+.3f}, LLR(ira)={llr_ira:+.3f}, margin={llr - llr_ira:+.3f}")

print(f"\nMean LLR for mike vs mike: {np.mean(mike_llrs):.3f}")

print("\n" + "="*60)
print("Testing cross-identity scoring (should have low/negative LLR)")
print("="*60)

# Score ira samples against mike identity (should be negative)
print(f"\nScoring ira embeddings against 'mike' identity:")
cross_llrs = []
for i, emb in enumerate(ira_embeddings[:5]):
    llr = plda.score(emb, 'mike')
    cross_llrs.append(llr)
    print(f"  ira sample {i} vs mike: LLR={llr:+.3f}")

print(f"\nMean LLR for ira vs mike (should be negative): {np.mean(cross_llrs):.3f}")

print("\n" + "="*60)
print("Testing unknown rejection")
print("="*60)

# Create random embeddings (simulating unknown person)
np.random.seed(42)
for i in range(3):
    unknown = np.random.randn(512)
    unknown = unknown / np.linalg.norm(unknown)

    best_id, best_llr, margin, scores = plda.score_against_all(unknown)
    unknown_prob = plda.compute_unknown_score(unknown)

    print(f"\nRandom probe {i}:")
    print(f"  Best match: {best_id}, LLR={best_llr:+.3f}")
    print(f"  Unknown probability: {unknown_prob:.3f}")
    print(f"  All scores: {scores}")

print("\n" + "="*60)
print("Testing score_with_uncertainty")
print("="*60)

# Test uncertainty for a known identity
test_emb = ira_embeddings[10]
mean_llr, uncertainty = plda.score_with_uncertainty(test_emb, 'ira')
print(f"\nIra embedding vs ira (should be confident):")
print(f"  Mean LLR: {mean_llr:+.3f}")
print(f"  Uncertainty: {uncertainty:.3f}")

mean_llr, uncertainty = plda.score_with_uncertainty(test_emb, 'mike')
print(f"\nIra embedding vs mike (should be uncertain/negative):")
print(f"  Mean LLR: {mean_llr:+.3f}")
print(f"  Uncertainty: {uncertainty:.3f}")

print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"""
PLDA LLR Scoring Results:
- Same identity (ira vs ira): mean LLR = {np.mean(ira_llrs):.3f}
- Same identity (mike vs mike): mean LLR = {np.mean(mike_llrs):.3f}  
- Cross identity (ira vs mike): mean LLR = {np.mean(cross_llrs):.3f}

Key insight: 
- LLR > 0 means "same person" is more likely
- LLR < 0 means "different person" is more likely
- Use LLR threshold (~0.5) for unknown rejection
""")

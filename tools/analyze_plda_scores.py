#!/usr/bin/env python3
"""Analyze PLDA score distributions for threshold tuning using Leave-One-Sample-Out (LOSO)."""

import numpy as np
import sys
from pathlib import Path

from watchbird.fusion.plda_scorer import PLDAScorer

def analyze_scores():
    """Analyze score distributions for threshold calibration."""

    print("=" * 70)
    print("PLDA Score Distribution Analysis (LOSO)")
    print("=" * 70)

    # Load PLDA model
    plda = PLDAScorer()
    if not plda.load("data/index/plda.npz"):
        print("Failed to load PLDA model")
        return

    print(f"\nLoaded: {plda}")
    print(f"Identities: {list(plda.identity_models.keys())}")
    print(f"Samples per identity: {plda.identity_counts}")

    # Collect all scores
    same_id_scores = []      # Probe matches correct identity
    diff_id_scores = []      # Probe matches wrong identity
    same_id_margins = []     # Margin when matching correct identity
    diff_id_margins = []     # Margin when matching wrong identity

    identities = list(plda.model.identity_embeddings.keys())

    print(f"\n{'='*70}")
    print("Leave-One-Sample-Out Analysis")
    print("="*70)

    for person_id in identities:
        embeddings = plda.model.identity_embeddings[person_id]
        print(f"\n--- {person_id}: {len(embeddings)} samples ---")

        for i, probe in enumerate(embeddings):
            # Score against all identities
            best_id, best_llr, margin, all_scores = plda.score_against_all(probe)

            # Get score for correct identity
            correct_score = all_scores.get(person_id, float('-inf'))

            # Get best wrong identity score
            wrong_scores = {k: v for k, v in all_scores.items() if k != person_id}
            best_wrong_id = max(wrong_scores, key=wrong_scores.get) if wrong_scores else None
            best_wrong_score = wrong_scores.get(best_wrong_id, float('-inf')) if best_wrong_id else float('-inf')

            # Compute margin (correct - best wrong)
            actual_margin = correct_score - best_wrong_score

            # Classify
            if best_id == person_id:
                same_id_scores.append(correct_score)
                same_id_margins.append(actual_margin)
            else:
                diff_id_scores.append(correct_score)
                diff_id_margins.append(actual_margin)

            # Print sample results
            if i < 3 or i >= len(embeddings) - 2:  # First 3 and last 2
                status = "✓" if best_id == person_id else "✗"
                print(f"  Sample {i:2d}: {status} best={best_id}, LLR={best_llr:.3f}, "
                      f"margin={actual_margin:.3f}, scores={all_scores}")

    # Analyze random probes (unknowns)
    print(f"\n{'='*70}")
    print("Unknown Probe Analysis (random embeddings)")
    print("="*70)

    unknown_scores = []
    unknown_margins = []

    np.random.seed(42)
    for i in range(20):
        unknown = np.random.randn(512)
        unknown = unknown / np.linalg.norm(unknown)

        best_id, best_llr, margin, all_scores = plda.score_against_all(unknown)
        unknown_scores.append(best_llr)
        unknown_margins.append(margin)

        if i < 5:
            print(f"  Unknown {i}: best={best_id}, LLR={best_llr:.3f}, margin={margin:.3f}")

    # Statistics
    print(f"\n{'='*70}")
    print("Score Distribution Statistics")
    print("="*70)

    same_id_scores = np.array(same_id_scores)
    same_id_margins = np.array(same_id_margins)
    unknown_scores = np.array(unknown_scores)
    unknown_margins = np.array(unknown_margins)

    print(f"\nSame-identity (correct matches): n={len(same_id_scores)}")
    print(f"  LLR:    min={same_id_scores.min():.3f}, max={same_id_scores.max():.3f}, "
          f"mean={same_id_scores.mean():.3f}, std={same_id_scores.std():.3f}")
    print(f"  Margin: min={same_id_margins.min():.3f}, max={same_id_margins.max():.3f}, "
          f"mean={same_id_margins.mean():.3f}, std={same_id_margins.std():.3f}")

    if len(diff_id_scores) > 0:
        diff_id_scores = np.array(diff_id_scores)
        diff_id_margins = np.array(diff_id_margins)
        print(f"\nDiff-identity (mismatches): n={len(diff_id_scores)}")
        print(f"  LLR:    min={diff_id_scores.min():.3f}, max={diff_id_scores.max():.3f}, "
              f"mean={diff_id_scores.mean():.3f}")

    print(f"\nUnknown probes: n={len(unknown_scores)}")
    print(f"  LLR:    min={unknown_scores.min():.3f}, max={unknown_scores.max():.3f}, "
          f"mean={unknown_scores.mean():.3f}, std={unknown_scores.std():.3f}")
    print(f"  Margin: min={unknown_margins.min():.3f}, max={unknown_margins.max():.3f}, "
          f"mean={unknown_margins.mean():.3f}")

    # Recommended thresholds
    print(f"\n{'='*70}")
    print("Recommended Thresholds")
    print("="*70)

    # LLR threshold: Should separate knowns from unknowns
    # Use percentiles to find good separation point
    known_min = same_id_scores.min()
    unknown_max = unknown_scores.max()

    # Good threshold is between unknown max and known min
    # With some safety margin toward known min
    llr_threshold = (unknown_max + known_min) / 2

    # If known min is lower than unknown max, there's overlap
    if known_min < unknown_max:
        # Use a more conservative threshold
        llr_threshold = np.percentile(same_id_scores, 5)  # 5th percentile of knowns
        print(f"  ⚠ Warning: Score overlap between known and unknown!")

    # Margin threshold: Should ensure confident discrimination
    margin_threshold = max(0.1, same_id_margins.min() * 0.8)  # 80% of minimum margin

    # For t_accept (calibrated score), use sigmoid of LLR
    calibrated_min = plda.calibrate_score(known_min)
    calibrated_threshold = plda.calibrate_score(llr_threshold)

    print(f"\nPLDA thresholds (config.yaml -> plda section):")
    print(f"  llr_threshold: {llr_threshold:.2f}  (separates known from unknown)")
    print(f"  margin_threshold: {margin_threshold:.2f}  (min margin for confidence)")

    print(f"\nDecision thresholds (config.yaml -> thresholds section):")
    print(f"  t_accept: {calibrated_threshold:.2f}  (calibrated score threshold)")
    print(f"  t_margin: {margin_threshold:.2f}  (min margin between candidates)")

    # Safety analysis
    print(f"\n{'='*70}")
    print("Safety Analysis")
    print("="*70)

    # What percentage of known samples would be accepted?
    accepted_known = (same_id_scores >= llr_threshold).mean() * 100
    print(f"  Known samples accepted (LLR >= {llr_threshold:.2f}): {accepted_known:.1f}%")

    # What percentage of unknowns would be rejected?
    rejected_unknown = (unknown_scores < llr_threshold).mean() * 100
    print(f"  Unknown samples rejected (LLR < {llr_threshold:.2f}): {rejected_unknown:.1f}%")

    # What percentage pass margin check?
    margin_pass = (same_id_margins >= margin_threshold).mean() * 100
    print(f"  Known with sufficient margin (>= {margin_threshold:.2f}): {margin_pass:.1f}%")

    return {
        'llr_threshold': llr_threshold,
        'margin_threshold': margin_threshold,
        't_accept': calibrated_threshold,
    }

if __name__ == "__main__":
    analyze_scores()

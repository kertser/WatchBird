"""Test PLDA calibration behavior."""

import numpy as np

def old_calibrate(llr: float) -> float:
    """Old calibration without capping."""
    k = 1.5
    clipped_llr = np.clip(-k * llr, -500, 500)
    sigmoid = 1.0 / (1.0 + np.exp(clipped_llr))
    return float(sigmoid)

def new_calibrate(llr: float) -> float:
    """New calibration with 0.98 cap."""
    k = 1.5
    clipped_llr = np.clip(-k * llr, -500, 500)
    sigmoid = 1.0 / (1.0 + np.exp(clipped_llr))
    max_confidence = 0.98
    calibrated = min(sigmoid, max_confidence)
    return float(calibrated)

if __name__ == "__main__":
    print("PLDA Calibration Test")
    print("=" * 60)
    print(f"{'LLR':<10} {'Old Score':<15} {'New Score':<15} {'Difference':<15}")
    print("-" * 60)

    test_llrs = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    for llr in test_llrs:
        old_score = old_calibrate(llr)
        new_score = new_calibrate(llr)
        diff = old_score - new_score
        print(f"{llr:<10.1f} {old_score:<15.6f} {new_score:<15.6f} {diff:<15.6f}")

    print("\n" + "=" * 60)
    print("Key observations:")
    print("- LLR >= 5.0 now caps at 0.98 instead of approaching 1.0")
    print("- This prevents unrealistic 100% certainty in face recognition")
    print("- Lower LLR values (0-2) remain unchanged for normal operation")

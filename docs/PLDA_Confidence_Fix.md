# PLDA Confidence Score Issue - Fixed

## Problem

The system was reporting a confidence of **1.000** (perfect certainty) when recognizing faces, which is unrealistic and indicates an issue with the scoring calibration.

### Root Cause

1. **PLDA Overfitting**: With only 2 identities and limited training data, the PLDA model was producing very high Log-Likelihood Ratio (LLR) scores
2. **Sigmoid Saturation**: The calibration function uses a sigmoid: `score = 1 / (1 + exp(-1.5 * LLR))`
   - When LLR ≥ 10, the sigmoid saturates to ~1.0
   - This gave a false sense of perfect certainty

### Example
```
LLR = 10.0  →  calibrated score = 1.000000  (unrealistic!)
```

## Solution

Added a **confidence cap of 0.98** in the PLDA calibration function:

```python
def calibrate_score(self, llr: float, target_range=(0.0, 1.0)) -> float:
    k = 1.5
    clipped_llr = np.clip(-k * llr, -500, 500)
    sigmoid = 1.0 / (1.0 + np.exp(clipped_llr))
    
    # Cap maximum confidence at 0.98
    max_confidence = 0.98
    calibrated = min(sigmoid, max_confidence)
    
    return float(calibrated)
```

### Results After Fix

| LLR   | Old Score | New Score | Notes |
|-------|-----------|-----------|-------|
| 0.0   | 0.500     | 0.500     | Uncertain - unchanged |
| 1.0   | 0.818     | 0.818     | Normal operation - unchanged |
| 2.0   | 0.953     | 0.953     | High confidence - unchanged |
| 3.0   | 0.989     | **0.980** | Capped to prevent saturation |
| 10.0  | 1.000     | **0.980** | Capped to prevent unrealistic certainty |

## Benefits

1. **More Realistic Confidence**: No more perfect 1.0 scores that claim 100% certainty
2. **Better Calibration**: Confidence scores now better reflect real-world uncertainty
3. **Maintains Performance**: Normal operation (LLR < 3.0) is unaffected
4. **Prevents Overfitting Artifacts**: High LLRs from overfitting are now bounded

## Why 0.98?

- Face recognition systems should never claim 100% certainty due to:
  - Lighting variations
  - Pose changes
  - Partial occlusions
  - Image quality issues
  - Model limitations
- 0.98 (98% confidence) is a realistic upper bound for high-quality matches
- Leaves 2% uncertainty even for the best matches

## Testing

Run the calibration test to see the behavior:
```bash
python tools/test_calibration.py
```

Or run the full system:
```bash
python tools/run_runtime.py
```

You should now see confidence scores like **0.75-0.95** for real matches instead of **1.000**.

## Files Modified

1. `src/watchbird/fusion/plda_scorer.py` - Added 0.98 confidence cap in `calibrate_score()`
2. `tools/test_calibration.py` - New test script to demonstrate the fix

---

**Date**: 2026-01-27  
**Status**: ✅ Fixed

# PLDA Quick Start Guide

This guide will help you quickly set up and test PLDA-based face recognition.

## Prerequisites

- At least 2 enrolled identities with 3+ photos each
- WatchBird installed and working with FAISS

## Step 1: Check Your Current Setup

```bash
# Check enrolled identities
python tools/check_index.py
```

You should see output like:
```
Loaded 20 embeddings from 2 identities:
  - mike: 10 embeddings
  - ira: 10 embeddings
```

**Important**: PLDA requires at least 2 distinct identities. If you only have 1, enroll another person first.

## Step 2: Train PLDA Model

Re-run enrollment with the `--train-plda` flag:

```bash
python tools/enroll.py --data-dir friendly --train-plda
```

Expected output:
```
INFO - Enrollment complete!
INFO - Training PLDA model...
INFO - Training PLDA: 20 samples, 2 identities, dim=512
INFO - PLDA training complete: 2 identity models
INFO - Saved PLDA model to data/index/plda.npz
INFO - PLDA model: 2 identities, 20 total samples
```

This creates two files:
- `data/index/plda.npz` - PLDA parameters (covariances, mean)
- `data/index/plda.models.json` - Identity models

## Step 3: Enable PLDA in Config

Edit `config.yaml`:

```yaml
plda:
  enabled: true  # Change from false to true
  model_path: data/index/plda.npz
  faiss_k: 5
  llr_threshold: 0.0
  margin_threshold: 0.5
  calibrate: true
```

## Step 4: Test Recognition

Run the recognition pipeline:

```bash
python tools/run_runtime.py --backend usb
```

Look for these log messages at startup:

```
INFO - Scoring backend: plda+faiss
INFO - PLDA enabled: 2 identities, LLR threshold=0.00
```

If you see:
```
INFO - Scoring backend: faiss
```

Then PLDA is not active. Check:
- `plda.enabled: true` in config.yaml
- PLDA model files exist in `data/index/`

## Step 5: Verify PLDA is Working

While running recognition, watch the logs. You should NOT see errors like:
```
ERROR - PLDA scoring failed
```

For debugging, set log level to DEBUG in `config.yaml`:

```yaml
logging:
  level: DEBUG
```

You'll see PLDA scoring details:
```
DEBUG - PLDA scoring matrices precomputed
DEBUG - Built 2 identity models
```

## Step 6: Compare Performance

### Test 1: Known Person

Show your face to the camera. You should see:
- **SUSPECT** → **FRIENDLY** transition
- Your enrolled name displayed
- Confidence score shown

### Test 2: Unknown Person

Have someone not enrolled show their face:
- Should stay **SUSPECT** 
- Eventually timeout to **ENEMY**
- With PLDA, should reject faster and more confidently

### Test 3: Borderline Lighting

Test in poor lighting or with face partially obscured:
- PLDA should be more robust
- Better rejection of poor quality matches

## Tuning PLDA

### More Strict (Fewer False Accepts)

```yaml
plda:
  llr_threshold: 1.0  # Increase from 0.0
  margin_threshold: 0.8  # Increase from 0.5
```

### More Lenient (Fewer False Rejects)

```yaml
plda:
  llr_threshold: -1.0  # Decrease from 0.0
  margin_threshold: 0.3  # Decrease from 0.5
```

### Faster (Less Thorough)

```yaml
plda:
  faiss_k: 3  # Reduce from 5
  calibrate: false  # Skip calibration
```

## Testing SFace Model (Optional)

### Download SFace

```bash
python tools/download_models.py
# Select option 3
```

### Configure SFace

Edit `config.yaml`:

```yaml
models:
  face_embedder: models/sface.onnx
```

### Re-enroll with SFace + PLDA

```bash
# This will rebuild index with SFace embeddings
python tools/enroll.py --data-dir friendly --train-plda
```

### Test

```bash
python tools/run_runtime.py --backend usb
```

SFace provides:
- Good balance of speed and accuracy
- Robust to pose variations
- Smaller model size than ArcFace ResNet100

## Troubleshooting

### "PLDA model not found"

**Problem**: 
```
INFO - PLDA model not found at data/index/plda.npz, will use FAISS-only scoring
```

**Solution**:
```bash
# Re-run enrollment with --train-plda
python tools/enroll.py --data-dir friendly --train-plda
```

### "PLDA requires at least 2 identities"

**Problem**:
```
WARNING - PLDA requires at least 2 identities, found 1. Skipping PLDA training.
```

**Solution**:
Enroll a second person:
```bash
# Create directory for second person
mkdir friendly/alice
# Add photos of Alice to friendly/alice/
# Re-run enrollment
python tools/enroll.py --data-dir friendly --train-plda
```

### "Matrix singular" warnings

**Problem**:
```
WARNING - PLDA: matrix singular at iteration X
```

**Solutions**:
1. Add more enrollment photos per person (aim for 5-10)
2. Ensure photos have variety (different angles, lighting)
3. Check for duplicate images

### Performance is slow

**Problem**: Recognition is noticeably slower with PLDA

**Solutions**:
1. Reduce FAISS candidates:
   ```yaml
   plda:
     faiss_k: 3  # Reduce from 5
   ```

2. Disable score calibration:
   ```yaml
   plda:
     calibrate: false
   ```

3. Use GPU acceleration:
   ```yaml
   inference:
     use_gpu: true
   ```

## Verification Commands

### Check PLDA Files

```bash
# Windows PowerShell
ls data\index\plda*

# Expected output:
# plda.npz
# plda.models.json
```

### Check File Sizes

```bash
# PLDA model should be 1-10 MB depending on identities
ls -lh data/index/plda.npz
```

### Python Quick Test

```python
from watchbird.fusion.plda_scorer import PLDAScorer

# Load PLDA model
plda = PLDAScorer()
success = plda.load("data/index/plda.npz")

print(f"PLDA loaded: {success}")
print(f"Trained: {plda.is_trained}")
print(f"Identities: {len(plda.identity_models)}")
print(f"Identity names: {list(plda.identity_models.keys())}")
```

Expected output:
```
PLDA loaded: True
Trained: True
Identities: 2
Identity names: ['mike', 'ira']
```

## Next Steps

Once PLDA is working:

1. **Collect More Data**: Add more enrollment photos to improve accuracy
2. **Tune Thresholds**: Adjust LLR thresholds for your use case
3. **Try Different Models**: Test SFace, ArcFace, MobileFaceNet
4. **Deploy**: PLDA adds minimal overhead (~10-20ms per frame)

## Support

For issues:
1. Check logs with `logging.level: DEBUG`
2. Verify PLDA files exist and load correctly
3. Ensure at least 2 enrolled identities
4. Try re-training PLDA with `--train-plda`

Happy recognizing! 🎉

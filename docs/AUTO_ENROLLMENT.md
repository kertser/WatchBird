# Auto-Enrollment Guide

## Overview

Auto-enrollment is an intelligent enrollment system that automatically captures photos until the person can be recognized with high confidence. Instead of capturing a fixed number of photos, it adapts to each person and stops when recognition quality is sufficient.

## Why Auto-Enrollment?

### Traditional Enrollment Problems

**Fixed photo count approach:**
- Some people need 5 photos, others need 25
- No way to know if photos are good enough
- May capture too few (poor recognition) or too many (waste time)
- No validation during capture

**Manual quality checking:**
- Time-consuming to verify each photo
- Subjective quality assessment
- No way to test recognition until after enrollment

### Auto-Enrollment Solution

**Confidence-driven approach:**
- ✅ Captures until recognition confidence reaches target
- ✅ Real-time testing and feedback
- ✅ Adapts to each person (some faces are easier to recognize)
- ✅ Validates against existing enrolled people
- ✅ Warns if being confused with another person
- ✅ Stops automatically when done

## How It Works

### Process Flow

```
1. Initialize camera and face detection
   ↓
2. Capture frame from camera
   ↓
3. Detect face and check quality
   ↓
4. If quality good → Save photo
   ↓
5. Every N photos → Test recognition
   ↓
6. Calculate confidence: How well can we recognize this person?
   ↓
7. If confidence ≥ target → DONE ✅
   If confidence < target → Continue capturing
   ↓
8. Auto-run enrollment (if --auto-enroll flag)
```

### Confidence Testing

After every 3 photos (configurable), the system:

1. **Builds temporary index** with captured photos
2. **Tests each photo** against the temporary index
3. **Calculates average similarity** (how consistent are the embeddings?)
4. **Checks against existing people** (are we being confused with someone else?)
5. **Reports confidence** (0-1 scale)

**Confidence interpretation:**
- `0.90+`: Excellent recognition (very distinctive photos)
- `0.85-0.90`: Good recognition (recommended target) ✅
- `0.70-0.85`: Acceptable recognition (may need more variety)
- `< 0.70`: Poor recognition (need more/better photos)

## Usage

### Basic Usage

```bash
# Auto-enroll with default settings (85% confidence target)
python tools/auto_enroll.py --person mike --auto-enroll
```

### Custom Confidence Target

```bash
# High security - 90% confidence (more photos needed)
python tools/auto_enroll.py --person alice --target-confidence 0.90 --auto-enroll

# Balanced - 85% confidence (default, recommended)
python tools/auto_enroll.py --person bob --target-confidence 0.85 --auto-enroll

# Fast enrollment - 70% confidence (fewer photos, less reliable)
python tools/auto_enroll.py --person charlie --target-confidence 0.70 --auto-enroll
```

### Advanced Options

```bash
python tools/auto_enroll.py \
  --person mike \
  --target-confidence 0.85 \
  --min-photos 5 \              # Test only after 5+ photos
  --max-photos 30 \             # Stop at 30 even if target not reached
  --test-interval 3 \           # Test every 3 photos
  --backend usb \               # Camera backend
  --device-id 0 \               # Camera device
  --auto-enroll                 # Auto-run enrollment when done
```

### Without Auto-Enrollment

```bash
# Capture photos only, manual enrollment later
python tools/auto_enroll.py --person mike --target-confidence 0.85

# Then manually enroll
python tools/enroll.py --data-dir friendly --config config.yaml
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--person` | Required | Person ID (folder name) |
| `--target-confidence` | 0.85 | Target recognition confidence (0-1) |
| `--min-photos` | 5 | Minimum photos before testing |
| `--max-photos` | 30 | Maximum photos to capture |
| `--test-interval` | 3 | Test recognition every N photos |
| `--backend` | usb | Camera backend (usb, picamera2) |
| `--device-id` | 0 | Camera device ID |
| `--config` | config.yaml | Configuration file |
| `--auto-enroll` | False | Auto-run enrollment after capture |

## Best Practices

### During Capture

**Do:**
- ✅ Position face clearly in camera view
- ✅ Move head slowly in different directions
- ✅ Vary expressions (neutral, smiling, serious)
- ✅ Maintain good lighting
- ✅ Stay at similar distance from camera
- ✅ Let system auto-capture (don't rush)

**Don't:**
- ❌ Move too quickly (blurry photos)
- ❌ Cover face with hands/objects
- ❌ Wear sunglasses or hats
- ❌ Look away from camera completely
- ❌ Press 'q' before minimum photos captured

### Confidence Targets

**Choose target based on use case:**

**High Security (0.90+):**
- Critical identification scenarios
- Small number of people (2-5)
- Acceptable to have some false rejections
- Examples: Access control, financial transactions

**Balanced (0.85):** ✅ **Recommended**
- General face recognition
- Medium number of people (5-20)
- Good balance of accuracy and usability
- Examples: Office access, home security

**Fast Enrollment (0.70-0.80):**
- Quick setup needed
- Large number of people (20+)
- Can tolerate some false positives
- Examples: Event check-in, casual applications

### Photo Variety

For best results, capture photos with variety in:

1. **Head angles:**
   - Front view (straight ahead)
   - Slight left turn (15-30°)
   - Slight right turn (15-30°)
   - Slight up angle (looking slightly up)
   - Slight down angle (looking slightly down)

2. **Expressions:**
   - Neutral (no expression)
   - Smiling
   - Talking
   - Serious

3. **Lighting:**
   - Normal lighting
   - Slightly darker/brighter (if expected in use)

## Understanding Output

### During Capture

```
INFO - AUTO-ENROLLMENT SESSION: mike
INFO - Target confidence: 0.85
INFO - Min photos: 5, Max photos: 30
INFO - 
INFO -   ✓ Captured photo 1/30 (quality=0.78)
INFO -   ✓ Captured photo 2/30 (quality=0.82)
INFO -   ✓ Captured photo 3/30 (quality=0.75)
INFO - 
INFO -   Testing recognition confidence...
INFO -   Current confidence: 0.62 (target: 0.85)
INFO - 
INFO -   ✓ Captured photo 4/30 (quality=0.88)
...
INFO -   Testing recognition confidence...
INFO -   Current confidence: 0.87 (target: 0.85)
INFO - 
INFO -   🎉 Target confidence reached: 0.87
INFO -   ✅ Auto-enrollment complete with 15 photos!
```

### Final Results

```
================================================================================
FINAL RESULTS
================================================================================
Photos captured: 15
Final confidence: 0.87
✅ SUCCESS - Target confidence reached!
================================================================================
```

### Warnings

```
⚠️  Being confused with 'alice' - need more distinctive photos!
```
This means your photos are too similar to someone already enrolled (alice). Capture more unique angles/expressions.

```
⚠️  Target confidence not reached (need 0.85)
   Consider capturing more photos or lowering target
```
Reached max photos but confidence still low. Either capture more manually or reduce target.

## Troubleshooting

### Issue: Confidence stays low (< 0.70)

**Causes:**
- Poor lighting
- Blurry photos (moving too fast)
- Not enough variety in angles/expressions
- Low-quality camera

**Solutions:**
1. Improve lighting
2. Move slower during capture
3. Vary head angles more
4. Use better camera if available
5. Lower minimum quality threshold in config.yaml

### Issue: Confidence increases slowly

**Causes:**
- Face is hard to distinguish (generic features)
- Similar to another enrolled person
- Inconsistent embeddings

**Solutions:**
1. Capture more photos (increase --max-photos)
2. Ensure more variety in angles
3. Lower target confidence to 0.75-0.80
4. Check if being confused with another person

### Issue: "Being confused with another person"

**Causes:**
- Two people look very similar
- Wrong person in front of camera
- Photos labeled incorrectly

**Solutions:**
1. **Verify correct person** is being photographed
2. Delete other person's photos and re-enroll
3. Capture very distinctive angles/expressions
4. Increase confidence target to 0.90+

### Issue: Takes too long to reach target

**Causes:**
- Target too high (0.95+)
- Test interval too large
- Face difficult to recognize

**Solutions:**
1. Lower target to 0.85 (recommended)
2. Reduce test interval to 2
3. Increase capture cooldown (faster capture rate)

### Issue: "Not enough photos captured"

**Causes:**
- Pressed 'q' too early
- No face detected in frames
- Quality too low (all photos rejected)

**Solutions:**
1. Don't quit before min_photos reached
2. Position face clearly in view
3. Lower quality threshold in config
4. Improve lighting

## Integration with Existing Workflow

### Option A: Full Auto-Enrollment

```bash
# One command - capture and enroll
python tools/auto_enroll.py --person mike --auto-enroll
```

### Option B: Separate Steps

```bash
# 1. Auto-capture with confidence testing
python tools/auto_enroll.py --person mike --target-confidence 0.85

# 2. Review photos
ls friendly/mike/

# 3. Manual enrollment
python tools/enroll.py --data-dir friendly --config config.yaml
```

### Option C: Mix with Manual Photos

```bash
# 1. Add some existing photos to folder
cp /path/to/photos/* friendly/mike/

# 2. Auto-capture additional photos
python tools/auto_enroll.py --person mike --target-confidence 0.85

# 3. Enroll all photos together
python tools/enroll.py --data-dir friendly --config config.yaml
```

## Comparison with Other Methods

| Feature | Auto-Enroll | Manual Count | Pre-existing Photos |
|---------|-------------|--------------|---------------------|
| Adaptive to person | ✅ Yes | ❌ No | ❌ No |
| Quality validation | ✅ Real-time | ⚠️ Manual | ❌ None |
| Confidence testing | ✅ Automatic | ❌ No | ❌ No |
| Stops when ready | ✅ Yes | ❌ Fixed count | N/A |
| Ease of use | ✅ Easy | ✅ Easy | ⚠️ Manual prep |
| Speed | ⚠️ Adaptive | ✅ Fast | ✅ Instant |
| Best for | New enrollment | Quick setup | Existing photos |

## Advanced Features

### Custom Quality Thresholds

Edit `config.yaml` before running:

```yaml
quality:
  min_face_quality: 0.5  # Higher = stricter (fewer photos captured)
  min_bbox_size: 60      # Larger = face must be closer to camera
```

### Capture Cooldown

Currently hardcoded to 0.5 seconds between captures. To change, edit `auto_enroll.py`:

```python
capture_cooldown = 0.5  # Increase for slower capture, decrease for faster
```

### Test Interval

```bash
# Test every photo (slower but more responsive)
python tools/auto_enroll.py --person mike --test-interval 1

# Test every 5 photos (faster but less responsive)
python tools/auto_enroll.py --person mike --test-interval 5
```

## Technical Details

### Confidence Calculation

1. Build temporary FAISS index with N captured photos
2. For each photo embedding:
   - Search index for top K similar embeddings
   - Exclude self-match
   - Average similarity to other photos of same person
3. Average all photo confidences → final confidence score

### Why This Works

- **Intra-class similarity**: Photos of same person should be similar
- **Consistency check**: If embeddings vary widely, recognition will be unreliable
- **Early stopping**: Detects when enough consistent data is collected
- **Adaptive**: Naturally captures more photos for harder-to-recognize faces

### Limitations

- Requires at least 5 photos for meaningful testing
- Confidence is relative to captured photos, not absolute
- May not detect all edge cases (extreme angles, lighting)
- First-person enrollment has no comparison baseline

## Future Enhancements

Possible improvements:

1. **Active learning**: Suggest specific poses/expressions to capture
2. **Quality heatmap**: Show which angles still need coverage
3. **Live confidence display**: Real-time confidence meter
4. **Multi-person comparison**: Test against all enrolled people
5. **Automatic re-enrollment**: Periodically add photos to improve recognition

## Conclusion

Auto-enrollment with confidence testing provides an intelligent, adaptive way to enroll people with high-quality recognition from the start. It eliminates guesswork about how many photos are needed and ensures reliable recognition before deployment.

**Recommended workflow:**
1. Use auto-enrollment for all new people
2. Target 0.85 confidence (good balance)
3. Let system capture 10-20 photos adaptively
4. Use --auto-enroll flag for one-step operation
5. Test runtime immediately after enrollment

This ensures production-ready recognition from day one! 🎉

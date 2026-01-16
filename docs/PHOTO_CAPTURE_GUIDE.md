# Interactive Photo Capture & Enrollment Guide

## Overview

The `capture_and_enroll.py` tool provides an interactive way to:
1. **Capture photos** using your camera (USB webcam, Raspberry Pi camera, or video file)
2. **Automatically select best quality photos** from what you capture
3. **Handle multiple faces** by selecting the largest face
4. **Save only the top 10 photos** (or custom number)
5. **Optionally auto-enroll** the captured photos

## Why Use This Tool?

### Problems with Manual Photo Collection:
- ❌ Photos may have multiple people (wrong face enrolled)
- ❌ Photos may have no faces detected
- ❌ Photos may be low quality (blurry, poor lighting)
- ❌ Too many or too few photos per person
- ❌ Manual quality checking is time-consuming

### Benefits of Interactive Capture:
- ✅ Real-time face detection feedback
- ✅ Automatically handles multiple faces (uses largest)
- ✅ Quality scoring and automatic selection of best photos
- ✅ Consistent number of photos per person
- ✅ Immediate visual feedback during capture
- ✅ One-step capture and enrollment

## Quick Start

### Capture 10 Photos and Auto-Enroll
```bash
python tools/capture_and_enroll.py --person mike --count 10 --auto-enroll
```

### Capture Only (Manual Enrollment Later)
```bash
python tools/capture_and_enroll.py --person alice --count 15
```

## Full Usage

```bash
python tools/capture_and_enroll.py \
  --person NAME \           # Required: person's name
  --count 10 \              # Number of photos to capture (default: 10)
  --backend usb \           # Camera backend: usb, picamera2, video_file
  --device-id 0 \           # USB camera device ID (default: 0)
  --output-dir friendly \   # Where to save photos (default: friendly)
  --config config.yaml \    # Config file (default: config.yaml)
  --auto-enroll             # Auto-run enrollment after capture
```

## Interactive Controls

During photo capture session:

| Key | Action |
|-----|--------|
| **SPACE** | Capture current frame |
| **Q** | Finish early (with photos captured so far) |
| **ESC** | Cancel session (discard all photos) |

## On-Screen Display

### Status Information
- **Captured: X/10** - Number of photos captured vs target
- **Ready to capture** - Single face detected (green box)
- **N faces - will use largest** - Multiple faces detected (yellow boxes)
- **No face detected** - Cannot capture (red text)

### Visual Indicators
- **Green box** - Single face detected or largest face (will be saved)
- **Gray boxes** - Other faces (will be ignored)
- **CAPTURED!** - Flash when photo is saved

## How It Works

### 1. Face Detection
Each frame is analyzed for faces using the YuNet detector.

### 2. Face Selection
- **1 face:** Use that face ✅
- **Multiple faces:** Use the **largest face** (by bounding box area) ⭐
- **No faces:** Cannot capture ❌

### 3. Quality Scoring
Each captured photo is scored based on:
- Face confidence score
- Face size
- Image sharpness (blur detection)
- Face bounding box quality

### 4. Best Photo Selection
After capture, photos are:
1. Sorted by quality score (highest first)
2. Top N photos selected (default: 10)
3. Saved with quality score in filename

### 5. Filename Format
```
{person}_{timestamp}_{index:02d}_q{quality:.2f}.jpg
```

Example:
```
mike_1768602000_01_q0.85.jpg  # Best quality
mike_1768602000_02_q0.82.jpg
mike_1768602000_10_q0.65.jpg  # 10th best
```

## Best Practices

### During Capture

1. **Face Position Variety**
   - Capture 1: Face straight ahead
   - Capture 2: Face slightly to left
   - Capture 3: Face slightly to right
   - Capture 4: Face slightly up
   - Capture 5: Face slightly down
   - Capture 6-10: Mix of angles and expressions

2. **Expression Variety**
   - 50% neutral expression
   - 30% smiling
   - 20% serious/other

3. **Lighting**
   - Ensure face is well-lit
   - Avoid backlighting (bright window behind you)
   - Front lighting or side lighting is best

4. **Distance**
   - Stay at similar distance from camera
   - Face should fill 20-50% of frame
   - Not too close (distortion) or too far (low resolution)

5. **Only One Person**
   - **BEST:** Only target person in frame
   - **OK:** Multiple people (tool will use largest face)
   - **RISKY:** Largest face might not be target person!

### Photo Quality

Good quality photos have:
- ✅ Face clearly visible
- ✅ Good lighting
- ✅ Sharp focus (not blurry)
- ✅ Face size >80 pixels
- ✅ Confidence score >0.7

## Examples

### Example 1: Enroll New Person
```bash
# Mike is new - capture 15 photos and auto-enroll
python tools/capture_and_enroll.py --person mike --count 15 --auto-enroll

# Output:
# Captured: 15/15
# Saving 15 best photos...
# ✓ Saved: mike_1768602000_01_q0.89.jpg
# ...
# ✓ Saved: mike_1768602000_15_q0.67.jpg
# Running enrollment...
# ✅ Enrollment complete!
```

### Example 2: Add More Photos to Existing Person
```bash
# Alice already has 5 photos, add 10 more
python tools/capture_and_enroll.py --person alice --count 10

# Then re-enroll everyone:
python tools/enroll.py --data-dir friendly --config config.yaml
```

### Example 3: Capture with Raspberry Pi Camera
```bash
# On Raspberry Pi with CSI camera
python tools/capture_and_enroll.py --person bob --backend picamera2 --count 12 --auto-enroll
```

### Example 4: Capture with External USB Camera
```bash
# Using second USB camera
python tools/capture_and_enroll.py --person charlie --backend usb --device-id 1 --count 10 --auto-enroll
```

### Example 5: Test with Video File
```bash
# Practice using a video file
python tools/capture_and_enroll.py --person test --backend video_file --video test.mp4 --count 10
```

## Troubleshooting

### "No face detected" Every Frame
**Causes:**
- Camera not positioned correctly
- Too far from camera
- Poor lighting
- Face detector threshold too high

**Solutions:**
1. Move closer to camera
2. Improve lighting
3. Lower face detection threshold in config.yaml:
   ```yaml
   detection:
     face_conf_threshold: 0.4  # Lower from 0.5
   ```

### Wrong Person Saved (Multiple Faces)
**Problem:** Photos have multiple people, tool selects largest face which isn't the target person.

**Solutions:**
1. **Best:** Ensure only target person in frame
2. **OK:** Position target person closer to camera (make them largest)
3. **Workaround:** Manually delete bad photos from `friendly/{person}/` folder

### Photos Are Blurry
**Causes:**
- Moving during capture
- Camera auto-focus not working
- Low light

**Solutions:**
1. Stay still for 1 second before pressing SPACE
2. Ensure good lighting
3. Use camera with better auto-focus

### Tool Crashes or Freezes
**Causes:**
- Camera not available
- OpenCV display issues
- Insufficient system resources

**Solutions:**
1. Check camera is connected and not in use:
   ```bash
   # Test camera
   python test_usb_camera.py
   ```
2. Try different backend or device ID
3. Close other camera applications

### "Failed to open camera"
**Causes:**
- Camera already in use
- Wrong device ID
- Picamera2 not available (Windows)

**Solutions:**
1. USB camera: Try `--device-id 1` or `--device-id 2`
2. Check camera is not in use by another app
3. Windows: Use `--backend usb` (not picamera2)
4. Linux/RPi: Install picamera2 if needed

## Workflow Comparison

### Old Workflow (Manual)
1. Take photos with phone/camera ⏱️ 5 min
2. Transfer to computer ⏱️ 2 min
3. Organize into folders ⏱️ 2 min
4. Check photo quality manually ⏱️ 5 min
5. Remove bad photos ⏱️ 3 min
6. Run enrollment ⏱️ 1 min
**Total:** ~18 minutes per person

### New Workflow (Interactive Tool)
1. Run capture tool ⏱️ 2 min
   - Take 10-15 photos
   - Auto-selects best quality
   - Auto-enrolls
**Total:** ~2 minutes per person ⚡

## Advanced Usage

### Capture Many Photos, Save Best 10
```bash
# Capture 30 photos, save only top 10 by quality
python tools/capture_and_enroll.py --person david --count 10

# During session, capture 30+ photos by pressing SPACE many times
# Tool will automatically select best 10
```

### Custom Output Directory
```bash
# Save to different directory
python tools/capture_and_enroll.py --person test --output-dir test_enrollment --count 5
```

### No Auto-Enrollment (Manual Later)
```bash
# Capture photos only
python tools/capture_and_enroll.py --person emily --count 12

# Enroll later (after capturing multiple people)
python tools/enroll.py --data-dir friendly --config config.yaml
```

## Integration with Enrollment Flow

### Recommended Flow for Multiple People

```bash
# 1. Capture photos for each person
python tools/capture_and_enroll.py --person alice --count 10
python tools/capture_and_enroll.py --person bob --count 10  
python tools/capture_and_enroll.py --person charlie --count 10

# 2. Check photo quality
python tools/check_photos.py --data-dir friendly

# 3. Enroll everyone
python tools/enroll.py --data-dir friendly --config config.yaml

# 4. Test recognition
python tools/run_runtime.py --backend usb --config config.yaml
```

## Files Created

After running capture for "mike":
```
friendly/mike/
├── mike_1768602000_01_q0.89.jpg  # Best quality
├── mike_1768602000_02_q0.87.jpg
├── mike_1768602000_03_q0.85.jpg
├── mike_1768602000_04_q0.83.jpg
├── mike_1768602000_05_q0.81.jpg
├── mike_1768602000_06_q0.79.jpg
├── mike_1768602000_07_q0.76.jpg
├── mike_1768602000_08_q0.74.jpg
├── mike_1768602000_09_q0.71.jpg
└── mike_1768602000_10_q0.68.jpg  # 10th best
```

## Summary

**The interactive capture tool solves the main enrollment problems:**
- ✅ Ensures all photos have faces
- ✅ Handles multiple faces automatically
- ✅ Selects best quality photos
- ✅ Consistent photo count per person
- ✅ Real-time visual feedback
- ✅ One command for capture + enroll

**Use this tool for all new enrollments to save time and ensure quality!** 🎯

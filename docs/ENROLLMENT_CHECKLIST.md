# ✅ Enrollment Verification Checklist

Before running runtime, ALWAYS verify your enrollment:

## Quick Check

```powershell
# Check who's enrolled
Get-Content "data\index\meta.jsonl" | ConvertFrom-Json | Group-Object person_id | Select-Object Name,Count
```

**Expected output:**
```
Name   Count
----   -----
ira    10   
mike   14   
```

## Detailed Check

```powershell
# See all enrollment details
Get-Content "data\index\meta.jsonl" | ConvertFrom-Json | Select-Object person_id, source, quality | Format-Table
```

## Common Issues

### Issue 1: Person Missing from Enrollment
**Symptom:** Low scores (0.1-0.3) and wrong person detected  
**Example:** Mike appears, but system shows "ira" with score 0.21  
**Fix:** Re-run enrollment to include ALL people

### Issue 2: Too Few Photos Per Person
**Symptom:** Inconsistent recognition, high false negative rate  
**Minimum:** 5 photos per person  
**Recommended:** 10-15 photos per person  
**Best:** 20+ photos with varied angles, lighting, expressions

### Issue 3: Multiple People in Same Photo
**Symptom:** Warning "Multiple faces detected, using first"  
**Problem:** May enroll wrong person's face  
**Fix:** Use photos with only ONE person visible

## Re-enrollment Process

If you need to add someone or fix enrollment:

```powershell
# Delete old index
Remove-Item -Recurse -Force data\index\*

# Re-run enrollment
python tools/enroll.py --data-dir friendly --config config.yaml

# Verify it worked
Get-Content "data\index\meta.jsonl" | ConvertFrom-Json | Group-Object person_id | Select-Object Name,Count
```

## Enrollment Best Practices

### Photo Requirements
- ✅ One person per photo
- ✅ Good lighting (front-lit, not backlit)
- ✅ Face clearly visible (not blurry, not obscured)
- ✅ Varied angles (straight-on, slight left/right, up/down)
- ✅ Varied expressions (neutral, smiling, serious)
- ✅ Different times/conditions

### Photo Directory Structure
```
friendly/
├── ira/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── photo3.jpg
└── mike/
    ├── photo1.jpg
    ├── photo2.jpg
    └── photo3.jpg
```

### Quality Thresholds
Check `config.yaml`:
```yaml
quality:
  min_face_quality: 0.3  # Lower = accept more faces
  min_bbox_size: 40      # Minimum face size in pixels
```

## Troubleshooting Enrollment

### No faces detected
1. Check image quality (not too blurry)
2. Check face size (at least 40px)
3. Try lower quality threshold
4. Check if image file corrupted

### Wrong person enrolled
1. Check for multiple faces in photos
2. Remove photos with multiple people
3. Use photos showing only target person

### Low enrollment count
1. Add more photos (aim for 10-15 per person)
2. Lower quality threshold in config.yaml
3. Check enrollment logs for rejected photos

## After Enrollment

Always verify:
```powershell
# 1. Check count per person
Get-Content "data\index\meta.jsonl" | ConvertFrom-Json | Group-Object person_id | Select-Object Name,Count

# 2. Check index file exists
Test-Path "data\index\face.index"

# 3. Check index size (should be > 0)
Get-Item "data\index\face.index" | Select-Object Length
```

## Expected Results

For 2 people with 10-15 photos each:
- Total embeddings: 20-30
- Index file size: ~40-60 KB
- Meta file size: ~4-8 KB

---

**Always verify enrollment before running runtime!** 🎯

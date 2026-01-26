#!/usr/bin/env python3
"""Test camera supported resolutions."""
import cv2

# Common resolutions to test
resolutions = [
    (640, 480),    # VGA
    (800, 600),    # SVGA
    (1024, 768),   # XGA
    (1280, 720),   # 720p HD
    (1280, 960),   # SXGA-
    (1920, 1080),  # 1080p Full HD
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit(1)

print("Testing USB camera resolutions:")
print("=" * 50)

for w, h in resolutions:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if actual_w == w and actual_h == h:
        print(f"  ✅ {w}x{h} - SUPPORTED")
    else:
        print(f"  ❌ {w}x{h} - Not supported (got {actual_w}x{actual_h})")

cap.release()
print("=" * 50)


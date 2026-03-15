"""
kfs/detect.py — Basic KFS detection via webcam

Single-frame inference. Every frame is independently classified and displayed.
Use this to verify the model loads and detects correctly during initial testing.

For competition use, prefer detect_stable.py (temporal smoothing) or
detect_dualmode.py (dual-mode with navigation assistance).

Usage:
    python3 detect.py

Controls:
    q — quit

Notes:
    source=1 targets camera device index 1 — the camera physically mounted on R2
    for forest navigation. Change to source=0 if testing with a single webcam.
"""

import cv2
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"
SOURCE  = 1       # R2 camera — change to 0 for single-webcam testing
CONF    = 0.70
IOU     = 0.40
IMGSZ   = 800

model = YOLO(WEIGHTS)

results = model.predict(
    source=SOURCE,
    conf=CONF,
    iou=IOU,
    imgsz=IMGSZ,
    stream=True,
)

for result in results:
    frame = result.plot()
    cv2.imshow("KFS Detection — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

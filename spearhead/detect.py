"""
spearhead/detect.py — Real-time spearhead detection via webcam

Runs the trained YOLOv8s model on a live webcam feed and draws bounding boxes
around detected SPEAR / FIST / PALM gestures in real time.

Usage:
    python3 detect.py

Controls:
    q — quit

Requirements:
    - weights/best.pt must exist (copy from training output or download)
    - A connected webcam at device index 0
    - CUDA GPU recommended for smooth real-time inference
"""

import cv2
from ultralytics import YOLO

WEIGHTS = "weights/best.pt"
SOURCE  = 0       # Webcam device index — 0 = default camera
CONF    = 0.70
IOU     = 0.70
IMGSZ   = 768

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
    cv2.imshow("Spearhead Detection — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

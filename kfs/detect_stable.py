"""
kfs/detect_stable.py — KFS detection with temporal smoothing

Adds a sliding window majority vote over the last N frames before displaying
a detection. A class label is only shown if the same class appears in at least
STABILITY_THRESHOLD of the last HISTORY_SIZE frames.

Why this matters: During R2's movement between forest blocks, the camera feed
is subject to motion blur and partial occlusion. Single-frame classifications
during motion are unreliable. The temporal buffer ensures R2 only acts on
stable, consistent detections — preventing reactions to transient misclassifications
caused by motion artefacts.

Usage:
    python3 detect_stable.py

Controls:
    q — quit
"""

import cv2
from collections import deque, Counter
from ultralytics import YOLO

WEIGHTS            = "weights/best.pt"
SOURCE             = 1      # R2 camera
CONF               = 0.60   # Slightly relaxed vs detect.py — smoothing handles noise
IOU                = 0.60
IMGSZ              = 800
HISTORY_SIZE       = 5      # Number of frames to keep in rolling window
STABILITY_THRESHOLD = 3     # Minimum count in window to display detection

model = YOLO(WEIGHTS)
history = deque(maxlen=HISTORY_SIZE)

results = model.predict(
    source=SOURCE,
    conf=CONF,
    iou=IOU,
    imgsz=IMGSZ,
    augment=True,   # Test Time Augmentation — improves per-frame accuracy
    max_det=20,
    stream=True,
)

for r in results:
    frame = r.orig_img.copy()

    if r.boxes is not None and len(r.boxes) > 0:
        # Take the single highest-confidence detection per frame
        top_box = r.boxes[r.boxes.conf.argmax()]
        cls = int(top_box.cls)
        history.append(cls)

        most_common = Counter(history).most_common(1)[0][0]

        # Only render detection if it has been stable across enough recent frames
        if history.count(most_common) >= STABILITY_THRESHOLD:
            frame = r.plot()

    cv2.imshow("KFS Stable Detection — press Q to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()

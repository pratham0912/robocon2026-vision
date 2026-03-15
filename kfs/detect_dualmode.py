"""
kfs/detect_dualmode.py — Dual-mode KFS detection and navigation assistance

Combines two detection modes in a single runtime session, switchable live
with keyboard keys:

  Mode 1 — YOLO KFS classification:
      Majority-vote temporal smoothing (same as detect_stable.py).
      Use when R2 is stationary on a block and needs to classify the scroll.

  Mode 2 — Blue square detection (HSV-based):
      Detects near-square blue contours for positional navigation.
      Per ABU Robocon 2026 rulebook Section 15, the Meihua Forest Zone 2
      Pathway (blue team side) is painted RGB 128-191-209, and the blue
      Start Zone is RGB 50-0-255. This mode helps R2 confirm it has reached
      the correct block position before attempting KFS pickup.

Usage:
    python3 detect_dualmode.py

Controls:
    1 — switch to YOLO KFS mode
    2 — switch to blue square navigation mode
    q — quit
"""

import cv2
import numpy as np
from collections import deque, Counter
from ultralytics import YOLO

# ── Configuration ─────────────────────────────────────────────────────────────

WEIGHTS             = "weights/best.pt"
CAMERA_INDEX        = 0          # Change to 1 for R2's dedicated camera
FRAME_WIDTH         = 1280
FRAME_HEIGHT        = 720

# YOLO inference
CONF                = 0.60
IOU                 = 0.60
IMGSZ               = 800
HISTORY_SIZE        = 5
STABILITY_THRESHOLD = 3

# Blue HSV range — tuned for Robocon 2026 field colours (Section 15)
LOWER_BLUE          = np.array([90, 100, 70])
UPPER_BLUE          = np.array([140, 255, 255])
MIN_CONTOUR_AREA    = 1000       # Ignore noise below this pixel area
SQUARE_ASPECT_MIN   = 0.85       # Accept contours with width/height ratio in
SQUARE_ASPECT_MAX   = 1.15       # this range as "square"

# ── Initialise ────────────────────────────────────────────────────────────────

model   = YOLO(WEIGHTS)
history = deque(maxlen=HISTORY_SIZE)
mode    = 1   # Start in YOLO mode

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# ── Main loop ─────────────────────────────────────────────────────────────────

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()

    # ── Mode 1: YOLO KFS classification ──────────────────────────────────────
    if mode == 1:
        results = model.predict(
            source=frame,
            conf=CONF,
            iou=IOU,
            imgsz=IMGSZ,
            verbose=False,
        )
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            top_box = r.boxes[r.boxes.conf.argmax()]
            cls = int(top_box.cls)
            history.append(cls)

            most_common = Counter(history).most_common(1)[0][0]
            if history.count(most_common) >= STABILITY_THRESHOLD:
                display_frame = r.plot()

        cv2.putText(display_frame, "MODE 1: YOLO KFS DETECTION",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # ── Mode 2: Blue square navigation detection ──────────────────────────────
    elif mode == 2:
        hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)

        kernel = np.ones((5, 5), np.uint8)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < MIN_CONTOUR_AREA:
                continue

            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h

            if SQUARE_ASPECT_MIN < aspect_ratio < SQUARE_ASPECT_MAX:
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(display_frame, "BLUE SQUARE",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(display_frame, "MODE 2: BLUE SQUARE DETECTION",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Blue Mask (debug)", mask)

    # ── Display ───────────────────────────────────────────────────────────────
    cv2.imshow("KFS Detection System — 1: YOLO | 2: Blue | Q: Quit", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("1"):
        mode = 1
        print("Switched to Mode 1 — YOLO KFS classification")
    elif key == ord("2"):
        mode = 2
        history.clear()   # Reset voting history when switching modes
        print("Switched to Mode 2 — Blue square navigation")

cap.release()
cv2.destroyAllWindows()

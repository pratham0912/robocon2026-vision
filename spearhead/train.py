"""
spearhead/train.py — YOLOv8s training for Spearhead detection

Trains a YOLOv8s model to classify three spearhead types (SPEAR / FIST / PALM)
from live camera input. This model runs on R2 during the Martial Club phase of
ABU Robocon 2026 to identify which spearhead to pick up from the shared rack.

Usage:
    python3 train.py

Output:
    SPEARHEAD_FULL/spearhead_train/weights/best.pt
    SPEARHEAD_FULL/spearhead_train/results.png
    SPEARHEAD_FULL/spearhead_train/confusion_matrix.png

After training completes, copy best.pt to weights/best.pt
"""

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    # ── Dataset ──────────────────────────────────────────────────────────────
    data="data.yaml",

    # ── Training schedule ────────────────────────────────────────────────────
    epochs=150,
    patience=40,

    # ── Image ────────────────────────────────────────────────────────────────
    imgsz=768,

    # ── Batch ────────────────────────────────────────────────────────────────
    batch=16,

    # ── Hardware ─────────────────────────────────────────────────────────────
    device=0,
    workers=12,
    amp=True,

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer="AdamW",
    lr0=0.003,
    weight_decay=0.0005,
    cos_lr=True,

    # ── Augmentation ─────────────────────────────────────────────────────────
    augment=True,
    mosaic=1.0,
    mixup=0.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    degrees=5,
    translate=0.1,
    scale=0.5,
    shear=2,

    # ── Output ───────────────────────────────────────────────────────────────
    project="SPEARHEAD_FULL",
    name="spearhead_train",
    exist_ok=True,
)

metrics = model.val()
print(metrics)

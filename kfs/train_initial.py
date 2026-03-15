"""
kfs/train_initial.py — Stage 1: Initial KFS classification training

Trains YOLOv8s from the pretrained ImageNet backbone on the full 30-class
Merged_Dataset (15 Fake KFS + 15 Real KFS symbols).

This is Stage 1 of a two-stage training pipeline. The best checkpoint
produced here is used as the starting point for Stage 2 fine-tuning
in train_finetune.py.

Usage:
    python3 train_initial.py

Output:
    FAKE_KFS_TRAINING_2/exp_yolov8s_best/weights/best.pt

After Stage 1 completes, run train_finetune.py for Stage 2.
"""

from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    # ── Dataset ──────────────────────────────────────────────────────────────
    data="data.yaml",

    # ── Training schedule ────────────────────────────────────────────────────
    epochs=300,       # Long run — full exploration before fine-tuning
    patience=30,

    # ── Image ────────────────────────────────────────────────────────────────
    imgsz=800,        # High resolution critical for fine-grained stroke recognition

    # ── Batch ────────────────────────────────────────────────────────────────
    batch=16,

    # ── Hardware ─────────────────────────────────────────────────────────────
    device=0,
    workers=8,
    amp=True,

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer="SGD",  # Better generalization over long training runs
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True,

    # ── Augmentation ─────────────────────────────────────────────────────────
    augment=True,
    mosaic=1.0,
    mixup=0.1,        # Softens decision boundaries between similar classes
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,

    # ── Output ───────────────────────────────────────────────────────────────
    project="FAKE_KFS_TRAINING_2",
    name="exp_yolov8s_best",
    exist_ok=True,
)

metrics = model.val()
print(metrics)

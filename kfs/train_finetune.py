"""
kfs/train_finetune.py — Stage 2: Fine-tuning KFS classification model

Loads the best Stage 1 checkpoint and continues training with adjusted
hyperparameters optimized for fine-tuning rather than initial learning.

Key differences from Stage 1:
  - Lower lr0 (0.005 vs 0.01) — avoid overwriting learned features
  - Higher patience (50 vs 30) — fine-tuning converges slower
  - Added copy_paste=0.2 — improves small object generalization
  - Added perspective=0.0005 — simulate slight camera tilt variation
  - Added degrees=5.0 — allow minor rotation augmentation

Usage:
    python3 train_finetune.py

Prerequisites:
    Stage 1 must have completed. The checkpoint at
    FAKE_KFS_TRAINING_2/exp_yolov8s_best/weights/best.pt must exist.

Output:
    FULL_KFS_TRAINING/kfs_v8_fine_tuned/weights/best.pt

After fine-tuning completes, copy the best weights to weights/best.pt.
Note: The production model used in this repo was further selected from
multiple runs and lives at FULL_KFS_TRAINING/best_model_kfs/weights/best.pt.
"""

from ultralytics import YOLO

# Load Stage 1 best checkpoint as starting point
model = YOLO("FAKE_KFS_TRAINING_2/exp_yolov8s_best/weights/best.pt")

model.train(
    # ── Dataset ──────────────────────────────────────────────────────────────
    data="data.yaml",

    # ── Training schedule ────────────────────────────────────────────────────
    epochs=200,
    patience=50,      # Fine-tuning converges slower — needs more patience

    # ── Image ────────────────────────────────────────────────────────────────
    imgsz=800,

    # ── Batch ────────────────────────────────────────────────────────────────
    batch=12,         # Slightly reduced from Stage 1

    # ── Hardware ─────────────────────────────────────────────────────────────
    device=0,
    workers=12,
    amp=True,

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer="SGD",
    lr0=0.005,        # Lower than Stage 1 — preserve learned representations
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    cos_lr=True,

    # ── Augmentation ─────────────────────────────────────────────────────────
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=5.0,      # Small rotation — scrolls may be slightly tilted
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0005,  # Simulate slight camera angle variation
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.2,   # Paste objects across images — improves small object detection

    # ── Regularization ───────────────────────────────────────────────────────
    dropout=0.0,      # Disabled — fine-tuning benefits from full model capacity

    # ── Validation / Saving ──────────────────────────────────────────────────
    val=True,
    plots=True,
    save=True,

    # ── Output ───────────────────────────────────────────────────────────────
    project="FULL_KFS_TRAINING",
    name="kfs_v8_fine_tuned",
    exist_ok=True,
)

metrics = model.val()
print(metrics)

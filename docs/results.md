# Training Results

Full training metrics, confusion matrices, precision-recall curves, and sample validation predictions for both vision subsystems.

---

## Subsystem 1 — Spearhead Detection

**Model:** YOLOv8s | **Classes:** SPEAR, FIST, PALM | **Epochs:** 150 (patience=40) | **imgsz:** 768

### Training Curves

![Training Curves](../spearhead/results/results.png)

### Confusion Matrix

![Confusion Matrix](../spearhead/results/confusion_matrix.png)

![Confusion Matrix Normalized](../spearhead/results/confusion_matrix_normalized.png)

### Precision, Recall, and F1 Curves

| Curve | Plot |
|-------|------|
| Precision-Recall | ![PR Curve](../spearhead/results/BoxPR_curve.png) |
| Precision | ![P Curve](../spearhead/results/BoxP_curve.png) |
| Recall | ![R Curve](../spearhead/results/BoxR_curve.png) |
| F1 | ![F1 Curve](../spearhead/results/BoxF1_curve.png) |

### Sample Validation Predictions

| Batch | Ground Truth | Predictions |
|-------|-------------|-------------|
| 0 | ![Labels](../spearhead/results/val_batch0_labels.jpg) | ![Pred](../spearhead/results/val_batch0_pred.jpg) |
| 1 | ![Labels](../spearhead/results/val_batch1_labels.jpg) | ![Pred](../spearhead/results/val_batch1_pred.jpg) |
| 2 | ![Labels](../spearhead/results/val_batch2_labels.jpg) | ![Pred](../spearhead/results/val_batch2_pred.jpg) |

---

## Subsystem 2 — KFS Classification

**Model:** YOLOv8s (2-stage fine-tuned) | **Classes:** 30 (REAL_1–15, FAKE_1–15) | **imgsz:** 800

Training used a two-stage pipeline: initial training from the pretrained YOLOv8s backbone (300 epochs, SGD), followed by fine-tuning from the Stage 1 best checkpoint (200 epochs, lower LR, additional augmentation). The results below are from the final fine-tuned model.

### Training Curves

![Training Curves](../kfs/results/results.png)

### Confusion Matrix

![Confusion Matrix](../kfs/results/confusion_matrix.png)

![Confusion Matrix Normalized](../kfs/results/confusion_matrix_normalized.png)

### Precision, Recall, and F1 Curves

| Curve | Plot |
|-------|------|
| Precision-Recall | ![PR Curve](../kfs/results/BoxPR_curve.png) |
| Precision | ![P Curve](../kfs/results/BoxP_curve.png) |
| Recall | ![R Curve](../kfs/results/BoxR_curve.png) |
| F1 | ![F1 Curve](../kfs/results/BoxF1_curve.png) |

### Sample Validation Predictions

| Batch | Ground Truth | Predictions |
|-------|-------------|-------------|
| 0 | ![Labels](../kfs/results/val_batch0_labels.jpg) | ![Pred](../kfs/results/val_batch0_pred.jpg) |
| 1 | ![Labels](../kfs/results/val_batch1_labels.jpg) | ![Pred](../kfs/results/val_batch1_pred.jpg) |
| 2 | ![Labels](../kfs/results/val_batch2_labels.jpg) | ![Pred](../kfs/results/val_batch2_pred.jpg) |

---

*[← Back to README](../README.md)*

# Annotation Pipeline — Full Issue Log

Complete record of every issue encountered during the semi-automatic CVAT annotation pipeline for the Spearhead dataset. This supplements the summary in `README.md § 8.3`.

---

## Environment

- OS: Ubuntu 24.04
- Python: 3.10
- Ultralytics YOLOv8 (latest at time of development)
- Annotation tool: CVAT

---

## Goal

Use a YOLOv8 model trained on a small initial set of manually annotated images to automatically predict bounding box labels for ~1200 images per class (FIST, PALM, SPEAR), import those labels into CVAT as pre-annotations, and produce a corrected full dataset for final training. Reduces manual labeling effort by ~80–90%.

---

## Issue 1 — `FileNotFoundError` on `model.predict()`

**Trigger:** Passing `Creating_data/SPEARHEAD_DATA` as the source directory.

**Error:**
```
FileNotFoundError: No images or videos found in Creating_data/SPEARHEAD_DATA.
Supported formats: {'bmp', 'jpg', 'png', ...}
```

**Root cause:** `model.predict()` does not recursively scan subdirectories in some Ultralytics versions. `SPEARHEAD_DATA/` contains only subdirectories (`FIST/`, `PALM/`, `SPEAR/`) — no image files are directly inside the parent folder.

**Fix:** Pass each class subfolder individually:

```python
folders = [
    "Creating_data/SPEARHEAD_DATA/FIST",
    "Creating_data/SPEARHEAD_DATA/PALM",
    "Creating_data/SPEARHEAD_DATA/SPEAR",
]
for folder in folders:
    model.predict(source=folder, ...)
```

---

## Issue 2 — Labels overwritten on repeated `model.predict()` calls

**Trigger:** Running `model.predict()` three times in a loop without specifying unique output directories.

**Symptom:** After prediction on FIST, PALM, and SPEAR, only `spear_*.txt` files existed. FIST and PALM labels were silently replaced.

**Root cause:** `model.predict()` defaults to `runs/detect/predict/` for all output. Every subsequent call without a unique `name` increments to `predict2/`, `predict3/`, etc. — but if `exist_ok=True` or the folder already exists, files with identical names overwrite each other.

**Fix:**
```python
model.predict(source=fist_path,  project="runs/detect", name="fist_pred",  save_txt=True)
model.predict(source=palm_path,  project="runs/detect", name="palm_pred",  save_txt=True)
model.predict(source=spear_path, project="runs/detect", name="spear_pred", save_txt=True)
```

---

## Issue 3 — Filename collisions when combining labels

**Trigger:** Copying all `.txt` files from three prediction folders into one `combined_labels/` directory.

**Symptom:** After all three `cp` commands, `combined_labels/` contained ~400 files instead of ~1200.

**Root cause:** All three class folders originally had identically numbered images (`0.jpg`, `1.jpg`, etc.). YOLO generates labels with the same basename as the source image. When combined, files silently overwrote each other.

**Fix:** Rename all images with a class prefix before prediction:

```python
import os

base = "Creating_data/SPEARHEAD_DATA"
for folder in ["FIST", "PALM", "SPEAR"]:
    path = os.path.join(base, folder)
    files = sorted([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    for i, filename in enumerate(files):
        ext = os.path.splitext(filename)[1]
        new_name = f"{folder.lower()}_{i}{ext}"
        os.rename(os.path.join(path, filename), os.path.join(path, new_name))
```

After renaming, YOLO generates: `fist_0.txt`, `palm_0.txt`, `spear_0.txt` — no collisions.

---

## Issue 4 — CVAT import fails with `datumaro._ImportFail`

This was the most significant and least obvious issue. Multiple hours were spent investigating it.

**Trigger:** Importing `labels.zip` into CVAT with format **YOLO 1.1**.

**Errors seen (multiple forms during investigation):**
```
datumaro.components.contexts.importer._ImportFail:
Failed to import dataset 'yolo' at '/home/django/data/tmp/tmp6rwbfm10'
```
```
Dataset must contain a file: "obj.data"
```

**Initial (incorrect) diagnosis:** The `obj.data` error message implied the wrong format was selected or that Darknet YOLO structure (`obj.data`, `obj.names`, `obj/` folder) was required. Significant time was spent building this structure before realising it was not the actual problem.

**Actual root cause:** Inspection of the generated `.txt` files showed that every line had **six values** instead of the five that YOLO format requires:

```
# What was in the files (wrong — 6 values):
0 0.699021 0.474084 0.0717145 0.256695 0.735146
1 0.565751 0.516867 0.0770987 0.113702 0.453691

# What CVAT requires (correct — 5 values):
0 0.699021 0.474084 0.0717145 0.256695
```

The sixth value is the detection confidence score, present because `save_conf=True` was used:

```python
model.predict(source=folder, save_txt=True, save_conf=True, ...)
#                                            ^^^^^^^^^^^^^^^^
#                                            appends confidence as column 6
```

CVAT's Datumaro importer strictly parses YOLO format as exactly five space-separated values per line. Any deviation causes an internal exception that surfaces as the generic `_ImportFail` error. The error message gives no indication of the column count mismatch.

**Fix — remove `save_conf=True`** (already done in `annotate.py`).

**Fix for already-generated files:**
```bash
cd combined_labels
for f in *.txt; do awk '{print $1,$2,$3,$4,$5}' "$f" > tmp && mv tmp "$f"; done
```

Verify:
```bash
head -3 fist_0.txt
# Correct — exactly 5 values:
# 0 0.699021 0.474084 0.0717145 0.256695
```

---

## Issue 5 — ZIP structure rejected by CVAT

**Trigger:** Creating the import ZIP from the parent directory:
```bash
zip -r labels.zip combined_labels/
```

**Symptom:** Import fails with the same `_ImportFail` error even after the confidence column was stripped.

**Root cause:** The ZIP contained files nested inside a subdirectory:
```
labels.zip
└── combined_labels/
    ├── fist_0.txt   ← CVAT rejects this nesting
    └── palm_0.txt
```

CVAT YOLO 1.1 requires `.txt` files at the ZIP root level, not inside any folder.

**Fix:**
```bash
cd combined_labels
zip -r ../labels.zip *.txt
```

Resulting structure (correct):
```
labels.zip
├── fist_0.txt
├── palm_0.txt
└── spear_0.txt
```

Verify before uploading:
```bash
unzip -l ../labels.zip | head
# Must NOT show a subdirectory prefix
```

---

## Complete working command sequence

```bash
# From project root

# 1. Rename images (run once before any annotation)
python3 - <<'EOF'
import os
base = "Creating_data/SPEARHEAD_DATA"
for folder in ["FIST", "PALM", "SPEAR"]:
    path = os.path.join(base, folder)
    files = sorted([f for f in os.listdir(path) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    for i, f in enumerate(files):
        ext = os.path.splitext(f)[1]
        os.rename(os.path.join(path, f), os.path.join(path, f"{folder.lower()}_{i}{ext}"))
EOF

# 2. Run annotation (save_conf NOT used)
cd spearhead && python3 annotate.py && cd ..

# 3. Combine labels
mkdir -p combined_labels
cp spearhead/runs/detect/fist_pred/labels/*.txt combined_labels/
cp spearhead/runs/detect/palm_pred/labels/*.txt combined_labels/
cp spearhead/runs/detect/spear_pred/labels/*.txt combined_labels/

# 4. Strip confidence column (only needed if save_conf was used accidentally)
cd combined_labels
for f in *.txt; do awk '{print $1,$2,$3,$4,$5}' "$f" > tmp && mv tmp "$f"; done
cd ..

# 5. Create import ZIP (files at root, no subfolder)
cd combined_labels && zip -r ../labels.zip *.txt && cd ..

# 6. Import in CVAT:
#    Actions → Upload annotations → Format: YOLO 1.1 → Upload labels.zip
```

"""
spearhead/annotate.py — Auto-annotation script for CVAT import

Uses the trained YOLOv8 model to predict bounding boxes on all images in
SPEARHEAD_DATA and saves YOLO-format .txt label files for import into CVAT.

Usage:
    python3 annotate.py

Output:
    runs/detect/fist_pred/labels/
    runs/detect/palm_pred/labels/
    runs/detect/spear_pred/labels/

After running, follow README.md § 8.2 to combine, strip the confidence column,
zip, and import into CVAT using format YOLO 1.1.

──────────────────────────────────────────────────────────────────────────────
IMPORTANT: save_conf is intentionally NOT used.
──────────────────────────────────────────────────────────────────────────────

Using save_conf=True appends a 6th column (confidence score) to each line:
    <class_id> <x> <y> <w> <h> <confidence>

CVAT's YOLO 1.1 importer (via Datumaro) strictly requires 5 values per line:
    <class_id> <x> <y> <w> <h>

Presence of the confidence column causes a hard import failure:
    datumaro.components.contexts.importer._ImportFail

See README.md § 8.3 Issue 4 for the full explanation.

If you already generated labels with save_conf=True, strip column 6:
    cd combined_labels
    for f in *.txt; do awk '{print $1,$2,$3,$4,$5}' "$f" > tmp && mv tmp "$f"; done
"""

import os
from ultralytics import YOLO

WEIGHTS  = "weights/best.pt"
DATA_DIR = os.path.join("Creating_data", "SPEARHEAD_DATA")

CLASS_FOLDERS = {
    "FIST":  "fist_pred",
    "PALM":  "palm_pred",
    "SPEAR": "spear_pred",
}

model = YOLO(WEIGHTS)

for class_name, output_name in CLASS_FOLDERS.items():
    source = os.path.join(DATA_DIR, class_name)

    if not os.path.isdir(source):
        print(f"[SKIP] Not found: {source}")
        continue

    print(f"[RUN] {source} → runs/detect/{output_name}/labels/")

    model.predict(
        source=source,
        save_txt=True,
        # save_conf intentionally omitted — see docstring
        project="runs/detect",
        name=output_name,
        exist_ok=True,
    )

print("\nDone. Next steps:")
print("  mkdir -p combined_labels")
print("  cp runs/detect/fist_pred/labels/*.txt combined_labels/")
print("  cp runs/detect/palm_pred/labels/*.txt combined_labels/")
print("  cp runs/detect/spear_pred/labels/*.txt combined_labels/")
print("  cd combined_labels && zip -r ../labels.zip *.txt")
print("  Import labels.zip into CVAT → Format: YOLO 1.1")

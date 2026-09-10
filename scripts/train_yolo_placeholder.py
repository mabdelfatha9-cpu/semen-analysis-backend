#!/usr/bin/env python3
"""
Self-learning training entrypoint (YOLO / classifier).

IMPORTANT
---------
1. You need labeled images with bounding boxes for real YOLO detection.
   Technician concentration/morphology numbers alone are NOT enough for YOLO;
   they train a regression/classifier head, not a detector.

2. Railway free tier is for serving API, not heavy GPU training.
   Run this script on a machine with GPU (local PC, Colab, or paid GPU host):

     pip install ultralytics
     python scripts/train_yolo_placeholder.py --data-dir ./learning_data

3. Workflow we implement now:
   - App sends reviewed labels (+ optional image) to POST /feedback
   - Server appends to learning_data/labels.jsonl and images/
   - When you have enough images, convert to YOLO format and train
   - Copy best.pt into models/ and set MODEL_PATH for inference

This file documents the pipeline and exports a simple dataset summary.
Full YOLO box labeling is still a manual/tooling step (Label Studio, CVAT, etc.).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Mono Chrome self-learning trainer scaffold")
    parser.add_argument("--data-dir", type=Path, default=Path("learning_data"))
    parser.add_argument("--export-summary", action="store_true", default=True)
    args = parser.parse_args()

    labels_path = args.data_dir / "labels.jsonl"
    images_dir = args.data_dir / "images"

    if not labels_path.exists():
        print("No labels.jsonl yet. Collect reviews via POST /feedback first.")
        return 1

    rows = []
    with labels_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    n_img = sum(1 for r in rows if r.get("has_image"))
    n_morph = sum(1 for r in rows if r.get("human_normal_forms_percent") is not None)
    n_conc = sum(1 for r in rows if r.get("human_concentration") is not None)

    print("=== Mono Chrome learning library ===")
    print(f"labels: {len(rows)}")
    print(f"with image: {n_img}")
    print(f"human concentration: {n_conc}")
    print(f"human morphology %: {n_morph}")
    print(f"images dir exists: {images_dir.exists()}")

    if n_img < 50:
        print("\nNeed ~50+ labeled images before YOLO fine-tuning is meaningful.")
        print("Next: use Label Studio/CVAT to draw sperm boxes on images/ then:")
        print("  yolo detect train data=data.yaml model=yolov8n.pt epochs=50")
        return 0

    print("\nEnough images to start annotation → YOLO train.")
    print("Install: pip install ultralytics")
    print("Example:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolov8n.pt')")
    print("  model.train(data='data.yaml', epochs=50, imgsz=640)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

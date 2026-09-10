"""
Self-learning library: stores technician corrections so they can later
train a detector / morphology classifier (e.g. YOLO).

Layout under DATA_DIR (default /data/learning or ./learning_data):
  labels.jsonl          — one JSON object per reviewed sample
  images/<sample_id>.jpg — optional attached frame/image
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _data_dir() -> Path:
    root = Path(os.environ.get("LEARNING_DATA_DIR", "learning_data"))
    root.mkdir(parents=True, exist_ok=True)
    (root / "images").mkdir(parents=True, exist_ok=True)
    return root


@dataclass
class LabeledSample:
    sample_id: str
    timestamp_epoch: float
    # Model outputs at analysis time
    estimated_concentration: Optional[float] = None
    estimated_normal_forms_percent: Optional[float] = None
    # Human ground truth
    human_concentration: Optional[float] = None
    human_normal_forms_percent: Optional[float] = None
    human_progressive_motility_percent: Optional[float] = None
    reviewer_note: Optional[str] = None
    has_image: bool = False
    microns_per_pixel: Optional[float] = None
    chamber_depth_microns: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


def append_label(sample: LabeledSample) -> Path:
    path = _data_dir() / "labels.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(sample), ensure_ascii=False) + "\n")
    return path


def save_image(sample_id: str, data: bytes, suffix: str = ".jpg") -> Path:
    path = _data_dir() / "images" / f"{sample_id}{suffix}"
    path.write_bytes(data)
    return path


def list_labels(limit: int = 500) -> List[Dict[str, Any]]:
    path = _data_dir() / "labels.jsonl"
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def stats() -> Dict[str, Any]:
    rows = list_labels(limit=100_000)
    with_image = sum(1 for r in rows if r.get("has_image"))
    with_morph = sum(1 for r in rows if r.get("human_normal_forms_percent") is not None)
    with_conc = sum(1 for r in rows if r.get("human_concentration") is not None)
    return {
        "total_labels": len(rows),
        "with_image": with_image,
        "with_human_concentration": with_conc,
        "with_human_morphology": with_morph,
        "ready_for_yolo_hint": with_image >= 50,
        "data_dir": str(_data_dir().resolve()),
    }

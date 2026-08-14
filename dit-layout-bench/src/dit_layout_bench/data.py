"""PubLayNet layout and annotation validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .spec import PUBLAYNET_CATEGORIES


@dataclass(frozen=True)
class DatasetSummary:
    split: str
    images: int
    annotations: int


def validate_publaynet(root: str | Path) -> tuple[DatasetSummary, DatasetSummary]:
    root = Path(root)
    summaries = []
    for split in ("train", "val"):
        image_dir = root / split
        annotation_path = root / "annotations" / f"{split}.json"
        if not image_dir.is_dir():
            raise FileNotFoundError(f"PubLayNet image directory not found: {image_dir}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"PubLayNet annotation not found: {annotation_path}")
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {int(item["id"]): item["name"] for item in document.get("categories", [])}
        if set(categories) != set(PUBLAYNET_CATEGORIES):
            raise ValueError(
                f"{annotation_path} category IDs must be {sorted(PUBLAYNET_CATEGORIES)}, "
                f"got {sorted(categories)}"
            )
        image_ids = {int(item["id"]) for item in document.get("images", [])}
        dangling = [
            item["id"]
            for item in document.get("annotations", [])
            if int(item["image_id"]) not in image_ids
            or int(item["category_id"]) not in PUBLAYNET_CATEGORIES
        ]
        if dangling:
            raise ValueError(f"{annotation_path} has invalid annotations: {dangling[:5]}")
        summaries.append(
            DatasetSummary(split, len(image_ids), len(document.get("annotations", [])))
        )
    return tuple(summaries)


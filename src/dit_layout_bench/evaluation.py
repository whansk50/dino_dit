"""Common COCO result summaries."""

from __future__ import annotations

from typing import Mapping

from .spec import PUBLAYNET_CATEGORIES


def per_category_ap(coco_evaluation) -> dict[str, float]:
    """Return AP@[.50:.95] for each PubLayNet category from COCOeval."""
    import numpy as np

    precision = coco_evaluation.eval["precision"]  # IoU, recall, category, area, maxDets
    category_ids = list(coco_evaluation.params.catIds)
    result = {}
    for category_id, name in PUBLAYNET_CATEGORIES.items():
        if category_id not in category_ids:
            result[name] = float("nan")
            continue
        values = precision[:, :, category_ids.index(category_id), 0, -1]
        valid = values[values > -1]
        result[name] = float(np.mean(valid) * 100) if valid.size else float("nan")
    return result


def print_per_category_ap(coco_evaluation) -> Mapping[str, float]:
    result = per_category_ap(coco_evaluation)
    print("PubLayNet per-category AP: " + ", ".join(f"{k}={v:.2f}" for k, v in result.items()))
    return result


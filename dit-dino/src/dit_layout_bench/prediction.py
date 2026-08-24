"""Backend-neutral prediction serialization."""

from __future__ import annotations

from typing import Any

from .spec import PUBLAYNET_CATEGORIES


def prediction_record(
    image_id: str | int,
    box_xyxy: Any,
    score: float,
    category_id: int,
) -> dict[str, Any]:
    if category_id not in PUBLAYNET_CATEGORIES:
        raise ValueError(f"Invalid PubLayNet prediction category: {category_id}")
    return {
        "image_id": image_id,
        "box_xyxy": box_xyxy.tolist(),
        "score": float(score),
        "category_id": category_id,
    }

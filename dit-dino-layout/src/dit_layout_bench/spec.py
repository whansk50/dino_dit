"""Dependency-free public data contracts used by both detector backends."""

from __future__ import annotations

from typing import Iterable


PUBLAYNET_CATEGORIES = {
    1: "Text",
    2: "Title",
    3: "List",
    4: "Table",
    5: "Figure",
}

def category_id_to_train_label(category_id: int) -> int:
    """Map PubLayNet category IDs (1..5) to contiguous labels (0..4)."""
    if category_id not in PUBLAYNET_CATEGORIES:
        raise ValueError(f"Unknown PubLayNet category_id: {category_id}")
    return category_id - 1


def train_label_to_category_id(label: int) -> int:
    """Map contiguous labels (0..4) back to public PubLayNet category IDs."""
    category_id = label + 1
    if category_id not in PUBLAYNET_CATEGORIES:
        raise ValueError(f"Unknown PubLayNet training label: {label}")
    return category_id


def validate_category_ids(category_ids: Iterable[int]) -> None:
    for category_id in category_ids:
        category_id_to_train_label(category_id)

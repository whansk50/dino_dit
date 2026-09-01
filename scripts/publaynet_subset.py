"""Shared PubLayNet subset utilities for runtime validation scripts."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
from typing import Any


_CATEGORIES = {1: "Text", 2: "Title", 3: "List", 4: "Table", 5: "Figure"}


def _read_coco(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"COCO annotation not found: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"COCO annotation root must be an object: {path}")
    for key in ("images", "annotations", "categories"):
        if not isinstance(document.get(key), list):
            raise ValueError(f"COCO annotation {key} must be an array: {path}")
    categories = {
        int(category["id"]): str(category["name"])
        for category in document["categories"]
    }
    if categories != _CATEGORIES:
        raise ValueError(
            f"PubLayNet category mapping must be exactly {_CATEGORIES}: {path}"
        )
    return document


def _select_images(
    document: dict[str, Any], limit: int, *, seed: int
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("subset image counts must be positive")
    annotated_ids = {
        int(annotation["image_id"]) for annotation in document["annotations"]
    }
    annotated = [
        image for image in document["images"] if int(image["id"]) in annotated_ids
    ]
    unannotated = [
        image
        for image in document["images"]
        if int(image["id"]) not in annotated_ids
    ]
    generator = random.Random(seed)
    generator.shuffle(annotated)
    generator.shuffle(unannotated)
    selected = (annotated + unannotated)[:limit]
    if len(selected) < limit:
        raise ValueError(
            f"Requested {limit} images but the split contains only {len(selected)}"
        )
    return selected


def create_publaynet_subset(
    source_root: Path,
    destination_root: Path,
    *,
    split: str,
    image_count: int,
    seed: int,
) -> None:
    """Create one COCO split using symlinks without modifying source data."""
    source_root = source_root.resolve()
    source_image_root = (source_root / split).resolve()
    document = _read_coco(source_root / "annotations" / f"{split}.json")
    selected_images = _select_images(document, image_count, seed=seed)
    selected_ids = {int(image["id"]) for image in selected_images}
    selected_annotations = [
        annotation
        for annotation in document["annotations"]
        if int(annotation["image_id"]) in selected_ids
    ]

    destination_image_root = destination_root / split
    for image in selected_images:
        relative_path = Path(str(image["file_name"]))
        if relative_path.is_absolute():
            raise ValueError(f"COCO file_name must be relative: {relative_path}")
        source_image = (source_image_root / relative_path).resolve()
        try:
            source_image.relative_to(source_image_root)
        except ValueError as error:
            raise ValueError(
                f"Image path escapes the {split} directory: {relative_path}"
            ) from error
        if not source_image.is_file():
            raise FileNotFoundError(f"PubLayNet image not found: {source_image}")
        destination_image = destination_image_root / relative_path
        destination_image.parent.mkdir(parents=True, exist_ok=True)
        if destination_image.is_symlink():
            if destination_image.resolve() == source_image:
                continue
            raise FileExistsError(
                f"Subset symlink points to another source: {destination_image}"
            )
        if destination_image.exists():
            raise FileExistsError(
                f"Subset destination is not a symlink: {destination_image}"
            )
        destination_image.symlink_to(source_image)

    subset = {
        key: deepcopy(value)
        for key, value in document.items()
        if key not in {"images", "annotations"}
    }
    subset["images"] = selected_images
    subset["annotations"] = selected_annotations
    annotation_dir = destination_root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_path = annotation_dir / f"{split}.json"
    temporary_annotation = annotation_dir / f".{split}.json.tmp"
    temporary_annotation.write_text(
        json.dumps(subset, ensure_ascii=False), encoding="utf-8"
    )
    temporary_annotation.replace(annotation_path)


def prepare_empty_work_dir(requested: Path | None, *, prefix: str) -> tuple[Path, bool]:
    """Return an empty artifact directory and whether it is temporary."""
    if requested is None:
        return Path(tempfile.mkdtemp(prefix=prefix)), True
    requested = requested.resolve()
    if requested.exists() and not requested.is_dir():
        raise ValueError(f"--work-dir must be a directory: {requested}")
    if requested.exists() and any(requested.iterdir()):
        raise ValueError(f"--work-dir must be empty: {requested}")
    requested.mkdir(parents=True, exist_ok=True)
    return requested, False

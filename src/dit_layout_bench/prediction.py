"""Backend-neutral prediction serialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .spec import PUBLAYNET_CATEGORIES


IMAGE_EXTENSIONS = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)

_CATEGORY_COLORS = {
    1: "#1f77b4",
    2: "#ff7f0e",
    3: "#2ca02c",
    4: "#d62728",
    5: "#9467bd",
}


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


def collect_image_paths(path: str | Path) -> list[Path]:
    """Return one image or the directly contained images in stable order."""
    path = Path(path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {path.suffix or '<none>'}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Image input not found: {path}")
    images = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError(f"No supported images found in directory: {path}")
    return images


def save_visualization(
    image_path: str | Path,
    predictions: Iterable[dict[str, Any]],
    output_path: str | Path,
) -> Path:
    """Draw backend-neutral predictions in original-image coordinates."""
    from PIL import Image, ImageDraw

    image_path = Path(image_path)
    output_path = Path(output_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = max(2, round(min(image.size) / 400))
    for prediction in predictions:
        category_id = int(prediction["category_id"])
        box = [float(value) for value in prediction["box_xyxy"]]
        color = _CATEGORY_COLORS[category_id]
        label = (
            f"{PUBLAYNET_CATEGORIES[category_id]} "
            f"{float(prediction['score']):.2f}"
        )
        draw.rectangle(box, outline=color, width=line_width)
        text_box = draw.textbbox((box[0], box[1]), label)
        text_height = text_box[3] - text_box[1]
        label_top = max(0.0, box[1] - text_height - 4)
        label_box = (box[0], label_top, box[0] + text_box[2] - text_box[0] + 4, box[1])
        draw.rectangle(label_box, fill=color)
        draw.text((box[0] + 2, label_top + 2), label, fill="white")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path

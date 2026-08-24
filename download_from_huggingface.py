"""Stream Hugging Face PubLayNet shards into the local COCO layout.

This is a data-preparation command, not a runtime dependency of the training
package. By default, Hugging Face streams one Parquet shard at a time instead
of materializing the full dataset as an Arrow table first.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterator, Mapping

from PIL import Image as PILImage


DATASET_ID = "jordanparker6/publaynet"
SPLITS = {"train": "train", "validation": "val"}
CATEGORIES = [
    {"id": 1, "name": "Text", "supercategory": "layout"},
    {"id": 2, "name": "Title", "supercategory": "layout"},
    {"id": 3, "name": "List", "supercategory": "layout"},
    {"id": 4, "name": "Table", "supercategory": "layout"},
    {"id": 5, "name": "Figure", "supercategory": "layout"},
]
CATEGORY_IDS = {category["id"] for category in CATEGORIES}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert sharded Hugging Face PubLayNet to COCO files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("publaynet"))
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--revision", help="Optional immutable Hub revision/commit")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Process remote/cached Parquet shards sequentially (default: true)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse image files already exported by an interrupted conversion",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        help="Limit each split for a conversion smoke test",
    )
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@contextmanager
def _source_image(value: Any) -> Iterator[tuple[PILImage.Image, bytes | None, Path | None]]:
    """Yield a PIL image plus an optional original byte/path representation."""
    if isinstance(value, PILImage.Image):
        yield value, None, None
        return
    if not isinstance(value, Mapping):
        raise TypeError(f"Unsupported Hugging Face image value: {type(value).__name__}")
    raw = value.get("bytes")
    source_path = Path(value["path"]) if value.get("path") else None
    if raw is not None:
        with PILImage.open(BytesIO(raw)) as image:
            yield image, raw, None
        return
    if source_path is not None:
        with PILImage.open(source_path) as image:
            yield image, None, source_path
        return
    raise ValueError("Image record contains neither bytes nor path")


def _write_image(value: Any, destination: Path, *, reuse: bool) -> tuple[int, int]:
    if destination.exists() and not reuse:
        raise FileExistsError(
            f"Image already exists: {destination}; use --resume to reuse it"
        )
    with _source_image(value) as (image, raw, source_path):
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image dimensions: {(width, height)}")
        if destination.exists():
            return width, height
        destination.parent.mkdir(parents=True, exist_ok=True)
        if image.format in {"JPEG", "JPG"} and raw is not None:
            destination.write_bytes(raw)
        elif image.format in {"JPEG", "JPG"} and source_path is not None:
            shutil.copyfile(source_path, destination)
        else:
            image.convert("RGB").save(destination, format="JPEG", quality=95)
        return width, height


def _annotation(record: Mapping[str, Any], image_id: int) -> dict[str, Any]:
    source_image_id = int(record.get("image_id", image_id))
    if source_image_id != image_id:
        raise ValueError(
            f"Annotation image_id {source_image_id} does not match row id {image_id}"
        )
    category_id = int(record["category_id"])
    if category_id not in CATEGORY_IDS:
        raise ValueError(f"Unexpected PubLayNet category_id: {category_id}")
    bbox = [float(value) for value in record["bbox"]]
    if len(bbox) != 4 or not all(math.isfinite(value) for value in bbox):
        raise ValueError(f"Invalid bbox for annotation {record.get('id')}: {bbox}")
    if bbox[2] < 0 or bbox[3] < 0:
        raise ValueError(f"Negative bbox size for annotation {record.get('id')}: {bbox}")
    area = float(record.get("area", bbox[2] * bbox[3]))
    return {
        "id": int(record["id"]),
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": area,
        "iscrowd": int(record.get("iscrowd", 0)),
    }


def export_split(
    rows,
    output_root: Path,
    destination_split: str,
    *,
    dataset_id: str = DATASET_ID,
    revision: str | None = None,
    resume: bool = False,
    max_images: int | None = None,
    progress_every: int = 1000,
) -> tuple[int, int]:
    """Export one iterable split without retaining all annotations in memory."""
    if max_images is not None and max_images < 1:
        raise ValueError("max_images must be positive")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    image_dir = output_root / destination_split
    annotation_dir = output_root / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    final_path = annotation_dir / f"{destination_split}.json"
    json_part = annotation_dir / f".{destination_split}.json.part"
    annotation_part = annotation_dir / f".{destination_split}.annotations.jsonl.part"
    if final_path.exists():
        raise FileExistsError(
            f"COCO annotation already exists: {final_path}; choose another output directory"
        )

    seen_image_ids: set[int] = set()
    image_count = 0
    annotation_count = 0
    with json_part.open("w", encoding="utf-8") as output, annotation_part.open(
        "w", encoding="utf-8"
    ) as annotation_output:
        output.write('{"info":')
        output.write(
            _json(
                {
                    "description": "PubLayNet exported from Hugging Face shards",
                    "source": dataset_id,
                    "revision": revision or "default",
                    "split": destination_split,
                }
            )
        )
        output.write(',"images":[')
        first_image = True
        for row in rows:
            if max_images is not None and image_count >= max_images:
                break
            image_id = int(row["id"])
            if image_id in seen_image_ids:
                raise ValueError(f"Duplicate image id in {destination_split}: {image_id}")
            seen_image_ids.add(image_id)
            file_name = f"{image_id:012d}.jpg"
            width, height = _write_image(
                row["image"], image_dir / file_name, reuse=resume
            )
            image_record = {
                "id": image_id,
                "file_name": file_name,
                "width": width,
                "height": height,
            }
            if not first_image:
                output.write(",")
            output.write(_json(image_record))
            first_image = False

            for source_annotation in row.get("annotations", []):
                annotation_output.write(_json(_annotation(source_annotation, image_id)))
                annotation_output.write("\n")
                annotation_count += 1
            image_count += 1
            if image_count % progress_every == 0:
                print(
                    f"[{destination_split}] images={image_count:,}, "
                    f"annotations={annotation_count:,}",
                    flush=True,
                )

        output.write('],"annotations":[')
        annotation_output.flush()
        first_annotation = True
        with annotation_part.open("r", encoding="utf-8") as annotations:
            for line in annotations:
                if not first_annotation:
                    output.write(",")
                output.write(line.rstrip("\n"))
                first_annotation = False
        output.write('],"categories":')
        output.write(_json(CATEGORIES))
        output.write("}\n")

    os.replace(json_part, final_path)
    annotation_part.unlink()
    print(
        f"[{destination_split}] complete: images={image_count:,}, "
        f"annotations={annotation_count:,}, coco={final_path}",
        flush=True,
    )
    return image_count, annotation_count


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.max_images_per_split is not None and args.max_images_per_split < 1:
        raise SystemExit("--max-images-per-split must be positive")
    if args.progress_every < 1:
        raise SystemExit("--progress-every must be positive")

    from datasets import Image as HuggingFaceImage
    from datasets import load_dataset

    load_options: dict[str, Any] = {
        "streaming": args.streaming,
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
    }
    if args.revision:
        load_options["revision"] = args.revision

    for source_split, destination_split in SPLITS.items():
        completed = args.output_dir / "annotations" / f"{destination_split}.json"
        if args.resume and completed.is_file():
            print(f"[{destination_split}] already complete, skipping: {completed}")
            continue
        rows = load_dataset(
            args.dataset_id,
            split=source_split,
            **load_options,
        )
        # Preserve encoded JPEG bytes where possible and avoid eager pixel decoding.
        rows = rows.cast_column("image", HuggingFaceImage(decode=False))
        export_split(
            rows,
            args.output_dir,
            destination_split,
            dataset_id=args.dataset_id,
            revision=args.revision,
            resume=args.resume,
            max_images=args.max_images_per_split,
            progress_every=args.progress_every,
        )


if __name__ == "__main__":
    main()

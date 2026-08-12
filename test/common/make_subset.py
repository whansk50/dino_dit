"""
PubLayNet COCO json 유틸리티. 두 가지 용도:

1. `subset` - 스모크 테스트용으로 처음 N장 이미지 + 그 annotation만 담은 작은 json을 만든다.
   (baseline_dit_cascade / dit_dino 양쪽 모두 같은 subset json을 가리키게 해서 50 iter 학습이
   crash 없이 도는지 빠르게 확인하는 용도. 계획서 §7 Step 1-2)

2. `check` - 두 COCO json이 같은 이미지 집합·category id(1~5)를 쓰는지 assert한다.
   본 벤치마크는 두 실험이 동일한 data/publaynet/val.json 하나를 공유하도록 구성했지만,
   복사본을 따로 두거나 나중에 데이터가 갱신됐을 때 두 실험이 실제로 같은 GT를 보고 있는지
   재확인하는 용도로 쓴다.

pycocotools/json만 다루고 detectron2·DINO를 import하지 않는다.
"""
import argparse
import json

EXPECTED_CATEGORY_IDS = {1, 2, 3, 4, 5}


def make_subset(src_json, dst_json, n):
    with open(src_json) as f:
        coco = json.load(f)

    images = coco["images"][:n]
    image_ids = {img["id"] for img in images}
    annotations = [ann for ann in coco["annotations"] if ann["image_id"] in image_ids]

    subset = {
        "images": images,
        "annotations": annotations,
        "categories": coco["categories"],
    }
    with open(dst_json, "w") as f:
        json.dump(subset, f)

    print(f"{src_json} ({len(coco['images'])} images) -> {dst_json}: "
          f"{len(images)} images, {len(annotations)} annotations")


def check_consistency(json_a, json_b):
    with open(json_a) as f:
        coco_a = json.load(f)
    with open(json_b) as f:
        coco_b = json.load(f)

    ids_a = {img["id"] for img in coco_a["images"]}
    ids_b = {img["id"] for img in coco_b["images"]}
    assert ids_a == ids_b, (
        f"image_id 집합이 다르다: {json_a}에만 있는 것 {len(ids_a - ids_b)}개, "
        f"{json_b}에만 있는 것 {len(ids_b - ids_a)}개"
    )
    assert len(coco_a["images"]) == len(coco_b["images"]), "이미지 수가 다르다"

    for name, coco in [(json_a, coco_a), (json_b, coco_b)]:
        cat_ids = {cat["id"] for cat in coco["categories"]}
        assert cat_ids == EXPECTED_CATEGORY_IDS, (
            f"{name}의 category id가 PubLayNet 5-class(1~5)와 다르다: {sorted(cat_ids)}"
        )

    print(f"OK: {json_a} <-> {json_b} - {len(ids_a)} images, category id {sorted(EXPECTED_CATEGORY_IDS)} 일치")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_subset = sub.add_parser("subset", help="N장짜리 스모크 테스트용 subset json 생성")
    p_subset.add_argument("--src", required=True, help="원본 json (예: ../data/publaynet/train.json)")
    p_subset.add_argument("--dst", required=True, help="생성할 subset json 경로")
    p_subset.add_argument("--n", type=int, default=100, help="포함할 이미지 수 (기본 100)")

    p_check = sub.add_parser("check", help="두 COCO json의 이미지/카테고리 일치 여부 확인")
    p_check.add_argument("json_a")
    p_check.add_argument("json_b")

    args = parser.parse_args()
    if args.cmd == "subset":
        make_subset(args.src, args.dst, args.n)
    elif args.cmd == "check":
        check_consistency(args.json_a, args.json_b)


if __name__ == "__main__":
    main()

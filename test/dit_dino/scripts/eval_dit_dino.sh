#!/usr/bin/env bash
# DiT + DINO 체크포인트 평가 (계획서 7절 Step 4).
# 실행 위치: test/dit_dino/ (예: bash scripts/eval_dit_dino.sh ../data/publaynet logs/dit_dino/checkpoint.pth)
set -e

publaynet_root=${1:-../data/publaynet}
checkpoint=${2:?"사용법: eval_dit_dino.sh <publaynet_root> <checkpoint.pth> [output_dir]"}
output_dir=${3:-logs/dit_dino_eval}

python main.py \
	--output_dir "$output_dir" -c config/DINO/DINO_4scale_dit.py \
	--dataset_file publaynet --coco_path "$publaynet_root" \
	--eval --resume "$checkpoint" \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

#!/usr/bin/env bash
# DiT + DINO PubLayNet 학습 (계획서 7절 Step 3).
# 실행 위치: test/dit_dino/ (예: bash scripts/train_dit_dino.sh)
set -e

publaynet_root=${1:-../data/publaynet}
output_dir=${2:-logs/dit_dino}

python main.py \
	--output_dir "$output_dir" -c config/DINO/DINO_4scale_dit.py \
	--dataset_file publaynet --coco_path "$publaynet_root" --amp \
	--options dn_scalar=100 embed_init_tgt=TRUE \
	dn_label_coef=1.0 dn_bbox_coef=1.0 use_ema=False \
	dn_box_noise_scale=1.0

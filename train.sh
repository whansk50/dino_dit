#!/usr/bin/env bash
set -euo pipefail

TRAIN_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TRAIN_SCRIPT_DIR"

nohup python train.py --devices 2 --config configs/dino_train.yaml > train.log 2>&1 &

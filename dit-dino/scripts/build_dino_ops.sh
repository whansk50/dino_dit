#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_dir}/src/dit_layout_bench/_vendor/dino/models/dino/ops"
python setup.py build_ext --inplace

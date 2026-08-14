# DiTLayoutBench workflow

## 1. Repository layout

```text
sample/
├── DINO/                         # official IDEA DINO source
├── dit/                          # source/checkpoint reference only
└── dit-layout-bench/
    ├── configs/dino_publaynet.py
    ├── src/dit_layout_bench/
    │   ├── backends/
    │   │   ├── cascade_rcnn.py
    │   │   └── dino.py
    │   ├── models/
    │   │   ├── dit.py
    │   │   └── pyramid.py
    │   ├── checkpoint.py
    │   ├── cli.py
    │   ├── config.py
    │   ├── data.py
    │   └── spec.py
    ├── tests/
    ├── train.py
    ├── evaluate.py
    └── inference.py
```

The package discovers the sibling `DINO/` directory from its own location, so
commands do not depend on the caller's current directory after installation.

## 2. Environment

Use Python 3.9–3.11. The supplied upstream repositories target older PyTorch,
torchvision and CUDA combinations; choose mutually compatible versions rather
than installing the newest packages independently.

```bash
cd dit-layout-bench
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Install Detectron2 using its official instructions for the selected PyTorch and
CUDA versions. Then compile DINO multi-scale deformable attention:

```bash
cd ../DINO/models/dino/ops
python setup.py build install
python test.py
cd ../../../../dit-layout-bench
```

Confirm imports before training:

```bash
python -c "import torch, torchvision, detectron2, pycocotools"
```

## 3. PubLayNet data

The required layout is:

```text
publaynet/
├── train/
│   └── *.jpg
├── val/
│   └── *.jpg
└── annotations/
    ├── train.json
    └── val.json
```

The annotation category IDs must be exactly:

| ID | Name |
|---:|---|
| 1 | Text |
| 2 | Title |
| 3 | List |
| 4 | Table |
| 5 | Figure |

Both backends run the same structural validation before training or evaluation.

## 4. DiT checkpoint

Download the self-supervised `DiT-base` patch16 checkpoint and keep it outside
version control, for example:

```text
weights/dit-base-224-p16-500k-62d53a.pth
```

The loader records the absolute path, SHA-256, loaded keys, missing keys,
unexpected keys and shape mismatches. It refuses to continue if any expected
`blocks.*` transformer parameter is absent. Absolute position embeddings are
bicubically resized when the configured pretraining grid differs.

## 5. Common input and feature flow

```text
RGB image
  → random horizontal flip (train only)
  → short edge sampled from 480..800, long edge ≤ 1333
  → tensor normalization: mean=0.5, std=0.5
  → bottom/right padding to a multiple of 32
  → DiT-base transformer taps: layers 3, 5, 7, 11
  → shared pyramid: p2/p3/p4/p5 at stride 4/8/16/32
  → selected detector
```

The DINO adapter propagates a boolean padding mask for every pyramid level.
Detectron2 owns batch padding for Cascade R-CNN; the backbone adds any remaining
stride padding internally.

## 6. Training

Cascade R-CNN:

```bash
python train.py \
  --detector cascade_rcnn \
  --data-root /data/publaynet \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth \
  --output-dir outputs/cascade \
  --batch-size 2 \
  --epochs 12 \
  --amp
```

DINO:

```bash
python train.py \
  --detector dino \
  --data-root /data/publaynet \
  --pretrained /weights/dit-base-224-p16-500k-62d53a.pth \
  --output-dir outputs/dino \
  --batch-size 2 \
  --epochs 12 \
  --amp
```

Both use detector/pyramid LR `1e-4` and pretrained DiT encoder LR `1e-5`.
Cascade R-CNN translates
epochs to Detectron2 iterations from the annotation image count. DINO uses its
native epoch loop. For a controlled comparison, keep device count, effective
batch size, precision, seed and epoch count equal.

Resume with `--resume PATH`. Cascade checkpoints are normally named
`model_*.pth`; DINO writes `checkpoint.pth` and periodic checkpoints.

## 7. Evaluation

```bash
python evaluate.py \
  --detector dino \
  --data-root /data/publaynet \
  --resume outputs/dino/checkpoint.pth \
  --output-dir outputs/dino-eval
```

Replace the detector and checkpoint for Cascade R-CNN. Evaluation uses COCO bbox
metrics. Record AP, AP50, AP75, APs, APm, APl and per-class AP for Text, Title,
List, Table and Figure.

## 8. Inference

```bash
python inference.py \
  --detector dino \
  --resume outputs/dino/checkpoint.pth \
  --data-root /data/publaynet \
  --image page.jpg \
  --score-threshold 0.5 \
  --json-output prediction.json
```

Each JSON record contains `image_id`, `box_xyxy`, `score`, and the original
PubLayNet `category_id` in the range 1–5.

## 9. Verification

Dependency-free source and contract checks:

```bash
python -m unittest discover -s tests
```

Full tests in the training environment:

```bash
python -m pytest
python -m compileall src
```

Before a full run, use a tiny PubLayNet subset and verify:

1. DiT transformer keys load without missing `blocks.*` parameters.
2. `p2/p3/p4/p5` shapes match strides `4/8/16/32` for rectangular images.
3. Both models complete forward, backward and one optimizer step.
4. Loss decreases in a small-subset overfit run.
5. Saved checkpoints resume and produce COCO evaluation output.
6. Predicted boxes map back to original image coordinates.

## 10. Benchmark reporting

For every result record the git revision, complete config, checkpoint SHA-256,
dependency versions, GPU, precision, batch size and seed. Measure preprocessing,
model, postprocessing and total latency separately after warm-up. Also record
training/inference peak VRAM and backbone/detector/total parameter counts.

## 11. License boundary

The runtime deliberately avoids the MPViT-derived files in the sibling DiT
object-detection tree. See `THIRD_PARTY_NOTICES.md` for the source inventory and
distribution checklist. Preserve the upstream DINO Apache-2.0 license when
redistributing its source or binaries.

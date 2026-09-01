# Third-party notices

| Component | Use | License / location |
|---|---|---|
| Microsoft UniLM / DiT | Architecture and checkpoint-compatible names | MIT; preserve the upstream checkpoint/source notice |
| IDEA-Research DINO | Detector, training loop, deformable-attention extension | Apache-2.0; `src/dit_layout_bench/_vendor/dino/LICENSE` |
| Detectron2 | Cascade R-CNN backend | Apache-2.0 |
| PyTorch / torchvision | Tensor and image operations | BSD-style licenses |
| COCO API | Evaluation | BSD-2-Clause |
| MLflow | Experiment tracking | Apache-2.0 |

`src/dit_layout_bench/models/dit.py` and `pyramid.py` were implemented with
standard PyTorch operations against the published DiT/BEiT architecture and
checkpoint shape contract. They do not copy or import the MPViT-derived files
under the workspace's `dit/object_detection/ditod/` tree.

The vendored DINO source is intentionally kept separate and retains its full
Apache-2.0 license. Local modifications are limited to the integration callback
documented in `_vendor/dino/engine.py` and Torch 2.10 op compatibility; the adapter lives in this project's own
source tree. This notice is engineering documentation, not legal advice.

# Third-party notices

DiTLayoutBench connects to source trees supplied alongside this project. Those
projects retain their own copyright and license terms.

| Component | Use | License |
|---|---|---|
| Microsoft UniLM / DiT | Architecture, checkpoint format and pretrained weights | MIT; verify checkpoint terms at download source |
| IDEA-Research DINO | Detector, training loop and deformable-attention extension | Apache-2.0 |
| Detectron2 | Cascade R-CNN backend | Apache-2.0 |
| PyTorch / torchvision | Tensor operations and image transforms | BSD-style licenses |
| COCO API | PubLayNet evaluation | BSD-2-Clause |

The runtime does not import `dit/object_detection/ditod/backbone.py`,
`dit/object_detection/train_net.py`, or `dit/object_detection/ditod/__init__.py`.
Those files carry MPViT provenance and are outside this package. The shared DiT
encoder and pyramid in `src/dit_layout_bench/models/` use standard PyTorch
operations and were implemented for this package.

Before publishing a combined repository or binary distribution, preserve the
complete license/notice files of every bundled dependency and have the final
source inventory reviewed. This notice is engineering documentation, not legal
advice.


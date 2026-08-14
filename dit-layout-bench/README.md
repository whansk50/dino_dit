# DiTLayoutBench

DiTLayoutBench compares two PubLayNet document-layout detectors while keeping a
shared DiT-base backbone, input policy, label contract and evaluation dataset.

```text
DiT-base → shared 4/8/16/32 pyramid → Cascade R-CNN
                                    → DINO
```

The detector is selected once at startup:

```bash
python train.py --detector cascade_rcnn ...
python train.py --detector dino ...
```

See [WORKFLOW.md](WORKFLOW.md) for installation, data preparation, training,
evaluation and inference. The original implementation decisions are preserved
in [IMPLEMENTATION_PLAN.md](../IMPLEMENTATION_PLAN.md).

## Design constraints

- Dataset: PubLayNet only, with category IDs 1 through 5.
- Backbone initialization: the same self-supervised DiT-base checkpoint.
- New detector and pyramid parameters start randomly initialized.
- Input normalization is `(x - 0.5) / 0.5` in both backends.
- Training uses the same short-edge resize set and no detector-specific crop.
- Predictions use original-image `xyxy` boxes and PubLayNet category IDs.
- The package does not execute the MPViT-derived wrapper in the supplied
  `dit/object_detection` source tree.

## Status

The implementation includes the shared encoder/pyramid, audited checkpoint
loader, both backend adapters, unified CLIs, dataset validation and source-level
tests. A compatible PyTorch/CUDA environment is required for numerical and GPU
smoke tests; see the verification section in the workflow.


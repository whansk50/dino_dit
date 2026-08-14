"""Consistent pretrained/new-parameter learning-rate groups."""

from __future__ import annotations


def parameter_groups(model, *, detector_lr: float, backbone_lr: float):
    """Use the low LR only for the pretrained DiT encoder.

    The randomly initialized shared pyramid remains in the detector-rate group.
    Both framework adapters name the encoder below ``pyramid.backbone``.
    """
    encoder, detector = [], []
    counts = {"encoder": 0, "detector_and_pyramid": 0}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "pyramid.backbone." in name:
            encoder.append(parameter)
            counts["encoder"] += parameter.numel()
        else:
            detector.append(parameter)
            counts["detector_and_pyramid"] += parameter.numel()
    if not encoder or not detector:
        raise RuntimeError(
            f"Invalid optimizer split: encoder={len(encoder)}, detector={len(detector)}"
        )
    print(
        "Optimizer parameters: "
        f"DiT encoder={counts['encoder']:,} @ {backbone_lr:g}; "
        f"detector+pyramid={counts['detector_and_pyramid']:,} @ {detector_lr:g}"
    )
    return [
        {"params": detector, "lr": detector_lr},
        {"params": encoder, "lr": backbone_lr},
    ]


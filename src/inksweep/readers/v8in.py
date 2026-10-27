"""v8-in (Youssef Nader's 8 µm reader, huggingface.co/YoussefMoNader/ink-8um-v8in, MIT).

Reference: ``ink8um.inference.predict_stack`` at revision ``d89166b4`` with one fix:

* window: 24 layers, ``s = L//2 + d - 12`` (layers ``d - 11.5 .. d + 11.5``); ``reverse`` flips the depth axis;
* intensities clipped to 0..200, then ``* float32(1/255)``;
* coverage = window maximum > 0, closed with a 7x7 ellipse; only 64 px tiles lying entirely inside it, stride 21
  (no edge-aligned tile);
* fp16 autocast on CUDA, sigmoid on the fp16 logits, bilinear resize to the tile if needed;
* blending with ``gkern(64, 1) / max``. **Stitching fix**: the released code divides the weighted sum by the number
  of overlapping tiles (``count += 1``), leaving a 21 px grid of +-10-19 % on every map; inksweep divides by the summed
  weight (``count += blend``), the one-line patch of EconLearn/vesuvius-first-letters-stage-loss
  (``patches/v8in-stitching-normalise-by-blend-weight.patch``, MIT). ``stitch="count"`` reproduces the released maps.

The model class (``ink8um.modeling.InkDetector``) and weights are fetched from the model repository at the pinned
revision and checked against their sha256 before import; nothing of them is copied into inksweep.
"""
from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoints import V8IN_FILES
from .spec import ReaderSpec

CLIP_MAX = 200


def _load(paths: Any, device: torch.device | str) -> nn.Module:
    """``paths``: the fetched v8-in files (all in one folder); loads ``InkDetector`` from that folder."""
    folder = Path(str(list(paths)[0])).parent
    if (folder / "config.json").exists() is False:
        raise FileNotFoundError(f"{folder}: config.json missing")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    from ink8um import InkDetector  # noqa: E402  (pinned model code)

    model = InkDetector.from_pretrained(str(folder))
    if (model.in_depth, model.tile_size, model.stride, model.clip_max) != (24, 64, 21, CLIP_MAX):
        raise ValueError("unexpected v8-in configuration")
    return model.eval().to(device)


def _preprocess(stack: torch.Tensor) -> torch.Tensor:
    """uint8 ``[B, 24, 64, 64]`` -> float32, clipped to 0..200 and scaled by float32(1/255)."""
    if stack.ndim == 3:
        stack = stack[None]
    return torch.clamp(stack, 0, CLIP_MAX).to(torch.float32).mul_(1.0 / 255.0)


def _forward(module: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return module(batch)


def _postprocess(logits: torch.Tensor, *, tile: int) -> torch.Tensor:
    p = torch.sigmoid(logits).float()           # sigmoid in the logits' dtype (fp16 under autocast), as released
    if tuple(p.shape[-2:]) != (tile, tile):
        p = F.interpolate(p, size=(tile, tile), mode="bilinear")
    return p


def v8in_reader(released_stitching: bool = False) -> ReaderSpec:
    name = "v8in_released" if released_stitching else "v8in"
    return ReaderSpec(
        name=name, window_layers=24, voxel_size_um=8.0, tile=64, stride=21,
        load=_load, preprocess=_preprocess, forward=_forward, postprocess=partial(_postprocess, tile=64),
        batched=True, blend="v8in_gaussian", grid="inside_mask", min_weight=0.0, schedule="grid", skip_empty="none",
        stitch="count" if released_stitching else "threshold", zero_empty_columns=False,
        resample_tolerance=0.20, calibration="identity", reference_storage=None, amp_dtype=torch.float16,
        conv_tf32=None, matmul_precision=None, reference_batch=16, checkpoints=V8IN_FILES, family="v8in",
        citation="Y. Nader, v8-in (huggingface.co/YoussefMoNader/ink-8um-v8in), MIT", licence="MIT",
        notes=("released stitching (divide by tile count)" if released_stitching
               else "stitching fixed: divide by the summed blend weight (EconLearn patch)"),
    )


def builtin_readers() -> dict[str, ReaderSpec]:
    return {"v8in": v8in_reader(False), "v8in_released": v8in_reader(True)}

"""Hecate (the organisers' resolution-distilled reader), 9.6 µm and 2.4 µm checkpoints.

Reference: ``hecate.py`` released with the weights (huggingface.co/scrollprize/hecate, MIT, Copyright (c) 2026
Vesuvius Challenge), called by the PHerc. 343 recipe as ``hecate.py --spacing-um 9.6 --batch-size 4 [--reverse]`` on a
``hecate_prep.py`` input:

* input: the render resampled to 9.6 µm in x, y and z (16 planes centred on layer ``(L-1)/2 + d``; done by the engine,
  :mod:`inksweep.resample`); ``--reverse`` reverses the 16 planes;
* tiles 64 px, stride 32, sliding grid with a final edge-aligned tile; all-zero tiles skipped;
* input ``uint8 / 255`` (float32 division, no clipping); the model also receives the per-pixel support mask (any
  non-zero plane), and empty CT columns get probability 0;
* float32 (no autocast), PyTorch's default TF32 policy for convolutions;
* blending: Hann window floored at 0.001, ``sum(p w) / sum(w)``;
* storage: ``round(255 p)`` PNG on the 9.6 µm grid; the ensemble resizes ``v / 255`` to the render grid bilinearly.

The model class is the released ``hecate.py`` itself, fetched at a pinned revision and checked against its sha256
before it is imported; nothing of it is copied into inksweep.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .spec import ReaderSpec

CODE_SHA256 = "c232c18a1a86cfb91257a00db13288202bb8ae33732ba26f7291a9b8adb8da59"


def import_hecate(code_path: str | os.PathLike) -> Any:
    """Import the released ``hecate.py`` (after checking its sha256) as module ``hecate_released``."""
    from .checkpoints import sha256_file

    code_path = Path(code_path)
    got = sha256_file(code_path)
    if got != CODE_SHA256:
        raise RuntimeError(f"{code_path}: sha256 {got} != pinned {CODE_SHA256}")
    name = "hecate_released"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(code_path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module


def _load(paths: Any, device: torch.device | str, *, sampling_um: float) -> nn.Module:
    """``paths`` = (checkpoint, hecate.py). Strict load of the released EMA checkpoint."""
    ckpt, code = (str(p) for p in paths)
    hecate = import_hecate(code)
    model = hecate.load_model(ckpt, str(device))
    if abs(float(model.sampling_um) - sampling_um) > 1e-6:
        raise ValueError(f"{ckpt}: checkpoint samples {model.sampling_um} µm, reader expects {sampling_um} µm")
    return model


def division_table(divisor: float) -> torch.Tensor:
    """``v / divisor`` for every uint8 value ``v`` as numpy computes it (true float32 division).

    On CUDA, ``tensor / python_scalar`` is evaluated as ``tensor * (1 / scalar)``, which differs from the reference's
    numpy division by one ulp for some values; indexing this table reproduces the reference exactly.
    """
    return torch.from_numpy(np.arange(256, dtype=np.float32) / np.float32(divisor))


def _preprocess(stack: torch.Tensor, *, divisor: float, table: torch.Tensor) -> torch.Tensor:
    """uint8 ``[B, Z, p, p]`` -> float32 ``[B, 2, Z, p, p]``: channel 0 the image (``/ divisor``), 1 the support."""
    if stack.ndim == 3:
        stack = stack[None]
    if table.device != stack.device:
        table = table.to(stack.device)
    image = table[stack.long()]
    support = stack.any(1, keepdim=True).expand_as(stack)        # [B, Z, p, p], per-pixel support on every plane
    return torch.stack([image, support.to(torch.float32)], dim=1)


def _forward(module: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    image = batch[:, :1]
    valid = batch[:, 1:2] > 0.5
    return module(image, valid)


def _postprocess(logits: torch.Tensor) -> torch.Tensor:
    return logits.float().sigmoid()


def hecate_reader(sampling_um: float = 9.6) -> ReaderSpec:
    if sampling_um == 9.6:
        name, key, depth, tile, divisor = "hecate", "hecate_9.6um", 16, 64, 255.0
    elif sampling_um == 2.4:
        name, key, depth, tile, divisor = "hecate_2.4", "hecate_2.4um", 64, 256, 200.0
    else:
        raise ValueError("Hecate checkpoints exist for 9.6 and 2.4 µm only")
    return ReaderSpec(
        name=name, window_layers=depth, voxel_size_um=sampling_um, tile=tile, stride=tile // 2,
        load=partial(_load, sampling_um=sampling_um),
        preprocess=partial(_preprocess, divisor=divisor, table=division_table(divisor)),
        forward=_forward, postprocess=_postprocess,
        batched=True, blend="hann", blend_floor=0.001, grid="sliding", min_weight=0.0, stitch="threshold",
        schedule="grid", skip_empty="tile", zero_empty_columns=True, resample_tolerance=0.02, calibration="identity", reference_storage="u8_round",
        amp_dtype=None, conv_tf32=None, matmul_precision=None, reference_batch=4,
        checkpoints=(key, "hecate_code"), family="hecate",
        citation="Vesuvius Challenge, Hecate (huggingface.co/scrollprize/hecate), MIT", licence="MIT",
        notes="input resampled to the model's sampling in all three axes (hecate_prep.py rule)",
        options=dict(divisor=divisor, depth_mean_recommended=[-1, 0, 1] if sampling_um == 9.6 else None),
    )


def builtin_readers() -> dict[str, ReaderSpec]:
    return {"hecate": hecate_reader(9.6), "hecate_2.4": hecate_reader(2.4)}

"""The organisers' canonical 2 µm reader (ResNet-152 3D encoder + 3D decoder, ``r152_3ddec_v2_l5_epoch13``) for
2.4 µm surface volumes (huggingface.co/scrollprize/ink_canonical_2um, MIT).

Reference: ScrollPrize/villa ``ink-detection/optimized_inference`` (``inference.py``, ``processing.py``, MIT), with the
settings the community uses for this checkpoint (``TILE_SIZE=256 STRIDE=128``, a **centred** 62-layer window, villa
#1765 / #1766):

* window: 62 layers, ``s = L//2 + d - 31`` (``START_LAYER = 23`` for a 109-layer render at ``d = 0``);
  ``FORCE_REVERSE`` flips the window;
* per-pixel validity = any non-zero layer (before clipping); intensities clipped to 0..200, then ``/ 200``;
* tiles 256 px, stride 128, sliding grid with a final edge-aligned tile;
* fp16 autocast, sigmoid, bilinear resize of the 64 px output to 256 px;
* blending: ``outer(np.hanning(256), np.hanning(256)) / sum``, multiplied by validity in the sum and the weight;
  result ``sum / max(weight, 1e-6)``, clipped, stored as ``trunc(255 p)``.

The network is villa's own ``model_resnet3d_3d_decoder.RegressionModel`` (imported from the villa checkout installed
by ``node/setup_inksweep.sh``; set ``INKSWEEP_VILLA_DIR`` to use another checkout).
"""
from __future__ import annotations

import os
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .spec import ReaderSpec
from .villa import load_torch

CLIP_MAX = 200.0


def villa_root() -> Path:
    """The villa checkout: ``$INKSWEEP_VILLA_DIR``, else the one the ``vesuvius`` package was installed from."""
    env = os.environ.get("INKSWEEP_VILLA_DIR")
    if env:
        return Path(env)
    import vesuvius

    return Path(vesuvius.__file__).resolve().parents[3]


def _import_decoder_module() -> Any:
    folder = villa_root() / "ink-detection" / "optimized_inference"
    if not (folder / "model_resnet3d_3d_decoder.py").is_file():
        raise FileNotFoundError(f"{folder}: villa optimized_inference not found (sparse checkout needs ink-detection)")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    import model_resnet3d_3d_decoder as module  # noqa: E402

    return module


def _load(paths: Any, device: torch.device | str) -> nn.Module:
    """villa ``model_resnet3d_3d_decoder.load_model`` without its DataParallel wrapping, missing weights fatal."""
    path = str(list(paths)[0]) if not isinstance(paths, (str, os.PathLike)) else str(paths)
    mod = _import_decoder_module()
    checkpoint = load_torch(path)
    state = mod._strip_known_prefixes(mod._extract_state_dict(checkpoint))
    with_norm = any(k.startswith("normalization.") for k in state)
    model = mod.RegressionModel(with_norm=with_norm)
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise ValueError(f"{path}: {len(result.missing_keys)} missing weights, e.g. {result.missing_keys[:3]}")
    return model.to(device).eval()


_TABLE = torch.from_numpy(np.minimum(np.arange(256), 200).astype(np.float32) / np.float32(CLIP_MAX))


def _preprocess(stack: torch.Tensor) -> torch.Tensor:
    """uint8 ``[B, 62, p, p]`` -> float32 ``[B, 1, 62, p, p]`` = ``clip(x, 0, 200) / 200`` (true division, as the
    reference's numpy; a table lookup, because CUDA divides by a Python scalar via its reciprocal)."""
    if stack.ndim == 3:
        stack = stack[None]
    return _TABLE.to(stack.device)[stack.long()][:, None]


def _forward(module: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return module(batch)


def _postprocess(logits: torch.Tensor, *, tile: int) -> torch.Tensor:
    p = torch.sigmoid(logits)
    return F.interpolate(p.float(), size=(tile, tile), mode="bilinear", align_corners=False)


def canonical_reader(window: int = 62, tile: int = 256, stride: int = 128) -> ReaderSpec:
    return ReaderSpec(
        name="canonical_2um", window_layers=window, voxel_size_um=2.4, tile=tile, stride=stride,
        load=_load, preprocess=_preprocess, forward=_forward, postprocess=partial(_postprocess, tile=tile),
        batched=True, blend="hann_sum1", grid="sliding", min_weight=1e-6, stitch="clamp", schedule="grid",
        skip_empty="batch", zero_empty_columns=True,
        resample_tolerance=0.10, calibration="identity", reference_storage="u8_floor", amp_dtype=torch.float16,
        conv_tf32=None, matmul_precision=None, reference_batch=8, checkpoints=("canonical_2um",),
        family="canonical", citation="Vesuvius Challenge, ink_canonical_2um (huggingface.co/scrollprize/"
                                     "ink_canonical_2um), MIT",
        licence="MIT", notes="centred 62-layer window (villa #1765/#1766), 256 px tiles, stride 128",
    )


def builtin_readers() -> dict[str, ReaderSpec]:
    return {"canonical_2um": canonical_reader()}

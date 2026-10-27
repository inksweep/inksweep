"""Readers built on the organisers' flat ink models ("villa family"): ink_9um, dense_native, the seed-42 soup, Reader
v2, Reader V3 and d9v4C. All share one architecture family (a local 3D stem feeding a 2D U-Net, ``hybrid_3d2d``) and
one inference recipe, villa ``vesuvius.ink_detection.inference.infer`` as the PHerc. 343 recipe calls it
(``--overlap 0.5 --blend-mode hann``):

* window: ``crop_z`` (17) layers, ``s = L//2 + d - 8``; ``--direction reverse`` feeds them in reverse order;
* tiles: ``crop_y`` px (128; 384 for Reader V3), stride ``round(tile * 0.5)``, sliding grid with a final edge-aligned
  tile; a tile whose raw window is all zero is skipped;
* normalisation: per tile, robust median / MAD over the whole ``17 x tile x tile`` patch (:mod:`.norm`);
* forward: autocast with the checkpoint's training precision (``--amp-dtype auto``: fp16 for every released
  checkpoint), sigmoid, bilinear resize to the tile if needed;
* blending: Hann window floored at 0.001, ``sum(p w) / sum(w)`` where ``sum(w) > 1e-6``;
* storage: ``trunc(255 p)`` as uint8; the ensemble reads ``clip((v - 64) / 128, 0, 1)``, the model-card rescale
  ``(p - 0.25) / 0.5`` of the label-smoothed models.

The network is built by villa's own code (``vesuvius.ink_detection.models.model.make_model``, MIT), which must be
importable; ``node/setup_inksweep.sh`` installs ScrollPrize/villa at the commit the PHerc. 343 maps were made with.
:func:`flat_preprocessing` and the checkpoint-ensemble arithmetic transcribe villa's ``infer.py`` (MIT, Copyright (c)
2024 Vesuvius Challenge).
"""
from __future__ import annotations

import json
import logging
import os
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .norm import RobustMadGPU
from .spec import ReaderSpec

LOG = logging.getLogger(__name__)
VILLA_COMMIT = "0fb38c45b488e192566ea83ae36369cb6db92652"     # the commit of every PHerc. 343 ink map


# ----------------------------------------------------------------------------------------------- loading
def load_torch(path: str | os.PathLike, trusted: bool = True) -> Any:
    """``torch.load`` on CPU, ``weights_only=True`` first; pickled extras only for trusted (sha256-pinned) files."""
    try:
        return torch.load(str(path), map_location="cpu", weights_only=True)
    except Exception as exc:
        if not trusted and os.environ.get("INKSWEEP_ALLOW_PICKLE") != "1":
            raise RuntimeError(f"{path}: needs a full pickle load; set INKSWEEP_ALLOW_PICKLE=1 if you trust it") from exc
        LOG.info("%s: weights_only load failed (%s); loading the pinned file with weights_only=False", path, exc)
        return torch.load(str(path), map_location="cpu", weights_only=False)


def payload_from_files(paths: Sequence[str | os.PathLike], kind: str) -> dict:
    """Assemble villa's checkpoint payload ``{'model', 'config', ...}`` from the fetched files of a reader.

    ``kind``: ``"pth"`` (one villa checkpoint), ``"safetensors+json"`` (weights + training config, the dense_native
    release; this is the in-memory equivalent of the PHerc. 343 ``rebuild_dense_native_pth.py``) or
    ``"safetensors+pth"`` (weights + a checkpoint whose config is used, the seed-42 soup).
    """
    if kind == "pth":
        return load_torch(paths[0])
    from safetensors.torch import load_file

    state = load_file(str(paths[0]))
    if kind == "safetensors+json":
        config = json.loads(Path(paths[1]).read_text())
        return {"model": state, "config": config, "step": 16000}
    if kind == "safetensors+pth":
        config = load_torch(paths[1])["config"]
        return {"model": state, "config": config}
    raise ValueError(f"unknown payload kind {kind!r}")


def flat_preprocessing(norm_config: Any) -> str:
    """villa ``flat_preprocessing_from_config``: robust_mad (1, 99) -> ``tifxyz_robust``; divide 255 -> ``divide_255``."""
    mode = norm_config.mode
    if mode == "robust_mad":
        if (norm_config.percentile_lower, norm_config.percentile_upper) != (1.0, 99.0):
            raise ValueError("flat inference supports robust_mad only with percentiles (1, 99)")
        return "tifxyz_robust"
    if mode == "divide":
        if norm_config.divisor != 255.0:
            raise ValueError("flat divide preprocessing requires divisor 255")
        return "divide_255"
    raise ValueError(f"flat inference does not support image_normalization mode {mode!r}")


def build_villa_model(payload: Mapping[str, Any], source: str) -> tuple[nn.Module, dict]:
    """Rebuild one flat villa model exactly as villa ``infer.configure_model`` does; returns (model, info)."""
    from vesuvius.ink_detection.config import InkConfig
    from vesuvius.ink_detection.inference.inference_runtime import TargetModel, resolve_amp_dtype
    from vesuvius.ink_detection.models.checkpoint import select_inference_weights
    from vesuvius.ink_detection.models.model import make_model

    if not isinstance(payload, Mapping) or not isinstance(payload.get("config"), Mapping):
        raise ValueError(f"{source}: a villa checkpoint needs a 'config' mapping")
    config = InkConfig.from_mapping(payload["config"])
    if config.data.mode != "flat":
        raise ValueError(f"{source}: flat inference requires mode 'flat', got {config.data.mode!r}")
    selected, state = select_inference_weights(payload, source=source)
    base = make_model(config)
    if state and all(str(k).startswith("module.") for k in state):
        state = {str(k)[len("module."):]: v for k, v in state.items()}
    result = base.load_state_dict(state, strict=False)               # villa loads permissively ...
    if result.missing_keys:                                          # ... but random weights must never pass
        raise ValueError(f"{source}: {len(result.missing_keys)} missing weights, e.g. {result.missing_keys[:3]}")
    if result.unexpected_keys:
        LOG.warning("%s: %d unexpected keys ignored (as villa does)", source, len(result.unexpected_keys))
    base.eval()
    crop_z, crop_y, crop_x = config.model.crop_size
    if crop_y != crop_x:
        raise ValueError(f"{source}: flat inference needs square patches, got {config.model.crop_size}")
    model = TargetModel(base, input_pad_depth_to=config.model.input_pad_depth_to).eval()
    info = dict(selected_state=selected, crop_size=[int(crop_z), int(crop_y), int(crop_x)],
                preprocessing=flat_preprocessing(config.data.normalization),
                amp_dtype=resolve_amp_dtype("auto", payload, source),
                unexpected_keys=len(result.unexpected_keys))
    return model, info


class ProbabilityMeanEnsemble(nn.Module):
    """Checkpoint ensemble of villa ``--ensemble-checkpoint``: ``logit(mean(sigmoid(member)))`` (eps 1e-6)."""

    def __init__(self, members: Sequence[nn.Module], eps: float = 1e-6) -> None:
        super().__init__()
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        self.members = nn.ModuleList(members)
        self.eps = float(eps)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        total: Optional[torch.Tensor] = None
        for member in self.members:
            p = member(image).float().sigmoid()
            total = p if total is None else total + p
        assert total is not None
        return torch.logit(total / float(len(self.members)), eps=self.eps)


def _load(paths: Any, device: torch.device | str, *, kind: str, members: int, expect: dict) -> nn.Module:
    """ReaderSpec.load for the villa family. ``paths``: the fetched files in the order of ``spec.checkpoints``."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    paths = [str(p) for p in paths]
    per = len(paths) // members
    models, infos = [], []
    for m in range(members):
        group = paths[m * per:(m + 1) * per]
        model, info = build_villa_model(payload_from_files(group, kind), group[0])
        models.append(model)
        infos.append(info)
    for info in infos:
        for key, want in expect.items():
            if info[key] != want:
                raise ValueError(f"checkpoint {key}={info[key]!r}, the reader expects {want!r}")
    if len({str(i["amp_dtype"]) for i in infos}) != 1:
        raise ValueError("ensemble members were trained with different mixed precision")
    module = models[0] if members == 1 else ProbabilityMeanEnsemble(models)
    module.inksweep_info = infos  # type: ignore[attr-defined]
    # villa --amp-dtype auto: the autocast dtype comes from the checkpoint's training config; with none, villa still
    # enters torch.autocast("cuda", dtype=None), i.e. PyTorch's default CUDA autocast dtype, float16
    module.inksweep_amp_dtype = infos[0]["amp_dtype"] or torch.float16  # type: ignore[attr-defined]
    return module.to(device).eval()


# ----------------------------------------------------------------------------------------------- per-tile steps
def _preprocess(stack: torch.Tensor, *, normaliser: RobustMadGPU, mode: str) -> torch.Tensor:
    """uint8 ``[B, Z, p, p]`` (model depth order) -> float32 ``[B, 1, Z, p, p]``."""
    if stack.ndim == 3:
        stack = stack[None]
    if mode == "tifxyz_robust":
        x = normaliser(stack)
    elif mode == "divide_255":
        x = stack.to(torch.float32) * (1.0 / 255.0)
    else:
        raise ValueError(mode)
    return x[:, None]


def _forward(module: nn.Module, batch: torch.Tensor, *, tta: bool) -> torch.Tensor:
    """Logits ``[B, 1, h, w]``; with mirror TTA the mean probability over the four Y/X flips instead."""
    if not tta:
        return module(batch)
    hw = tuple(int(v) for v in batch.shape[-2:])
    total = None
    for dims in ((), (3,), (4,), (3, 4)):                 # villa's combination order over the equal Y/X axes
        x = torch.flip(batch, dims=list(dims)) if dims else batch
        p = _to_probability(module(x), hw)
        if dims:
            p = torch.flip(p, dims=[d - 1 for d in dims])
        total = p if total is None else total + p
    return total / 4.0


def _to_probability(logits: torch.Tensor, hw: tuple[int, int]) -> torch.Tensor:
    """villa ``logits_to_probabilities``: float sigmoid, bilinear resize to the patch size if needed."""
    p = logits.float().sigmoid_()
    if tuple(p.shape[-2:]) != tuple(hw):
        p = F.interpolate(p, size=hw, mode="bilinear", align_corners=False)
    return p


def _postprocess(output: torch.Tensor, *, tile: int, tta: bool) -> torch.Tensor:
    return output if tta else _to_probability(output, (tile, tile))


# ----------------------------------------------------------------------------------------------- reader factory
def villa_reader(name: str, checkpoints: Sequence[str], *, kind: str = "pth", members: int = 1, tile: int = 128,
                 depth: int = 17, tta_mirror: bool = False, citation: str = "", notes: str = "",
                 preprocessing: str = "tifxyz_robust") -> ReaderSpec:
    """A ReaderSpec for villa-family checkpoints (see the module docstring for the recipe)."""
    normaliser = RobustMadGPU(depth * tile * tile)
    stride = max(1, int(round(tile * (1.0 - 0.5))))
    expect = dict(crop_size=[depth, tile, tile], preprocessing=preprocessing)
    return ReaderSpec(
        name=name, window_layers=depth, voxel_size_um=9.36, tile=tile, stride=stride,
        load=partial(_load, kind=kind, members=members, expect=expect),
        preprocess=partial(_preprocess, normaliser=normaliser, mode=preprocessing),
        forward=partial(_forward, tta=tta_mirror),
        postprocess=partial(_postprocess, tile=tile, tta=tta_mirror),
        batched=True, blend="hann", blend_floor=0.001, grid="sliding", min_weight=1e-6,
        schedule="occupancy", skip_empty="tile", zero_empty_columns=False, resample_tolerance=0.12, calibration="label_smoothing_0.5",
        reference_storage="u8_floor", amp_dtype=torch.float16, conv_tf32=None, matmul_precision="high",
        reference_batch=4, checkpoints=tuple(checkpoints), family="villa", citation=citation, licence="MIT",
        notes=notes, options=dict(villa_commit=VILLA_COMMIT, payload=kind, members=members, tta_mirror=tta_mirror,
                                  preprocessing=preprocessing),
    )


INK9_CITE = "Vesuvius Challenge, ink_9um (huggingface.co/scrollprize/ink_9um), MIT"


def builtin_readers() -> dict[str, ReaderSpec]:
    """The villa-family readers shipped with inksweep."""
    from .checkpoints import INK9_STEPS

    out: dict[str, ReaderSpec] = {}
    for seed in (42, 43):
        for step in INK9_STEPS:
            key = f"ink9_s{seed}_{step // 1000:03d}k"
            out[key] = villa_reader(key, [key], citation=INK9_CITE)
        out[f"ink9_s{seed}"] = villa_reader(f"ink9_s{seed}", [f"ink9_s{seed}_075k"], citation=INK9_CITE,
                                            notes="step 75 000, as in the PHerc. 343 recipe")
    out["ink9_s43_50-75k"] = villa_reader(
        "ink9_s43_50-75k", ["ink9_s43_050k", "ink9_s43_060k", "ink9_s43_075k"], members=3, citation=INK9_CITE,
        notes="probability mean of seed 43 steps 50k/60k/75k per tile (villa --ensemble-checkpoint)")
    out["ink9_soup42"] = villa_reader(
        "ink9_soup42", ["soup42_last4", "ink9_s42_075k"], kind="safetensors+pth",
        citation="E. Nieuwlaar, soup42_last4 (huggingface.co/Nieuwlaar/ink9um-dense-native), MIT",
        notes="weight average of seed-42 steps 40k-75k; config from the seed-42 75k checkpoint")
    out["dnative"] = villa_reader(
        "dnative", ["dense_native_016000", "dense_native_config"], kind="safetensors+json",
        citation="E. Nieuwlaar, dense_native_016000 (huggingface.co/Nieuwlaar/ink9um-dense-native), MIT",
        notes="public safetensors + training config, wrapped in memory as rebuild_dense_native_pth.py does")
    out["readerv2"] = villa_reader(
        "readerv2", ["readerv2_040000"],
        citation="D. Russo, Reader v2 (huggingface.co/domenicor046/reader-v2), MIT")
    out["readerv3"] = villa_reader(
        "readerv3", ["readerv3"], tile=384,
        citation="D. Russo, Reader V3 (huggingface.co/domenicor046/reader-v3), MIT",
        notes="384 px tiles, stride 192")
    out["d9v4c"] = villa_reader(
        "d9v4c", ["d9v4c_006000"],
        citation="TAUIL Abd Elilah, d9v4C (github.com/TAUIL-Abd-Elilah/9um-reader-letter-shape v1.0), MIT")
    out["d9v4c_seed1"] = villa_reader(
        "d9v4c_seed1", ["d9v4c_seed1_006000"],
        citation="TAUIL Abd Elilah, d9v4C seed 1 (github.com/TAUIL-Abd-Elilah/9um-reader-letter-shape v1.0), MIT")
    return out

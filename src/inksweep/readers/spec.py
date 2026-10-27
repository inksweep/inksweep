"""The reader contract: what an ink model must provide so that the sweep engine can run it.

A *reader* is one ink-detection model plus the exact recipe its authors use to apply it to a surface volume (a stack of
layers rendered along a mesh normal). The engine owns everything that is the same for every model: loading the volume
onto the GPU once, choosing the depth window for each depth ``d``, the tile grid, batching, blending the overlapping
tiles, padding for tile phases, orientation, resampling to the model's voxel size, writing the maps. The reader owns
what differs: the network, its normalisation, its forward pass and the conversion of its output to a probability.

Depth convention (the "surface volume" contract): layer ``i`` of an ``L``-layer stack sits at signed depth
``d_i = i - L // 2`` voxels along the surface normal, positive away from the scroll axis.

Window placement. For depth ``d`` the reader sees ``window_layers = W`` consecutive layers starting at
``s = L // 2 + d - W // 2`` (layers ``s .. s + W - 1``). For odd ``W`` the window is centred on layer ``L // 2 + d``;
for even ``W`` on the midpoint between layers ``L // 2 + d - 1`` and ``L // 2 + d``. This is the rule of the PHerc. 343
recipe (``run_ink.sh``: ``s = N/2 + d - 8`` for the 17-layer villa models). When the render's voxel size differs from
``voxel_size_um`` by more than ``resample_tolerance`` (relative), the engine resamples the window to the reader's voxel
size in all three axes first (see :mod:`inksweep.resample`).

Orientation. ``"forward"`` feeds the window in increasing layer order (outward under the contract convention),
``"reverse"`` in decreasing order. The released 9 µm readers read inward, so on contract renders the *reading side* is
``"reverse"``; the other side is the *blind* control.

Only the first nine fields are the plugin contract (``READER: ReaderSpec`` in a plugin module). Every other field
is an optional extension with a default that keeps a minimal plugin working; the built-in adapters set them to
reproduce their reference implementations exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Optional, Sequence

import torch

#: Blend windows understood by the engine (see :mod:`inksweep.tiling`).
BLEND_MODES = ("hann", "gaussian", "v8in_gaussian", "hann_sum1", "constant")
#: How the blended sums become a map: ``threshold`` = ``num / den`` where ``den > min_weight`` else 0;
#: ``clamp`` = ``num / max(den, min_weight)``; ``count`` = divide by the number of tiles (the released v8-in code).
STITCHES = ("threshold", "clamp", "count")
#: Tile grids understood by the engine (see :mod:`inksweep.tiling`).
GRIDS = ("sliding", "inside_mask")
#: Which grid positions the reference schedules: ``occupancy`` = positions with CT data in any layer of the stack
#: (villa's occupancy scan of a level-0-only render), ``grid`` = every position of the grid.
SCHEDULES = ("occupancy", "grid")
#: How the reference drops empty tiles: ``tile`` = tiles whose raw window is all zero are dropped from their batch
#: (villa, Hecate); ``batch`` = a batch is skipped only when all its tiles are empty (optimized_inference);
#: ``none`` = every scheduled tile runs (v8-in).
SKIPS = ("tile", "batch", "none")
#: Map-level calibrations applied after stitching.
CALIBRATIONS = ("identity", "label_smoothing_0.5")
#: Reference storage round trips that can be emulated (the reference pipelines store maps as 8-bit files).
STORAGES = (None, "u8_floor", "u8_round")


@dataclass
class ReaderSpec:
    """One ink reader. Required fields first (the contract), optional extensions after.

    Required (contract)
    -------------------
    name:
        Short identifier; used for output folders (``ink/<name>/``) and in ``ink.json``.
    window_layers:
        Number of layers (depth planes) the model consumes, ``W``.
    voxel_size_um:
        The input resolution the model expects, in µm (isotropic).
    tile:
        Tile (patch) edge length in pixels on the model's grid.
    stride:
        Step between neighbouring tiles in pixels on the model's grid.
    load(checkpoint_path, device) -> torch.nn.Module:
        Build the network and load its weights. ``checkpoint_path`` is a path, or a sequence of paths for a
        checkpoint ensemble. Must return the module in eval mode on ``device``.
    preprocess(stack_uint8) -> Tensor:
        ``stack_uint8`` is one tile, a ``uint8`` tensor ``[W, tile, tile]`` already in model depth order (the engine
        applies the orientation), on the GPU. Returns the model input for that tile, normalisation included. With
        ``batched=True`` the engine passes ``[B, W, tile, tile]`` instead and expects a batched result.
    forward(module, batch) -> Tensor:
        Runs the network on a batch: ``batch`` is the stack (along a new dim 0) of ``preprocess`` outputs, or the
        batched ``preprocess`` output. Returns the raw network output for the batch (logits for the built-in readers).
    postprocess(output) -> Tensor:
        Converts the ``forward`` output to probabilities in [0, 1] of shape ``[B, tile, tile]`` (or
        ``[B, 1, tile, tile]``), float32.

    Optional extensions
    -------------------
    batched:
        ``preprocess`` accepts a leading batch dimension (faster; the built-in readers set it).
    blend:
        Overlap-add window: ``"hann"`` (villa and Hecate: outer product of symmetric Hann windows divided by its
        maximum, floored at ``blend_floor``), ``"gaussian"`` (villa's Gaussian mode), ``"v8in_gaussian"`` (v8-in's
        CDF-based kernel divided by its maximum), ``"hann_sum1"`` (numpy Hann outer product divided by its sum, the
        organisers' optimized_inference) or ``"constant"``.
    blend_floor:
        Floor of the Hann window (0.001 in villa and Hecate).
    grid:
        ``"sliding"``: positions ``0, stride, 2*stride, ...`` plus a final position flush with the far edge (villa
        ``iter_blocks`` and Hecate ``axis_starts`` give the same grid); tiles whose raw window is all zero are skipped.
        ``"inside_mask"``: v8-in's rule, only tiles lying entirely inside the coverage mask (window maximum > 0 followed
        by a 7x7 elliptical closing), no final edge-aligned position, no empty-tile skip.
    min_weight:
        With ``stitch="threshold"`` a pixel gets a value only where the summed blend weight exceeds this (villa 1e-6;
        Hecate and v8-in 0); with ``stitch="clamp"`` the weight sum is clamped to at least this.
    stitch:
        ``"threshold"``, ``"clamp"`` or ``"count"`` (see ``STITCHES``). ``"count"`` reproduces the released v8-in
        code, which divides by the number of tiles instead of the summed weight; the v8-in reader uses the fixed
        ``"threshold"`` rule by default.
    schedule, skip_empty:
        Which tiles the reference forms its batches from and which it drops (see ``SCHEDULES`` and ``SKIPS``). Exact
        mode cuts the scheduled tiles into batches of ``reference_batch`` and then drops tiles exactly as the
        reference does, so every forward pass has the reference's batch composition; fast mode drops empty tiles
        first and uses large batches. Outputs are the same either way; only cuDNN's algorithm choice can depend on the
        batch shape.
    zero_empty_columns:
        Multiply each tile's probabilities by the per-pixel support (any non-zero voxel in the window column) before
        blending, as Hecate does; pixels without CT data then come out 0.
    resample_tolerance:
        Relative voxel-size mismatch tolerated without resampling (Hecate 0.02: it refuses renders more than 2 % off
        9.6 µm; the villa readers run unresampled on 8.64 and 9.36 µm renders).
    calibration:
        Map-level conversion after stitching: ``"identity"`` or ``"label_smoothing_0.5"`` (``(p - 0.25) / 0.5``
        clipped to [0, 1], the model-card rescale of the label-smoothed villa readers).
    reference_storage:
        The 8-bit storage of the reference pipeline, emulated when the sweep runs with ``reference_storage=True`` so
        that the maps equal what the reference ensemble read: ``"u8_floor"`` (villa ``infer``: ``trunc(255 p)``),
        ``"u8_round"`` (Hecate: ``round(255 p)``) or ``None``.
    amp_dtype:
        Autocast dtype of the reference implementation (exact mode uses it; ``None`` = no autocast).
    conv_tf32:
        TF32 policy of the reference for float32 convolutions in exact mode: ``True`` / ``False`` to force it, ``None``
        to keep PyTorch's default (the reference scripts never change it).
    matmul_precision:
        ``torch.set_float32_matmul_precision`` value of the reference in exact mode (villa ``infer`` sets ``"high"``),
        or ``None`` to leave it.
    reference_batch:
        Batch size of the reference command; exact mode uses it unless overridden.
    checkpoints:
        Registry keys (:mod:`inksweep.readers.checkpoints`) of the default weights.
    family, citation, licence, notes:
        Documentation, copied into ``ink.json``.
    options:
        Free-form adapter settings, recorded in ``ink.json``.
    """

    # ---- required contract fields ----
    name: str
    window_layers: int
    voxel_size_um: float
    tile: int
    stride: int
    load: Callable[..., torch.nn.Module]
    preprocess: Callable[[torch.Tensor], torch.Tensor]
    forward: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
    postprocess: Callable[[torch.Tensor], torch.Tensor]
    # ---- optional extensions ----
    batched: bool = False
    blend: str = "hann"
    blend_floor: float = 0.001
    grid: str = "sliding"
    min_weight: float = 1e-6
    stitch: str = "threshold"
    schedule: str = "occupancy"
    skip_empty: str = "tile"
    zero_empty_columns: bool = False
    resample_tolerance: float = 0.10
    calibration: str = "identity"
    reference_storage: Optional[str] = None
    amp_dtype: Optional[torch.dtype] = None
    conv_tf32: Optional[bool] = None
    matmul_precision: Optional[str] = None
    reference_batch: int = 4
    checkpoints: Sequence[str] = ()
    family: str = "plugin"
    citation: str = ""
    licence: str = ""
    notes: str = ""
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Raise ``ValueError`` on an inconsistent spec (called on construction)."""
        if not self.name or any(c in self.name for c in "/\\ "):
            raise ValueError(f"reader name must be non-empty without '/', '\\\\' or spaces: {self.name!r}")
        for key in ("window_layers", "tile", "stride"):
            if int(getattr(self, key)) <= 0:
                raise ValueError(f"{self.name}: {key} must be positive")
        if self.stride > self.tile:
            raise ValueError(f"{self.name}: stride {self.stride} > tile {self.tile} leaves gaps")
        if self.voxel_size_um <= 0:
            raise ValueError(f"{self.name}: voxel_size_um must be positive")
        if self.blend not in BLEND_MODES:
            raise ValueError(f"{self.name}: blend must be one of {BLEND_MODES}, got {self.blend!r}")
        if self.stitch not in STITCHES:
            raise ValueError(f"{self.name}: stitch must be one of {STITCHES}, got {self.stitch!r}")
        if self.schedule not in SCHEDULES:
            raise ValueError(f"{self.name}: schedule must be one of {SCHEDULES}, got {self.schedule!r}")
        if self.skip_empty not in SKIPS:
            raise ValueError(f"{self.name}: skip_empty must be one of {SKIPS}, got {self.skip_empty!r}")
        if self.grid not in GRIDS:
            raise ValueError(f"{self.name}: grid must be one of {GRIDS}, got {self.grid!r}")
        if self.calibration not in CALIBRATIONS:
            raise ValueError(f"{self.name}: calibration must be one of {CALIBRATIONS}")
        if self.reference_storage not in STORAGES:
            raise ValueError(f"{self.name}: reference_storage must be one of {STORAGES}")
        for fn in ("load", "preprocess", "forward", "postprocess"):
            if not callable(getattr(self, fn)):
                raise ValueError(f"{self.name}: {fn} must be callable")

    def with_options(self, **changes: Any) -> "ReaderSpec":
        """A copy with some fields changed (e.g. ``stride``), validated again."""
        return replace(self, **changes)

    def needs_resampling(self, render_voxel_um: float) -> bool:
        """True if a render at ``render_voxel_um`` must be resampled to this reader's voxel size."""
        return abs(float(render_voxel_um) / float(self.voxel_size_um) - 1.0) > float(self.resample_tolerance)

    def describe(self) -> dict[str, Any]:
        """JSON-serialisable description for ``ink.json``."""
        return dict(
            name=self.name, family=self.family, window_layers=int(self.window_layers),
            voxel_size_um=float(self.voxel_size_um), tile=int(self.tile), stride=int(self.stride),
            batched=bool(self.batched), blend=self.blend, blend_floor=float(self.blend_floor), grid=self.grid,
            min_weight=float(self.min_weight), stitch=self.stitch, schedule=self.schedule,
            skip_empty=self.skip_empty, zero_empty_columns=bool(self.zero_empty_columns),
            resample_tolerance=float(self.resample_tolerance), calibration=self.calibration,
            reference_storage=self.reference_storage,
            amp_dtype=None if self.amp_dtype is None else str(self.amp_dtype).replace("torch.", ""),
            conv_tf32=self.conv_tf32, matmul_precision=self.matmul_precision,
            reference_batch=int(self.reference_batch), checkpoints=list(self.checkpoints),
            citation=self.citation, licence=self.licence, notes=self.notes, options=dict(self.options),
        )


def as_probability_batch(prob: torch.Tensor, tile: int) -> torch.Tensor:
    """Normalise a ``postprocess`` result to float32 ``[B, tile, tile]`` and check its shape."""
    if prob.ndim == 4:
        if prob.shape[1] != 1:
            raise ValueError(f"postprocess returned {tuple(prob.shape)}; expected one channel")
        prob = prob[:, 0]
    if prob.ndim != 3 or tuple(prob.shape[-2:]) != (tile, tile):
        raise ValueError(f"postprocess must return [B, {tile}, {tile}], got {tuple(prob.shape)}")
    return prob.float()

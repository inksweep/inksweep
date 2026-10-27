"""The sweep engine: every reader x depth x tile phase x orientation in one GPU-resident process per GPU.

The surface volume is copied to the GPU once. For each reader (loaded once) and each depth, orientation and tile phase
the engine builds the depth window on the GPU (or the resampled window, for readers at another voxel size), lays out
the reader's tile grid, drops empty tiles, normalises and runs the tiles in batches, and blends them into the map.
Nothing goes through disk between steps, no process is started per map, and the CPU never touches a tile.

Two modes:

* ``exact``: the reference numerics. Autocast dtype, TF32 and matmul-precision policy of each reference command, its
  batch size, deterministic cuDNN, eager execution, and the same per-pixel accumulation order as the reference (tiles
  added one by one in grid order). The resampling of Hecate inputs and the resize of its maps run in OpenCV exactly as
  in the reference scripts. Where the reference is deterministic the maps are bit-identical; elsewhere they differ only
  by the GPU's own run-to-run noise (cuDNN algorithm choice).
* ``fast``: TF32, the reader's training autocast dtype (bf16 when it has none), cuDNN autotuning, channels-last
  weights, large batches, vectorised accumulation, GPU resampling, and optionally ``torch.compile`` and CUDA graphs.

Tile phases (``pad_rows, pad_cols``) are virtual zero padding on the top and left of the canvas, as the PHerc. 343
recipe's ``pad_render.py`` does physically. With a ``select`` merge each phase only runs the tiles that touch the
pixels the merge takes from it, which halves the work and changes no output pixel (every tile covering a taken pixel
still runs).

Outputs per reader ``R``, orientation side ``S`` (``reading`` or ``blind``) and depth ``d`` (see docs/guide.md,
"Inputs and outputs"):

* ``ink/R/dNN.tif`` (reading side) or ``ink/R/blind/dNN.tif``: float32 [0, 1] on the render grid (the contract map:
  the reader's calibrated probability, from the emulated reference storage when ``reference_storage`` is on);
* ``ink/R/raw/S/dNN.tif``: the blended probability before calibration, on the reader's own grid (for
  ``inksweep compare``);
* ``ink/R/.done/S_dNN.json``: the resume marker with the map's timing and settings.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np
import torch

from . import io as iox
from .readers.spec import ReaderSpec, as_probability_batch
from .resample import (crop_offsets, mask_to_grid, resample_window_cpu, resample_window_gpu, resampled_shape,
                       to_render_grid, xy_scale)
from .tiling import (blend_window, centre_out, depth_tag, inside_mask_positions, sliding_grid, tiles_touching,
                     v8in_coverage, window_indices, window_positions)

LOG = logging.getLogger("inksweep")
ORIENTATIONS = ("forward", "reverse")


@dataclass
class SweepConfig:
    """Everything that defines a sweep (recorded in ``ink.json`` and hashed into the resume markers)."""

    depths: list[int]
    orientations: tuple[str, ...] = ("reverse", "forward")
    reading: str = "reverse"
    phases: tuple[tuple[int, int], ...] = ((0, 0),)
    phase_merge: str = "none"                 # none | select | mean
    mode: str = "exact"                       # exact | fast
    precision: str = "reference"              # reference | fp32 | bf16 | fp16 (fast: reference = training dtype)
    batch: Optional[int] = None
    compile: bool = False
    cuda_graphs: bool = False
    channels_last: bool = True                # fast mode only
    reference_storage: bool = True
    keep_raw: bool = True
    keep_phases: bool = False
    need_masks: bool = True
    resample_threads: int = 8
    group_bytes: int = 1 << 29                # tiles preprocessed together (bytes of float32 input), see phase_map
    max_gpu_mem_gb: Optional[float] = None    # cap of the CUDA caching allocator (GiB), see cap_gpu_memory

    def validate(self) -> None:
        for o in self.orientations:
            if o not in ORIENTATIONS:
                raise ValueError(f"orientation {o!r} not in {ORIENTATIONS}")
        if self.reading not in ORIENTATIONS:
            raise ValueError("reading orientation must be forward or reverse")
        if self.mode not in ("exact", "fast"):
            raise ValueError("mode must be exact or fast")
        if self.precision not in ("reference", "fp32", "bf16", "fp16"):
            raise ValueError("precision must be reference, fp32, bf16 or fp16")
        if self.phase_merge not in ("none", "select", "mean"):
            raise ValueError("phase_merge must be none, select or mean")
        if self.phase_merge == "select" and len(self.phases) != 2:
            raise ValueError("a select merge needs exactly two phases")
        if self.phase_merge == "none" and len(self.phases) != 1:
            raise ValueError("several phases need a merge (select or mean)")
        if self.max_gpu_mem_gb is not None and self.max_gpu_mem_gb <= 0:
            raise ValueError("max_gpu_mem_gb must be positive")

    def side(self, orientation: str) -> str:
        return "reading" if orientation == self.reading else "blind"

    def digest(self, spec: ReaderSpec) -> str:
        """Hash of every setting that changes a map of ``spec`` (resume markers refuse to mix settings)."""
        keys = dict(mode=self.mode, precision=self.precision, phases=[list(p) for p in self.phases],
                    phase_merge=self.phase_merge, reference_storage=self.reference_storage,
                    batch=self.batch if self.mode == "exact" else None, reader=spec.describe())
        return hashlib.sha256(json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:16]


@dataclass
class MapRecord:
    reader: str
    orientation: str
    side: str
    depth: int
    seconds: float
    tiles: int
    shape_model: tuple[int, int]
    files: dict[str, str] = field(default_factory=dict)
    phases: list[dict] = field(default_factory=list)
    digest: str = ""
    skipped: str = ""
    gpu_peak_allocated_mb: Optional[float] = None     # peak of torch.cuda.max_memory_allocated during the map (MiB)
    gpu_peak_reserved_mb: Optional[float] = None      # peak held by the caching allocator (MiB); None off CUDA
    profile: Optional[dict] = None                    # seconds and calls per stage (StageTimer), when profiling


# ----------------------------------------------------------------------------------------------- profiling
class StageTimer:
    """Wall time per stage of a map, with a CUDA synchronisation at every stage boundary.

    Stages: ``read`` (the depth window: gather or resampling, and the tiles taken from it), ``schedule`` (tile grid,
    empty and needed tiles), ``normalise`` (the reader's preprocess), ``forward`` (network and postprocess), ``blend``
    (weighting and accumulation of the tiles, the final division, the phase merge) and ``storage`` (copy to the host,
    calibration, file writes). Only used when profiling: the synchronisations make each stage's time its own (GPU work
    is otherwise asynchronous) but cost throughput, so a profiled run gives the split, not the speed.
    """

    STAGES = ("read", "schedule", "normalise", "forward", "blend", "storage")

    def __init__(self, device: torch.device) -> None:
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        self.seconds = dict.fromkeys(self.STAGES, 0.0)
        self.calls = dict.fromkeys(self.STAGES, 0)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.seconds[name] = self.seconds.get(name, 0.0) + time.perf_counter() - t0
            self.calls[name] = self.calls.get(name, 0) + 1

    def snapshot(self) -> dict:
        return dict(seconds={k: round(v, 6) for k, v in self.seconds.items()}, calls=dict(self.calls))


_NO_STAGE = contextlib.nullcontext()                  # reusable: what Sweep._stage returns when not profiling


# ----------------------------------------------------------------------------------------------- GPU memory
MIB = float(1 << 20)
_CAPPED: set[int] = set()                             # CUDA devices on which this process has set a cap


def cap_gpu_memory(device: torch.device, gb: Optional[float]) -> Optional[float]:
    """Cap this process's CUDA caching allocator on ``device`` at ``gb`` GiB, to emulate a smaller card.

    Calls ``torch.cuda.set_per_process_memory_fraction`` (so call it before anything is allocated) and returns the
    fraction set, or ``None`` when there is no cap or the device is not CUDA; ``gb=None`` lifts a cap that an earlier
    sweep of this process set. A test aid: the CUDA context and memory that libraries allocate outside PyTorch's
    allocator are not counted, so emulate a card with a margin below its size.
    """
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if gb is None:
        if index in _CAPPED:
            torch.cuda.set_per_process_memory_fraction(1.0, device)
            _CAPPED.discard(index)
        return None
    if gb <= 0:
        raise ValueError("max_gpu_mem_gb must be positive")
    total = float(torch.cuda.get_device_properties(device).total_memory)
    fraction = min(1.0, float(gb) * (1 << 30) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    _CAPPED.add(index)
    return fraction


def reset_gpu_peaks(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def gpu_peaks(device: torch.device) -> tuple[Optional[float], Optional[float]]:
    """(peak allocated, peak reserved) in MiB since the last :func:`reset_gpu_peaks`; ``(None, None)`` off CUDA."""
    if device.type != "cuda":
        return None, None
    return (round(torch.cuda.max_memory_allocated(device) / MIB, 1),
            round(torch.cuda.max_memory_reserved(device) / MIB, 1))


# ----------------------------------------------------------------------------------------------- numerics
@contextlib.contextmanager
def numerics(spec: ReaderSpec, cfg: SweepConfig) -> Iterable[None]:
    """Global PyTorch flags for one reader (exact: the reference's; fast: speed), restored afterwards."""
    cudnn = torch.backends.cudnn
    saved = (cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32,
             torch.get_float32_matmul_precision())
    try:
        if cfg.mode == "exact":
            cudnn.deterministic, cudnn.benchmark = True, False
            if cfg.precision == "fp32":
                cudnn.allow_tf32 = False
                torch.set_float32_matmul_precision("highest")
            else:
                if spec.conv_tf32 is not None:
                    cudnn.allow_tf32 = bool(spec.conv_tf32)
                if spec.matmul_precision:
                    torch.set_float32_matmul_precision(spec.matmul_precision)
        else:
            cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32 = False, True, True
            torch.set_float32_matmul_precision("high")
        yield
    finally:
        cudnn.deterministic, cudnn.benchmark, cudnn.allow_tf32 = saved[0], saved[1], saved[2]
        torch.backends.cuda.matmul.allow_tf32 = saved[3]
        torch.set_float32_matmul_precision(saved[4])


def autocast_dtype(spec: ReaderSpec, module: torch.nn.Module, cfg: SweepConfig) -> Optional[torch.dtype]:
    """The autocast dtype for this reader and mode (``None`` = no autocast)."""
    if cfg.precision == "fp32":
        return None
    if cfg.precision == "bf16":
        return torch.bfloat16
    if cfg.precision == "fp16":
        return torch.float16
    ref = getattr(module, "inksweep_amp_dtype", spec.amp_dtype)
    if cfg.mode == "exact":
        return ref
    return ref if ref is not None else torch.bfloat16


def fast_batch(spec: ReaderSpec) -> int:
    if "fast_batch" in spec.options:
        return int(spec.options["fast_batch"])
    return {"villa": 64 if spec.tile <= 128 else 16, "hecate": 32 if spec.tile <= 64 else 2, "v8in": 32,
            "canonical": 4}.get(spec.family, 16)


# ----------------------------------------------------------------------------------------------- the engine
class Sweep:
    """GPU-resident sweep of one surface volume on one device."""

    def __init__(self, surface: iox.SurfaceVolume, out_dir: str | os.PathLike, cfg: SweepConfig,
                 device: str | torch.device = "cuda", phase_mask: Optional[np.ndarray] = None,
                 profile: bool = False) -> None:
        cfg.validate()
        if surface.voxel_size_um is None:
            raise ValueError("the surface volume's voxel size is unknown: pass --voxel-um or a surface.json")
        self.cfg = cfg
        self.out = Path(out_dir)
        self.surface = surface
        self.device = torch.device(device)
        if self.device.type == "cuda":
            if self.device.index is None:                     # "cuda" -> the current device, with its index
                self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        self.memory_fraction = cap_gpu_memory(self.device, cfg.max_gpu_mem_gb)   # before anything is allocated
        self._last_reader: Optional[str] = None
        self.profiler: Optional[StageTimer] = StageTimer(self.device) if profile else None
        t0 = time.time()
        self.vol_cpu = surface.read()                                   # (L, H, W) uint8, kept for exact resampling
        self.vol = torch.from_numpy(self.vol_cpu).to(self.device)       # GPU-resident for the whole sweep
        self.load_seconds = time.time() - t0
        self.L, self.H, self.W = (int(v) for v in self.vol.shape)
        self.voxel_um = float(surface.voxel_size_um)
        if cfg.phase_merge == "select":
            if phase_mask is None or phase_mask.shape != (self.H, self.W):
                raise ValueError("a select merge needs a phase mask of the render shape")
        self.phase_mask = phase_mask
        self.occupancy = (self.vol != 0).any(0)                      # CT data in any layer (villa's occupancy scan)
        self._graphs: dict[Any, Any] = {}

    def _stage(self, name: str) -> contextlib.AbstractContextManager:
        """A timed stage when profiling, else a no-op context (no synchronisation, nothing changes)."""
        return self.profiler.stage(name) if self.profiler is not None else _NO_STAGE

    # ------------------------------------------------------------------ paths
    def map_path(self, spec: ReaderSpec, side: str, d: int) -> Path:
        base = self.out / "ink" / spec.name
        return (base if side == "reading" else base / "blind") / f"{depth_tag(d)}.tif"

    def raw_path(self, spec: ReaderSpec, side: str, d: int, phase: Optional[tuple[int, int]] = None) -> Path:
        base = self.out / "ink" / spec.name / "raw" / side
        if phase is not None:
            base = base / f"phase_{phase[0]}_{phase[1]}"
        return base / f"{depth_tag(d)}.tif"

    def done_path(self, spec: ReaderSpec, side: str, d: int) -> Path:
        return self.out / "ink" / spec.name / ".done" / f"{side}_{depth_tag(d)}.json"

    def is_done(self, spec: ReaderSpec, orientation: str, d: int) -> bool:
        p = self.done_path(spec, self.cfg.side(orientation), d)
        if not p.is_file():
            return False
        rec = json.loads(p.read_text())
        if rec.get("digest") != self.cfg.digest(spec):
            raise RuntimeError(f"{p}: made with other settings (digest {rec.get('digest')} != "
                               f"{self.cfg.digest(spec)}); use a new output folder or delete the reader's maps")
        return self.map_path(spec, self.cfg.side(orientation), d).is_file()

    # ------------------------------------------------------------------ geometry
    def model_shape(self, spec: ReaderSpec, pad: tuple[int, int] = (0, 0)) -> tuple[int, int]:
        h, w = self.H + pad[0], self.W + pad[1]
        if spec.needs_resampling(self.voxel_um):
            return resampled_shape(h, w, self.voxel_um, spec.voxel_size_um)
        return h, w

    def depth_ok(self, spec: ReaderSpec, d: int) -> bool:
        try:
            if spec.needs_resampling(self.voxel_um):
                window_positions(self.L, d, spec.window_layers, spec.voxel_size_um / self.voxel_um)
            else:
                window_indices(self.L, d, spec.window_layers)
            return True
        except ValueError:
            return False

    def canvas(self, spec: ReaderSpec, d: int, orientation: str, pad: tuple[int, int]) -> torch.Tensor:
        """The reader's input canvas for one depth: uint8 ``[W, Hc, Wc]`` on the GPU, in model depth order."""
        if not spec.needs_resampling(self.voxel_um):
            idx = torch.tensor(window_indices(self.L, d, spec.window_layers, orientation), device=self.device)
            planes = self.vol.index_select(0, idx)
            if pad != (0, 0):
                planes = torch.nn.functional.pad(planes, (pad[1], 0, pad[0], 0))
            return planes
        step = spec.voxel_size_um / self.voxel_um
        pos = window_positions(self.L, d, spec.window_layers, step)
        if self.cfg.mode == "exact":
            planes_np = resample_window_cpu(self.vol_cpu, pos, self.voxel_um, spec.voxel_size_um, pad=pad,
                                            threads=self.cfg.resample_threads)
            planes = torch.from_numpy(planes_np).to(self.device)
        else:
            planes = resample_window_gpu(self.vol, pos, self.voxel_um, spec.voxel_size_um, pad=pad)
        return planes.flip(0) if orientation == "reverse" else planes

    def need_mask(self, spec: ReaderSpec, phase_index: int, pad: tuple[int, int],
                  canvas_shape: tuple[int, int]) -> Optional[np.ndarray]:
        """Pixels of the padded canvas that the merge takes from this phase (None = all)."""
        if self.cfg.phase_merge != "select" or not self.cfg.need_masks or self.phase_mask is None:
            return None
        plain = self.model_shape(spec)
        take_second = mask_to_grid(self.phase_mask, plain)
        mine = take_second if phase_index == 1 else ~take_second
        scale = xy_scale(self.voxel_um, spec.voxel_size_um) if spec.needs_resampling(self.voxel_um) else 1.0
        dr, dc = crop_offsets(canvas_shape, plain, pad, scale)
        need = np.zeros(canvas_shape, bool)
        need[dr:dr + plain[0], dc:dc + plain[1]] = mine
        return need

    # ------------------------------------------------------------------ one map
    def _box_counts(self, mask: torch.Tensor, origins: np.ndarray, tile: int) -> np.ndarray:
        """Number of True pixels of ``mask`` (GPU, ``[h, w]``) inside each tile, for origins ``[N, 2]``."""
        h, w = mask.shape
        integral = torch.nn.functional.pad(mask.to(torch.int32).cumsum(0).cumsum(1), (1, 0, 1, 0))
        o = torch.from_numpy(origins).to(mask.device)
        y0 = torch.clamp(o[:, 0], 0, h)
        x0 = torch.clamp(o[:, 1], 0, w)
        y1 = torch.clamp(o[:, 0] + tile, 0, h)
        x1 = torch.clamp(o[:, 1] + tile, 0, w)
        return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]).cpu().numpy()

    def tile_batches(self, spec: ReaderSpec, canvas: torch.Tensor, need: Optional[np.ndarray], pad: tuple[int, int],
                     batch_size: int) -> list[np.ndarray]:
        """The batches of tile origins ``(y, x)`` to run, in the reference's order.

        Exact mode reproduces the reference's batch composition: the scheduled grid (``spec.schedule``) is cut into
        batches of ``batch_size`` and tiles are dropped within each batch as the reference drops them
        (``spec.skip_empty``). A batch with no tile touching a needed pixel is skipped as a whole, the others run
        whole, so every forward pass has the reference's shape. Fast mode drops everything not needed first.
        """
        _, hc, wc = canvas.shape
        tile = spec.tile
        if spec.grid == "sliding":
            grid = sliding_grid(hc, wc, tile, spec.stride)
        else:
            grid = inside_mask_positions(v8in_coverage(canvas.amax(0).cpu().numpy()), tile, spec.stride)
        if len(grid) == 0:
            return []
        if spec.schedule == "occupancy":
            if spec.needs_resampling(self.voxel_um):
                occ = (canvas != 0).any(0)                                # resampled readers: the window's data
            else:
                occ = torch.nn.functional.pad(self.occupancy, (pad[1], 0, pad[0], 0))
            grid = grid[self._box_counts(occ, grid, tile) > 0]
            if len(grid) == 0:
                return []
        if spec.skip_empty == "none":
            nonempty = np.ones(len(grid), bool)
        else:
            nonempty = self._box_counts((canvas != 0).any(0), grid, tile) > 0
        needed = tiles_touching(grid, tile, need) if need is not None else np.ones(len(grid), bool)
        if self.cfg.mode == "fast":
            keep = grid[nonempty & needed]
            return [keep[i:i + batch_size] for i in range(0, len(keep), batch_size)]
        batches = []
        for i in range(0, len(grid), batch_size):
            sl = slice(i, i + batch_size)
            ne, nd = nonempty[sl], needed[sl]
            if spec.skip_empty == "tile":
                if (ne & nd).any():
                    batches.append(grid[sl][ne])
            elif spec.skip_empty == "batch":
                if ne.any() and nd.any():
                    batches.append(grid[sl])
            elif nd.any():
                batches.append(grid[sl])
        return batches

    def _forward_fn(self, spec: ReaderSpec, module: torch.nn.Module, batch_size: int,
                    amp: Optional[torch.dtype]) -> Callable[[torch.Tensor], torch.Tensor]:
        """forward(x) under the reader's autocast; with CUDA graphs a captured replay for full batches."""
        def plain(x: torch.Tensor) -> torch.Tensor:
            ctx = torch.autocast("cuda", dtype=amp) if (amp is not None and self.device.type == "cuda") \
                else contextlib.nullcontext()
            with ctx:
                return spec.forward(module, x)

        if not (self.cfg.cuda_graphs and self.device.type == "cuda" and not self.cfg.compile):
            return plain
        state: dict[str, Any] = {}
        any_shape = self.cfg.mode == "exact"          # exact batches vary in size (the reference's composition)

        def graphed(x: torch.Tensor) -> torch.Tensor:
            if x.shape[0] != batch_size and not any_shape:
                return plain(x)
            key = (spec.name, tuple(x.shape), x.dtype)
            if key not in self._graphs:
                static_in = x.clone()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        with torch.autocast("cuda", dtype=amp, cache_enabled=False, enabled=amp is not None):
                            spec.forward(module, static_in)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    with torch.autocast("cuda", dtype=amp, cache_enabled=False, enabled=amp is not None):
                        static_out = spec.forward(module, static_in)
                self._graphs[key] = (graph, static_in, static_out)
            graph, static_in, static_out = self._graphs[key]
            static_in.copy_(x)
            graph.replay()
            return static_out.clone()

        state["fn"] = graphed
        return graphed

    @torch.inference_mode()
    def phase_map(self, spec: ReaderSpec, module: torch.nn.Module, d: int, orientation: str,
                  pad: tuple[int, int], phase_index: int, forward: Callable[[torch.Tensor], torch.Tensor],
                  batch_size: int) -> tuple[torch.Tensor, int]:
        """The blended probability map of one tile phase, cropped to the unpadded reader grid; and the tile count.

        The ``with stage(...)`` blocks only time the work when profiling (``inksweep bench --profile``); otherwise
        they are no-ops and the kernels run exactly as without them.
        """
        stage = self._stage
        with stage("read"):
            canvas = self.canvas(spec, d, orientation, pad)
        _, hc, wc = canvas.shape
        with stage("schedule"):
            need = self.need_mask(spec, phase_index, pad, (hc, wc))
            batches = self.tile_batches(spec, canvas, need, pad, batch_size)
        p = spec.tile
        hb, wb = max(hc, p), max(wc, p)                              # canvases smaller than a tile: zero-pad
        if (hb, wb) != (hc, wc):
            canvas = torch.nn.functional.pad(canvas, (0, wb - wc, 0, hb - hc))
        num = torch.zeros((hb, wb), dtype=torch.float32, device=self.device)
        den = torch.zeros_like(num)
        w = blend_window(spec.blend, p, spec.blend_floor).to(self.device)
        wden = torch.ones_like(w) if spec.stitch == "count" else w
        ar = torch.arange(p, device=self.device)
        exact = self.cfg.mode == "exact"
        if not batches:
            n = 0
        else:
            # All origins go to the GPU once; tiles are gathered and preprocessed in groups of whole batches (the
            # preprocessing is per tile, so grouping changes no value) and the forward passes then run batch by batch
            # in the reference's composition, with no host synchronisation inside a group.
            sizes = [len(b) for b in batches]
            bounds = np.cumsum([0] + sizes)
            n = int(bounds[-1])
            origins = torch.from_numpy(np.concatenate(batches)).to(self.device)
            group_tiles = max(batch_size, int(self.cfg.group_bytes // max(1, spec.window_layers * p * p * 4)))
            bi = 0
            while bi < len(batches):
                bj, count = bi, 0
                while bj < len(batches) and (count == 0 or count + sizes[bj] <= group_tiles):
                    count += sizes[bj]
                    bj += 1
                g0, g1 = int(bounds[bi]), int(bounds[bj])
                with stage("read"):
                    og = origins[g0:g1]
                    rows_g = (og[:, 0:1] + ar)[:, :, None]                   # [G, p, 1]
                    cols_g = (og[:, 1:2] + ar)[:, None, :]                   # [G, 1, p]
                    tiles_g = canvas[:, rows_g, cols_g].permute(1, 0, 2, 3)  # [G, W, p, p] uint8
                with stage("normalise"):
                    x_g = (spec.preprocess(tiles_g) if spec.batched
                           else torch.stack([spec.preprocess(t) for t in tiles_g]))
                    support_g = tiles_g.any(1) if spec.zero_empty_columns else None
                for b in range(bi, bj):
                    a0, a1 = int(bounds[b]) - g0, int(bounds[b + 1]) - g0
                    k = a1 - a0
                    with stage("forward"):
                        x = x_g[a0:a1]
                        if self.cfg.compile and k < batch_size:              # fixed shapes for the compiled graph
                            x = torch.cat([x, x.new_zeros((batch_size - k,) + tuple(x.shape[1:]))])
                        prob = as_probability_batch(spec.postprocess(forward(x)), p)[:k]
                    with stage("blend"):
                        if support_g is not None:
                            prob = prob * support_g[a0:a1].to(prob.dtype)
                        contrib = prob * w
                        if exact:                                            # the reference's addition order
                            for j, (y, xx) in enumerate(batches[b].tolist()):
                                num[y:y + p, xx:xx + p] += contrib[j]
                                den[y:y + p, xx:xx + p] += wden
                        else:
                            lin = (rows_g[a0:a1] * wb + cols_g[a0:a1]).reshape(-1)   # linear pixel indices
                            num.view(-1).index_add_(0, lin, contrib.reshape(-1))
                            den.view(-1).index_add_(0, lin, wden.expand(k, p, p).reshape(-1))
                bi = bj
        with stage("blend"):
            if spec.stitch == "clamp":
                m = num / torch.clamp(den, min=spec.min_weight)
            else:
                m = torch.where(den > spec.min_weight, num / torch.where(den > 0, den, torch.ones_like(den)),
                                torch.zeros_like(num))
            m = torch.clamp(torch.nan_to_num(m[:hc, :wc]), 0.0, 1.0)
        plain = self.model_shape(spec)
        scale = xy_scale(self.voxel_um, spec.voxel_size_um) if spec.needs_resampling(self.voxel_um) else 1.0
        dr, dc = crop_offsets((hc, wc), plain, pad, scale) if pad != (0, 0) else (0, 0)
        return m[dr:dr + plain[0], dc:dc + plain[1]], n

    # ------------------------------------------------------------------ calibration and writing
    def calibrate(self, spec: ReaderSpec, m_model: np.ndarray) -> np.ndarray:
        """Contract map (float32 [0, 1], render grid) from a blended reader-grid map; docs/guide.md, "Map values"."""
        m = np.clip(np.asarray(m_model, dtype=np.float32), 0, 1)
        emulate = self.cfg.reference_storage and spec.reference_storage is not None
        if emulate:
            if spec.reference_storage == "u8_floor":
                q = (m * np.float32(255)).astype(np.uint8)          # villa: probability *= 255; astype(uint8)
            else:
                q = np.rint(m * np.float32(255)).astype(np.uint8)   # Hecate: rint(prob.clip(0, 1) * 255)
            if spec.calibration == "label_smoothing_0.5":
                u = np.clip((q.astype(np.float32) - 64.0) / 128.0, 0, 1)
            else:
                u = q.astype(np.float32) / 255.0
        elif spec.calibration == "label_smoothing_0.5":
            u = np.clip((m - np.float32(0.25)) / np.float32(0.5), 0, 1)
        else:
            u = m
        return to_render_grid(u.astype(np.float32), (self.H, self.W))

    def merge(self, spec: ReaderSpec, maps: list[torch.Tensor]) -> torch.Tensor:
        if len(maps) == 1:
            return maps[0]
        if self.cfg.phase_merge == "mean":
            return torch.stack(maps).mean(0)
        take_second = torch.from_numpy(mask_to_grid(self.phase_mask, tuple(maps[0].shape))).to(maps[0].device)
        return torch.where(take_second, maps[1], maps[0])

    def run_map(self, spec: ReaderSpec, module: torch.nn.Module, d: int, orientation: str) -> MapRecord:
        cfg = self.cfg
        side = cfg.side(orientation)
        reset_gpu_peaks(self.device)                    # the record's peaks cover this map (weights and volume included)
        if self.profiler is not None:
            self.profiler.reset()
        t0 = time.time()
        amp = autocast_dtype(spec, module, cfg)
        bs = cfg.batch or (spec.reference_batch if cfg.mode == "exact" else fast_batch(spec))
        forward = self._forward_fn(spec, module, bs, amp)
        maps, tiles, phase_recs = [], 0, []
        for i, pad in enumerate(cfg.phases):
            tp = time.time()
            m, n = self.phase_map(spec, module, d, orientation, tuple(pad), i, forward, bs)
            maps.append(m)
            tiles += n
            phase_recs.append(dict(pad=list(pad), tiles=n, seconds=round(time.time() - tp, 3)))
            if cfg.keep_phases and len(cfg.phases) > 1:
                with self._stage("storage"):
                    iox.write_tif(self.raw_path(spec, side, d, tuple(pad)), m.cpu().numpy())
        with self._stage("blend"):
            merged_gpu = self.merge(spec, maps)
        with self._stage("storage"):
            merged = merged_gpu.cpu().numpy().astype(np.float32)
            del merged_gpu
            files = {}
            if cfg.keep_raw:
                files["raw"] = str(iox.write_tif(self.raw_path(spec, side, d), merged).relative_to(self.out))
            files["map"] = str(iox.write_tif(self.map_path(spec, side, d), self.calibrate(spec, merged))
                               .relative_to(self.out))
        peak_allocated, peak_reserved = gpu_peaks(self.device)
        rec = MapRecord(reader=spec.name, orientation=orientation, side=side, depth=int(d),
                        seconds=round(time.time() - t0, 3), tiles=tiles, shape_model=tuple(merged.shape),
                        files=files, phases=phase_recs, digest=cfg.digest(spec),
                        gpu_peak_allocated_mb=peak_allocated, gpu_peak_reserved_mb=peak_reserved,
                        profile=self.profiler.snapshot() if self.profiler is not None else None)
        iox.write_json(self.done_path(spec, side, d), asdict(rec) | dict(batch=bs, amp=str(amp),
                                                                       device=str(self.device)))
        return rec

    def prepare_module(self, spec: ReaderSpec, module: torch.nn.Module) -> torch.nn.Module:
        """Fast-mode module transforms (channels-last weights, torch.compile)."""
        if self.cfg.mode != "fast":
            return module
        if self.cfg.channels_last:
            for prm in module.parameters():
                if prm.ndim == 4:
                    prm.data = prm.data.contiguous(memory_format=torch.channels_last)
                elif prm.ndim == 5:
                    prm.data = prm.data.contiguous(memory_format=torch.channels_last_3d)
        if self.cfg.compile:
            mode = "reduce-overhead" if self.cfg.cuda_graphs else "max-autotune-no-cudagraphs"
            module = torch.compile(module, mode=mode, dynamic=False)
        return module

    def run_reader(self, spec: ReaderSpec, checkpoint_paths: Any,
                   depths: Optional[Sequence[int]] = None,
                   orientations: Optional[Sequence[str]] = None,
                   module: Optional[torch.nn.Module] = None) -> list[MapRecord]:
        """Every (orientation, depth) map of one reader that is not done yet."""
        depths = centre_out(depths if depths is not None else self.cfg.depths)
        orientations = list(orientations or self.cfg.orientations)
        todo = [(o, d) for o in orientations for d in depths
                if self.depth_ok(spec, d) and not self.is_done(spec, o, d)]
        records: list[MapRecord] = []
        for o in orientations:
            for d in depths:
                if not self.depth_ok(spec, d):
                    records.append(MapRecord(spec.name, o, self.cfg.side(o), d, 0.0, 0, (0, 0),
                                             skipped="window outside the stack"))
        if not todo:
            return records
        if self.device.type == "cuda" and self._last_reader != spec.name:
            torch.cuda.empty_cache()                         # release the previous reader's cached blocks, so that
        self._last_reader = spec.name                        # each reader's memory peaks start from its own needs
        t0 = time.time()
        if module is None:                                   # a module passed in is already prepared
            module = self.prepare_module(spec, spec.load(checkpoint_paths, self.device))
        LOG.info("%s: loaded in %.1fs; %d maps to do", spec.name, time.time() - t0, len(todo))
        with numerics(spec, self.cfg):
            for o, d in todo:
                rec = self.run_map(spec, module, d, o)
                LOG.info("%s %s %s: %d tiles in %.1fs (%.0f tiles/s)", spec.name, rec.side, depth_tag(d),
                         rec.tiles, rec.seconds, rec.tiles / max(rec.seconds, 1e-9))
                records.append(rec)
        self._graphs.clear()
        return records


# ----------------------------------------------------------------------------------------------- orchestration
@dataclass
class Job:
    reader: str
    orientation: str
    depth: int


def plan_jobs(specs: Sequence[ReaderSpec], cfg: SweepConfig) -> list[Job]:
    """Jobs reader-major, centre-out in depth, reading side first."""
    orientations = sorted(cfg.orientations, key=lambda o: o != cfg.reading)
    return [Job(s.name, o, d) for s in specs for o in orientations for d in centre_out(cfg.depths)]


def _worker(rank: int, device: str, surface_args: dict, out_dir: str, cfg: SweepConfig, specs_factory: Callable,
            checkpoints: dict, jobs: "Any", results: "Any", phase_mask: Optional[np.ndarray]) -> None:
    """One GPU: pull jobs, keep the current reader loaded, push records (or errors)."""
    logging.basicConfig(level=logging.INFO, format=f"[gpu{rank}] %(asctime)s %(message)s")
    try:
        specs = {s.name: s for s in specs_factory()}
        surface = iox.open_surface(**surface_args)
        sweep = Sweep(surface, out_dir, cfg, device=device, phase_mask=phase_mask)
        current, module = None, None
        while True:
            job = jobs.get()
            if job is None:                                   # one sentinel per worker ends the queue
                break
            spec = specs[job.reader]
            if current != spec.name:
                module = None
                torch.cuda.empty_cache()
                module = sweep.prepare_module(spec, spec.load(checkpoints[spec.name], sweep.device))
                current = spec.name
            recs = sweep.run_reader(spec, checkpoints[spec.name], depths=[job.depth], orientations=[job.orientation],
                                    module=module)
            for r in recs:
                results.put(("ok", asdict(r)))
        results.put(("done", rank))
    except Exception:
        results.put(("error", f"gpu{rank}: {traceback.format_exc()}"))


def run_parallel(specs_factory: Callable[[], Sequence[ReaderSpec]], surface_args: dict, out_dir: str,
                 cfg: SweepConfig, checkpoints: dict, devices: Sequence[str],
                 phase_mask: Optional[np.ndarray] = None) -> list[dict]:
    """Run the sweep over several GPUs (one process each, a shared job queue). Returns the map records."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    jobs, results = ctx.Queue(), ctx.Queue()
    for job in plan_jobs(list(specs_factory()), cfg):
        jobs.put(job)
    for _ in devices:
        jobs.put(None)
    procs = [ctx.Process(target=_worker, args=(i, dev, surface_args, out_dir, cfg, specs_factory, checkpoints, jobs,
                                               results, phase_mask)) for i, dev in enumerate(devices)]
    for p in procs:
        p.start()
    records, finished, errors = [], 0, []
    while finished < len(procs):
        kind, payload = results.get()
        if kind == "ok":
            records.append(payload)
        elif kind == "done":
            finished += 1
        else:
            errors.append(payload)
            finished += 1
    for p in procs:
        p.join()
    if errors:
        raise RuntimeError("worker failure:\n" + "\n".join(errors))
    return records

"""``inksweep bench``: wall time, tiles/s and peak GPU memory per reader and mode on one surface volume (or a crop).

Each configuration runs every reader at the given depths in a fresh output folder; the first map of each reader is
reported separately (it includes cuDNN autotuning and, with ``--compile``, compilation), the rest give the steady
state. Reference timings (``timings.tsv`` of the recipe's ``run_depth_sweep.sh``: model, d, seconds, rc; one job =
both orientations of one depth) can be passed to report speedups against the reference pipeline.

Peak GPU memory is the largest per-map peak of PyTorch's caching allocator (``gpu_peak_reserved_mb``, what it held
from the driver; ``gpu_peak_allocated_mb``, what tensors used), the surface volume and the weights included. With
``max_gpu_mem_gb`` the allocator is capped to emulate a smaller card; a reader that runs out of memory is recorded as
such and the benchmark goes on.

With ``profile`` each map's time is split into the stages of :class:`inksweep.sweep.StageTimer` (read, schedule,
normalise, forward, blend, storage), timed with a CUDA synchronisation at every stage boundary. Off by default: the
synchronisations cost throughput, so profiled runs give the split, and unprofiled runs the speed.
"""
from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch

from . import io as iox
from .compare import REFERENCE_KEYS
from .readers.spec import ReaderSpec
from .runner import PEAK_KEYS, make_specs, resolve_checkpoints
from .sweep import MapRecord, StageTimer, Sweep, SweepConfig


def crop_surface(surface: iox.SurfaceVolume, rows: Optional[tuple[int, int]],
                 cols: Optional[tuple[int, int]]) -> iox.SurfaceVolume:
    if rows is None and cols is None:
        return surface
    data = surface.read(rows=rows, cols=cols)
    return iox.SurfaceVolume(path=surface.path, name=surface.name + "_crop", layers=data, kind="crop",
                             voxel_size_um=surface.voxel_size_um, meta=dict(surface.meta, crop=dict(rows=rows,
                                                                                                    cols=cols)))


def reference_seconds(tsv: str | Path) -> dict[str, dict[int, float]]:
    """``{recipe model key: {d: seconds}}`` from a ``run_depth_sweep.sh`` timings file (successful jobs only)."""
    out: dict[str, dict[int, float]] = {}
    with open(tsv) as f:
        for row in csv.reader(f, delimiter="\t"):
            if len(row) < 4 or row[3] != "0":
                continue
            out.setdefault(row[0], {})[int(row[1])] = float(row[2])
    return out


def is_out_of_memory(exc: BaseException) -> bool:
    """CUDA's out-of-memory error (also when a library reports it as a plain RuntimeError)."""
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom is not None and isinstance(exc, oom)) or "out of memory" in str(exc).lower()


def _first_line(exc: BaseException, limit: int = 200) -> str:
    text = str(exc).strip().splitlines()
    return (text[0] if text else type(exc).__name__)[:limit]


def _free_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def summarise_reader(recs: Sequence[MapRecord], load_s: float) -> dict:
    """Timing and memory summary of one reader's maps (the first map apart: autotuning, compilation)."""
    secs = [r.seconds for r in recs]
    tiles = [r.tiles for r in recs]
    steady = secs[1:] if len(secs) > 1 else secs
    steady_tiles = tiles[1:] if len(tiles) > 1 else tiles
    out = dict(load_s=round(load_s, 2), maps=len(recs), first_map_s=secs[0] if secs else None,
               seconds_per_map=float(np.mean(steady)) if steady else None,
               tiles_per_map=float(np.mean(tiles)) if tiles else None,
               tiles_per_s=(sum(steady_tiles) / max(sum(steady), 1e-9)) if steady else None,
               total_s=round(sum(secs) + load_s, 2))
    for k in PEAK_KEYS:
        vals = [getattr(r, k) for r in recs if getattr(r, k) is not None]
        out[k] = max(vals) if vals else None
    prof = summarise_profile(recs)
    if prof is not None:
        out["profile"] = prof
    return out


def summarise_profile(recs: Sequence[MapRecord]) -> Optional[dict]:
    """Mean stage times of the steady maps (all but the first): seconds and share per map, calls per map, the time
    per forward call, and the map time no stage accounts for (``other_s``)."""
    timed = [r for r in recs if r.profile]
    if not timed:
        return None
    steady = timed[1:] if len(timed) > 1 else timed
    n = len(steady)
    per_map = sum(r.seconds for r in steady) / n
    seconds = {s: sum(r.profile["seconds"].get(s, 0.0) for r in steady) / n for s in StageTimer.STAGES}
    calls = {s: sum(r.profile["calls"].get(s, 0) for r in steady) / n for s in StageTimer.STAGES}
    return dict(maps=n, seconds_per_map=per_map, seconds=seconds, calls_per_map=calls,
                share={s: (seconds[s] / per_map if per_map > 0 else None) for s in StageTimer.STAGES},
                other_s=per_map - sum(seconds.values()),
                ms_per_forward=(1000.0 * seconds["forward"] / calls["forward"]) if calls["forward"] else None,
                tiles_per_s=sum(r.tiles for r in steady) / max(sum(r.seconds for r in steady), 1e-9))


def bench_config(surface: iox.SurfaceVolume, specs: Sequence[ReaderSpec], paths: dict, conf: dict,
                 out: str | Path, *, depths: Sequence[int], orientations: Sequence[str] = ("reverse",),
                 phases: Sequence[tuple[int, int]] = ((0, 0),), device: str = "cuda",
                 max_gpu_mem_gb: Optional[float] = None, profile: bool = False) -> dict:
    """One configuration (``conf``: mode, precision, batch, compile, cuda_graphs, label) over every reader."""
    label = conf.get("label") or conf["mode"]
    merge = "none" if len(phases) == 1 else "mean"
    cfg = SweepConfig(depths=list(depths), orientations=tuple(orientations), reading="reverse",
                      phases=tuple(tuple(p) for p in phases), phase_merge=merge, mode=conf["mode"],
                      precision=conf.get("precision", "reference"), batch=conf.get("batch"),
                      compile=bool(conf.get("compile", False)), cuda_graphs=bool(conf.get("cuda_graphs", False)),
                      keep_raw=True, max_gpu_mem_gb=max_gpu_mem_gb)
    out = Path(out)
    if out.exists():                       # a benchmark always computes: no resume markers from earlier runs
        shutil.rmtree(out)
    t0 = time.time()
    entry: dict = dict(label=label, config=conf, readers={})
    try:
        sweep = Sweep(surface, out, cfg, device=device, profile=profile)
    except Exception as exc:
        if not is_out_of_memory(exc):
            raise
        _free_cuda()
        entry.update(error=f"out of memory loading the surface volume: {_first_line(exc)}",
                     wall_s=round(time.time() - t0, 2))
        return entry
    entry["volume_to_gpu_s"] = round(sweep.load_seconds, 3)
    for spec in specs:
        tl = time.time()
        module = None
        try:
            module = sweep.prepare_module(spec, spec.load(paths[spec.name], sweep.device))
            load_s = time.time() - tl
            recs = [r for r in sweep.run_reader(spec, paths[spec.name], module=module) if not r.skipped]
            entry["readers"][spec.name] = summarise_reader(recs, load_s)
        except Exception as exc:
            if not is_out_of_memory(exc):
                raise
            entry["readers"][spec.name] = dict(oom=True, error=_first_line(exc), load_s=round(time.time() - tl, 2),
                                               maps=0)
        finally:
            module = None
            sweep._graphs.clear()
            _free_cuda()
    entry["wall_s"] = round(time.time() - t0, 2)
    del sweep
    _free_cuda()
    return entry


def bench(surface_path: str, readers: Sequence[str], depths: Sequence[int], configs: Sequence[dict],
          out_dir: str, *, ckpt_dir: str, download: bool = True, device: str = "cuda",
          rows: Optional[tuple[int, int]] = None, cols: Optional[tuple[int, int]] = None,
          voxel_um: Optional[float] = None, orientations: Sequence[str] = ("reverse",),
          phases: Sequence[tuple[int, int]] = ((0, 0),), reference_tsv: Optional[str] = None,
          plugins: Sequence[str] = (), max_gpu_mem_gb: Optional[float] = None, profile: bool = False) -> dict:
    surface = crop_surface(iox.open_surface(surface_path, voxel_size_um=voxel_um), rows, cols)
    specs = make_specs(readers, plugins)
    paths, _ = resolve_checkpoints(specs, ckpt_dir, download)
    report: dict = dict(surface=surface.name, shape=list(surface.shape), depths=list(depths),
                        orientations=list(orientations), phases=[list(p) for p in phases],
                        max_gpu_mem_gb=max_gpu_mem_gb, profile=bool(profile), configs=[])
    for conf in configs:
        label = conf.get("label") or conf["mode"]
        report["configs"].append(bench_config(surface, specs, paths, conf, Path(out_dir) / label, depths=depths,
                                              orientations=orientations, phases=phases, device=device,
                                              max_gpu_mem_gb=max_gpu_mem_gb, profile=profile))
    if reference_tsv:
        ref = reference_seconds(reference_tsv)
        report["reference"] = {}
        for spec in specs:
            key = REFERENCE_KEYS.get(spec.name, spec.name)
            if key in ref:
                vals = [ref[key][d] for d in depths if d in ref[key]]
                if vals:
                    # one reference job runs both orientations of a depth
                    per_map = float(np.mean(vals)) / 2.0
                    report["reference"][spec.name] = dict(seconds_per_map=per_map, jobs=len(vals))
                    for entry in report["configs"]:
                        r = entry["readers"].get(spec.name)
                        if r and r.get("seconds_per_map"):
                            r["speedup_vs_reference"] = round(per_map / r["seconds_per_map"], 2)
    return report


def _gib(mb: Optional[float]) -> str:
    return "-" if mb is None else f"{mb / 1024:.1f}"


def markdown(report: dict) -> str:
    head = f"surface {report['surface']} {report['shape']}, depths {report['depths']}"
    if report.get("max_gpu_mem_gb"):
        head += f", CUDA allocator capped at {report['max_gpu_mem_gb']:g} GiB"
    lines = [head, "",
             "| config | reader | s/map | tiles/map | tiles/s | first map s | load s | peak GPU GiB (reserved) | "
             "speedup vs reference |",
             "|---|---|---|---|---|---|---|---|---|"]
    for entry in report["configs"]:
        if entry.get("error"):
            lines.append(f"| {entry['label']} | all | {entry['error']} | - | - | - | - | - | - |")
            continue
        for name, r in entry["readers"].items():
            spm = r.get("seconds_per_map")
            tps = r.get("tiles_per_s")
            if r.get("oom"):
                lines.append(f"| {entry['label']} | {name} | out of memory | - | - | - | {r['load_s']:.1f} | "
                             f"over the cap | - |")
            elif spm is not None and tps is not None:
                lines.append(f"| {entry['label']} | {name} | {spm:.2f} | {r.get('tiles_per_map') or 0:.0f} | "
                             f"{tps:.0f} | {r.get('first_map_s') or 0:.2f} | {r['load_s']:.1f} | "
                             f"{_gib(r.get('gpu_peak_reserved_mb'))} | {r.get('speedup_vs_reference', '-')} |")
            else:
                lines.append(f"| {entry['label']} | {name} | - | - | - | - | {r['load_s']:.1f} | "
                             f"{_gib(r.get('gpu_peak_reserved_mb'))} | - |")
    prof = profile_markdown(report)
    return "\n".join(lines) + ("\n\n" + prof if prof else "")


def profile_markdown(report: dict) -> str:
    """The stage split of a profiled benchmark: seconds per map and share of the map time, per stage."""
    rows = [(e["label"], name, r["profile"]) for e in report["configs"] for name, r in e.get("readers", {}).items()
            if r.get("profile")]
    if not rows:
        return ""
    stages = StageTimer.STAGES
    lines = ["stage split per map, steady maps (CUDA-synchronised timers: compare shares, not throughput, with an "
             "unprofiled run)", "",
             "| config | reader | s/map | " + " | ".join(stages) + " | other | forward calls/map | ms/forward | "
             "tiles/s (profiled) |",
             "|" + "---|" * (len(stages) + 7)]
    for label, name, p in rows:
        cells = [f"{p['seconds'][s]:.3f} ({100 * p['share'][s]:.0f} %)" if p["share"][s] is not None
                 else f"{p['seconds'][s]:.3f}" for s in stages]
        ms = "-" if p.get("ms_per_forward") is None else f"{p['ms_per_forward']:.2f}"
        lines.append(f"| {label} | {name} | {p['seconds_per_map']:.3f} | " + " | ".join(cells) +
                     f" | {p['other_s']:.3f} | {p['calls_per_map']['forward']:.0f} | {ms} | {p['tiles_per_s']:.0f} |")
    return "\n".join(lines)

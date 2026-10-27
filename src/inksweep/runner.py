"""``inksweep run``: resolve readers and checkpoints, run the sweep (one or several GPUs), the ensemble, ``ink.json``."""
from __future__ import annotations

import logging
import os
import platform
import sys
import time
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from . import io as iox
from .readers import builtin_readers, load_plugin
from .readers import checkpoints as ck
from .readers.spec import ReaderSpec
from .sweep import Sweep, SweepConfig, run_parallel

LOG = logging.getLogger("inksweep")


def make_specs(names: Sequence[str], plugins: Sequence[str] = (), overrides: Optional[dict] = None) -> list[ReaderSpec]:
    """Reader specs in the requested order (picklable entry point for the GPU workers)."""
    builtin = builtin_readers()
    plug = {s.name: s for s in (load_plugin(p) for p in plugins)}
    out = []
    for n in names:
        if n in plug:
            spec = plug[n]
        elif n in builtin:
            spec = builtin[n]
        else:
            raise KeyError(f"unknown reader {n!r}; built-in: {', '.join(sorted(builtin))}; plugins: {sorted(plug)}")
        changes = dict((overrides or {}).get(n, {}))
        if "tile" in changes and spec.family == "villa":    # rebuild: the normaliser depends on the tile size
            from .readers.villa import villa_reader

            o = spec.options
            spec = villa_reader(spec.name, spec.checkpoints, kind=o["payload"], members=o["members"],
                                tile=int(changes.pop("tile")), depth=spec.window_layers, tta_mirror=o["tta_mirror"],
                                citation=spec.citation, notes=spec.notes, preprocessing=o["preprocessing"])
        if changes:
            spec = spec.with_options(**changes)
        out.append(spec)
    names_seen = [s.name for s in out]
    if len(set(names_seen)) != len(names_seen):
        raise ValueError(f"duplicate reader names: {names_seen}")
    return out


def resolve_checkpoints(specs: Sequence[ReaderSpec], ckpt_dir: str, download: bool,
                        explicit: Optional[dict] = None) -> tuple[dict, dict]:
    """Paths per reader (``explicit`` wins) and their records (pinned key, sha256 of the file actually used)."""
    paths, records = {}, {}
    explicit = explicit or {}
    for s in specs:
        if s.name in explicit:
            files = [str(Path(p)) for p in explicit[s.name]]
            paths[s.name] = files
            records[s.name] = [dict(path=Path(f).name, sha256=ck.sha256_file(f), source="given on the command line")
                               for f in files]
            continue
        if not s.checkpoints:
            raise ValueError(f"{s.name}: no default checkpoint; pass --checkpoint {s.name}=PATH")
        files = [str(ck.fetch(k, ckpt_dir, download)) for k in s.checkpoints]
        paths[s.name] = files
        records[s.name] = ck.record(s.checkpoints)
    return paths, records


def software() -> dict:
    info: dict = dict(inksweep=__version__, python=platform.python_version(), platform=platform.platform())
    for mod in ("numpy", "torch", "cv2", "zarr", "tifffile", "scipy", "safetensors"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "?")
        except Exception:
            info[mod] = None
    try:
        import torch

        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gib"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    try:
        import vesuvius

        from .doctor import villa_source

        src = villa_source(vesuvius.__file__)      # a villa git checkout, or pip's record of a git install
        info["villa_source"] = src["kind"]
        info["villa_checkout"] = src["where"]
        info["villa_commit"] = src["commit"]
    except Exception:
        info["villa_commit"] = None
    return info


PEAK_KEYS = ("gpu_peak_allocated_mb", "gpu_peak_reserved_mb")


def summarise_records(records: Sequence[dict]) -> dict:
    """Per reader: maps, tiles and seconds summed over its computed maps, and the largest per-map GPU memory peaks
    (MiB; ``None`` when not measured, e.g. on the CPU)."""
    per_reader: dict = {}
    for r in records:
        if r.get("skipped"):
            continue
        d = per_reader.setdefault(r["reader"], dict(maps=0, tiles=0, seconds=0.0, **{k: None for k in PEAK_KEYS}))
        d["maps"] += 1
        d["tiles"] += r["tiles"]
        d["seconds"] = round(d["seconds"] + r["seconds"], 2)
        for k in PEAK_KEYS:
            v = r.get(k)
            if v is not None:
                d[k] = v if d[k] is None else max(d[k], v)
    return per_reader


def run(surface_path: str, out_dir: str, reader_names: Sequence[str], cfg: SweepConfig, *,
        plugins: Sequence[str] = (), ckpt_dir: str = "checkpoints", download: bool = True,
        devices: Sequence[str] = ("cuda",), phase_mask_spec: Optional[str] = None,
        voxel_um: Optional[float] = None, explicit_checkpoints: Optional[dict] = None,
        overrides: Optional[dict] = None, ensemble_cfg: Optional[dict] = None, command: str = "",
        name: Optional[str] = None) -> dict:
    """The whole run; returns the ``ink.json`` dict (also written to ``out_dir/ink.json``)."""
    t_start = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    surface = iox.open_surface(surface_path, voxel_size_um=voxel_um, name=name)
    specs = make_specs(reader_names, plugins, overrides)
    paths, ck_records = resolve_checkpoints(specs, ckpt_dir, download, explicit_checkpoints)
    phase_mask = iox.load_phase_mask(phase_mask_spec, (surface.height, surface.width)) if phase_mask_spec else None
    surface_args = dict(path=str(surface_path), voxel_size_um=voxel_um, name=name)
    t_sweep = time.time()
    if len(devices) > 1:
        factory = partial(make_specs, list(reader_names), list(plugins), overrides)
        records = run_parallel(factory, surface_args, str(out), cfg, paths, list(devices), phase_mask)
        load_s = None
    else:
        sweep = Sweep(surface, out, cfg, device=devices[0], phase_mask=phase_mask)
        load_s = sweep.load_seconds
        records = []
        for s in specs:
            records += [asdict(r) for r in sweep.run_reader(s, paths[s.name])]
        del sweep
    sweep_s = time.time() - t_sweep
    ink = dict(schema="inksweep/ink/1", command=command,
               surface=dict(path=str(surface_path), name=surface.name, kind=surface.kind, shape=list(surface.shape),
                            voxel_size_um=surface.voxel_size_um, meta=surface.meta,
                            volume_mb=round(float(np.prod(surface.shape)) / 2**20, 1)),   # its size on the GPU
               readers={s.name: dict(spec=s.describe(), checkpoints=ck_records[s.name]) for s in specs},
               settings=asdict(cfg) | dict(phase_mask=phase_mask_spec, devices=list(devices)),
               timings=dict(volume_load_s=load_s, sweep_wall_s=round(sweep_s, 2)), maps=records,
               software=software())
    ink["timings"]["per_reader"] = summarise_records(records)
    iox.write_json(out / "ink.json", ink)
    if ensemble_cfg is not None and ensemble_cfg.get("models"):
        from . import ensemble as ens

        ecfg = ens.EnsembleConfig(models=list(ensemble_cfg["models"]), stat_depths=list(ensemble_cfg["stat_depths"]),
                                  write_depths=list(ensemble_cfg["write_depths"]),
                                  display_depth=int(ensemble_cfg["display_depth"]),
                                  orientation=ensemble_cfg.get("orientation", "reading"),
                                  area=ensemble_cfg.get("area"), voxel_um=surface.voxel_size_um,
                                  bins=ens.ensemble_bins(ink), grey=ensemble_cfg.get("grey", True))
        t_e = time.time()
        ink["ensemble"] = ens.compute(out, surface, ecfg)
        ink["ensemble_settings"] = asdict(ecfg)
        ink["timings"]["ensemble_s"] = round(time.time() - t_e, 2)
    ink["timings"]["total_s"] = round(time.time() - t_start, 2)
    iox.write_json(out / "ink.json", ink)
    return ink

"""Command line: ``inksweep run | ensemble | depth-mean | compare | export-reference | bench | fetch | readers | render |
compare-render | export-ct | doctor``.

Nothing here imports torch at start-up: each command imports what it needs, so that ``inksweep doctor`` can report a
torch that crashes on import instead of crashing with it.
"""
from __future__ import annotations

import argparse
import json
import logging
import shlex
import sys
from pathlib import Path
from typing import Optional, Sequence

from . import __version__


def parse_depths(spec: str | Sequence[str]) -> list[int]:
    """:func:`inksweep.tiling.parse_depths`, imported on first use (``inksweep.tiling`` imports torch)."""
    from .tiling import parse_depths as parse

    return parse(spec)


RECIPE_DEFAULTS = {
    "pherc0343": dict(readers="dnative,hecate,ink9_s42,ink9_s43", depths="-15..15", orientations="both",
                      stat_depths="-15..15", write_depths="1", display_depth=1),
}


def _csv(s: Optional[str]) -> list[str]:
    return [t for t in (s or "").split(",") if t]


def _pairs(values: Sequence[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for v in values or ():
        name, _, rest = v.partition("=")
        if not rest:
            raise SystemExit(f"expected NAME=VALUE, got {v!r}")
        out[name] = rest.split(",")
    return out


def _range(s: Optional[str]) -> Optional[tuple[int, int]]:
    if not s:
        return None
    a, b = s.split(":")
    return int(a), int(b)


def _orientations(s: str, reading: str) -> tuple[str, ...]:
    if s == "both":
        other = "forward" if reading == "reverse" else "reverse"
        return (reading, other)
    if s == "reading":
        return (reading,)
    return tuple(_csv(s))


def add_run_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("surface", help="surface-volume directory (layers.zarr + surface.json), zarr render or TIFF stack")
    p.add_argument("--out", required=True, help="output directory (ink/, ink.json)")
    p.add_argument("--name", help="segment name used in image and statistics file names (default: the input's)")
    p.add_argument("--recipe", choices=sorted(RECIPE_DEFAULTS), help="named defaults (pherc0343 = the winning recipe)")
    p.add_argument("--readers", help="comma list of reader names (see `inksweep readers`)")
    p.add_argument("--plugin", action="append", default=[], help="reader plugin module or .py file (defines READER)")
    p.add_argument("--checkpoint", action="append", default=[], help="NAME=PATH[,PATH...]: weights for a reader")
    p.add_argument("--stride", action="append", default=[], help="NAME=N: override a reader's tile stride")
    p.add_argument("--tile", action="append", default=[],
                   help="NAME=N: override a villa-family reader's tile (must equal its checkpoint's crop size)")
    p.add_argument("--depths", help="e.g. -15..15 (default 0, or the recipe's)")
    p.add_argument("--orientations", default=None, help="both | reading | forward | reverse")
    p.add_argument("--reading", default="reverse", choices=("reverse", "forward"),
                   help="orientation of the reading side (reverse for renders with +d away from the scroll axis)")
    p.add_argument("--phase", action="append", default=[], help="tile phase PAD_ROWS,PAD_COLS (repeatable)")
    p.add_argument("--phase-merge", choices=("none", "select", "mean"), default=None)
    p.add_argument("--phase-mask", help="select-merge mask: npz:NODE_SOURCE.npz[:1,4[:20]] | npy:MASK.npy | tif:MASK")
    p.add_argument("--mode", choices=("exact", "fast"), default="exact")
    p.add_argument("--precision", choices=("reference", "fp32", "bf16", "fp16"), default="reference")
    p.add_argument("--batch", type=int)
    p.add_argument("--compile", action="store_true", help="fast mode: torch.compile the readers")
    p.add_argument("--cuda-graphs", action="store_true",
                   help="CUDA graphs for the forward passes (exact mode: replays the same kernels as eager execution)")
    p.add_argument("--no-channels-last", action="store_true")
    p.add_argument("--no-reference-storage", action="store_true",
                   help="do not emulate the reference's 8-bit storage in the contract maps")
    p.add_argument("--no-raw", action="store_true", help="do not keep the raw (uncalibrated) maps")
    p.add_argument("--keep-phases", action="store_true", help="keep the per-phase raw maps")
    p.add_argument("--no-need-masks", action="store_true", help="select merge: run every tile in both phases")
    p.add_argument("--gpus", default=None, help="comma list of CUDA device indices (one process each)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-gpu-mem-gb", type=float, default=None,
                   help="cap PyTorch's CUDA allocator at this many GiB per GPU (a test aid to emulate a smaller card)")
    p.add_argument("--ckpt-dir", default="checkpoints")
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--voxel-um", type=float, help="render voxel size (overrides surface.json)")
    p.add_argument("--resample-threads", type=int, default=8)
    p.add_argument("--no-ensemble", action="store_true")
    add_ensemble_args(p, with_surface=False)


def add_ensemble_args(p: argparse.ArgumentParser, with_surface: bool = True) -> None:
    if with_surface:
        p.add_argument("--surface", required=True)
        p.add_argument("--voxel-um", type=float)
    p.add_argument("--ensemble-models", help="comma list, in ensemble order (default: the readers)")
    p.add_argument("--stat-depths", default=None)
    p.add_argument("--write-depths", default=None)
    p.add_argument("--display-depth", type=int, default=None)
    p.add_argument("--display-orientation", default="reading", choices=("reading", "raw", "ud", "lr"))
    p.add_argument("--area", help="preview area x0,y0,x1,y1 (raw render pixels, exclusive)")
    p.add_argument("--no-grey", action="store_true", help="skip the recipe-style grey images")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="inksweep", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--version", action="version", version=f"inksweep {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)
    add_run_args(sub.add_parser("run", help="depth sweep + ensemble"))
    e = sub.add_parser("ensemble", help="(re)compute the ensemble and images from existing maps")
    e.add_argument("out")
    e.add_argument("--name")
    add_ensemble_args(e)
    dm = sub.add_parser("depth-mean", help="derived maps: mean of a model over depth offsets (e.g. Hecate +-1)")
    dm.add_argument("out")
    dm.add_argument("--model", required=True)
    dm.add_argument("--offsets", default="-1,0,1")
    dm.add_argument("--depths", default="-15..15")
    dm.add_argument("--name")
    c = sub.add_parser("compare", help="equivalence with the reference pipeline's maps")
    c.add_argument("out")
    c.add_argument("reference")
    c.add_argument("--name", required=True, help="the reference maps' name prefix")
    c.add_argument("--readers", default="dnative,hecate,ink9_s42,ink9_s43")
    c.add_argument("--depths", default="-15..15")
    c.add_argument("--orientations", default="reverse,forward")
    c.add_argument("--reading", default="reverse")
    c.add_argument("--r-min", type=float, default=0.9999)
    c.add_argument("--max-diff", type=float, default=1 / 255)
    c.add_argument("--layers", type=int, default=66)
    c.add_argument("--report", default=None, help="report path (.json; a .md table is written next to it)")
    c.add_argument("--ref-stats", help="the reference ensemble_maps.py stats.json")
    c.add_argument("--ref-grey", help="folder of the reference ensemble_maps.py images")
    x = sub.add_parser("export-reference", help="write maps under the reference pipeline's names and encodings")
    x.add_argument("out")
    x.add_argument("dest")
    x.add_argument("--name", required=True)
    x.add_argument("--readers", default="dnative,hecate,ink9_s42,ink9_s43")
    x.add_argument("--depths", default="-15..15")
    x.add_argument("--orientations", default="reverse,forward")
    x.add_argument("--reading", default="reverse")
    x.add_argument("--layers", type=int, default=66)
    b = sub.add_parser("bench", help="wall time and tiles/s per reader and mode")
    b.add_argument("surface")
    b.add_argument("--out", required=True)
    b.add_argument("--readers", default="dnative,hecate,ink9_s42,ink9_s43")
    b.add_argument("--depths", default="0")
    b.add_argument("--configs", default="exact,fast", help="comma list of exact | fast | fast-compile | fast-graphs")
    b.add_argument("--rows")
    b.add_argument("--cols")
    b.add_argument("--orientations", default="reverse")
    b.add_argument("--phase", action="append", default=[])
    b.add_argument("--reference-tsv")
    b.add_argument("--ckpt-dir", default="checkpoints")
    b.add_argument("--no-download", action="store_true")
    b.add_argument("--device", default="cuda")
    b.add_argument("--voxel-um", type=float)
    b.add_argument("--batch", type=int)
    b.add_argument("--max-gpu-mem-gb", type=float, default=None,
                   help="cap PyTorch's CUDA allocator at this many GiB (emulates a smaller card; out of memory is "
                        "recorded per reader)")
    b.add_argument("--profile", action="store_true",
                   help="split each map's time into read, schedule, normalise, forward, blend and storage "
                        "(CUDA-synchronised timers; slower, so off by default)")
    f = sub.add_parser("fetch", help="download and verify checkpoints")
    f.add_argument("--readers", default="dnative,hecate,ink9_s42,ink9_s43")
    f.add_argument("--ckpt-dir", default="checkpoints")
    sub.add_parser("readers", help="list the built-in readers")
    r = sub.add_parser("render", help="render a tifxyz mesh into a surface volume (clean-room renderer)")
    from .render import add_render_args

    add_render_args(r)
    cr = sub.add_parser("compare-render", help="compare two surface volumes of one mesh layer by layer (e.g. VC3D's "
                                               "render and inksweep's)")
    cr.add_argument("a", help="first surface volume (zarr, surface-volume folder or TIFF stack), e.g. VC3D's zarr")
    cr.add_argument("b", help="second surface volume, e.g. the inksweep render folder")
    cr.add_argument("--names", default="a,b", help="labels of the two renders in the report, e.g. vc3d,inksweep")
    cr.add_argument("--report", help="write the report here (.json; a .md table is written next to it)")
    cr.add_argument("--max-shift", type=int, default=0,
                    help="also correlate the --shift-depth layer at row/column offsets up to this many pixels")
    cr.add_argument("--shift-depth", type=int, default=1, help="depth label of the shift search (default +1)")
    cr.add_argument("--block-rows", type=int, help="rows read at a time (default: a multiple of the chunk height)")
    ex = sub.add_parser("export-ct", help="write the CT chunks cached by `inksweep render` as a local OME-Zarr "
                                          "(one level), e.g. for VC3D")
    ex.add_argument("--volume", required=True, help="the volume the cache was filled from (as given to inksweep render)")
    ex.add_argument("--cache", required=True, help="the --cache folder of inksweep render")
    ex.add_argument("--out", required=True, help="the local OME-Zarr to write")
    ex.add_argument("--level", default="0", help="resolution level (inksweep render reads level 0)")
    ex.add_argument("--copy", action="store_true", help="copy the chunk files instead of hard-linking them")
    ex.add_argument("--keep-attrs", action="store_true",
                    help="keep the volume's .zattrs as is (it then lists levels the copy does not have)")
    dr = sub.add_parser("doctor", help="check Python, torch and CUDA, OpenCV, the vesuvius package, checkpoints, disk")
    dr.add_argument("--ckpt-dir", default="checkpoints", help="checkpoint folder to verify (as for run and fetch)")
    dr.add_argument("--readers", default="dnative,hecate,ink9_s42,ink9_s43",
                    help="readers whose checkpoints to verify (default: the pherc0343 recipe)")
    dr.add_argument("--out", default=".", help="where outputs will go (free-space check)")
    dr.add_argument("--min-free-gb", type=float, default=50.0, help="warn below this much free space for outputs")
    dr.add_argument("--no-hash", action="store_true", help="check checkpoint sizes only, not their sha256")
    dr.add_argument("--json", action="store_true", help="print the checks as JSON (e.g. for a bug report)")
    return ap


def _ensemble_cfg(a: argparse.Namespace, readers: Sequence[str], defaults: dict) -> dict:
    models = _csv(a.ensemble_models) or list(readers)
    return dict(models=models,
                stat_depths=parse_depths(a.stat_depths or defaults.get("stat_depths") or a.depths or "0"),
                write_depths=parse_depths(a.write_depths or defaults.get("write_depths") or "0"),
                display_depth=a.display_depth if a.display_depth is not None else defaults.get("display_depth", 0),
                orientation=a.display_orientation,
                area=tuple(int(v) for v in a.area.split(",")) if a.area else None, grey=not a.no_grey)


def cmd_run(a: argparse.Namespace) -> int:
    from .runner import run
    from .sweep import SweepConfig

    d = RECIPE_DEFAULTS.get(a.recipe or "", {})
    readers = _csv(a.readers or d.get("readers"))
    if not readers:
        raise SystemExit("give --readers or --recipe")
    a.depths = a.depths or d.get("depths") or "0"
    phases = tuple(tuple(int(v) for v in p.split(",")) for p in a.phase) or ((0, 0),)
    merge = a.phase_merge or ("none" if len(phases) == 1 else ("select" if a.phase_mask else "mean"))
    cfg = SweepConfig(depths=parse_depths(a.depths), orientations=_orientations(a.orientations or
                                                                                d.get("orientations", "reading"),
                                                                                a.reading),
                      reading=a.reading, phases=phases, phase_merge=merge, mode=a.mode, precision=a.precision,
                      batch=a.batch, compile=a.compile, cuda_graphs=a.cuda_graphs,
                      channels_last=not a.no_channels_last, reference_storage=not a.no_reference_storage,
                      keep_raw=not a.no_raw, keep_phases=a.keep_phases, need_masks=not a.no_need_masks,
                      resample_threads=a.resample_threads, max_gpu_mem_gb=a.max_gpu_mem_gb)
    devices = [f"cuda:{g}" for g in _csv(a.gpus)] if a.gpus else [a.device]
    overrides: dict = {}
    for k, v in _pairs(a.tile).items():
        overrides.setdefault(k, {}).update(tile=int(v[0]), stride=int(v[0]) // 2)
    for k, v in _pairs(a.stride).items():
        overrides.setdefault(k, {})["stride"] = int(v[0])
    ens = None if a.no_ensemble else _ensemble_cfg(a, readers, d)
    ink = run(a.surface, a.out, readers, cfg, plugins=a.plugin, ckpt_dir=a.ckpt_dir, download=not a.no_download,
              devices=devices, phase_mask_spec=a.phase_mask, voxel_um=a.voxel_um,
              explicit_checkpoints=_pairs(a.checkpoint), overrides=overrides, ensemble_cfg=ens,
              command=" ".join(shlex.quote(t) for t in sys.argv), name=a.name)
    t = ink["timings"]
    print(json.dumps(dict(out=a.out, maps=len([m for m in ink["maps"] if not m.get("skipped")]),
                          sweep_wall_s=t.get("sweep_wall_s"), total_s=t.get("total_s"),
                          per_reader=t.get("per_reader")), indent=1))
    return 0


def cmd_ensemble(a: argparse.Namespace) -> int:
    from . import ensemble as ens
    from . import io as iox

    ink = ens.load_ink_json(a.out)
    readers = list(ink.get("readers", {})) or _csv(a.ensemble_models)
    a.depths = None
    prev = ink.get("ensemble_settings", {})                 # defaults: the run's own ensemble settings
    if prev and not a.ensemble_models:
        a.ensemble_models = ",".join(prev.get("models", []))
    if prev and a.area is None and prev.get("area"):
        a.area = ",".join(str(v) for v in prev["area"])
    defaults = dict(stat_depths=",".join(str(d) for d in prev.get("stat_depths", [])) or None,
                    write_depths=",".join(str(d) for d in prev.get("write_depths", [])) or None,
                    display_depth=prev.get("display_depth", 0))
    cfgd = _ensemble_cfg(a, readers, defaults)
    surface = iox.open_surface(a.surface, voxel_size_um=a.voxel_um, name=a.name)
    cfg = ens.EnsembleConfig(models=cfgd["models"], stat_depths=cfgd["stat_depths"],
                             write_depths=cfgd["write_depths"], display_depth=cfgd["display_depth"],
                             orientation=cfgd["orientation"], area=cfgd["area"], voxel_um=surface.voxel_size_um,
                             bins=ens.ensemble_bins(ink), grey=cfgd["grey"])
    stats = ens.compute(a.out, surface, cfg)
    print(json.dumps(stats.get("ensemble", {}), indent=1))
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    from . import compare as cmp

    report = cmp.compare(a.out, a.reference, a.name, _csv(a.readers), parse_depths(a.depths),
                         orientations=_csv(a.orientations), reading=a.reading, r_min=a.r_min, max_diff=a.max_diff,
                         n_layers=a.layers)
    if a.ref_stats:
        ours = Path(a.out) / "ink" / "stats.json"
        if ours.is_file():
            report["stats"] = cmp.compare_stats(cmp.load_json(ours), cmp.load_json(a.ref_stats))
    if a.ref_grey:
        report["grey_images"] = cmp.compare_images(Path(a.out) / "ink" / "grey", a.ref_grey)
    path = a.report or str(Path(a.out) / "compare.json")
    cmp.save_report(report, path)
    print(cmp.markdown_table(report, limit=20))
    s = report["summary"]
    print(f"ACCEPTANCE: {'PASS' if s['all_passed'] else 'FAIL'} ({s['passed']}/{s['maps']} maps; "
          f"r >= {a.r_min}, max |diff| <= {a.max_diff * 255:.0f}/255)")
    return 0 if s["all_passed"] else 1


def cmd_export(a: argparse.Namespace) -> int:
    from . import compare as cmp

    n = cmp.export_reference(a.out, a.dest, a.name, _csv(a.readers), parse_depths(a.depths),
                             orientations=_csv(a.orientations), reading=a.reading, n_layers=a.layers)
    print(f"{n} maps written to {a.dest}")
    return 0 if n else 1


BENCH_CONFIGS = {"exact": dict(mode="exact"), "fast": dict(mode="fast"),
                 "fast-compile": dict(mode="fast", compile=True),
                 "fast-graphs": dict(mode="fast", cuda_graphs=True),
                 "exact-fp32": dict(mode="exact", precision="fp32")}


def cmd_bench(a: argparse.Namespace) -> int:
    from .bench import bench, markdown
    from . import io as iox

    configs = []
    for c in _csv(a.configs):
        conf = dict(BENCH_CONFIGS[c], label=c)
        if a.batch:
            conf["batch"] = a.batch
        configs.append(conf)
    phases = tuple(tuple(int(v) for v in p.split(",")) for p in a.phase) or ((0, 0),)
    report = bench(a.surface, _csv(a.readers), parse_depths(a.depths), configs, a.out, ckpt_dir=a.ckpt_dir,
                   download=not a.no_download, device=a.device, rows=_range(a.rows), cols=_range(a.cols),
                   voxel_um=a.voxel_um, orientations=_csv(a.orientations), phases=phases,
                   reference_tsv=a.reference_tsv, max_gpu_mem_gb=a.max_gpu_mem_gb, profile=a.profile)
    iox.write_json(Path(a.out) / "bench.json", report)
    md = markdown(report)
    (Path(a.out) / "bench.md").write_text(md + "\n")
    print(md)
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    from .readers import checkpoints as ck
    from .runner import make_specs

    for s in make_specs(_csv(a.readers)):
        for k in s.checkpoints:
            p = ck.fetch(k, a.ckpt_dir)
            print(f"ok {k} {ck.REGISTRY[k].sha256[:12]} {p}")
    return 0


def cmd_readers(a: argparse.Namespace) -> int:
    from .readers import RECIPES, builtin_readers

    for name, s in sorted(builtin_readers().items()):
        print(f"{name:20s} {s.family:9s} W={s.window_layers:<3d} tile={s.tile:<4d} stride={s.stride:<4d} "
              f"{s.voxel_size_um:5.2f} um  {s.citation}")
    for k, v in RECIPES.items():
        print(f"recipe {k}: {', '.join(v)}")
    return 0


def cmd_depth_mean(a: argparse.Namespace) -> int:
    from .ensemble import depth_mean

    name = depth_mean(a.out, a.model, [int(v) for v in a.offsets.split(",")], parse_depths(a.depths), a.name)
    print(name)
    return 0


def cmd_compare_render(a: argparse.Namespace) -> int:
    from . import compare as cmp

    report = cmp.compare_renders(a.a, a.b, names=(_csv(a.names) + ["a", "b"])[:2], block_rows=a.block_rows,
                                 max_shift=a.max_shift, shift_depth=a.shift_depth)
    if a.report:
        cmp.save_render_report(report, a.report)
    print(cmp.render_table(report))
    return 0


def cmd_export_ct(a: argparse.Namespace) -> int:
    from .render import export_level

    print(json.dumps(export_level(a.volume, a.cache, a.out, level=a.level, copy=a.copy, keep_attrs=a.keep_attrs),
                     indent=1))
    return 0


def cmd_doctor(a: argparse.Namespace) -> int:
    from dataclasses import asdict

    from . import doctor

    checks = doctor.run_checks(ckpt_dir=a.ckpt_dir, readers=_csv(a.readers), out_dir=a.out,
                               hash_files=not a.no_hash, min_free_gb=a.min_free_gb)
    if a.json:
        print(json.dumps(dict(inksweep=__version__, python=sys.executable, checks=[asdict(c) for c in checks]),
                         indent=1))
    else:
        print(f"inksweep {__version__} doctor, Python {sys.executable}")
        print(doctor.format_checks(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    handlers = {"run": cmd_run, "ensemble": cmd_ensemble, "compare": cmd_compare, "export-reference": cmd_export,
                "bench": cmd_bench, "fetch": cmd_fetch, "readers": cmd_readers, "depth-mean": cmd_depth_mean,
                "doctor": cmd_doctor, "compare-render": cmd_compare_render, "export-ct": cmd_export_ct}
    if a.cmd == "render":
        from .render import cmd_render

        return cmd_render(a)
    return handlers[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

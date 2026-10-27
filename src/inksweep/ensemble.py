"""The PHerc. 343 ensemble: one contrast stretch per model over all depths, the mean of per-model z-scores, images.

Transcription of ``ink/scripts/ensemble_maps.py`` and of the overlay step of ``make_submission_image.py``
(Nieuwlaar/pherc343-first-letters, MIT, Copyright (c) 2026 Erwin Nieuwlaar). All statistics come from the reading side;
the blind side is shown with the reading side's statistics, never its own.

* valid pixels: CT > 0 in every layer of the surface volume;
* per model: ``p1`` and ``p99.5`` of all valid pixels of all reading-side maps at the stat depths, exact from a
  histogram (129 bins for the 8-bit villa maps, 65536 bins for continuous values); pooled mean and SD (float64 sums,
  ``sd = sqrt(max(E[v^2] - mean^2, 0)) + 1e-6``);
* ensemble: mean of the models' z-scores ``(map - mean) / sd`` (float32, summed in recipe order); one ensemble stretch
  ``p1 .. p99.5`` over a fixed grid (every 8th row and column, from 4) of valid pixels at every depth where all
  models exist;
* CT: layer ``L//2 + d``, one stretch ``p1 .. p99.5`` of the valid pixels of layers ``L//2 - 15 .. L//2 + 15``;
* images: ``uint8(clip((x - lo) / (hi - lo), 0, 1) * 255)``, invalid pixels black, reading orientation = rotated
  180 degrees (configurable).

Outputs (the "ink outputs" contract plus extras): ``ink/ensemble.tif`` (the stretched ensemble at the display depth,
float32 [0, 1]), ``ink/ensemble/dNN.tif`` (raw z-mean per depth; ``blind/`` for the blind side),
``ink/preview_dNN.png`` (ensemble in yellow over the dimmed CT, 1 cm scale bar), ``ink/grey/*.png`` (the recipe's grey
images, named as the recipe names them), ``ink/stats.json`` (the recipe's statistics file) and ``ink/valid.tif``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import io as iox
from .tiling import depth_tag

P_LO, P_HI = 1.0, 99.5
FINE = 65536
SUB = 8
# the recipe's short names in its stats.json / image names
RECIPE_NAMES = {"dnative": "dnative", "hecate": "hecate", "ink9_s42": "ink9-42", "ink9_s43": "ink9-43"}
STYLES = {"dimmed-ct": dict(color=(255, 235, 120), alpha=0.9, threshold=0.35, full=0.75, ct_dim=0.45),
          "default": dict(color=(255, 140, 0), alpha=0.75, threshold=0.5, full=0.9, ct_dim=1.0)}


def pct_hist(hist: np.ndarray, values: np.ndarray, qs: Sequence[float]) -> list[float]:
    """``np.percentile`` (linear) computed exactly from a histogram over discrete values (recipe ``pct_hist``)."""
    c = np.cumsum(hist)
    n = int(c[-1])
    out = []
    for q in qs:
        pos = q / 100.0 * (n - 1)
        lo = int(math.floor(pos))
        fr = pos - lo
        i_lo = int(np.searchsorted(c, lo + 1))
        i_hi = int(np.searchsorted(c, min(lo + 2, n))) if fr > 0 else i_lo
        out.append(float(values[i_lo] + fr * (values[i_hi] - values[i_lo])))
    return out


def to8(m: np.ndarray, lo: float, hi: float, valid: np.ndarray, flip: tuple[bool, bool]) -> np.ndarray:
    """Recipe ``to8``: stretch, clip, 8 bit (truncating), invalid black, then the display flips (ud, lr)."""
    b = (np.clip((m - lo) / max(hi - lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    b[~valid] = 0
    return orient(b, flip)


def orient(a: np.ndarray, flip: tuple[bool, bool]) -> np.ndarray:
    if flip[0]:
        a = a[::-1]
    if flip[1]:
        a = a[:, ::-1]
    return np.ascontiguousarray(a)


def parse_flip(orientation: str) -> tuple[bool, bool]:
    """``reading`` (the recipe default: rows and columns reversed), ``raw``, ``ud`` or ``lr``."""
    return {"reading": (True, True), "both": (True, True), "raw": (False, False), "none": (False, False),
            "ud": (True, False), "lr": (False, True)}[orientation]


@dataclass
class EnsembleConfig:
    models: list[str]
    stat_depths: list[int]
    write_depths: list[int]
    display_depth: int = 1
    orientation: str = "reading"
    style: str = "dimmed-ct"
    area: Optional[tuple[int, int, int, int]] = None        # x0, y0, x1, y1 in raw render pixels (exclusive)
    voxel_um: Optional[float] = None
    bins: dict[str, str] = field(default_factory=dict)      # model -> "villa129" | "fine"
    grey: bool = True


def map_file(out: Path, model: str, side: str, d: int) -> Path:
    base = out / "ink" / model
    return (base if side == "reading" else base / "blind") / f"{depth_tag(d)}.tif"


def _ct_stats(surface: iox.SurfaceVolume, write_depths: Sequence[int]) -> tuple[np.ndarray, list, dict]:
    mid = surface.n_layers // 2
    lo_l, hi_l = max(0, mid - 15), min(surface.n_layers - 1, mid + 15)
    valid = np.zeros((surface.height, surface.width), bool)
    hist = np.zeros(256, np.int64)
    layers: dict[int, list[np.ndarray]] = {}
    for r0, blk in surface.iter_row_blocks(128):
        g = (blk > 0).all(0)
        valid[r0:r0 + blk.shape[1]] = g
        sel = blk[lo_l:hi_l + 1]
        hist += np.bincount(sel[:, g].ravel(), minlength=256)
        for d in write_depths:
            layers.setdefault(d, []).append(blk[mid + d])
    ct_lo, ct_hi = pct_hist(hist, np.arange(256, dtype=float), [P_LO, P_HI])
    ct = {d: np.concatenate(v, axis=0) for d, v in layers.items()}
    return valid, [ct_lo, ct_hi], dict(ct=ct, layers=[lo_l, hi_l], mid=mid)


def compute(out_dir: str | Path, surface: iox.SurfaceVolume, cfg: EnsembleConfig) -> dict:
    """The recipe's statistics, the ensemble maps and the images; returns the stats dict (also ``ink/stats.json``)."""
    out = Path(out_dir)
    flip = parse_flip(cfg.orientation)
    h, w = surface.height, surface.width
    valid, (ct_lo, ct_hi), ctx = _ct_stats(surface, sorted(set(cfg.write_depths) | {cfg.display_depth}))
    sub = np.zeros_like(valid)
    sub[SUB // 2::SUB, SUB // 2::SUB] = True
    sub &= valid
    stats: dict = dict(surface=surface.name, H=h, W=w, valid_px=int(valid.sum()),
                       ct=dict(p1=ct_lo, p99_5=ct_hi, layers=ctx["layers"]), models={})
    subs: dict[str, np.ndarray] = {}
    for model in cfg.models:
        villa = cfg.bins.get(model, "fine") == "villa129"
        hist = np.zeros(129 if villa else FINE, np.int64)
        s1 = s2 = 0.0
        n = 0
        ds, sv = [], []
        for d in cfg.stat_depths:
            p = map_file(out, model, "reading", d)
            if not p.is_file():
                continue
            m = iox.read_tif(p).astype(np.float32, copy=False)
            if m.shape != (h, w):
                raise ValueError(f"{p}: shape {m.shape} != surface {(h, w)}")
            v = m[valid]
            if villa:
                hist += np.bincount(np.rint(v * 128).astype(np.int64), minlength=129)[:129]
            else:
                hist += np.bincount(np.minimum((v * FINE).astype(np.int64), FINE - 1), minlength=FINE)
            s1 += float(v.sum(dtype=np.float64))
            s2 += float(np.square(v, dtype=np.float64).sum())
            n += v.size
            sv.append(m[sub].astype(np.float32))
            ds.append(d)
        if not ds:
            stats["models"][model] = dict(depths=[])
            continue
        values = (np.arange(129) / 128.0) if villa else (np.arange(FINE) / FINE)
        lo, hi = pct_hist(hist, values, [P_LO, P_HI])
        mu = s1 / n
        sd = math.sqrt(max(s2 / n - mu * mu, 0.0)) + 1e-6
        stats["models"][model] = dict(depths=ds, p1=lo, p99_5=hi, mean=mu, sd=sd, n_px=n,
                                      bins="villa129" if villa else "fine")
        subs[model] = np.stack(sv)
    ens_models = [m for m in cfg.models if stats["models"][m].get("depths")]
    common = sorted(set.intersection(*[set(stats["models"][m]["depths"]) for m in ens_models])) if ens_models else []
    ens: dict = dict(models=ens_models, depths=common)
    if len(ens_models) >= 2 and common:
        e = None
        for m in ens_models:
            s = stats["models"][m]
            idx = [s["depths"].index(d) for d in common]
            z = (subs[m][idx] - s["mean"]) / s["sd"]
            e = z if e is None else e + z
        e /= len(ens_models)
        ens["p1"], ens["p99_5"] = (float(x) for x in np.percentile(e.ravel(), [P_LO, P_HI]))
        ens["n_grid"] = int(e.size)
    stats["ensemble"] = ens
    stats["orientation"] = cfg.orientation
    iox.write_tif(out / "ink" / "valid.tif", valid.astype(np.uint8))

    for d in sorted(set(cfg.write_depths) | {cfg.display_depth}):
        ct = ctx["ct"][d].astype(np.float32)
        if cfg.grey:
            iox.write_png(out / "ink" / "grey" / f"{surface.name}__ct__d{d:+03d}.png",
                          to8(ct, ct_lo, ct_hi, valid, flip))
        for side, suffix in (("reading", ""), ("blind", "_blind")):
            acc, nmod = None, 0
            for model in cfg.models:
                s = stats["models"][model]
                if "p1" not in s:
                    continue
                p = map_file(out, model, side, d)
                if not p.is_file():
                    continue
                m = iox.read_tif(p).astype(np.float32, copy=False)
                if cfg.grey:
                    short = RECIPE_NAMES.get(model, model)
                    iox.write_png(out / "ink" / "grey" / f"{surface.name}__{short}{suffix}__d{d:+03d}.png",
                                  to8(m, s["p1"], s["p99_5"], valid, flip))
                if ens.get("p1") is not None and model in ens["models"]:
                    zc = (m - s["mean"]) / s["sd"]
                    acc = zc if acc is None else acc + zc
                    nmod += 1
            if ens.get("p1") is None or acc is None or nmod != len(ens["models"]):
                continue
            e = (acc / nmod).astype(np.float32)
            e_store = np.where(valid, e, np.float32(0))
            edir = out / "ink" / "ensemble" if side == "reading" else out / "ink" / "ensemble" / "blind"
            iox.write_tif(edir / f"{depth_tag(d)}.tif", e_store)
            if cfg.grey:
                iox.write_png(out / "ink" / "grey" / f"{surface.name}__ensemble{suffix}__d{d:+03d}.png",
                              to8(e, ens["p1"], ens["p99_5"], valid, flip))
            if side == "reading":
                u = np.clip((e - ens["p1"]) / max(ens["p99_5"] - ens["p1"], 1e-9), 0, 1).astype(np.float32)
                u[~valid] = 0
                if d == cfg.display_depth:
                    iox.write_tif(out / "ink" / "ensemble.tif", u)
                if d in cfg.write_depths or d == cfg.display_depth:
                    preview = overlay(ct, ct_lo, ct_hi, e, ens["p1"], ens["p99_5"], valid, flip, cfg, d,
                                      len(ens["models"]))
                    iox.write_png(out / "ink" / f"preview_{depth_tag(d)}.png", preview)
    # stats.json under the recipe's model names (dnative, hecate, ink9-42, ink9-43), so that the recipe's
    # make_submission_image.py can read it with --ink-stats
    out_stats = dict(stats)
    out_stats["models"] = {RECIPE_NAMES.get(m, m): dict(v, reader=m) for m, v in stats["models"].items()}
    out_stats["ensemble"] = dict(ens, models=[RECIPE_NAMES.get(m, m) for m in ens["models"]])
    iox.write_json(out / "ink" / "stats.json", out_stats)
    return out_stats


def overlay(ct: np.ndarray, ct_lo: float, ct_hi: float, e: np.ndarray, e_lo: float, e_hi: float, valid: np.ndarray,
            flip: tuple[bool, bool], cfg: EnsembleConfig, d: int, n_models: int) -> np.ndarray:
    """Ink in colour over the dimmed CT (the recipe's ``dimmed-ct`` overlay), with a 1 cm scale bar; BGR uint8."""
    import cv2

    st = STYLES[cfg.style]
    ct_u8 = (np.clip((ct - ct_lo) / max(ct_hi - ct_lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    ct_u8[~valid] = 0
    ink_u8 = (np.clip((e - e_lo) / max(e_hi - e_lo, 1e-9), 0, 1) * 255).astype(np.uint8)
    ink_u8[~valid] = 0
    v = valid
    if cfg.area is not None:
        x0, y0, x1, y1 = cfg.area
        ct_u8, ink_u8, v = ct_u8[y0:y1, x0:x1], ink_u8[y0:y1, x0:x1], v[y0:y1, x0:x1]
    ct_r, ink_r, v_r = orient(ct_u8, flip), orient(ink_u8, flip), orient(v, flip)
    img = np.repeat(ct_r[..., None], 3, -1).astype(np.float32)
    if st["ct_dim"] != 1.0:
        img *= np.float32(st["ct_dim"])
    u = ink_r.astype(np.float32) / 255.0
    alpha = st["alpha"] * np.clip((u - st["threshold"]) / max(st["full"] - st["threshold"], 1e-6), 0, 1)
    r, g, b = st["color"]
    col = np.broadcast_to(np.array([b, g, r], np.float32), img.shape)
    img = img * (1 - alpha[..., None]) + col * alpha[..., None]
    img[~v_r] = 0
    panel = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    # bottom margin with a 1 cm scale bar and a one-line caption
    vox = cfg.voxel_um or 0.0
    margin = 70
    canvas = np.full((panel.shape[0] + margin, panel.shape[1], 3), 24, np.uint8)
    canvas[:panel.shape[0]] = panel
    y = panel.shape[0] + 20
    if vox > 0:
        bar = int(round(10000.0 / vox))
        x0b = 20
        cv2.rectangle(canvas, (x0b, y), (min(canvas.shape[1] - 1, x0b + bar), y + 10), (255, 255, 255), -1)
        cv2.putText(canvas, "1 cm", (x0b, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
        xt = x0b + bar + 40
    else:
        xt = 20
    text = f"ink: mean z-score of {n_models} models, d = {d:+d}, stretch {e_lo:.2f}..{e_hi:.2f}"
    cv2.putText(canvas, text, (xt, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
    return canvas


def depth_mean(out_dir: str | Path, model: str, offsets: Sequence[int], depths: Sequence[int],
               new_name: Optional[str] = None) -> str:
    """A derived model whose map at ``d`` is the mean of ``model`` at ``d + offset`` (e.g. Hecate's +-1 ensemble or
    the z-window average of ``zavg_infer.sh``); written as ``ink/<new_name>/dNN.tif`` (and ``blind/``)."""
    out = Path(out_dir)
    name = new_name or f"{model}_dmean{'_'.join(str(o) for o in offsets)}"
    for side in ("reading", "blind"):
        for d in depths:
            files = [map_file(out, model, side, d + o) for o in offsets]
            if not all(f.is_file() for f in files):
                continue
            acc = None
            for f in files:
                m = iox.read_tif(f).astype(np.float32)
                acc = m if acc is None else acc + m
            iox.write_tif(map_file(out, name, side, d), (acc / len(files)).astype(np.float32))
    return name


def ensemble_bins(ink_json: dict) -> dict[str, str]:
    """Which histogram each model's stretch uses, from the readers recorded in ``ink.json``."""
    bins = {}
    for name, r in ink_json.get("readers", {}).items():
        spec = r.get("spec", {})
        emulated = ink_json.get("settings", {}).get("reference_storage") and spec.get("reference_storage") == "u8_floor"
        bins[name] = "villa129" if emulated and spec.get("calibration") == "label_smoothing_0.5" else "fine"
    return bins


def load_ink_json(out_dir: str | Path) -> dict:
    p = Path(out_dir) / "ink.json"
    return json.loads(p.read_text()) if p.is_file() else {}

"""Equivalence of inksweep maps with the reference pipeline's maps, export under the reference's file names, and the
layer-by-layer comparison of two renders.

The PHerc. 343 recipe stores villa maps as ``<name>__<model>__L66s<s>[_reverse].tif`` (uint8 ``trunc(255 p)``,
``s = L//2 + d - 8``) and Hecate maps as ``<name>__hecate__L66zc<zc>[_reverse].png`` (uint8 ``round(255 p)`` on the
9.6 µm grid, ``zc = (L-1)/2 + d``); ``_reverse`` marks the reverse orientation (its reading side).

:func:`compare` quantises inksweep's raw maps with the reference's own rule and compares them with the reference files
in those 8-bit units: Pearson r, the largest absolute difference (in units of 1/255), the share of equal pixels.
A map passes when ``r >= r_min`` and ``max_abs_diff <= max_diff`` (exact mode: 0.9999 and 1/255; fast mode: 0.999).
:func:`export_reference` writes inksweep maps under the reference names, so the recipe's own ``ensemble_maps.py`` and
``make_submission_image.py`` can be run on them unchanged.

:func:`compare_renders` compares two surface volumes of one mesh (e.g. VC3D's render and ``inksweep render``'s) layer
by layer; ``inksweep compare-render`` prints its table.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import io as iox
from .tiling import depth_tag

#: inksweep reader name -> the recipe's model key in file names
REFERENCE_KEYS = {"dnative": "dnative", "ink9_s42": "hybrid_3d2d-seed42", "ink9_s43": "hybrid_3d2d-seed43",
                  "hecate": "hecate", "readerv2": "readerv2", "readerv3": "readerv3", "d9v4c": "d9v4c"}


def reference_name(name: str, reader: str, orientation: str, d: int, n_layers: int = 66) -> str:
    rev = "_reverse" if orientation == "reverse" else ""
    key = REFERENCE_KEYS.get(reader, reader)
    if reader.startswith("hecate"):
        return f"{name}__{key}__L{n_layers}zc{(n_layers - 1) / 2 + d:.2f}{rev}.png"
    return f"{name}__{key}__L{n_layers}s{n_layers // 2 + d - 8}{rev}.tif"


def quantise(m: np.ndarray, storage: str) -> np.ndarray:
    """The reference's 8-bit storage of a float map in [0, 1]."""
    m = np.clip(np.asarray(m, dtype=np.float32), 0, 1)
    if storage == "u8_round":
        return np.rint(m * np.float32(255)).astype(np.uint8)
    return (m * np.float32(255)).astype(np.uint8)


def read_reference(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".png":
        import cv2

        a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if a is None:
            raise OSError(f"cannot read {path}")
        return a[..., 0] if a.ndim == 3 else a
    return iox.read_tif(path)


@dataclass
class MapComparison:
    reader: str
    side: str
    orientation: str
    depth: int
    reference: str
    shape: tuple
    pearson_r: float
    max_abs_diff: float          # in [0, 1] units (1/255 = one grey level)
    mean_abs_diff: float
    frac_equal: float
    frac_off_by_more_than_one: float
    passed: bool
    note: str = ""


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else (1.0 if np.array_equal(a, b) else 0.0)


def compare_map(ours: np.ndarray, ref: np.ndarray, r_min: float, max_diff: float,
                mask: Optional[np.ndarray] = None) -> dict:
    if ours.shape != ref.shape:
        return dict(pearson_r=float("nan"), max_abs_diff=float("nan"), mean_abs_diff=float("nan"), frac_equal=0.0,
                    frac_off_by_more_than_one=1.0, passed=False, note=f"shape {ours.shape} != {ref.shape}")
    o = ours.astype(np.int32)
    r = ref.astype(np.int32)
    if mask is not None:
        o, r = o[mask], r[mask]
    diff = np.abs(o - r)
    res = dict(pearson_r=pearson(o, r), max_abs_diff=float(diff.max()) / 255.0 if diff.size else 0.0,
               mean_abs_diff=float(diff.mean()) / 255.0 if diff.size else 0.0,
               frac_equal=float((diff == 0).mean()) if diff.size else 1.0,
               frac_off_by_more_than_one=float((diff > 1).mean()) if diff.size else 0.0)
    res["passed"] = bool(res["pearson_r"] >= r_min and res["max_abs_diff"] <= max_diff + 1e-12)
    return res


def compare(out_dir: str | Path, reference_dir: str | Path, name: str, readers: Sequence[str],
            depths: Sequence[int], orientations: Sequence[str] = ("reverse", "forward"), reading: str = "reverse",
            r_min: float = 0.9999, max_diff: float = 1 / 255, n_layers: int = 66,
            storages: Optional[dict] = None) -> dict:
    """Compare every (reader, orientation, depth) raw map with the reference file; returns the report dict."""
    out, ref_dir = Path(out_dir), Path(reference_dir)
    storages = storages or {}
    rows: list[MapComparison] = []
    for reader in readers:
        storage = storages.get(reader, "u8_round" if reader.startswith("hecate") else "u8_floor")
        for orientation in orientations:
            side = "reading" if orientation == reading else "blind"
            for d in depths:
                raw = out / "ink" / reader / "raw" / side / f"{depth_tag(d)}.tif"
                refp = ref_dir / reference_name(name, reader, orientation, d, n_layers)
                if not raw.is_file() or not refp.is_file():
                    rows.append(MapComparison(reader, side, orientation, d, refp.name, (), float("nan"), float("nan"),
                                              float("nan"), 0.0, 1.0, False,
                                              "missing: " + ("ours" if not raw.is_file() else "reference")))
                    continue
                ours = quantise(iox.read_tif(raw), storage)
                ref = read_reference(refp)
                res = compare_map(ours, ref, r_min, max_diff)
                rows.append(MapComparison(reader, side, orientation, d, refp.name, tuple(ours.shape), **res))
    compared = [r for r in rows if not r.note.startswith("missing")]
    summary = dict(maps=len(rows), compared=len(compared), passed=sum(r.passed for r in rows),
                   missing=len(rows) - len(compared),
                   worst_r=min((r.pearson_r for r in compared), default=float("nan")),
                   worst_max_abs_diff=max((r.max_abs_diff for r in compared), default=float("nan")),
                   mean_frac_equal=float(np.mean([r.frac_equal for r in compared])) if compared else float("nan"),
                   criteria=dict(r_min=r_min, max_abs_diff=max_diff))
    summary["all_passed"] = bool(compared) and summary["passed"] == len(rows)
    return dict(summary=summary, maps=[asdict(r) for r in rows])


def compare_images(ours_dir: str | Path, reference_dir: str | Path, pattern: str = "*.png") -> dict:
    """Pixel equality of same-named 8-bit images (e.g. the recipe's grey ensemble images)."""
    import cv2

    rows = []
    for ref in sorted(Path(reference_dir).glob(pattern)):
        mine = Path(ours_dir) / ref.name
        if not mine.is_file():
            rows.append(dict(image=ref.name, note="missing in ours"))
            continue
        a = cv2.imread(str(mine), cv2.IMREAD_UNCHANGED)
        b = cv2.imread(str(ref), cv2.IMREAD_UNCHANGED)
        if a is None or b is None or a.shape != b.shape:
            rows.append(dict(image=ref.name, note="unreadable or shape differs"))
            continue
        diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
        rows.append(dict(image=ref.name, frac_equal=float((diff == 0).mean()), max_abs_diff=int(diff.max()),
                         pearson_r=pearson(a, b)))
    return dict(images=rows)


def compare_stats(ours: dict, ref: dict) -> dict:
    """Relative differences of the recipe's statistics (per-model stretch, mean, SD; ensemble stretch)."""
    out: dict = {}
    for m, s in ref.get("models", {}).items():
        o = ours.get("models", {}).get(m) or ours.get("models", {}).get(_ours_name(m), {})
        out[m] = {k: _rel(o.get(k), s.get(k)) for k in ("p1", "p99_5", "mean", "sd")}
    out["ensemble"] = {k: _rel(ours.get("ensemble", {}).get(k), ref.get("ensemble", {}).get(k))
                       for k in ("p1", "p99_5")}
    out["ct"] = {k: _rel(ours.get("ct", {}).get(k), ref.get("ct", {}).get(k)) for k in ("p1", "p99_5")}
    return out


def _ours_name(recipe_name: str) -> str:
    return {"ink9-42": "ink9_s42", "ink9-43": "ink9_s43"}.get(recipe_name, recipe_name)


def _rel(a, b):
    if a is None or b is None:
        return None
    return dict(ours=a, reference=b, abs_diff=abs(a - b), rel_diff=abs(a - b) / max(abs(b), 1e-12))


def export_reference(out_dir: str | Path, dest: str | Path, name: str, readers: Sequence[str],
                     depths: Sequence[int], orientations: Sequence[str] = ("reverse", "forward"),
                     reading: str = "reverse", n_layers: int = 66, storages: Optional[dict] = None) -> int:
    """Write inksweep raw maps as the recipe's files (names and 8-bit encodings). Returns the number written."""
    import cv2

    out, dest = Path(out_dir), Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    storages = storages or {}
    n = 0
    for reader in readers:
        storage = storages.get(reader, "u8_round" if reader.startswith("hecate") else "u8_floor")
        for orientation in orientations:
            side = "reading" if orientation == reading else "blind"
            for d in depths:
                raw = out / "ink" / reader / "raw" / side / f"{depth_tag(d)}.tif"
                if not raw.is_file():
                    continue
                q = quantise(iox.read_tif(raw), storage)
                target = dest / reference_name(name, reader, orientation, d, n_layers)
                if target.suffix == ".png":
                    if not cv2.imwrite(str(target), q):
                        raise OSError(target)
                else:
                    iox.write_tif(target, q, compression="lzw")
                n += 1
    return n


def markdown_table(report: dict, limit: int = 40) -> str:
    s = report["summary"]
    lines = [f"maps {s['maps']}, compared {s['compared']}, passed {s['passed']}, missing {s['missing']}; "
             f"worst r {s['worst_r']:.6f}, worst max |diff| {s['worst_max_abs_diff'] * 255:.0f}/255; "
             f"criteria r >= {s['criteria']['r_min']}, max |diff| <= {s['criteria']['max_abs_diff'] * 255:.0f}/255",
             "", "| reader | side | d | r | max abs diff | equal px | pass |", "|---|---|---|---|---|---|---|"]
    rows = sorted(report["maps"], key=lambda r: (r["passed"], r["pearson_r"] if r["pearson_r"] == r["pearson_r"]
                                                  else -1))
    for r in rows[:limit]:
        lines.append(f"| {r['reader']} | {r['side']} | {r['depth']:+d} | {r['pearson_r']:.6f} | "
                     f"{r['max_abs_diff'] * 255:.0f}/255 | {r['frac_equal']:.4f} | {'yes' if r['passed'] else 'NO'} |")
    return "\n".join(lines)


def save_report(report: dict, path: str | Path) -> None:
    iox.write_json(path, report)
    Path(path).with_suffix(".md").write_text(markdown_table(report) + "\n")


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------------------------------------- renders
def pearson_from_sums(n: int, sa: int, sb: int, saa: int, sbb: int, sab: int, n_equal: int) -> float:
    """Pearson r from exact integer sums (Python integers: no cancellation however many voxels)."""
    if n == 0:
        return float("nan")
    num = n * sab - sa * sb
    va = n * saa - sa * sa
    vb = n * sbb - sb * sb
    if va <= 0 or vb <= 0:                       # a constant layer: r is undefined; 1 if the two are identical
        return 1.0 if n_equal == n else 0.0
    return num / math.sqrt(va * vb)


def _chunk_rows(layers: object) -> int:
    chunks = getattr(layers, "chunks", None)
    try:
        return int(chunks[1]) if chunks is not None and len(chunks) == 3 else 0
    except (TypeError, ValueError):
        return 0


def _block_rows(arrays: Sequence[object], want: int = 256) -> int:
    """Row-block height: a multiple of every input's chunk height (each chunk is then decoded once)."""
    heights = [h for h in (_chunk_rows(a) for a in arrays) if h > 0]
    step = math.lcm(*heights) if heights else 1
    if step > 2048:
        step = max(heights)
    return step * max(1, -(-want // step))


def shift_search(a: np.ndarray, b: np.ndarray, max_shift: int) -> dict:
    """Pearson r of ``a[y, x]`` against ``b[y + dy, x + dx]`` for every ``|dy|, |dx| <= max_shift``, over the interior
    pixels where both are non-zero. ``best`` is the offset at which B's pixels match A's."""
    s = int(max_shift)
    h, w = a.shape
    if s <= 0 or h <= 2 * s or w <= 2 * s:
        raise ValueError(f"max_shift {s} does not fit a {h} x {w} layer")
    core_a = a[s:h - s, s:w - s]
    grid = []
    for dy in range(-s, s + 1):
        for dx in range(-s, s + 1):
            core_b = b[s + dy:h - s + dy, s + dx:w - s + dx]
            m = (core_a > 0) & (core_b > 0)
            r = pearson(core_a[m], core_b[m]) if m.any() else float("nan")
            grid.append(dict(dy=dy, dx=dx, pearson_r=r, n_px=int(m.sum())))
    finite = [g for g in grid if g["pearson_r"] == g["pearson_r"]]
    best = max(finite, key=lambda g: g["pearson_r"]) if finite else None
    zero = next(g for g in grid if g["dy"] == 0 and g["dx"] == 0)
    return dict(max_shift=s, best=best, at_zero=zero, grid=grid)


def compare_renders(path_a: str | Path, path_b: str | Path, names: Sequence[str] = ("a", "b"),
                    block_rows: Optional[int] = None, max_shift: int = 0, shift_depth: int = 1) -> dict:
    """Layer-by-layer comparison of two uint8 surface volumes of one mesh (any input ``inksweep run`` accepts).

    Layers are matched by depth label ``d = i - L // 2`` (a 66-layer and a 28-layer render of one mesh line up) over the
    common top-left ``H x W``. For each depth, over the pixels where both renders have CT data in some layer: Pearson r
    (from exact integer sums), the largest and the mean absolute difference in grey levels, the share of equal values
    and of values at most one grey level apart; over every pixel, the agreement of the two layers' valid masks
    (value > 0) and their intersection over union. The renders' valid masks (data in any layer; data in every layer,
    the recipe's valid rule) are compared too. With ``max_shift`` the layer at ``shift_depth`` is also correlated at
    every row and column offset up to ``max_shift`` pixels, which exposes a pixel-centre convention difference.
    """
    name_a, name_b = (list(names) + ["a", "b"])[:2]
    sa, sb = iox.open_surface(path_a), iox.open_surface(path_b)
    la, lb = sa.n_layers, sb.n_layers
    index_a = {i - la // 2: i for i in range(la)}
    index_b = {j - lb // 2: j for j in range(lb)}
    depths = sorted(set(index_a) & set(index_b))
    if not depths:
        raise ValueError("the two renders share no depth label")
    h, w = min(sa.height, sb.height), min(sa.width, sb.width)
    rows = int(block_rows) if block_rows else _block_rows([sa.layers, sb.layers])
    keys = ("n", "sa", "sb", "saa", "sbb", "sab", "max", "sum_abs", "equal", "within1", "agree", "inter", "union")
    acc = {d: dict.fromkeys(keys, 0) for d in depths}
    valid = {k: dict(both=0, only_a=0, only_b=0, neither=0) for k in ("any", "all")}
    keep = max_shift > 0 and shift_depth in index_a and shift_depth in index_b
    keep_a = np.zeros((h, w), np.uint8) if keep else None
    keep_b = np.zeros((h, w), np.uint8) if keep else None
    for r0 in range(0, h, rows):
        r1 = min(h, r0 + rows)
        A = sa.read(rows=(r0, r1), cols=(0, w))                 # (La, rows, w) uint8
        B = sb.read(rows=(r0, r1), cols=(0, w))
        pa, pb = A > 0, B > 0
        any_a, any_b = pa.any(0), pb.any(0)
        for key, x, y in (("any", any_a, any_b), ("all", pa.all(0), pb.all(0))):
            v = valid[key]
            v["both"] += int((x & y).sum())
            v["only_a"] += int((x & ~y).sum())
            v["only_b"] += int((~x & y).sum())
            v["neither"] += int((~x & ~y).sum())
        both = any_a & any_b                                    # pixels on the mesh in both renders
        for d in depths:
            ia, ib = index_a[d], index_b[d]
            s = acc[d]
            ma, mb = pa[ia], pb[ib]
            s["agree"] += int((ma == mb).sum())
            s["inter"] += int((ma & mb).sum())
            s["union"] += int((ma | mb).sum())
            x = A[ia][both].astype(np.int64)
            y = B[ib][both].astype(np.int64)
            if not x.size:
                continue
            diff = np.abs(x - y)
            s["n"] += int(x.size)
            s["sa"] += int(x.sum())
            s["sb"] += int(y.sum())
            s["saa"] += int((x * x).sum())
            s["sbb"] += int((y * y).sum())
            s["sab"] += int((x * y).sum())
            s["max"] = max(s["max"], int(diff.max()))
            s["sum_abs"] += int(diff.sum())
            s["equal"] += int((diff == 0).sum())
            s["within1"] += int((diff <= 1).sum())
        if keep:
            keep_a[r0:r1] = A[index_a[shift_depth]]
            keep_b[r0:r1] = B[index_b[shift_depth]]
    total = h * w
    layers = []
    for d in depths:
        s = acc[d]
        n = s["n"]
        layers.append(dict(
            depth=d, layer_a=index_a[d], layer_b=index_b[d], n_px=n,
            pearson_r=pearson_from_sums(n, s["sa"], s["sb"], s["saa"], s["sbb"], s["sab"], s["equal"]),
            max_abs_diff=s["max"] if n else None, mean_abs_diff=s["sum_abs"] / n if n else None,
            frac_equal=s["equal"] / n if n else None, frac_within_1=s["within1"] / n if n else None,
            valid_agreement=s["agree"] / total if total else None,
            valid_iou=s["inter"] / s["union"] if s["union"] else None))
    for v in valid.values():
        v["agreement"] = (v["both"] + v["neither"]) / total if total else None
    measured = [L for L in layers if L["n_px"] and L["pearson_r"] == L["pearson_r"]]
    voxels = sum(acc[d]["n"] for d in depths)
    worst = min(measured, key=lambda L: L["pearson_r"]) if measured else None
    summary = dict(
        depths=[depths[0], depths[-1]], voxels=voxels,
        min_r=worst["pearson_r"] if worst else None, min_r_depth=worst["depth"] if worst else None,
        max_abs_diff=max((L["max_abs_diff"] for L in measured), default=None),
        mean_abs_diff=sum(acc[d]["sum_abs"] for d in depths) / voxels if voxels else None,
        frac_equal=sum(acc[d]["equal"] for d in depths) / voxels if voxels else None,
        frac_within_1=sum(acc[d]["within1"] for d in depths) / voxels if voxels else None,
        min_valid_agreement=min((L["valid_agreement"] for L in layers if L["valid_agreement"] is not None),
                                default=None))
    summary["d+1"] = next((L for L in layers if L["depth"] == 1), None)
    report = dict(
        a=dict(name=name_a, path=str(path_a), kind=sa.kind, shape=list(sa.shape), chunk_rows=_chunk_rows(sa.layers)),
        b=dict(name=name_b, path=str(path_b), kind=sb.kind, shape=list(sb.shape), chunk_rows=_chunk_rows(sb.layers)),
        compared_hw=[h, w], block_rows=rows, depth_rule="layer i is depth d = i - L // 2",
        note=None if la % 2 == lb % 2 else "layer counts of different parity: matched layers sit half a voxel apart",
        valid=valid, layers=layers, summary=summary)
    if keep:
        report["shift_search"] = dict(depth=shift_depth, **shift_search(keep_a, keep_b, max_shift))
    return report


def _fmt(v: object, spec: str) -> str:
    return "-" if v is None or (isinstance(v, float) and v != v) else format(v, spec)


def render_table(report: dict) -> str:
    """The printed form of :func:`compare_renders`: header, valid masks, one row per depth, summary."""
    a, b, s = report["a"], report["b"], report["summary"]
    lines = [f"A = {a['name']}: {a['path']} {a['shape']}; B = {b['name']}: {b['path']} {b['shape']}; compared "
             f"{report['compared_hw'][0]} x {report['compared_hw'][1]} px at depths {s['depths'][0]:+d}..{s['depths'][1]:+d}"]
    if report.get("note"):
        lines.append(f"note: {report['note']}")
    for key, what in (("any", "data in any layer"), ("all", "data in every layer (the recipe's valid rule)")):
        v = report["valid"][key]
        lines.append(f"valid pixels ({what}): both {v['both']}, only {a['name']} {v['only_a']}, only {b['name']} "
                     f"{v['only_b']}; agreement {_fmt(v['agreement'], '.6f')}")
    lines += ["", f"| d | layer {a['name']} | layer {b['name']} | r | max abs diff | mean abs diff | equal | within 1 | "
                  "valid agreement |", "|---|---|---|---|---|---|---|---|---|"]
    for L in report["layers"]:
        mark = " *" if L["depth"] == 1 else ""
        lines.append(f"| {L['depth']:+d}{mark} | {L['layer_a']} | {L['layer_b']} | {_fmt(L['pearson_r'], '.6f')} | "
                     f"{_fmt(L['max_abs_diff'], 'd')} | {_fmt(L['mean_abs_diff'], '.4f')} | "
                     f"{_fmt(L['frac_equal'], '.4f')} | {_fmt(L['frac_within_1'], '.4f')} | "
                     f"{_fmt(L['valid_agreement'], '.6f')} |")
    lines.append("")
    pct = (lambda v: "-" if v is None else f"{100 * v:.2f} %")
    lines.append(f"summary: min r {_fmt(s['min_r'], '.6f')} (d {_fmt(s['min_r_depth'], '+d')}), max |diff| "
                 f"{_fmt(s['max_abs_diff'], 'd')} grey levels, mean |diff| {_fmt(s['mean_abs_diff'], '.4f')}, "
                 f"{pct(s['frac_equal'])} of {s['voxels']} voxels equal, {pct(s['frac_within_1'])} within one grey "
                 f"level, min valid agreement {_fmt(s['min_valid_agreement'], '.6f')}")
    d1 = s.get("d+1")
    if d1 and d1["n_px"]:
        lines.append(f"d = +1 (* above, the image's layer): r {_fmt(d1['pearson_r'], '.6f')}, max |diff| "
                     f"{_fmt(d1['max_abs_diff'], 'd')}, {pct(d1['frac_equal'])} equal, {pct(d1['frac_within_1'])} "
                     f"within one grey level, valid agreement {_fmt(d1['valid_agreement'], '.6f')}")
    sh = report.get("shift_search")
    if sh and sh.get("best"):
        best, zero = sh["best"], sh["at_zero"]
        lines.append(f"shift search at d = {sh['depth']:+d} (offsets up to {sh['max_shift']} px): B matches A best at "
                     f"(dy, dx) = ({best['dy']:+d}, {best['dx']:+d}), r {_fmt(best['pearson_r'], '.6f')}; at (0, 0) r "
                     f"{_fmt(zero['pearson_r'], '.6f')}")
    return "\n".join(lines)


def save_render_report(report: dict, path: str | Path) -> None:
    iox.write_json(path, report)
    Path(path).with_suffix(".md").write_text(render_table(report) + "\n")

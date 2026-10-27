"""Render a tifxyz mesh into a surface volume (a stack of layers along the mesh normal) without VC3D.

Port of the clean-room Python renderer of EconLearn/vesuvius-first-letters-stage-loss (``tools/g12lib.py``, MIT,
Copyright (c) 2026 Jude Wallis), which reimplements the *behaviour* of ``vc_render_tifxyz`` (linear mode) from its
documentation and measured outputs; no VC3D code is used. Its author validated it against an organiser render
(PHerc1447 w058: r 0.99987, every voxel within one grey level); it is not bit-identical to VC3D.

Conventions (those of ``vc_render_tifxyz``):

* a tifxyz folder holds ``x.tif``, ``y.tif``, ``z.tif`` (level-0 voxel position of each grid node; -1 = no node) and
  ``meta.json`` with ``scale`` (grid nodes per render pixel, e.g. 0.05 = 20 px per node);
* render pixel ``(r, c)`` samples the grid at ``((r + 0.5) * s, (c + 0.5) * s)``, ``s = scale / render_scale``,
  position bilinear between the four nodes;
* normal: ``N = (P[r, c+1] - P[r, c-1]) x (P[r+1, c] - P[r-1, c])`` at the nearest node, normalised;
* layer ``i`` of ``n`` samples the CT at ``(i - (n - 1) / 2) * step`` voxels along ``N`` (``--flip-normals``: along
  ``-N``), trilinear, zero outside the volume, truncated to uint8.

So without flipping, layer ``i`` lies at depth ``d = i - n // 2`` (+ 0.5 voxel) along ``N``. The surface-volume
contract needs +d away from the scroll axis: ``--normal-sign auto`` measures, against the scroll's umbilicus, the share
of pixels whose ``N`` points outward and flips the normals when it is below one half; it refuses meshes whose normals
point both ways (share between 0.2 and 0.8) unless ``--force``. ``keep`` renders like ``vc_render_tifxyz`` without
``--flip-normals`` (the PHerc. 343 render), ``flip`` like ``--flip-normals``. The decision and the measured share are
written to ``surface.json``. (For model-order renders others use ``flip-normals = not left_handed_coordinates``; with the
contract's +d-outward stacks the readers' reading side is then the ``reverse`` orientation.)

CT access: level-0 chunks of an OME-Zarr v2 volume (local path or HTTPS URL of the open-data bucket), fetched once
into a disk cache. A missing chunk (HTTP 404) is recorded, never cached as zeros; if the volume's coarse level shows
data where a level-0 chunk is missing, the render stops (a fetch failure, not empty space).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import __version__
from . import io as iox


# ----------------------------------------------------------------------------------------------- mesh geometry
def load_tifxyz(folder: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, dict]:
    """``P`` (H, W, 3) float32 XYZ with NaN at missing nodes, the validity mask, ``meta.json``."""
    import tifffile

    d = Path(folder)
    x = tifffile.imread(d / "x.tif").astype(np.float32)
    y = tifffile.imread(d / "y.tif").astype(np.float32)
    z = tifffile.imread(d / "z.tif").astype(np.float32)
    meta = json.loads((d / "meta.json").read_text())
    p = np.stack([x, y, z], -1)
    valid = (x != -1) & np.isfinite(x) & ~((x == 0) & (y == 0) & (z == 0))
    p[~valid] = np.nan
    return p, valid, meta


def vertex_normals(p: np.ndarray) -> np.ndarray:
    """``(P[r, c+1] - P[r, c-1]) x (P[r+1, c] - P[r-1, c])``; NaN on the border and next to missing nodes."""
    n = np.full(p.shape, np.nan, np.float32)
    xv = p[1:-1, 2:] - p[1:-1, :-2]
    yv = p[2:, 1:-1] - p[:-2, 1:-1]
    n[1:-1, 1:-1] = np.cross(xv, yv)
    return n


def render_shape(p: np.ndarray, grid_scale: float, render_scale: float = 1.0) -> tuple[int, int]:
    return int(np.floor(p.shape[0] / grid_scale * render_scale)), int(np.floor(p.shape[1] / grid_scale * render_scale))


def surface_at_pixels(p: np.ndarray, n: np.ndarray, rows: np.ndarray, cols: np.ndarray, grid_scale: float,
                      render_scale: float = 1.0, pix_off: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """3D points (bilinear) and unit normals (nearest node) for render pixels ``rows x cols``; NaN where invalid."""
    s = grid_scale / render_scale
    gr = (rows.astype(np.float64) + pix_off) * s
    gc = (cols.astype(np.float64) + pix_off) * s
    GR, GC = np.meshgrid(gr, gc, indexing="ij")
    r0 = np.floor(GR).astype(np.int64)
    c0 = np.floor(GC).astype(np.int64)
    fr = (GR - r0)[..., None]
    fc = (GC - c0)[..., None]
    h, w = p.shape[:2]
    ok = (r0 >= 0) & (c0 >= 0) & (r0 + 1 < h) & (c0 + 1 < w)
    r0c = np.clip(r0, 0, h - 2)
    c0c = np.clip(c0, 0, w - 2)
    p00, p01 = p[r0c, c0c], p[r0c, c0c + 1]
    p10, p11 = p[r0c + 1, c0c], p[r0c + 1, c0c + 1]
    pts = (p00 * (1 - fr) * (1 - fc) + p01 * (1 - fr) * fc + p10 * fr * (1 - fc) + p11 * fr * fc).astype(np.float32)
    rn = np.clip(np.rint(GR).astype(np.int64), 0, h - 1)
    cn = np.clip(np.rint(GC).astype(np.int64), 0, w - 1)
    nrm = n[rn, cn].astype(np.float32)
    with np.errstate(invalid="ignore", divide="ignore"):
        nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True)
    pts[~ok] = np.nan
    nrm[~ok] = np.nan
    return pts, nrm


def load_umbilicus(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Control points ``{"control_points": [{"x", "y", "z"}, ...]}`` -> sorted (z, x(z), y(z))."""
    cp = json.loads(Path(path).read_text())["control_points"]
    z = np.array([q["z"] for q in cp], float)
    x = np.array([q["x"] for q in cp], float)
    y = np.array([q["y"] for q in cp], float)
    o = np.argsort(z)
    return z[o], x[o], y[o]


def outward_fraction(pts: np.ndarray, nrm: np.ndarray, umb: tuple[np.ndarray, np.ndarray, np.ndarray],
                     every: int = 4) -> tuple[float, int]:
    """Share of valid pixels whose normal points away from the scroll axis (horizontal ``n . r_hat > 0``)."""
    zu, xu, yu = umb
    q = pts[::every, ::every].reshape(-1, 3)
    m = nrm[::every, ::every].reshape(-1, 3)
    ok = np.isfinite(q).all(1) & np.isfinite(m).all(1)
    q, m = q[ok], m[ok]
    if not len(q):
        return float("nan"), 0
    ux = np.interp(q[:, 2], zu, xu)
    uy = np.interp(q[:, 2], zu, yu)
    r = np.stack([q[:, 0] - ux, q[:, 1] - uy], 1)
    r /= np.linalg.norm(r, axis=1, keepdims=True) + 1e-9
    return float(((m[:, 0] * r[:, 0] + m[:, 1] * r[:, 1]) > 0).mean()), int(len(q))


# ----------------------------------------------------------------------------------------------- CT chunks
class ChunkStore:
    """Level-0 chunks of an OME-Zarr v2 uint8 volume (local folder or HTTPS URL), cached on disk.

    404 = absent (listed in ``.absent``, never written as zeros); other errors are retried, then raised.
    """

    def __init__(self, volume: str, cache: str | os.PathLike, workers: int = 32, level: str = "0") -> None:
        self.volume = volume.rstrip("/")
        self.level = level
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers = workers
        self.meta = json.loads(self._get_bytes(f"{level}/.zarray"))
        if self.meta.get("dtype") not in ("|u1", "uint8"):
            raise ValueError(f"{volume}: dtype {self.meta.get('dtype')} is not uint8")
        self.shape = tuple(int(v) for v in self.meta["shape"])
        self.chunks = tuple(int(v) for v in self.meta["chunks"])
        self.sep = self.meta.get("dimension_separator", ".")
        comp = self.meta.get("compressor")
        if comp:
            import numcodecs

            self.codec = numcodecs.get_codec(comp)
        else:
            self.codec = None
        self.absent_file = self.cache / ".absent"
        self.absent = set(self.absent_file.read_text().split()) if self.absent_file.exists() else set()
        self.lock = threading.Lock()
        self.fetched_bytes = 0

    def _is_url(self) -> bool:
        return self.volume.startswith(("http://", "https://"))

    def _get_bytes(self, key: str) -> bytes:
        if self._is_url():
            req = urllib.request.Request(f"{self.volume}/{key}", headers={"User-Agent": f"inksweep/{__version__}"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.read()
        return (Path(self.volume) / key).read_bytes()

    def key(self, c: tuple[int, int, int]) -> str:
        return self.sep.join(str(v) for v in c)

    def path(self, c: tuple[int, int, int]) -> Path:
        return self.cache / f"{c[0]}" / f"{c[1]}" / f"{c[2]}"

    def _fetch(self, c: tuple[int, int, int]) -> None:
        k = self.key(c)
        if self.path(c).exists() or k in self.absent:
            return
        data = None
        for attempt in range(6):
            try:
                data = self._get_bytes(f"{self.level}/{k}")
                break
            except urllib.error.HTTPError as exc:
                if exc.code in (403, 404):
                    with self.lock:
                        self.absent.add(k)
                        with open(self.absent_file, "a") as f:
                            f.write(k + "\n")
                    return
                time.sleep(2 * (attempt + 1))
            except FileNotFoundError:
                with self.lock:
                    self.absent.add(k)
                return
            except Exception:
                time.sleep(2 * (attempt + 1))
        if data is None:
            raise RuntimeError(f"fetch failed: {self.volume}/{self.level}/{k}")
        raw = self.codec.decode(data) if self.codec is not None else data
        if len(raw) != int(np.prod(self.chunks)):
            raise RuntimeError(f"chunk {k}: {len(raw)} bytes, expected {int(np.prod(self.chunks))}")
        dest = self.path(c)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.{threading.get_ident()}")
        tmp.write_bytes(bytes(raw))
        os.replace(tmp, dest)
        with self.lock:
            self.fetched_bytes += len(raw)

    def ensure(self, keys: Sequence[tuple[int, int, int]]) -> int:
        todo = [c for c in keys if not self.path(c).exists() and self.key(c) not in self.absent]
        if todo:
            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(self._fetch, todo))
        return len(todo)

    def chunk(self, c: tuple[int, int, int]) -> Optional[np.ndarray]:
        p = self.path(c)
        if not p.exists():
            return None
        return np.fromfile(p, np.uint8).reshape(self.chunks)

    def read_box(self, z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """``[z0:z1, y0:y1, x0:x1]`` from cached chunks; outside the volume or absent = 0."""
        out = np.zeros((z1 - z0, y1 - y0, x1 - x0), np.uint8)
        cz, cy, cx = self.chunks
        for a in range(max(0, z0) // cz, (min(self.shape[0], z1) - 1) // cz + 1):
            for b in range(max(0, y0) // cy, (min(self.shape[1], y1) - 1) // cy + 1):
                for c in range(max(0, x0) // cx, (min(self.shape[2], x1) - 1) // cx + 1):
                    ch = self.chunk((a, b, c))
                    if ch is None:
                        continue
                    za, ya, xa = a * cz, b * cy, c * cx
                    sz0, sz1 = max(z0, za), min(z1, za + cz)
                    sy0, sy1 = max(y0, ya), min(y1, ya + cy)
                    sx0, sx1 = max(x0, xa), min(x1, xa + cx)
                    if sz0 < sz1 and sy0 < sy1 and sx0 < sx1:
                        out[sz0 - z0:sz1 - z0, sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = \
                            ch[sz0 - za:sz1 - za, sy0 - ya:sy1 - ya, sx0 - xa:sx1 - xa]
        return out


def check_absent_against_coarse(store: ChunkStore, keys: Sequence[tuple[int, int, int]], level: int = 5) -> list:
    """Absent level-0 chunks under which the coarse level shows data (fetch failures, not empty space)."""
    absent = [c for c in keys if store.key(c) in store.absent]
    if not absent:
        return []
    coarse = ChunkStore(store.volume, store.cache / f"level{level}", workers=store.workers, level=str(level))
    f = 2 ** level
    per = [max(1, ch // f) for ch in store.chunks]
    need = set()
    for c in absent:
        lo = [c[i] * store.chunks[i] // f for i in range(3)]
        need.add(tuple(lo[i] // coarse.chunks[i] for i in range(3)))
    coarse.ensure(sorted(need))
    bad = []
    for c in absent:
        lo = [c[i] * store.chunks[i] // f for i in range(3)]
        box = coarse.read_box(lo[0], lo[0] + per[0], lo[1], lo[1] + per[1], lo[2], lo[2] + per[2])
        if box.any():
            bad.append(store.key(c))
    return bad


def _single_level_attrs(data: bytes, level: str) -> bytes:
    """A volume's ``.zattrs`` with every OME multiscale listing only dataset ``level`` (the copy has no other)."""
    try:
        attrs = json.loads(data)
    except ValueError:
        return data
    if not isinstance(attrs, dict):
        return data
    for ms in attrs.get("multiscales") or []:
        if isinstance(ms, dict) and isinstance(ms.get("datasets"), list):
            ms["datasets"] = [d for d in ms["datasets"] if isinstance(d, dict) and str(d.get("path")) == str(level)]
    return (json.dumps(attrs, indent=1) + "\n").encode()


def export_level(volume: str, cache: str | os.PathLike, dest: str | os.PathLike, level: str = "0",
                 copy: bool = False, keep_attrs: bool = False) -> dict:
    """A local OME-Zarr v2 copy of ``volume`` that holds the chunks of one level found in an ``inksweep render`` cache.

    For programs that read a local volume (``vc_render_tifxyz --volume``): the copy has the source's metadata
    (``.zgroup``, ``.zattrs`` listing only this level unless ``keep_attrs``, ``<level>/.zarray``) and every cached
    chunk under its zarr key. Chunks that are not cached are absent and read as the fill value (0), as absent chunks of
    the bucket do. Uncompressed volumes (the open-data volumes are) get hard links to the cache files (``copy``: copies),
    so the copy holds the same bytes as the bucket; compressed ones are re-encoded with the volume's codec.
    """
    store = ChunkStore(volume, cache, level=level)
    dest = Path(dest)
    arr = dest / str(level)
    arr.mkdir(parents=True, exist_ok=True)
    for key in (".zgroup", ".zattrs"):
        try:
            data = store._get_bytes(key)
        except (urllib.error.URLError, OSError):            # absent (HTTP 404 or no such file)
            continue
        if key == ".zattrs" and not keep_attrs:
            data = _single_level_attrs(data, level)
        (dest / key).write_bytes(data)
    if not (dest / ".zgroup").exists():
        (dest / ".zgroup").write_text(json.dumps({"zarr_format": 2}) + "\n")
    (arr / ".zarray").write_bytes(store._get_bytes(f"{level}/.zarray"))
    try:
        (arr / ".zattrs").write_bytes(store._get_bytes(f"{level}/.zattrs"))
    except (urllib.error.URLError, OSError):
        pass
    size = int(np.prod(store.chunks))
    root = Path(cache)
    linked = copied = encoded = 0
    for p in sorted(root.glob("*/*/*")):
        parts = p.relative_to(root).parts
        if not p.is_file() or not all(s.isdigit() for s in parts):     # .absent, level5/, temporary files
            continue
        if p.stat().st_size != size:
            raise ValueError(f"{p}: {p.stat().st_size} bytes, a chunk of {volume} has {size}: another volume's cache?")
        target = arr / store.key(tuple(int(s) for s in parts))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            target.unlink()
        if store.codec is not None:
            target.write_bytes(bytes(store.codec.encode(np.fromfile(p, np.uint8))))
            encoded += 1
            continue
        if not copy:
            try:
                os.link(p, target)
                linked += 1
                continue
            except OSError:                                    # another file system: copy instead
                pass
        shutil.copyfile(p, target)
        copied += 1
    summary = dict(volume=volume, level=str(level), out=str(dest), chunks=linked + copied + encoded, linked=linked,
                   copied=copied, encoded=encoded, absent_listed=len(store.absent), shape=list(store.shape),
                   chunk_shape=list(store.chunks), attrs="kept" if keep_attrs else "this level only")
    iox.write_json(dest / "inksweep_export.json", summary)
    return summary


# ----------------------------------------------------------------------------------------------- rendering
def _tile_box(p: np.ndarray, n: np.ndarray, v: np.ndarray, offs: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    q = p[v]
    m = n[v]
    if not len(q):
        return None
    ends = np.concatenate([q + offs[0] * m, q + offs[-1] * m])
    lo = np.floor(np.nanmin(ends, 0)).astype(int) - 1
    hi = np.ceil(np.nanmax(ends, 0)).astype(int) + 2
    return lo, hi


def plan_chunks(pts: np.ndarray, nrm: np.ndarray, offs: np.ndarray, store: ChunkStore,
                tile: int = 64, max_span: int = 1500) -> list[tuple[int, int, int]]:
    """Level-0 chunks a render reads (the same read boxes as :func:`render_layers`)."""
    valid = np.isfinite(pts).all(-1) & np.isfinite(nrm).all(-1)
    keys = set()
    cz, cy, cx = store.chunks
    H, W = valid.shape
    for i0 in range(0, H, tile):
        for j0 in range(0, W, tile):
            v = valid[i0:i0 + tile, j0:j0 + tile]
            if not v.any():
                continue
            box = _tile_box(pts[i0:i0 + tile, j0:j0 + tile], nrm[i0:i0 + tile, j0:j0 + tile], v, offs)
            if box is None:
                continue
            lo, hi = box
            if np.any(hi - lo > max_span):
                continue
            for a in range(max(0, lo[2]) // cz, (min(store.shape[0], hi[2]) - 1) // cz + 1):
                for b in range(max(0, lo[1]) // cy, (min(store.shape[1], hi[1]) - 1) // cy + 1):
                    for c in range(max(0, lo[0]) // cx, (min(store.shape[2], hi[0]) - 1) // cx + 1):
                        keys.add((a, b, c))
    return sorted(keys)


_STORE: dict = {}


def _render_tile(args: tuple) -> tuple[int, int, Optional[np.ndarray]]:
    from scipy import ndimage

    i0, j0, p, n, v, offs, store_args, max_span = args
    if "store" not in _STORE:
        _STORE["store"] = ChunkStore(*store_args)
    store = _STORE["store"]
    box = _tile_box(p, n, v, offs)
    if box is None:
        return i0, j0, None
    lo, hi = box
    if np.any(hi - lo > max_span):          # folded or exploding mesh: leave the tile empty
        return i0, j0, None
    vol = store.read_box(lo[2], hi[2], lo[1], hi[1], lo[0], hi[0])
    s = p[None] + offs[:, None, None, None] * n[None]
    sv = s[:, v]
    coords = np.stack([sv[..., 2] - lo[2], sv[..., 1] - lo[1], sv[..., 0] - lo[0]], 0).reshape(3, -1)
    vals = ndimage.map_coordinates(vol, coords, order=1, mode="constant", cval=0, prefilter=False)
    blk = np.zeros((len(offs),) + v.shape, np.float32)
    blk[:, v] = vals.reshape(len(offs), -1)
    return i0, j0, np.clip(np.floor(blk + 1e-4), 0, 255).astype(np.uint8)   # C++ truncation (measured)


def render_layers(pts: np.ndarray, nrm: np.ndarray, n_layers: int, store_args: tuple, step: float = 1.0,
                  flip: bool = False, tile: int = 64, procs: int = 8, max_span: int = 1500) -> np.ndarray:
    """``(n_layers, H, W)`` uint8 for precomputed surface points and normals (render pixels)."""
    import multiprocessing as mp

    offs = (np.arange(n_layers) - 0.5 * (n_layers - 1)) * step
    d = -nrm if flip else nrm
    h, w = pts.shape[:2]
    out = np.zeros((n_layers, h, w), np.uint8)
    valid = np.isfinite(pts).all(-1) & np.isfinite(d).all(-1)
    jobs = []
    for i0 in range(0, h, tile):
        for j0 in range(0, w, tile):
            v = valid[i0:i0 + tile, j0:j0 + tile]
            if v.any():
                jobs.append((i0, j0, pts[i0:i0 + tile, j0:j0 + tile], d[i0:i0 + tile, j0:j0 + tile], v, offs,
                             store_args, max_span))
    if procs > 1 and len(jobs) > 1:
        with mp.get_context("fork").Pool(procs) as pool:
            for i0, j0, blk in pool.imap_unordered(_render_tile, jobs, chunksize=4):
                if blk is not None:
                    out[:, i0:i0 + blk.shape[1], j0:j0 + blk.shape[2]] = blk
    else:
        for job in jobs:
            i0, j0, blk = _render_tile(job)
            if blk is not None:
                out[:, i0:i0 + blk.shape[1], j0:j0 + blk.shape[2]] = blk
    return out


def tree_sha256(folder: Path) -> str:
    lines = []
    for p in sorted(folder.rglob("*")):
        if p.is_file():
            lines.append(f"{p.relative_to(folder)}\t{iox.sha256_file(p)}\n")
    return hashlib.sha256("".join(lines).encode()).hexdigest()


# ----------------------------------------------------------------------------------------------- command
def add_render_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("tifxyz", help="mesh folder (x.tif, y.tif, z.tif, meta.json)")
    p.add_argument("--volume", required=True, help="OME-Zarr volume (local folder or https URL), level 0 is read")
    p.add_argument("--cache", required=True, help="chunk cache folder")
    p.add_argument("--out", required=True, help="surface-volume folder to write (layers.zarr, surface.json, mesh)")
    p.add_argument("--layers", type=int, default=66)
    p.add_argument("--step", type=float, default=1.0)
    p.add_argument("--render-scale", type=float, default=1.0)
    p.add_argument("--normal-sign", choices=("auto", "keep", "flip"), default="auto")
    p.add_argument("--umbilicus", help="scroll axis JSON (control_points); needed for --normal-sign auto")
    p.add_argument("--force", action="store_true", help="render even if the normal sign is ambiguous")
    p.add_argument("--voxel-um", type=float, required=True, help="voxel size of the volume (from its metadata.json)")
    p.add_argument("--procs", type=int, default=8)
    p.add_argument("--fetch-workers", type=int, default=32)
    p.add_argument("--plan-only", action="store_true", help="list the chunks to fetch and stop")
    p.add_argument("--no-coarse-check", action="store_true")
    p.add_argument("--sample", default="", help="sample name for surface.json")
    p.add_argument("--volume-id", default="", help="volume id for surface.json")


def cmd_render(a: argparse.Namespace) -> int:
    t0 = time.time()
    p, _, meta = load_tifxyz(a.tifxyz)
    n = vertex_normals(p)
    gs = float(meta["scale"][0])
    H, W = render_shape(p, gs, a.render_scale)
    pts, nrm = surface_at_pixels(p, n, np.arange(H), np.arange(W), gs, a.render_scale)
    fout, npx = (float("nan"), 0)
    if a.umbilicus:
        fout, npx = outward_fraction(pts, nrm, load_umbilicus(a.umbilicus))
    if a.normal_sign == "auto":
        if not a.umbilicus:
            raise SystemExit("--normal-sign auto needs --umbilicus")
        if 0.2 < fout < 0.8 and not a.force:
            raise SystemExit(f"normal sign ambiguous: outward share {fout:.3f}; give --normal-sign or --force")
        flip = fout < 0.5
    else:
        flip = a.normal_sign == "flip"
    store = ChunkStore(a.volume, a.cache, workers=a.fetch_workers)
    offs = (np.arange(a.layers) - 0.5 * (a.layers - 1)) * a.step
    keys = plan_chunks(pts, -nrm if flip else nrm, offs, store)
    missing = [k for k in keys if not store.path(k).exists() and store.key(k) not in store.absent]
    print(json.dumps(dict(render_hw=[H, W], layers=a.layers, flip_normals=flip, outward_share=fout,
                          chunks=len(keys), to_fetch=len(missing), to_fetch_gb=round(len(missing) * 128 ** 3 / 1e9, 2))))
    if a.plan_only:
        return 0
    t_fetch = time.time()
    store.ensure(keys)
    fetch_s = time.time() - t_fetch
    bad = [] if a.no_coarse_check else check_absent_against_coarse(store, keys)
    if bad:
        raise SystemExit(f"{len(bad)} level-0 chunks are missing where the coarse level has data (fetch failure), "
                         f"e.g. {bad[:5]}")
    store_args = (a.volume, str(a.cache), 4)
    t_r = time.time()
    layers = render_layers(pts, nrm, a.layers, store_args, a.step, flip, procs=a.procs)
    render_s = time.time() - t_r
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mesh_out = out / "mesh.tifxyz"
    if mesh_out.exists():
        shutil.rmtree(mesh_out)
    shutil.copytree(a.tifxyz, mesh_out)
    plus_d = (1 - fout if flip else fout) if fout == fout else None
    if plus_d is None:
        positive_d = "unknown (no umbilicus given)"
    else:
        positive_d = "away from the scroll axis" if plus_d > 0.5 else "towards the scroll axis"
    surface = dict(
        sample=a.sample, volume_id=a.volume_id, volume=a.volume, voxel_size_um=float(a.voxel_um),
        n_layers=a.layers, layer_rule="layer i at (i - (n-1)/2) * step voxels along the normal; d = i - n//2",
        positive_d=positive_d, flip_normals=bool(flip), normal_sign_rule=a.normal_sign,
        outward_share_mesh_normals=fout, outward_share_plus_d=plus_d, measured_px=npx,
        render_scale=a.render_scale, step=a.step, mesh=Path(a.tifxyz).name,
        mesh_sha256={f: iox.sha256_file(Path(a.tifxyz) / f) for f in ("x.tif", "y.tif", "z.tif", "meta.json")
                     if (Path(a.tifxyz) / f).exists()},
        renderer="inksweep.render (port of the EconLearn clean-room vc_render_tifxyz renderer, MIT)",
        command=" ".join(sys.argv), chunks=len(keys), absent_chunks=sum(store.key(k) in store.absent for k in keys),
        seconds=dict(fetch=round(fetch_s, 1), render=round(render_s, 1), total=round(time.time() - t0, 1)),
        z_range=[float(np.nanmin(p[..., 2])), float(np.nanmax(p[..., 2]))])
    iox.write_zarr_layers(out / "layers.zarr", layers, attrs=dict(inksweep_render=surface))
    surface["layers_tree_sha256"] = tree_sha256(out / "layers.zarr")
    iox.write_json(out / "surface.json", surface)
    print(json.dumps(dict(out=str(out), shape=list(layers.shape), flip_normals=flip, seconds=surface["seconds"])))
    return 0

"""Resampling a depth window to a reader's voxel size, and maps back to the render grid.

Hecate 9.6 µm must see 9.6 µm sampling in x, y and z. The PHerc. 343 recipe (``ink/scripts/hecate_prep.py``,
Nieuwlaar/pherc343-first-letters, MIT) builds that input from a native render as follows; :func:`resample_window_cpu`
is a transcription of it:

* depth: ``W`` planes at fractional layer positions ``c + (i - (W - 1) / 2) * target_um / native_um``, linear
  interpolation between the two neighbouring layers in float32;
* in plane: each plane resized by ``s = native_um / target_um`` to ``round(H * s) x round(W * s)`` with
  ``cv2.INTER_AREA`` when shrinking by more than 5 % (``s < 0.95``, e.g. 8.64 -> 9.6 µm) and ``cv2.INTER_LINEAR``
  otherwise (e.g. 9.362 -> 9.6 µm);
* back to uint8 by clipping to 0..255 and truncating.

A tile phase (zero rows on top, zero columns on the left) is resampled the way the recipe did it: the padded canvas is
resampled as a whole, which equals padding the resampled canvas only when the padding is a multiple of the 10-pixel
period of the 8.64 -> 9.6 ratio (the recipe's 40 x 200 padding is).

:func:`resample_window_gpu` does the same on the GPU with separable banded weights computed like OpenCV's tables
(fast mode; it can differ from OpenCV by one grey level where a value lands within float rounding of an integer).

:func:`to_render_grid` brings a map from the reader grid back to the render grid with ``cv2.INTER_LINEAR``, exactly as
the recipe's ``ensemble_maps.py`` reads Hecate maps.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch


def xy_scale(native_um: float, target_um: float) -> float:
    """In-plane scale factor from the render grid to the reader grid (< 1 shrinks)."""
    return float(native_um) / float(target_um)


def resampled_shape(height: int, width: int, native_um: float, target_um: float) -> tuple[int, int]:
    """Reader-grid shape of a ``height x width`` render: ``(max(1, round(H s)), max(1, round(W s)))``."""
    s = xy_scale(native_um, target_um)
    return max(1, int(round(height * s))), max(1, int(round(width * s)))


def interpolation_mode(native_um: float, target_um: float) -> str:
    """``"area"`` when shrinking by more than 5 %, else ``"linear"`` (the recipe's rule)."""
    return "area" if xy_scale(native_um, target_um) < 0.95 else "linear"


def _depth_interp_numpy(volume: np.ndarray, positions: np.ndarray, i: int) -> np.ndarray:
    """Plane ``i``: ``vol[lo] * (1 - f) + vol[hi] * f`` in float32, exactly as ``hecate_prep.py`` computes it."""
    depth = volume.shape[0]
    lo = np.floor(positions).astype(int)
    frac = (positions - lo).astype(np.float32)
    hi = np.minimum(lo + 1, depth - 1)
    one_minus = (1 - frac)
    return (np.asarray(volume[lo[i]]).astype(np.float32) * one_minus[i]
            + np.asarray(volume[hi[i]]).astype(np.float32) * frac[i])


def resample_window_cpu(volume: np.ndarray, positions: np.ndarray, native_um: float, target_um: float,
                        pad: tuple[int, int] = (0, 0), threads: int = 8) -> np.ndarray:
    """Exact (OpenCV) reader-grid input for one depth window.

    ``volume`` is the uint8 ``(L, H, W)`` render (numpy or array-like), ``positions`` the fractional layer positions
    (:func:`inksweep.tiling.window_positions`), ``pad`` the tile-phase padding (rows on top, columns on the left) of
    the render canvas. Returns uint8 ``(W, h, w)`` with planes in increasing position order.
    """
    import cv2
    from concurrent.futures import ThreadPoolExecutor

    positions = np.asarray(positions, dtype=np.float64)
    _, height, width = volume.shape
    pr, pc = (int(v) for v in pad)
    hp, wp = height + pr, width + pc
    nh, nw = resampled_shape(hp, wp, native_um, target_um)
    interp = cv2.INTER_AREA if interpolation_mode(native_um, target_um) == "area" else cv2.INTER_LINEAR
    out = np.empty((len(positions), nh, nw), np.uint8)

    def one(i: int) -> None:
        plane = _depth_interp_numpy(volume, positions, i)
        if pr or pc:
            padded = np.zeros((hp, wp), np.float32)
            padded[pr:, pc:] = plane
            plane = padded
        out[i] = np.clip(cv2.resize(plane, (nw, nh), interpolation=interp), 0, 255).astype(np.uint8)

    with ThreadPoolExecutor(max(1, int(threads))) as pool:
        list(pool.map(one, range(len(positions))))
    return out


# ------------------------------------------------------------------------------------------- GPU (fast) path
def _area_taps(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV ``computeResizeAreaTab`` for one axis as (source index, weight) taps ``[n_out, K]`` (float64)."""
    scale = n_in / n_out
    rows: list[list[tuple[int, float]]] = []
    for dx in range(n_out):
        fsx1 = dx * scale
        fsx2 = fsx1 + scale
        cell = min(scale, n_in - fsx1)
        sx1 = math.ceil(fsx1)
        sx2 = math.floor(fsx2)
        sx2 = min(sx2, n_in - 1)
        sx1 = min(sx1, sx2)
        taps: list[tuple[int, float]] = []
        if sx1 - fsx1 > 1e-3:
            taps.append((sx1 - 1, (sx1 - fsx1) / cell))
        for sx in range(sx1, sx2):
            taps.append((sx, 1.0 / cell))
        if fsx2 - sx2 > 1e-3:
            taps.append((sx2, min(min(fsx2 - sx2, 1.0), cell) / cell))
        rows.append(taps)
    k = max(len(t) for t in rows)
    idx = np.zeros((n_out, k), np.int64)
    w = np.zeros((n_out, k), np.float64)
    for j, taps in enumerate(rows):
        for t, (s, a) in enumerate(taps):
            idx[j, t], w[j, t] = s, a
    return idx, w


def _linear_taps(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """OpenCV ``INTER_LINEAR`` coordinates for one axis (half-pixel centres, clamped borders) as two taps."""
    scale = n_in / n_out
    idx = np.zeros((n_out, 2), np.int64)
    w = np.zeros((n_out, 2), np.float64)
    for dx in range(n_out):
        fx = (dx + 0.5) * scale - 0.5
        sx = math.floor(fx)
        fx -= sx
        if sx < 0:
            fx, sx = 0.0, 0
        if sx >= n_in - 1:
            fx, sx = 0.0, n_in - 1
        idx[dx] = (sx, min(sx + 1, n_in - 1))
        w[dx] = (1.0 - fx, fx)
    return idx, w


def _apply_taps(x: torch.Tensor, idx: np.ndarray, w: np.ndarray, dim: int) -> torch.Tensor:
    """Sum over taps ``w[:, k] * x.index_select(dim, idx[:, k])`` in tap order (float32)."""
    ti = torch.from_numpy(idx).to(x.device)
    tw = torch.from_numpy(w.astype(np.float32)).to(x.device)
    shape = [1] * x.ndim
    shape[dim] = -1
    out = None
    for k in range(idx.shape[1]):
        term = x.index_select(dim, ti[:, k]) * tw[:, k].view(shape)
        out = term if out is None else out + term
    return out


def resample_window_gpu(volume: torch.Tensor, positions: np.ndarray, native_um: float, target_um: float,
                        pad: tuple[int, int] = (0, 0)) -> torch.Tensor:
    """Fast GPU version of :func:`resample_window_cpu` for a uint8 ``(L, H, W)`` tensor on the GPU."""
    positions = np.asarray(positions, dtype=np.float64)
    depth, height, width = volume.shape
    lo = np.floor(positions).astype(np.int64)
    frac = torch.from_numpy((positions - lo).astype(np.float32)).to(volume.device)
    hi = np.minimum(lo + 1, depth - 1)
    a = volume[torch.from_numpy(lo).to(volume.device)].to(torch.float32)
    b = volume[torch.from_numpy(hi).to(volume.device)].to(torch.float32)
    planes = a * (1 - frac)[:, None, None] + b * frac[:, None, None]
    pr, pc = (int(v) for v in pad)
    if pr or pc:
        planes = torch.nn.functional.pad(planes, (pc, 0, pr, 0))
    hp, wp = height + pr, width + pc
    nh, nw = resampled_shape(hp, wp, native_um, target_um)
    taps = _area_taps if interpolation_mode(native_um, target_um) == "area" else _linear_taps
    iy, wy = taps(hp, nh)
    ix, wx = taps(wp, nw)
    planes = _apply_taps(planes, ix, wx, dim=2)          # OpenCV resizes horizontally first, then vertically
    planes = _apply_taps(planes, iy, wy, dim=1)
    return torch.clamp(planes, 0, 255).to(torch.uint8)  # float -> uint8 conversion truncates (as astype)


def to_render_grid(map_model: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A reader-grid map on the render grid: ``cv2.resize(..., INTER_LINEAR)`` in float32 (no-op if same shape)."""
    import cv2

    a = np.ascontiguousarray(map_model, dtype=np.float32)
    if a.shape == tuple(shape):
        return a
    return cv2.resize(a, (int(shape[1]), int(shape[0])), interpolation=cv2.INTER_LINEAR)


def mask_to_grid(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """A render-grid boolean mask on another grid with nearest neighbour (the recipe's ``merge_tile_phase.py``)."""
    import cv2

    m = np.asarray(mask, dtype=bool)
    if m.shape == tuple(shape):
        return m
    return cv2.resize(m.astype(np.uint8), (int(shape[1]), int(shape[0])),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def crop_offsets(padded_shape: tuple[int, int], plain_shape: tuple[int, int],
                 pad: tuple[int, int], scale: Optional[float]) -> tuple[int, int]:
    """Rows / columns to drop from a padded-phase reader map to align it with the plain map.

    Matches ``merge_tile_phase.py``: the offset is the shape difference, which must equal ``round(pad * scale)``.
    """
    dr = int(padded_shape[0]) - int(plain_shape[0])
    dc = int(padded_shape[1]) - int(plain_shape[1])
    if scale is not None:
        er, ec = int(round(pad[0] * scale)), int(round(pad[1] * scale))
        if (dr, dc) != (er, ec):
            raise ValueError(f"padded map is offset by {(dr, dc)}, expected {(er, ec)}: the tile-phase padding is not "
                             "a multiple of the resampling period")
    return dr, dc

"""Tile grids, blend windows, depth windows and depth labels, as the reference implementations define them.

Every function here is a small, exact transcription of one rule of a reference implementation:

* ``sliding_positions``: villa ``infer._sliding_positions_1d`` (positions ``0, s, 2s, ...`` plus one flush with the
  far edge). Hecate's ``axis_starts`` yields the same positions (``tests/test_tiling.py`` checks both).
* ``inside_mask_positions`` and ``v8in_coverage``: v8-in ``ink8um.inference.tile_positions`` / ``coverage_mask``.
* ``hann_window``: villa ``compute_importance_map_2d(mode="hann")`` and Hecate's window (identical).
* ``villa_gaussian_window``: villa ``compute_importance_map_2d(mode="gaussian")``.
* ``v8in_gaussian_window``: v8-in ``gkern(64, 1) / max`` (float32).
* ``window_start`` / ``window_indices`` / ``window_positions``: the PHerc. 343 recipe's depth windows
  (``run_ink.sh``: ``s = N/2 + d - 8``; ``hecate_prep.py``: planes at ``(N-1)/2 + d + (i - 7.5) * 9.6 / um``).
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Sequence

import numpy as np
import torch


# ----------------------------------------------------------------------------------------------- tile grids
def sliding_positions(length: int, tile: int, stride: int) -> list[int]:
    """Tile starts along one axis: ``0, stride, ...`` and a final start ``length - tile`` (villa / Hecate rule).

    A canvas shorter than the tile gets the single start 0 (the tile is zero-padded past the edge).
    """
    length, tile, stride = int(length), int(tile), max(1, int(stride))
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def hecate_axis_starts(length: int, tile: int, stride: int) -> list[int]:
    """Hecate ``axis_starts`` verbatim (kept for the equivalence test)."""
    return list(range(0, max(0, length - tile), stride)) + [max(0, length - tile)]


def sliding_grid(height: int, width: int, tile: int, stride: int) -> np.ndarray:
    """Tile origins ``(y, x)``, int64 ``[N, 2]``, row-major (y outer, x inner) like villa's ``iter_blocks``."""
    ys = sliding_positions(height, tile, stride)
    xs = sliding_positions(width, tile, stride)
    grid = np.array([(y, x) for y in ys for x in xs], dtype=np.int64)
    return grid.reshape(-1, 2)


def v8in_coverage(window_max: np.ndarray) -> np.ndarray:
    """v8-in coverage mask: ``max over depth > 0`` then a 7x7 elliptical closing (uint8 0/255)."""
    import cv2

    coverage = (np.asarray(window_max) > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(coverage, cv2.MORPH_CLOSE, kernel)


def inside_mask_positions(mask: np.ndarray, tile: int, stride: int) -> np.ndarray:
    """v8-in ``tile_positions``: origins ``(y, x)`` of every tile lying entirely inside ``mask``, row-major.

    Unlike the sliding grid there is no final edge-aligned tile.
    """
    inside = np.asarray(mask) > 0
    height, width = inside.shape
    integral = np.pad(inside.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    ys = np.arange(0, height - tile + 1, stride, dtype=np.int64)
    xs = np.arange(0, width - tile + 1, stride, dtype=np.int64)
    if len(ys) == 0 or len(xs) == 0:
        return np.zeros((0, 2), np.int64)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    total = (integral[yy + tile, xx + tile] - integral[yy, xx + tile] - integral[yy + tile, xx] + integral[yy, xx])
    keep = total == tile * tile
    return np.stack([yy[keep], xx[keep]], axis=1).astype(np.int64).reshape(-1, 2)


def tiles_touching(origins: np.ndarray, tile: int, need: np.ndarray) -> np.ndarray:
    """Boolean per origin: does the tile overlap any True pixel of ``need`` (same grid as the origins)?"""
    if len(origins) == 0:
        return np.zeros(0, bool)
    h, w = need.shape
    integral = np.pad(need.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    y0 = np.clip(origins[:, 0], 0, h)
    x0 = np.clip(origins[:, 1], 0, w)
    y1 = np.clip(origins[:, 0] + tile, 0, h)
    x1 = np.clip(origins[:, 1] + tile, 0, w)
    return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) > 0


# ----------------------------------------------------------------------------------------------- blend windows
def hann_window(tile: int, floor: float = 0.001) -> torch.Tensor:
    """villa / Hecate overlap-add window: outer product of symmetric Hann windows, divided by its max, floored."""
    weight = torch.outer(torch.hann_window(tile, periodic=False, dtype=torch.float32),
                         torch.hann_window(tile, periodic=False, dtype=torch.float32))
    weight /= torch.clamp(weight.max(), min=torch.finfo(weight.dtype).eps)
    return torch.clamp(weight, min=floor)


def villa_gaussian_window(tile: int, sigma_scale: float = 0.125) -> torch.Tensor:
    """villa ``compute_importance_map_2d(mode="gaussian")``."""
    sigma = max(tile * float(sigma_scale), 1e-6)
    coords = torch.arange(tile, dtype=torch.float32) - (tile - 1) / 2
    gy, gx = torch.meshgrid(coords, coords, indexing="ij")
    weight = torch.exp(-0.5 * ((gy / sigma) ** 2 + (gx / sigma) ** 2))
    weight /= torch.clamp(weight.max(), min=torch.finfo(weight.dtype).eps)
    return torch.clamp(weight, min=torch.finfo(weight.dtype).eps)


def v8in_gaussian_window(tile: int, nsig: float = 1.0) -> torch.Tensor:
    """v8-in ``gkern(tile, nsig)`` (normal-CDF kernel) divided by its maximum, as float32."""
    x = np.linspace(-nsig, nsig, tile + 1)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    k1 = np.diff(cdf)
    k2 = np.outer(k1, k1)
    k2 = k2 / k2.sum()
    return torch.from_numpy((k2 / k2.max()).astype(np.float32))


def hann_sum1_window(tile: int) -> torch.Tensor:
    """The organisers' optimized_inference ``hann2d``: ``outer(np.hanning(h), np.hanning(w)) / sum`` (float32)."""
    wy = np.hanning(tile).astype(np.float32)
    k = np.outer(wy, wy)
    s = k.sum()
    return torch.from_numpy((k / (s if s > 0 else 1.0)).astype(np.float32))


def blend_window(mode: str, tile: int, floor: float = 0.001) -> torch.Tensor:
    """The overlap-add weight ``[tile, tile]`` float32 for a ReaderSpec ``blend`` mode."""
    if mode == "hann_sum1":
        return hann_sum1_window(tile)
    if mode == "hann":
        return hann_window(tile, floor)
    if mode == "gaussian":
        return villa_gaussian_window(tile)
    if mode == "v8in_gaussian":
        return v8in_gaussian_window(tile)
    if mode == "constant":
        return torch.ones((tile, tile), dtype=torch.float32)
    raise ValueError(f"unknown blend mode {mode!r}")


# ----------------------------------------------------------------------------------------------- depth windows
def window_start(n_layers: int, depth: int, window: int) -> int:
    """First layer of the ``window``-layer window for depth ``depth``: ``n_layers // 2 + depth - window // 2``."""
    return int(n_layers) // 2 + int(depth) - int(window) // 2


def window_indices(n_layers: int, depth: int, window: int, orientation: str = "forward") -> list[int]:
    """Layer indices fed to the model, in model order; raises ``ValueError`` if the window leaves the stack."""
    s = window_start(n_layers, depth, window)
    if s < 0 or s + window > n_layers:
        raise ValueError(f"depth {depth:+d} needs layers {s}..{s + window - 1}, outside 0..{n_layers - 1}")
    idx = list(range(s, s + window))
    if orientation == "reverse":
        idx.reverse()
    elif orientation != "forward":
        raise ValueError(f"orientation must be 'forward' or 'reverse', got {orientation!r}")
    return idx


def window_centre(n_layers: int, depth: int, window: int) -> float:
    """Centre of the depth window in layer units (``L//2 + d`` for odd windows, ``L//2 + d - 0.5`` for even)."""
    return window_start(n_layers, depth, window) + (int(window) - 1) / 2.0


def window_positions(n_layers: int, depth: int, window: int, step: float) -> np.ndarray:
    """Fractional layer positions of a resampled window (``hecate_prep.py``): ``c + (i - (W - 1) / 2) * step``.

    ``step`` is the reader voxel size over the render voxel size (9.6 / 8.64 for Hecate on an 8.64 µm render); with
    ``step = 1`` the positions are exactly :func:`window_indices`.
    """
    c = window_centre(n_layers, depth, window)
    pos = c + (np.arange(window) - (window - 1) / 2.0) * float(step)
    if pos.min() < 0 or pos.max() > n_layers - 1:
        raise ValueError(f"depth {depth:+d} needs planes {pos.min():.2f}..{pos.max():.2f}, outside 0..{n_layers - 1}")
    return pos


def depth_range(n_layers: int, window: int, step: float = 1.0) -> tuple[int, int]:
    """The smallest and largest depth whose window fits in an ``n_layers`` stack."""
    ok = []
    for d in range(-n_layers, n_layers + 1):
        try:
            window_positions(n_layers, d, window, step)
        except ValueError:
            continue
        ok.append(d)
    if not ok:
        raise ValueError(f"no depth fits a {window}-layer window in {n_layers} layers")
    return min(ok), max(ok)


# ----------------------------------------------------------------------------------------------- depth labels
def depth_tag(depth: int) -> str:
    """``d+01``, ``d-15``: the file label of a depth."""
    return f"d{int(depth):+03d}"


def parse_depths(spec: str | Sequence[str] | Iterable[int]) -> list[int]:
    """``"-15..15"``, ``"-4..4:2"``, ``"0,1,-1"`` or a list of those -> sorted unique depths."""
    if isinstance(spec, str):
        parts: list = [p for p in re.split(r"[,\s]+", spec.strip()) if p]
    else:
        parts = list(spec)
    out: set[int] = set()
    for part in parts:
        if isinstance(part, (int, np.integer)):
            out.add(int(part))
            continue
        m = re.fullmatch(r"(-?\d+)\.\.(-?\d+)(?::(\d+))?", str(part))
        if m:
            a, b, step = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
            lo, hi = min(a, b), max(a, b)
            out.update(range(lo, hi + 1, step))
        else:
            out.add(int(part))
    return sorted(out)


def centre_out(depths: Iterable[int]) -> list[int]:
    """Depths ordered 0, -1, +1, -2, +2, ... (``run_depth_sweep.sh``), so partial runs hold the central depths."""
    return sorted(set(int(d) for d in depths), key=lambda d: (abs(d), d > 0))

"""Per-patch robust (median / MAD) normalisation of the villa ink readers, on the GPU, equal to the CPU reference.

The villa flat inference (``vesuvius.ink_detection.inference.infer``) reads every ``17 x 128 x 128`` patch as float32
and passes it to ``vesuvius.image_proc.intensity.normalization.normalize_robust``: clip to the 1st..99th percentile,
subtract the median, divide by 1.4826 x MAD, with a fallback divisor when the MAD vanishes. On the CPU this costs
25-45 ms per patch and makes the reference pipeline CPU-bound.

:func:`normalize_robust_reference` is a transcription of that function for the unmasked call villa makes (source:
ScrollPrize/villa ``vesuvius/src/vesuvius/image_proc/intensity/normalization.py``, MIT, Copyright (c) 2024 Vesuvius
Challenge). It is used for the rare fallback patches and by the tests.

:class:`RobustMadGPU` computes the same values on the GPU for a batch of uint8 patches. The method follows the exact
GPU path of EconLearn/vesuvius-first-letters-stage-loss (``tools/ensemble/speed/fastvilla.py``, MIT, Copyright (c) 2026
Jude Wallis), re-implemented here:

* the input is uint8, so every value is an integer in 0..255 and a 256-bin histogram per patch gives every order
  statistic exactly;
* the two percentiles are numpy's own ``np.percentile`` evaluated on a proxy array that has the same order statistics
  around numpy's interpolation index (cached), so the result is numpy's whatever arithmetic numpy uses internally;
* clipping, the median and the MAD follow numpy's float32 arithmetic (NEP 50: Python floats adopt float32);
* patches that take numpy's fallback branch (scaled MAD < 1e-6, e.g. a patch that is more than half zero padding) are
  normalised on the CPU by :func:`normalize_robust_reference`.

The result is bit-identical to the CPU function for the same numpy version (``tests/test_norm.py``).
"""
from __future__ import annotations

import functools
from typing import Optional

import numpy as np
import torch

SCALE = 1.4826            # MAD -> standard deviation of a normal distribution (as in the reference)
MAD_FLOOR = 1e-6          # below this the reference falls back to another divisor


def normalize_robust_reference(image: np.ndarray, percentile_lower: float = 1.0,
                               percentile_upper: float = 99.0) -> np.ndarray:
    """Transcription of villa ``normalize_robust(image)`` without a mask (``clip_values=True``, float32)."""
    arr = np.asarray(image).astype(np.float32, copy=True)
    valid = arr.reshape(-1)
    if valid.size == 0:
        return arr
    std_preclip = float(np.std(valid))
    lower_val = float(np.percentile(valid, percentile_lower))
    upper_val = float(np.percentile(valid, percentile_upper))
    np.clip(arr, lower_val, upper_val, out=arr)
    valid = arr.reshape(-1)
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    scaled_mad = SCALE * mad
    if not np.isfinite(scaled_mad) or scaled_mad < MAD_FLOOR:
        percentile_span = upper_val - lower_val
        fallback_values = []
        if np.isfinite(std_preclip):
            fallback_values.append(abs(std_preclip))
        if np.isfinite(percentile_span):
            fallback_values.append(abs(percentile_span) / 2.0)
        scaled_mad = next((c for c in fallback_values if c >= MAD_FLOOR), None)
        if scaled_mad is None or not np.isfinite(scaled_mad):
            scaled_mad = 1.0
    arr -= median
    arr /= scaled_mad
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


@functools.lru_cache(maxsize=1 << 16)
def _numpy_percentile(n: int, q: float, first: int, window: tuple[int, ...]) -> float:
    """``float(np.percentile(x, q))`` for a float32 array ``x`` of ``n`` values whose sorted values at positions
    ``first, first + 1, ...`` are ``window``. numpy reads only the two order statistics at its interpolation index,
    which lies inside the window, so this is numpy's own result (including its interpolation arithmetic)."""
    if window[0] == window[-1]:
        return float(window[0])           # both interpolation points are equal: numpy returns that value exactly
    x = np.empty(n, np.float32)
    x[:first] = window[0]
    x[first:first + len(window)] = window
    x[first + len(window):] = window[-1]
    return float(np.percentile(x, q))


def _order_statistic(cum: torch.Tensor, k: int) -> torch.Tensor:
    """Value of the k-th smallest element (0-based) per row, from inclusive cumulative counts over values 0..255."""
    return (cum <= k).sum(1)


class RobustMadGPU:
    """Exact batched ``normalize_robust`` for uint8 patches of ``n_values`` voxels each.

    Call with a uint8 tensor ``[B, ...]`` on any device; returns float32 of the same shape on that device.
    ``n_fallback`` counts the patches normalised on the CPU so far.
    """

    def __init__(self, n_values: int, percentile_lower: float = 1.0, percentile_upper: float = 99.0) -> None:
        self.n = int(n_values)
        if self.n <= 0:
            raise ValueError("n_values must be positive")
        self.percentiles = (float(percentile_lower), float(percentile_upper))
        self.windows = []
        for q in self.percentiles:
            # numpy computes the interpolation index itself (in float32 for float32 input in numpy 2.x); a window of
            # +-(1 + n >> 21) order statistics around the float64 estimate always contains both points it reads.
            est = int(np.floor((self.n - 1) * (q / 100.0)))
            half = 1 + (self.n >> 21)
            self.windows.append((q, max(0, est - half), min(self.n - 1, est + half + 1)))
        if self.n % 2 == 0:
            self.middle = (self.n // 2 - 1, self.n // 2)
        else:
            self.middle = (self.n // 2, self.n // 2)
        self.n_fallback = 0
        # CPU threads for the fallback patches (INKSWEEP_NORM_THREADS, default: the usable cores, at most 16)
        import os

        avail = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        self.threads = int(os.environ.get("INKSWEEP_NORM_THREADS", min(16, avail)))

    def _percentile(self, cum: torch.Tensor, q: float, first: int, last: int) -> torch.Tensor:
        ks = torch.arange(first, last + 1, device=cum.device)
        stats = (cum[:, None, :] <= ks[None, :, None]).sum(2)          # [B, K] order statistics first..last
        values = [_numpy_percentile(self.n, q, first, tuple(row)) for row in stats.cpu().tolist()]
        return torch.tensor(values, dtype=torch.float64, device=cum.device)

    def _middle(self, values: torch.Tensor, cum: torch.Tensor) -> torch.Tensor:
        """numpy median (float32) from per-row sorted ``values`` [B, K] with inclusive cumulative counts ``cum``."""
        a = values.gather(1, _order_statistic(cum, self.middle[0])[:, None])[:, 0]
        if self.middle[0] == self.middle[1]:
            return a
        b = values.gather(1, _order_statistic(cum, self.middle[1])[:, None])[:, 0]
        return (a + b) / 2                                               # float32 sum, then an exact halving

    @torch.no_grad()
    def __call__(self, patches: torch.Tensor) -> torch.Tensor:
        if patches.dtype != torch.uint8:
            raise TypeError(f"RobustMadGPU expects uint8 patches, got {patches.dtype}")
        shape = patches.shape
        batch = int(shape[0])
        flat = patches.reshape(batch, -1)
        if flat.shape[1] != self.n:
            raise ValueError(f"patches hold {flat.shape[1]} values, normaliser was built for {self.n}")
        device = flat.device
        offsets = torch.arange(batch, device=device, dtype=torch.int64)[:, None] * 256
        hist = torch.bincount((flat.to(torch.int64) + offsets).reshape(-1), minlength=256 * batch).reshape(batch, 256)
        cum = hist.cumsum(1)
        lo64, hi64 = (self._percentile(cum, q, first, last) for q, first, last in self.windows)
        lo, hi = lo64.to(torch.float32), hi64.to(torch.float32)         # np.clip with Python floats -> float32 bounds
        # median of the clipped patch = median of the clipped order statistics (clipping is monotone)
        grey = torch.arange(256, device=device, dtype=torch.float32)[None, :].expand(batch, 256)
        clipped = torch.minimum(torch.maximum(grey, lo[:, None]), hi[:, None])   # clip(v) for every grey value v
        median = self._middle(clipped, cum)
        # MAD: median of |clip(v) - median| over the multiset given by the histogram
        dev_values, order = torch.sort(torch.abs(clipped - median[:, None]), dim=1)
        dev_cum = hist.gather(1, order).cumsum(1)
        mad = self._middle(dev_values, dev_cum)
        scaled = mad.to(torch.float64) * SCALE
        fallback = ~torch.isfinite(scaled) | (scaled < MAD_FLOOR)
        divisor = torch.where(fallback, torch.ones_like(scaled), scaled).to(torch.float32)
        view = (batch,) + (1,) * (len(shape) - 1)
        x = patches.to(torch.float32)
        x = torch.minimum(torch.maximum(x, lo.view(view)), hi.view(view))
        x = (x - median.view(view)) / divisor.view(view)
        if bool(fallback.any()):
            rows = torch.nonzero(fallback).flatten().cpu().tolist()
            self.n_fallback += len(rows)
            host = patches[rows].cpu().numpy()
            if len(rows) > 1 and self.threads > 1:                     # numpy releases the GIL in these steps
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(min(self.threads, len(rows))) as pool:
                    refs = list(pool.map(lambda a: normalize_robust_reference(a, *self.percentiles), host))
            else:
                refs = [normalize_robust_reference(a, *self.percentiles) for a in host]
            x[rows] = torch.from_numpy(np.stack(refs)).to(device)
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def robust_mad(patches: torch.Tensor, normaliser: Optional[RobustMadGPU] = None) -> torch.Tensor:
    """Convenience wrapper: normalise a uint8 batch ``[B, ...]`` with a (cached) :class:`RobustMadGPU`."""
    n = int(patches[0].numel())
    if normaliser is None or normaliser.n != n:
        normaliser = _cached_normaliser(n)
    return normaliser(patches)


@functools.lru_cache(maxsize=8)
def _cached_normaliser(n: int) -> RobustMadGPU:
    return RobustMadGPU(n)

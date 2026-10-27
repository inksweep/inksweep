"""Tile grids, coverage, blend windows and depth windows against transcriptions of the reference code."""
import math

import numpy as np
import pytest
import torch

from inksweep import tiling as T


# ---- transcriptions of the reference rules (kept verbatim in spirit, written independently of inksweep) ----
def villa_sliding_1d(length, patch_size, stride):        # villa infer._sliding_positions_1d
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, max(1, stride)))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def hecate_axis_starts(length, patch, stride):            # hecate.py axis_starts
    return list(range(0, max(0, length - patch), stride)) + [max(0, length - patch)]


def v8in_tile_positions(mask, tile_size=64, stride=21):  # ink8um.inference.tile_positions (returns (x, y))
    inside = mask > 0
    height, width = inside.shape
    integral = np.pad(inside.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    points = []
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            y1, x1 = y + tile_size, x + tile_size
            total = integral[y1, x1] - integral[y, x1] - integral[y1, x] + integral[y, x]
            if total == tile_size * tile_size:
                points.append((x, y))
    return points


@pytest.mark.parametrize("length", [1, 31, 64, 65, 127, 128, 129, 191, 192, 200, 1000, 2560, 6720, 2304, 6048])
@pytest.mark.parametrize("tile,stride", [(128, 64), (64, 32), (384, 192), (64, 21), (256, 128), (16, 16)])
def test_sliding_positions_match_villa_and_hecate(length, tile, stride):
    ours = T.sliding_positions(length, tile, stride)
    assert ours == villa_sliding_1d(length, tile, stride)
    assert ours == hecate_axis_starts(length, tile, stride)
    assert ours == T.hecate_axis_starts(length, tile, stride)


@pytest.mark.parametrize("h,w", [(100, 300), (128, 128), (2560, 6720), (37, 513)])
def test_sliding_grid_covers_every_pixel(h, w):
    tile, stride = 128, 64
    grid = T.sliding_grid(h, w, tile, stride)
    cover = np.zeros((max(h, tile), max(w, tile)), np.int32)
    for y, x in grid:
        cover[y:y + tile, x:x + tile] += 1
    assert (cover[:h, :w] >= 1).all()
    assert (grid[:, 0] + min(tile, h) <= max(h, tile)).all()
    ys = sorted(set(grid[:, 0].tolist()))
    assert grid.tolist() == [[y, x] for y in ys for x in sorted(set(grid[:, 1].tolist()))]   # row-major


def test_inside_mask_positions_match_v8in():
    rng = np.random.default_rng(0)
    for _ in range(5):
        mask = (rng.random((300, 420)) > 0.03).astype(np.uint8) * 255
        mask[:, :50] = 0
        ours = T.inside_mask_positions(mask, 64, 21)
        ref = v8in_tile_positions(mask, 64, 21)
        assert [(int(x), int(y)) for y, x in ours] == ref


def test_v8in_coverage_closing():
    import cv2

    m = np.zeros((50, 50), np.uint8)
    m[10:40, 10:40] = 7
    m[20, 20] = 0                                   # a hole the closing fills
    cov = T.v8in_coverage(m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ref = cv2.morphologyEx(((m > 0).astype(np.uint8) * 255), cv2.MORPH_CLOSE, kernel)
    assert np.array_equal(cov, ref) and cov[20, 20] == 255


def test_tiles_touching():
    need = np.zeros((100, 100), bool)
    need[60:62, 60:62] = True
    origins = T.sliding_grid(100, 100, 32, 16)
    hit = T.tiles_touching(origins, 32, need)
    for (y, x), h in zip(origins, hit):
        assert h == bool(need[y:y + 32, x:x + 32].any())


# ---- blend windows ----
def villa_hann(p):                                     # villa compute_importance_map_2d(mode="hann")
    weight = torch.outer(torch.hann_window(p, periodic=False, dtype=torch.float32),
                         torch.hann_window(p, periodic=False, dtype=torch.float32))
    weight /= torch.clamp(weight.max(), min=torch.finfo(weight.dtype).eps)
    return torch.clamp(weight, min=0.001)


def hecate_hann(p):                                    # hecate.py predict()
    win = torch.hann_window(p, periodic=False)
    win = win[:, None] * win[None, :]
    return (win / win.max().clamp_min(torch.finfo(torch.float32).eps)).clamp_min(.001)


def gkern(kernlen=64, nsig=1.0):                       # ink8um.inference.gkern
    x = np.linspace(-nsig, nsig, kernlen + 1)
    cdf = 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))
    kern1d = np.diff(cdf)
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()


@pytest.mark.parametrize("p", [64, 128, 384])
def test_hann_windows_are_bit_identical(p):
    ours = T.hann_window(p)
    assert torch.equal(ours, villa_hann(p))
    assert torch.equal(ours, hecate_hann(p))
    assert float(ours.min()) == pytest.approx(0.001) and float(ours.max()) == 1.0


def test_v8in_window():
    ref = gkern(64, 1)
    ref = (ref / ref.max()).astype(np.float32)
    assert np.array_equal(T.v8in_gaussian_window(64).numpy(), ref)


def test_hann_sum1_window():
    wy = np.hanning(256).astype(np.float32)
    k = np.outer(wy, wy)
    ref = k / k.sum()
    assert np.array_equal(T.hann_sum1_window(256).numpy(), ref.astype(np.float32))


# ---- depth windows ----
@pytest.mark.parametrize("d", range(-15, 16))
def test_villa_window_is_run_ink_rule(d):
    n = 66
    s = n // 2 + d - 8                                 # run_ink.sh: S = N/2 + D - 8, layers S..S+16
    assert T.window_indices(n, d, 17) == list(range(s, s + 17))
    assert T.window_indices(n, d, 17, "reverse") == list(range(s + 16, s - 1, -1))
    assert T.window_centre(n, d, 17) == 33 + d


@pytest.mark.parametrize("d", range(-15, 16))
def test_hecate_positions_are_hecate_prep_rule(d):
    n, native, target = 66, 8.64, 9.6
    zc = (n - 1) / 2.0 + d                             # run_ink.sh: ZC = (N-1)/2 + d
    step = target / native
    ref = zc + (np.arange(16) - (16 - 1) / 2.0) * step
    assert np.allclose(T.window_positions(n, d, 16, step), ref, atol=1e-12, rtol=0)


def test_v8in_window_rule():
    for d in (-12, 0, 5):
        assert T.window_indices(66, d, 24) == list(range(66 // 2 + d - 12, 66 // 2 + d + 12))


def test_window_positions_step_one_equal_indices():
    for w in (16, 17, 24, 62):
        for d in (-3, 0, 4):
            assert np.array_equal(T.window_positions(109, d, w, 1.0), np.array(T.window_indices(109, d, w), float))


def test_window_outside_stack_raises():
    with pytest.raises(ValueError):
        T.window_indices(66, 30, 17)
    lo, hi = T.depth_range(66, 17)
    assert (lo, hi) == (-25, 24)


def test_depth_labels_and_order():
    assert T.depth_tag(1) == "d+01" and T.depth_tag(-15) == "d-15" and T.depth_tag(0) == "d+00"
    assert T.parse_depths("-2..2") == [-2, -1, 0, 1, 2]
    assert T.parse_depths(["-4..4:4", "7"]) == [-4, 0, 4, 7]
    assert T.centre_out(range(-2, 3)) == [0, -1, 1, -2, 2]

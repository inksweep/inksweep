"""The ensemble statistics and images against a transcription of the PHerc. 343 ensemble_maps.py arithmetic."""
import math

import numpy as np
import pytest

from inksweep import ensemble as E
from inksweep import io as iox
from inksweep.tiling import depth_tag


def test_pct_hist_equals_numpy_percentile():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = rng.integers(0, 129, size=rng.integers(10, 5000)) / 128.0
        hist = np.bincount(np.rint(v * 128).astype(int), minlength=129)
        ours = E.pct_hist(hist, np.arange(129) / 128.0, [1.0, 99.5])
        ref = np.percentile(v, [1.0, 99.5])
        assert np.allclose(ours, ref, rtol=0, atol=1e-12)


def write_case(tmp_path, rng, models, depths, h=64, w=80, n_layers=66):
    vol = rng.integers(1, 255, size=(n_layers, h, w), dtype=np.uint8)
    vol[:, :5, :] = 0                                      # invalid rows
    out = tmp_path / "run"
    maps = {}
    for m in models:
        for side in ("reading", "blind"):
            for d in depths:
                if m.startswith("v"):                      # villa-like 8-bit decoded values
                    q = rng.integers(0, 256, size=(h, w)).astype(np.uint8)
                    a = np.clip((q.astype(np.float32) - 64.0) / 128.0, 0, 1)
                else:                                       # continuous (Hecate-like)
                    a = rng.random((h, w), dtype=np.float32)
                maps[(m, side, d)] = a
                iox.write_tif(E.map_file(out, m, side, d), a)
    return vol, out, maps


def reference_stats(vol, maps, models, villa, stat_depths):
    """Transcription of ensemble_maps.py main() up to the ensemble stretch."""
    n66 = vol.shape[0]
    mid = n66 // 2
    valid = (vol > 0).all(0)
    sub = np.zeros_like(valid)
    sub[4::8, 4::8] = True
    sub &= valid
    stats = {}
    subs = {}
    for m in models:
        hist = np.zeros(129 if villa[m] else 65536, np.int64)
        s1 = s2 = 0.0
        n = 0
        sv = []
        for d in stat_depths:
            M = maps[(m, "reading", d)]
            v = M[valid]
            if villa[m]:
                hist += np.bincount(np.rint(v * 128).astype(np.int64), minlength=129)[:129]
            else:
                hist += np.bincount(np.minimum((v * 65536).astype(np.int64), 65535), minlength=65536)
            s1 += float(v.sum(dtype=np.float64))
            s2 += float(np.square(v, dtype=np.float64).sum())
            n += v.size
            sv.append(M[sub].astype(np.float32))
        values = (np.arange(129) / 128.0) if villa[m] else (np.arange(65536) / 65536)
        lo, hi = E.pct_hist(hist, values, [1.0, 99.5])
        mu = s1 / n
        sd = math.sqrt(max(s2 / n - mu * mu, 0.0)) + 1e-6
        stats[m] = dict(p1=lo, p99_5=hi, mean=mu, sd=sd)
        subs[m] = np.stack(sv)
    acc = None
    for m in models:
        z = (subs[m] - stats[m]["mean"]) / stats[m]["sd"]
        acc = z if acc is None else acc + z
    acc /= len(models)
    ens = [float(x) for x in np.percentile(acc.ravel(), [1.0, 99.5])]
    return valid, stats, ens


def test_ensemble_matches_transcription(tmp_path):
    rng = np.random.default_rng(1)
    models = ["vA", "hB", "vC"]
    depths = [-2, -1, 0, 1, 2]
    vol, out, maps = write_case(tmp_path, rng, models, depths)
    surface = iox.SurfaceVolume(path=None, name="syn", layers=vol, kind="mem", voxel_size_um=8.64)
    bins = {"vA": "villa129", "vC": "villa129", "hB": "fine"}
    cfg = E.EnsembleConfig(models=models, stat_depths=depths, write_depths=[1], display_depth=1, bins=bins,
                           voxel_um=8.64)
    stats = E.compute(out, surface, cfg)
    villa = {m: bins[m] == "villa129" for m in models}
    valid, ref, ens = reference_stats(vol, maps, models, villa, depths)
    for m in models:
        for k in ("p1", "p99_5", "mean", "sd"):
            assert stats["models"][m][k] == ref[m][k], (m, k)
    assert [stats["ensemble"]["p1"], stats["ensemble"]["p99_5"]] == ens
    # the reading-side ensemble image at d = +1, rotated 180 degrees, invalid black
    acc = None
    for m in models:
        z = (maps[(m, "reading", 1)] - ref[m]["mean"]) / ref[m]["sd"]
        acc = z if acc is None else acc + z
    e = acc / len(models)
    b = (np.clip((e - ens[0]) / max(ens[1] - ens[0], 1e-9), 0, 1) * 255).astype(np.uint8)
    b[~valid] = 0
    import cv2

    img = cv2.imread(str(out / "ink" / "grey" / "syn__ensemble__d+01.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(img, b[::-1, ::-1])
    # stored z-mean and the stretched contract ensemble
    stored = iox.read_tif(out / "ink" / "ensemble" / f"{depth_tag(1)}.tif")
    assert np.array_equal(stored[valid], e.astype(np.float32)[valid])
    u = iox.read_tif(out / "ink" / "ensemble.tif")
    assert u.min() >= 0 and u.max() <= 1 and (u[~valid] == 0).all()
    assert (out / "ink" / "preview_d+01.png").is_file()


def test_blind_side_uses_reading_statistics(tmp_path):
    rng = np.random.default_rng(2)
    models = ["vA", "vB"]
    vol, out, maps = write_case(tmp_path, rng, models, [0])
    surface = iox.SurfaceVolume(path=None, name="syn", layers=vol, kind="mem", voxel_size_um=8.64)
    cfg = E.EnsembleConfig(models=models, stat_depths=[0], write_depths=[0], display_depth=0,
                           bins={"vA": "villa129", "vB": "villa129"})
    stats = E.compute(out, surface, cfg)
    s = stats["models"]["vA"]
    valid = (vol > 0).all(0)
    b = (np.clip((maps[("vA", "blind", 0)] - s["p1"]) / max(s["p99_5"] - s["p1"], 1e-9), 0, 1) * 255).astype(np.uint8)
    b[~valid] = 0
    import cv2

    img = cv2.imread(str(out / "ink" / "grey" / "syn__vA_blind__d+00.png"), cv2.IMREAD_UNCHANGED)
    assert np.array_equal(img, b[::-1, ::-1])


def test_depth_mean(tmp_path):
    out = tmp_path / "dm"
    for d in (-1, 0, 1):
        iox.write_tif(E.map_file(out, "hecate", "reading", d), np.full((4, 4), d + 2, np.float32))
    name = E.depth_mean(out, "hecate", [-1, 0, 1], [0])
    m = iox.read_tif(E.map_file(out, name, "reading", 0))
    assert np.allclose(m, 2.0)


def test_orientation_flags():
    a = np.arange(6).reshape(2, 3)
    assert np.array_equal(E.orient(a, E.parse_flip("reading")), a[::-1, ::-1])
    assert np.array_equal(E.orient(a, E.parse_flip("raw")), a)
    assert np.array_equal(E.orient(a, E.parse_flip("lr")), a[:, ::-1])

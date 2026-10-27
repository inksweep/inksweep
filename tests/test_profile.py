"""``inksweep bench --profile``: the stage timers change no value, count every forward call and fit in the map's time."""
import numpy as np
import pytest
import torch
from torch import nn

from inksweep import bench as B
from inksweep import io as iox
from inksweep import sweep as S
from inksweep.readers.spec import ReaderSpec

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


class Counting(nn.Module):
    """Logits from the window's depth mean; counts its forward calls."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):                                  # x [B, 1, Z, p, p]
        self.calls += 1
        return (x.mean(2) - 0.4) * 6.0


def spec_for(model, name="fake"):
    return ReaderSpec(name=name, window_layers=5, voxel_size_um=9.0, tile=32, stride=16,
                      load=lambda paths, device: model.to(device),
                      preprocess=lambda s: (s.to(torch.float32) / 255.0)[:, None],
                      forward=lambda m, x: m(x), postprocess=lambda lg: lg.float().sigmoid(), batched=True,
                      reference_batch=3)


def surface(seed=0):
    vol = np.random.default_rng(seed).integers(1, 255, size=(11, 70, 101), dtype=np.uint8)
    vol[:, :, :12] = 0
    vol[:, 50:, 80:] = 0
    return iox.SurfaceVolume(path=None, name="syn", layers=vol, kind="mem", voxel_size_um=9.0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("mode", ["exact", "fast"])
def test_profiling_changes_no_value(tmp_path, mode, device):
    cfg = dict(depths=[0], orientations=("reverse",), mode=mode, phases=((0, 0), (8, 20)), phase_merge="mean")
    if mode == "fast":
        cfg.update(precision="fp32", batch=7)
    maps = {}
    for profile in (False, True):
        model = Counting()
        sweep = S.Sweep(surface(), tmp_path / f"p{profile}", S.SweepConfig(**cfg), device=device, profile=profile)
        rec = sweep.run_map(spec_for(model), spec_for(model).load(None, sweep.device), 0, "reverse")
        maps[profile] = iox.read_tif(tmp_path / f"p{profile}" / rec.files["raw"])
        if not profile:
            assert rec.profile is None
            continue
        assert set(rec.profile["seconds"]) == set(S.StageTimer.STAGES)
        assert all(v >= 0 for v in rec.profile["seconds"].values())
        assert rec.profile["calls"]["forward"] == model.calls > 0
        assert rec.profile["calls"]["read"] >= 4                    # each phase: the window, then its tile groups
        assert rec.profile["calls"]["schedule"] == 2 and rec.profile["calls"]["storage"] == 1
        assert sum(rec.profile["seconds"].values()) <= rec.seconds + 5e-3
    if mode == "fast" and device == "cuda":       # index_add_ adds with atomics on CUDA: its order is not fixed
        np.testing.assert_allclose(maps[True], maps[False], rtol=0, atol=1e-6)
    else:
        assert np.array_equal(maps[False], maps[True])


def test_summarise_profile_uses_the_steady_maps():
    stages = S.StageTimer.STAGES

    def rec(seconds, forward_s, forward_calls):
        prof = dict(seconds=dict.fromkeys(stages, 0.1) | dict(forward=forward_s),
                    calls=dict.fromkeys(stages, 1) | dict(forward=forward_calls))
        return S.MapRecord("r", "reverse", "reading", 0, seconds, 100, (10, 10), profile=prof)

    p = B.summarise_profile([rec(9.0, 5.0, 10), rec(1.0, 0.4, 20), rec(1.2, 0.6, 20)])   # the first map: autotuning
    assert p["maps"] == 2 and p["seconds_per_map"] == pytest.approx(1.1)
    assert p["seconds"]["forward"] == pytest.approx(0.5) and p["calls_per_map"]["forward"] == 20
    assert p["ms_per_forward"] == pytest.approx(25.0)
    assert p["share"]["forward"] == pytest.approx(0.5 / 1.1)
    assert p["other_s"] == pytest.approx(1.1 - 0.5 - 5 * 0.1)
    assert p["tiles_per_s"] == pytest.approx(200 / 2.2)
    assert B.summarise_profile([S.MapRecord("r", "reverse", "reading", 0, 1.0, 1, (1, 1))]) is None


def test_bench_profile_table(tmp_path):
    entry = B.bench_config(surface(), [spec_for(Counting())], {"fake": None}, dict(mode="exact", label="exact"),
                           tmp_path / "b", depths=[0, 1, 2], device="cpu", profile=True)
    p = entry["readers"]["fake"]["profile"]
    assert p["maps"] == 2 and p["calls_per_map"]["forward"] > 0
    md = B.markdown(dict(surface="syn", shape=[11, 70, 101], depths=[0, 1, 2], configs=[entry]))
    assert "stage split per map" in md and "ms/forward" in md and md.count("| exact | fake |") == 2
    plain = B.bench_config(surface(), [spec_for(Counting())], {"fake": None}, dict(mode="exact"), tmp_path / "c",
                           depths=[0], device="cpu")
    assert "profile" not in plain["readers"]["fake"]
    assert "stage split" not in B.markdown(dict(surface="syn", shape=[11, 70, 101], depths=[0], configs=[plain]))

"""GPU memory: the allocator cap, per-map and per-reader peaks, and out-of-memory handling in ``inksweep bench``.

Everything but the last test runs on the CPU (the CUDA calls are faked or absent there).
"""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from inksweep import bench as B
from inksweep import io as iox
from inksweep import sweep as S
from inksweep.readers.spec import ReaderSpec
from inksweep.runner import summarise_records

W_LAYERS, TILE, STRIDE = 5, 32, 16


class DepthMean(nn.Module):
    def forward(self, x):                                  # x [B, 1, Z, p, p]
        return (x.mean(2) - 0.4) * 6.0


def fake_spec(name="fake"):
    def pre(stack):
        return (stack.to(torch.float32) / 255.0)[:, None]

    return ReaderSpec(name=name, window_layers=W_LAYERS, voxel_size_um=9.0, tile=TILE, stride=STRIDE,
                      load=lambda paths, device: DepthMean().to(device).eval(), preprocess=pre,
                      forward=lambda m, x: m(x), postprocess=lambda lg: lg.float().sigmoid(), batched=True,
                      reference_batch=3)


def surface():
    vol = np.random.default_rng(0).integers(1, 255, size=(11, 70, 101), dtype=np.uint8)
    vol[:, :, :12] = 0
    return iox.SurfaceVolume(path=None, name="syn", layers=vol, kind="mem", voxel_size_um=9.0)


def test_cap_gpu_memory(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: SimpleNamespace(total_memory=80 * 2**30))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f, dev=None: calls.append((f, dev)))
    dev = torch.device("cuda", 0)
    assert S.cap_gpu_memory(dev, 20) == pytest.approx(0.25)
    assert calls == [(pytest.approx(0.25), dev)]
    assert S.cap_gpu_memory(dev, 200) == 1.0                      # never above the card
    assert S.cap_gpu_memory(dev, None) is None                     # a later uncapped sweep lifts the cap ...
    assert calls[-1] == (1.0, dev) and len(calls) == 3
    assert S.cap_gpu_memory(dev, None) is None and len(calls) == 3  # ... once
    assert S.cap_gpu_memory(torch.device("cpu"), 20) is None and len(calls) == 3
    with pytest.raises(ValueError):
        S.cap_gpu_memory(dev, 0)
    with pytest.raises(ValueError):
        S.SweepConfig(depths=[0], max_gpu_mem_gb=-1).validate()


def test_map_records_carry_peaks(tmp_path):
    cfg = S.SweepConfig(depths=[0], orientations=("reverse",), mode="exact")
    sweep = S.Sweep(surface(), tmp_path / "m", cfg, device="cpu")
    spec = fake_spec()
    rec = sweep.run_map(spec, spec.load(None, sweep.device), 0, "reverse")
    assert rec.tiles > 0
    assert rec.gpu_peak_allocated_mb is None and rec.gpu_peak_reserved_mb is None      # not measured off CUDA


def test_per_reader_summary_takes_the_largest_peak():
    recs = [dict(reader="a", tiles=10, seconds=1.0, gpu_peak_allocated_mb=100.0, gpu_peak_reserved_mb=200.0),
            dict(reader="a", tiles=5, seconds=0.5, gpu_peak_allocated_mb=150.0, gpu_peak_reserved_mb=180.0),
            dict(reader="b", tiles=3, seconds=2.0, gpu_peak_allocated_mb=None, gpu_peak_reserved_mb=None),
            dict(reader="b", tiles=0, seconds=0.0, skipped="window outside the stack"),
            dict(reader="c", tiles=1, seconds=0.25)]                                    # a record without the keys
    s = summarise_records(recs)
    assert s["a"] == dict(maps=2, tiles=15, seconds=1.5, gpu_peak_allocated_mb=150.0, gpu_peak_reserved_mb=200.0)
    assert s["b"]["maps"] == 1 and s["b"]["gpu_peak_reserved_mb"] is None
    assert s["c"]["maps"] == 1 and s["c"]["gpu_peak_allocated_mb"] is None


def test_bench_records_out_of_memory_and_goes_on(tmp_path):
    def boom(paths, device):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB (test)")

    good, bad = fake_spec("good"), fake_spec("bad").with_options(load=boom)
    entry = B.bench_config(surface(), [bad, good], {"good": None, "bad": None}, dict(mode="exact", label="exact"),
                           tmp_path / "b", depths=[0, 1], device="cpu")
    assert entry["readers"]["bad"]["oom"] is True and "out of memory" in entry["readers"]["bad"]["error"]
    g = entry["readers"]["good"]
    assert g["maps"] == 2 and g["tiles_per_s"] > 0 and g["gpu_peak_reserved_mb"] is None
    md = B.markdown(dict(surface="syn", shape=[11, 70, 101], depths=[0, 1], configs=[entry], max_gpu_mem_gb=22))
    assert "capped at 22 GiB" in md and "| exact | bad | out of memory |" in md and "| exact | good | " in md


def test_bench_does_not_hide_other_errors(tmp_path):
    def broken(paths, device):
        raise ValueError("not a memory problem")

    with pytest.raises(ValueError):
        B.bench_config(surface(), [fake_spec("x").with_options(load=broken)], {"x": None},
                       dict(mode="exact"), tmp_path / "e", depths=[0], device="cpu")


def test_out_of_memory_detection():
    assert B.is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 1 GiB"))
    assert not B.is_out_of_memory(RuntimeError("shape mismatch"))
    oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if oom is not None:
        assert B.is_out_of_memory(oom("allocator says no"))


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_peaks_are_measured_on_cuda(tmp_path):
    cfg = S.SweepConfig(depths=[0], orientations=("reverse",), mode="exact")
    sweep = S.Sweep(surface(), tmp_path / "g", cfg, device="cuda")
    spec = fake_spec()
    rec = sweep.run_map(spec, spec.load(None, sweep.device), 0, "reverse")
    assert rec.gpu_peak_allocated_mb > 0 and rec.gpu_peak_reserved_mb >= rec.gpu_peak_allocated_mb

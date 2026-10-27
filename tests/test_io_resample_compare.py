"""Inputs, resampling (Hecate prep), map export names and the comparison arithmetic."""
import json

import numpy as np
import pytest
import torch

from inksweep import compare as C
from inksweep import io as iox
from inksweep import resample as R
from inksweep.tiling import window_positions


# ---------------------------------------------------------------- inputs
def test_tiff_stack_and_zarr_and_contract(tmp_path):
    import tifffile

    rng = np.random.default_rng(0)
    vol = rng.integers(0, 255, size=(6, 20, 30), dtype=np.uint8)
    d = tmp_path / "layers"
    d.mkdir()
    for i in range(6):
        tifffile.imwrite(d / f"{i:02d}.tif", vol[i])
    s = iox.open_surface(d, voxel_size_um=8.64)
    assert s.kind == "tiffdir" and np.array_equal(s.read(), vol)
    z = iox.write_zarr_layers(tmp_path / "r.zarr", vol)
    s2 = iox.open_surface(z, voxel_size_um=8.64)
    assert s2.kind == "zarr" and np.array_equal(s2.read(), vol) and s2.name == "r"
    c = tmp_path / "sv"
    c.mkdir()
    iox.write_zarr_layers(c / "layers.zarr", vol)
    (c / "surface.json").write_text(json.dumps({"voxel_size_um": 9.362}))
    s3 = iox.open_surface(c)
    assert s3.kind == "contract" and s3.voxel_size_um == 9.362
    assert np.array_equal(s3.valid_mask(), (vol > 0).all(0))


def test_uint16_layers_reduced(tmp_path):
    import tifffile

    d = tmp_path / "l16"
    d.mkdir()
    a = (np.arange(12, dtype=np.uint16).reshape(3, 4) * 5000)
    for i in range(3):
        tifffile.imwrite(d / f"{i}.tif", a)
    s = iox.open_surface(d, voxel_size_um=1.0)
    assert np.array_equal(s.read()[0], (a >> 8).astype(np.uint8))


def test_node_source_phase_mask(tmp_path):
    src = np.array([[0, 1], [4, 2]])
    np.savez(tmp_path / "ns.npz", source=src)
    m = iox.load_phase_mask(f"npz:{tmp_path / 'ns.npz'}", (40, 40))
    assert m[:20, 20:].all() and m[20:, :20].all() and not m[:20, :20].any() and not m[20:, 20:].any()


# ---------------------------------------------------------------- resampling
def hecate_prep(vol, z_center, nz, native_um, target_um):
    """Transcription of hecate_prep.py main() for a whole (D, H, W) render."""
    import cv2

    D, H, W = vol.shape
    step = target_um / native_um
    pos = z_center + (np.arange(nz) - (nz - 1) / 2.) * step
    lo = np.floor(pos).astype(int)
    frac = (pos - lo).astype(np.float32)
    hi = np.minimum(lo + 1, D - 1)
    zs = (vol[lo].astype(np.float32) * (1 - frac)[:, None, None] + vol[hi].astype(np.float32) * frac[:, None, None])
    s = native_um / target_um
    nh, nw = max(1, int(round(H * s))), max(1, int(round(W * s)))
    interp = cv2.INTER_AREA if s < 0.95 else cv2.INTER_LINEAR
    out = np.empty((nz, nh, nw), np.uint8)
    for i in range(nz):
        out[i] = np.clip(cv2.resize(zs[i], (nw, nh), interpolation=interp), 0, 255).astype(np.uint8)
    return out


@pytest.mark.parametrize("native", [8.64, 9.362])
@pytest.mark.parametrize("d", [-3, 0, 2])
def test_resample_cpu_is_hecate_prep(native, d):
    rng = np.random.default_rng(1)
    vol = rng.integers(0, 255, size=(66, 120, 200), dtype=np.uint8)
    pos = window_positions(66, d, 16, 9.6 / native)
    ours = R.resample_window_cpu(vol, pos, native, 9.6)
    ref = hecate_prep(vol, (66 - 1) / 2 + d, 16, native, 9.6)
    assert np.array_equal(ours, ref)


def test_resample_padded_canvas_equals_physical_padding():
    rng = np.random.default_rng(2)
    vol = rng.integers(0, 255, size=(66, 100, 150), dtype=np.uint8)
    pos = window_positions(66, 1, 16, 9.6 / 8.64)
    ours = R.resample_window_cpu(vol, pos, 8.64, 9.6, pad=(40, 200))
    padded = np.zeros((66, 140, 350), np.uint8)
    padded[:, 40:, 200:] = vol
    ref = hecate_prep(padded, 32.5 + 1, 16, 8.64, 9.6)
    assert np.array_equal(ours, ref)
    assert R.crop_offsets(ours.shape[1:], R.resampled_shape(100, 150, 8.64, 9.6), (40, 200), 0.9) == (36, 180)


@pytest.mark.parametrize("native", [8.64, 9.362])
def test_resample_gpu_close_to_cpu(native):
    rng = np.random.default_rng(3)
    vol = rng.integers(0, 255, size=(66, 90, 130), dtype=np.uint8)
    pos = window_positions(66, 0, 16, 9.6 / native)
    cpu = R.resample_window_cpu(vol, pos, native, 9.6).astype(np.int32)
    gpu = R.resample_window_gpu(torch.from_numpy(vol), pos, native, 9.6).numpy().astype(np.int32)
    assert cpu.shape == gpu.shape
    assert np.abs(cpu - gpu).max() <= 1
    assert (cpu == gpu).mean() > 0.99


def test_to_render_grid_is_cv2_linear():
    import cv2

    a = np.random.default_rng(4).random((90, 120), dtype=np.float32)
    ours = R.to_render_grid(a, (100, 133))
    assert np.array_equal(ours, cv2.resize(a, (133, 100), interpolation=cv2.INTER_LINEAR))


# ---------------------------------------------------------------- compare / export
def test_reference_names():
    n = "concat_w047-w048_R5B2_z9500-11000"
    assert C.reference_name(n, "dnative", "reverse", 1) == f"{n}__dnative__L66s26_reverse.tif"
    assert C.reference_name(n, "ink9_s42", "reverse", 1) == f"{n}__hybrid_3d2d-seed42__L66s26_reverse.tif"
    assert C.reference_name(n, "ink9_s43", "forward", -15) == f"{n}__hybrid_3d2d-seed43__L66s10.tif"
    assert C.reference_name(n, "hecate", "reverse", 1) == f"{n}__hecate__L66zc33.50_reverse.png"
    assert C.reference_name(n, "hecate", "forward", -15) == f"{n}__hecate__L66zc17.50.png"


def test_quantise_rules():
    m = np.array([0.0, 0.5, 0.999, 1.0, 0.2], np.float32)
    assert C.quantise(m, "u8_floor").tolist() == [0, 127, 254, 255, 51]
    assert C.quantise(m, "u8_round").tolist() == [0, 128, 255, 255, 51]


def test_compare_map_criteria():
    a = np.arange(100, dtype=np.uint8).reshape(10, 10)
    b = a.copy()
    b[0, 0] += 1
    ok = C.compare_map(a, b, 0.9999, 1 / 255)
    assert ok["passed"] and ok["max_abs_diff"] == pytest.approx(1 / 255)
    b[1, 1] += 2
    bad = C.compare_map(a, b, 0.9999, 1 / 255)
    assert not bad["passed"]


def test_export_then_compare_roundtrip(tmp_path):
    out = tmp_path / "run"
    rng = np.random.default_rng(5)
    for side in ("reading", "blind"):
        for d in (0, 1):
            iox.write_tif(out / "ink" / "dnative" / "raw" / side / f"d{d:+03d}.tif",
                          rng.random((20, 30), dtype=np.float32))
            iox.write_tif(out / "ink" / "hecate" / "raw" / side / f"d{d:+03d}.tif",
                          rng.random((18, 27), dtype=np.float32))
    n = C.export_reference(out, tmp_path / "ref", "seg", ["dnative", "hecate"], [0, 1])
    assert n == 8
    report = C.compare(out, tmp_path / "ref", "seg", ["dnative", "hecate"], [0, 1])
    assert report["summary"]["all_passed"] and report["summary"]["worst_max_abs_diff"] == 0.0

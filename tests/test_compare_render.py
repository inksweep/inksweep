"""``inksweep compare-render`` (layer-by-layer comparison of two renders) and ``inksweep export-ct`` (a local OME-Zarr
copy of the CT chunks a render cached), on synthetic volumes."""
import json
import os

import numpy as np
import pytest

from inksweep import compare as C
from inksweep import io as iox
from inksweep import render as R


def render(n=6, h=40, w=50, seed=0):
    vol = np.random.default_rng(seed).integers(1, 256, size=(n, h, w), dtype=np.uint8)
    vol[:, :, :6] = 0                                       # off the mesh: no data in any layer
    return vol


def write(tmp_path, name, vol):
    return iox.write_zarr_layers(tmp_path / f"{name}.zarr", vol)


# ---------------------------------------------------------------- compare-render
def test_identical_renders(tmp_path):
    vol = render()
    rep = C.compare_renders(write(tmp_path, "a", vol), write(tmp_path, "b", vol), names=("vc3d", "inksweep"))
    s = rep["summary"]
    assert s["min_r"] == pytest.approx(1.0, abs=1e-12) and s["max_abs_diff"] == 0 and s["frac_equal"] == 1.0
    assert s["voxels"] == 6 * 40 * 44 and s["depths"] == [-3, 2]
    assert rep["valid"]["any"] == dict(both=40 * 44, only_a=0, only_b=0, neither=40 * 6, agreement=1.0)
    assert all(L["valid_agreement"] == 1.0 and L["valid_iou"] == 1.0 for L in rep["layers"])
    assert s["d+1"]["layer_a"] == 4 and s["d+1"]["layer_b"] == 4
    text = C.render_table(rep)
    assert "| +1 * | 4 | 4 | 1.000000 | 0 |" in text and "100.00 % of 10560 voxels equal" in text


def test_differences_and_valid_masks(tmp_path):
    vol = render(seed=1)
    other = vol.copy()
    bump = (np.random.default_rng(2).random(vol.shape) < 0.1) & (vol > 0) & (vol < 255)
    other[bump] += 1                                       # one grey level on 10 % of the voxels
    other[:, :5, :] = 0                                    # five rows the other renderer leaves empty
    rep = C.compare_renders(write(tmp_path, "a", vol), write(tmp_path, "b", other))
    s = rep["summary"]
    assert s["max_abs_diff"] == 1 and s["frac_within_1"] == 1.0 and 0.85 < s["frac_equal"] < 0.95
    assert 0.999 < s["min_r"] < 1.0
    v = rep["valid"]["any"]
    assert v["only_a"] == 5 * 44 and v["only_b"] == 0 and v["both"] == 35 * 44
    assert s["voxels"] == 6 * 35 * 44                      # statistics only where both renders have data
    assert s["min_valid_agreement"] == pytest.approx(1 - 5 * 44 / (40 * 50))


def test_layers_match_by_depth_and_crop(tmp_path):
    vol = render(n=10, seed=3)
    rep = C.compare_renders(write(tmp_path, "a", vol), write(tmp_path, "b", vol[2:8, :30, :45]))
    assert [L["depth"] for L in rep["layers"]] == [-3, -2, -1, 0, 1, 2]      # 10 // 2 = 5, 6 // 2 = 3
    assert all(L["layer_a"] == L["layer_b"] + 2 for L in rep["layers"])
    assert rep["compared_hw"] == [30, 45] and rep["summary"]["max_abs_diff"] == 0
    odd = C.compare_renders(write(tmp_path, "c", vol), write(tmp_path, "d", vol[2:9]))
    assert odd["note"] and "half a voxel" in odd["note"]


def test_shift_search_finds_a_one_pixel_offset(tmp_path):
    vol = render(seed=4)
    shifted = np.zeros_like(vol)
    shifted[:, :, 1:] = vol[:, :, :-1]                     # B's pixel (y, x + 1) holds A's pixel (y, x)
    rep = C.compare_renders(write(tmp_path, "a", vol), write(tmp_path, "b", shifted), max_shift=2)
    best = rep["shift_search"]["best"]
    assert (best["dy"], best["dx"]) == (0, 1) and best["pearson_r"] == pytest.approx(1.0)
    assert rep["shift_search"]["at_zero"]["pearson_r"] < 0.5
    assert "(dy, dx) = (+0, +1)" in C.render_table(rep)


def test_pearson_from_sums_matches_numpy():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, 5000).astype(np.int64)
    b = np.clip(a + rng.integers(-3, 4, 5000), 0, 255)
    r = C.pearson_from_sums(a.size, int(a.sum()), int(b.sum()), int((a * a).sum()), int((b * b).sum()),
                            int((a * b).sum()), int((a == b).sum()))
    assert r == pytest.approx(np.corrcoef(a, b)[0, 1], abs=1e-12)
    assert C.pearson_from_sums(3, 6, 6, 12, 12, 12, 3) == 1.0                # constant and identical
    assert np.isnan(C.pearson_from_sums(0, 0, 0, 0, 0, 0, 0))


def test_report_files(tmp_path):
    vol = render()
    rep = C.compare_renders(write(tmp_path, "a", vol), write(tmp_path, "b", vol))
    C.save_render_report(rep, tmp_path / "out" / "cmp.json")
    assert json.loads((tmp_path / "out" / "cmp.json").read_text())["summary"]["max_abs_diff"] == 0
    assert (tmp_path / "out" / "cmp.md").read_text().startswith("A = a:")


# ---------------------------------------------------------------- export-ct
def raw_volume(tmp_path, vol, chunks=(4, 8, 8)):
    """A level-0 OME-Zarr v2 volume without compression, written by hand (the open-data volumes' layout)."""
    root = tmp_path / "bucket.zarr"
    (root / "0").mkdir(parents=True)
    (root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (root / ".zattrs").write_text(json.dumps({"multiscales": [{"version": "0.4", "datasets": [{"path": "0"},
                                                                                             {"path": "1"}]}]}))
    meta = dict(zarr_format=2, shape=list(vol.shape), chunks=list(chunks), dtype="|u1", compressor=None, fill_value=0,
                order="C", filters=None, dimension_separator="/")
    (root / "0" / ".zarray").write_text(json.dumps(meta))
    grid = [-(-s // c) for s, c in zip(vol.shape, chunks)]
    for a in range(grid[0]):
        for b in range(grid[1]):
            for c in range(grid[2]):
                blk = np.zeros(chunks, np.uint8)
                src = vol[a * chunks[0]:(a + 1) * chunks[0], b * chunks[1]:(b + 1) * chunks[1],
                          c * chunks[2]:(c + 1) * chunks[2]]
                blk[:src.shape[0], :src.shape[1], :src.shape[2]] = src
                p = root / "0" / str(a) / str(b) / str(c)
                p.parent.mkdir(parents=True, exist_ok=True)
                blk.tofile(p)
    return root


def expected(vol, keys, chunks):
    want = np.zeros_like(vol)
    for c in keys:
        sl = tuple(slice(i * n, (i + 1) * n) for i, n in zip(c, chunks))
        want[sl] = vol[sl]
    return want


def test_export_uncompressed_links_the_cached_chunks(tmp_path):
    vol = np.random.default_rng(6).integers(1, 256, size=(8, 16, 24), dtype=np.uint8)
    src = raw_volume(tmp_path, vol)
    cache = tmp_path / "cache"
    keys = [(0, 0, 0), (1, 1, 2), (0, 1, 1)]
    R.ChunkStore(str(src), cache).ensure(keys)
    (cache / "level5").mkdir()                              # what the coarse-level check leaves behind: skipped
    out = R.export_level(str(src), cache, tmp_path / "local.zarr")
    assert out["chunks"] == 3 and out["linked"] == 3 and out["encoded"] == 0
    local = tmp_path / "local.zarr"
    got = iox.open_surface(local).read()
    assert np.array_equal(got, expected(vol, keys, (4, 8, 8)))
    assert os.stat(cache / "1" / "1" / "2").st_ino == os.stat(local / "0" / "1" / "1" / "2").st_ino
    attrs = json.loads((local / ".zattrs").read_text())
    assert [d["path"] for d in attrs["multiscales"][0]["datasets"]] == ["0"]
    assert json.loads((local / "0" / ".zarray").read_text()) == json.loads((src / "0" / ".zarray").read_text())
    kept = R.export_level(str(src), cache, tmp_path / "kept.zarr", copy=True, keep_attrs=True)
    kept_attrs = json.loads((tmp_path / "kept.zarr" / ".zattrs").read_text())
    assert kept["copied"] == 3 and len(kept_attrs["multiscales"][0]["datasets"]) == 2


def test_export_compressed_reencodes(tmp_path):
    vol = np.random.default_rng(7).integers(1, 256, size=(6, 20, 30), dtype=np.uint8)
    src = iox.write_zarr_layers(tmp_path / "c.zarr", vol)     # Blosc, one chunk for so small a volume
    cache = tmp_path / "cache"
    store = R.ChunkStore(str(src), cache)
    assert store.codec is not None
    store.ensure([(0, 0, 0)])
    out = R.export_level(str(src), cache, tmp_path / "local.zarr")
    assert out["encoded"] == 1
    assert np.array_equal(iox.open_surface(tmp_path / "local.zarr").read(), vol)


def test_export_refuses_another_volumes_cache(tmp_path):
    vol = np.random.default_rng(8).integers(1, 256, size=(8, 16, 24), dtype=np.uint8)
    src = raw_volume(tmp_path, vol)
    bad = tmp_path / "cache" / "0" / "0" / "0"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"\0" * 10)
    with pytest.raises(ValueError):
        R.export_level(str(src), tmp_path / "cache", tmp_path / "local.zarr")

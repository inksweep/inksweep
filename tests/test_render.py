"""The clean-room renderer on a synthetic plane through a synthetic volume (layer order and normal sign)."""
import json

import numpy as np
import pytest

from inksweep import io as iox
from inksweep import render as R


def make_mesh(tmp_path, z0=30.0, rows=8, cols=10, step=20.0):
    import tifffile

    d = tmp_path / "plane.tifxyz"
    d.mkdir()
    r, c = np.mgrid[0:rows, 0:cols].astype(np.float32)
    x = 40 + step * c
    y = 40 + step * r
    z = np.full_like(x, z0)
    for name, a in (("x", x), ("y", y), ("z", z)):
        tifffile.imwrite(d / f"{name}.tif", a.astype(np.float32))
    (d / "meta.json").write_text(json.dumps({"scale": [1 / step, 1 / step], "format": "tifxyz"}))
    return d


def make_volume(tmp_path, shape=(64, 256, 300)):
    z = np.arange(shape[0], dtype=np.uint8)[:, None, None]
    vol = np.broadcast_to(z, shape).copy()
    return iox.write_zarr_layers(tmp_path / "vol.zarr", vol)


def test_normals_point_along_z(tmp_path):
    p, valid, meta = R.load_tifxyz(make_mesh(tmp_path))
    n = R.vertex_normals(p)
    inner = n[1:-1, 1:-1]
    assert np.allclose(inner[..., :2], 0) and (inner[..., 2] > 0).all()
    assert np.isnan(n[0]).all()


def test_layer_order_and_flip(tmp_path):
    mesh = make_mesh(tmp_path)
    vol = make_volume(tmp_path)
    p, _, meta = R.load_tifxyz(mesh)
    n = R.vertex_normals(p)
    gs = float(meta["scale"][0])
    H, W = R.render_shape(p, gs)
    pts, nrm = R.surface_at_pixels(p, n, np.arange(H), np.arange(W), gs)
    store_args = (str(vol), str(tmp_path / "cache"), 2)
    store = R.ChunkStore(*store_args)
    offs = (np.arange(7) - 3.0)
    store.ensure(R.plan_chunks(pts, nrm, offs, store))
    layers = R.render_layers(pts, nrm, 7, store_args, procs=1)
    ok = np.isfinite(pts).all(-1) & np.isfinite(nrm).all(-1)
    assert ok.sum() > 0
    for i in range(7):                       # layer i samples z = 30 + (i - 3): the ramp value itself
        assert (layers[i][ok] == 27 + i).all(), i
    flipped = R.render_layers(pts, nrm, 7, store_args, flip=True, procs=1)
    assert np.array_equal(flipped[:, ok], layers[::-1][:, ok])


def test_outward_fraction(tmp_path):
    pts = np.zeros((2, 2, 3), np.float32)
    pts[..., 0] = 10.0                       # points at x = 10 around an axis at x = 0
    nrm = np.zeros_like(pts)
    nrm[..., 0] = 1.0                        # normals along +x: outward
    umb = (np.array([0.0, 100.0]), np.array([0.0, 0.0]), np.array([0.0, 0.0]))
    f, n = R.outward_fraction(pts, nrm, umb, every=1)
    assert f == 1.0 and n == 4
    f2, _ = R.outward_fraction(pts, -nrm, umb, every=1)
    assert f2 == 0.0


def test_absent_chunks_are_not_cached_as_zero(tmp_path):
    vol = make_volume(tmp_path)
    store = R.ChunkStore(str(vol), tmp_path / "c2")
    store.ensure([(5, 5, 5)])                # outside the volume: absent
    assert store.key((5, 5, 5)) in store.absent
    assert not store.path((5, 5, 5)).exists()

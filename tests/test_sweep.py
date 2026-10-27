"""The sweep engine on synthetic data with a tiny deterministic reader, against a transcription of villa's tiling.

The reference functions below re-implement, independently of the engine, what villa ``infer`` does for one map:
schedule the sliding grid, skip all-zero raw patches, add ``probability * weight`` patch by patch in grid order into
float32 sums, divide where the weight exceeds 1e-6, clip. Tile phases are reproduced by physically padding the volume
(``pad_render.py``) and cropping (``merge_tile_phase.py``).
"""
import numpy as np
import pytest
import torch
from torch import nn

from inksweep import io as iox
from inksweep.readers.spec import ReaderSpec
from inksweep.sweep import Sweep, SweepConfig
from inksweep.tiling import hann_window, inside_mask_positions, sliding_positions, v8in_coverage, \
    v8in_gaussian_window, window_indices

W_LAYERS, TILE, STRIDE = 5, 32, 16


class DepthWeighted(nn.Module):
    """Logits = depth-weighted mean of the input (asymmetric in depth, so orientation matters)."""

    def __init__(self, depth):
        super().__init__()
        self.register_buffer("w", torch.arange(1, depth + 1, dtype=torch.float32))

    def forward(self, x):                              # x [B, 1, Z, p, p]
        w = self.w.view(1, 1, -1, 1, 1)
        return ((x * w).sum(2) / self.w.sum() - 0.4) * 6.0


def fake_spec(grid="sliding", blend="hann", stitch="threshold", zero_empty=False, min_weight=1e-6,
              calibration="identity", storage=None, batched=True):
    def pre(stack):
        if stack.ndim == 3:
            return (stack.to(torch.float32) / 255.0)[None]
        return (stack.to(torch.float32) / 255.0)[:, None]

    return ReaderSpec(name="fake", window_layers=W_LAYERS, voxel_size_um=9.0, tile=TILE, stride=STRIDE,
                      load=lambda paths, device: DepthWeighted(W_LAYERS).to(device).eval(),
                      preprocess=pre, forward=lambda m, x: m(x), postprocess=lambda lg: lg.float().sigmoid(),
                      batched=batched, blend=blend, grid=grid, stitch=stitch, min_weight=min_weight,
                      zero_empty_columns=zero_empty, calibration=calibration, reference_storage=storage,
                      reference_batch=3)


def synthetic_volume(seed=0, n=11, h=70, w=101):
    rng = np.random.default_rng(seed)
    vol = rng.integers(1, 255, size=(n, h, w), dtype=np.uint8)
    vol[:, :, :12] = 0                                 # an empty margin: skipped tiles
    vol[:, 50:, 80:] = 0
    return vol


def surface_of(vol):
    return iox.SurfaceVolume(path=None, name="syn", layers=vol, kind="mem", voxel_size_um=9.0)


def tile_probability(model, stack_u8):
    """One tile through the fake reader, batch of one (the engine batches; the model is batch-independent)."""
    x = (torch.from_numpy(stack_u8).float() / 255.0)[None, None]
    with torch.inference_mode():
        return model(x).float().sigmoid()[0, 0].numpy()


def villa_reference_map(vol, d, orientation, pad=(0, 0)):
    """villa-style map for one depth on the (physically padded) volume, cropped back to the render grid."""
    pr, pc = pad
    n, h, w = vol.shape
    padded = np.zeros((n, h + pr, w + pc), np.uint8)
    padded[:, pr:, pc:] = vol
    idx = window_indices(n, d, W_LAYERS, orientation)
    win = padded[idx]
    hh, ww = win.shape[1:]
    model = DepthWeighted(W_LAYERS).eval()
    wmap = hann_window(TILE).numpy()
    prob_sum = np.zeros((hh, ww), np.float32)
    weight_sum = np.zeros((hh, ww), np.float32)
    for y0 in sliding_positions(hh, TILE, STRIDE):
        for x0 in sliding_positions(ww, TILE, STRIDE):
            patch = np.zeros((W_LAYERS, TILE, TILE), np.uint8)
            src = win[:, y0:y0 + TILE, x0:x0 + TILE]
            patch[:, :src.shape[1], :src.shape[2]] = src
            if not patch.any():
                continue
            prob = tile_probability(model, patch)
            vh, vw = min(TILE, hh - y0), min(TILE, ww - x0)
            prob_sum[y0:y0 + vh, x0:x0 + vw] += prob[:vh, :vw] * wmap[:vh, :vw]
            weight_sum[y0:y0 + vh, x0:x0 + vw] += wmap[:vh, :vw]
    out = np.zeros_like(prob_sum)
    np.divide(prob_sum, weight_sum, out=out, where=weight_sum > 1e-6)
    np.clip(out, 0, 1, out=out)
    return out[pr:, pc:]


def run_one(tmp_path, vol, spec, cfg, d, orientation, phase_mask=None, sub="out"):
    sweep = Sweep(surface_of(vol), tmp_path / sub, cfg, device="cpu", phase_mask=phase_mask)
    module = spec.load(None, sweep.device)
    rec = sweep.run_map(spec, module, d, orientation)
    raw = iox.read_tif(tmp_path / sub / rec.files["raw"])
    contract = iox.read_tif(tmp_path / sub / rec.files["map"])
    return raw, contract, rec


@pytest.mark.parametrize("d", [-2, 0, 3])
@pytest.mark.parametrize("orientation", ["forward", "reverse"])
def test_exact_map_equals_villa_reference(tmp_path, d, orientation):
    vol = synthetic_volume()
    cfg = SweepConfig(depths=[d], orientations=(orientation,), reading=orientation, mode="exact")
    raw, contract, rec = run_one(tmp_path, vol, fake_spec(), cfg, d, orientation)
    ref = villa_reference_map(vol, d, orientation)
    assert raw.shape == vol.shape[1:]
    assert np.array_equal(raw, ref)
    assert np.array_equal(contract, ref)               # identity calibration, no storage emulation
    assert rec.tiles > 0


def test_orientation_changes_the_map(tmp_path):
    vol = synthetic_volume(1)
    cfg = SweepConfig(depths=[0], orientations=("forward", "reverse"), mode="exact")
    f, _, _ = run_one(tmp_path, vol, fake_spec(), cfg, 0, "forward", sub="f")
    r, _, _ = run_one(tmp_path, vol, fake_spec(), cfg, 0, "reverse", sub="r")
    assert not np.array_equal(f, r)
    assert np.array_equal(r, villa_reference_map(vol, 0, "reverse"))


def test_virtual_padding_equals_physical_padding(tmp_path):
    vol = synthetic_volume(2)
    pad = (8, 20)
    mask = np.zeros(vol.shape[1:], bool)
    mask[:, :50] = True                                 # left half from the padded phase
    cfg = SweepConfig(depths=[1], orientations=("reverse",), phases=((0, 0), pad), phase_merge="select",
                      mode="exact", need_masks=False)
    raw, _, _ = run_one(tmp_path, vol, fake_spec(), cfg, 1, "reverse", phase_mask=mask)
    plain = villa_reference_map(vol, 1, "reverse")
    padded = villa_reference_map(vol, 1, "reverse", pad)
    assert np.array_equal(raw, np.where(mask, padded, plain))


def test_need_masks_change_no_taken_pixel(tmp_path):
    vol = synthetic_volume(3)
    mask = np.zeros(vol.shape[1:], bool)
    mask[30:38, 40:50] = True                          # small: most padded-phase tiles are not needed
    base = dict(depths=[0], orientations=("reverse",), phases=((0, 0), (8, 20)), phase_merge="select", mode="exact")
    full, _, r_full = run_one(tmp_path, vol, fake_spec(), SweepConfig(**base, need_masks=False), 0, "reverse",
                              phase_mask=mask, sub="full")
    need, _, r_need = run_one(tmp_path, vol, fake_spec(), SweepConfig(**base, need_masks=True), 0, "reverse",
                              phase_mask=mask, sub="need")
    assert np.array_equal(full, need)
    assert r_need.tiles < r_full.tiles


def test_fast_mode_close_to_exact(tmp_path):
    vol = synthetic_volume(4)
    e, _, _ = run_one(tmp_path, vol, fake_spec(), SweepConfig(depths=[0], orientations=("reverse",), mode="exact"),
                      0, "reverse", sub="e")
    f, _, _ = run_one(tmp_path, vol, fake_spec(), SweepConfig(depths=[0], orientations=("reverse",), mode="fast",
                                                              precision="fp32", batch=7), 0, "reverse", sub="f")
    assert np.abs(e - f).max() < 1e-5


def test_unbatched_plugin_contract(tmp_path):
    vol = synthetic_volume(5)
    cfg = SweepConfig(depths=[0], orientations=("reverse",), mode="exact")
    a, _, _ = run_one(tmp_path, vol, fake_spec(batched=False), cfg, 0, "reverse", sub="u")
    assert np.array_equal(a, villa_reference_map(vol, 0, "reverse"))


def test_zero_empty_columns_and_threshold(tmp_path):
    vol = synthetic_volume(6)
    vol[:, 20:30, 40:60] = 0                            # a hole inside covered tiles
    cfg = SweepConfig(depths=[0], orientations=("reverse",), mode="exact")
    raw, _, _ = run_one(tmp_path, vol, fake_spec(zero_empty=True, min_weight=0.0), cfg, 0, "reverse")
    win = vol[window_indices(vol.shape[0], 0, W_LAYERS)]
    support = win.any(0)
    assert (raw[~support] == 0).all() and (raw[support] > 0).all()


def test_v8in_grid_and_count_stitching(tmp_path):
    vol = synthetic_volume(7)
    cfg = SweepConfig(depths=[0], orientations=("reverse",), mode="exact")
    spec_fix = fake_spec(grid="inside_mask", blend="v8in_gaussian", min_weight=0.0)
    spec_bug = fake_spec(grid="inside_mask", blend="v8in_gaussian", stitch="count", min_weight=0.0)
    fixed, _, _ = run_one(tmp_path, vol, spec_fix, cfg, 0, "reverse", sub="fix")
    bug, _, _ = run_one(tmp_path, vol, spec_bug, cfg, 0, "reverse", sub="bug")
    # transcription of ink8um.predict_stack (count += 1, the released code) for the same tiles
    win = vol[window_indices(vol.shape[0], 0, W_LAYERS, "reverse")]
    pts = inside_mask_positions(v8in_coverage(win.max(0)), TILE, STRIDE)
    blend = v8in_gaussian_window(TILE).numpy()
    model = DepthWeighted(W_LAYERS).eval()
    num = np.zeros(win.shape[1:], np.float32)
    count = np.zeros_like(num)
    wsum = np.zeros_like(num)
    for y, x in pts:
        p = tile_probability(model, win[:, y:y + TILE, x:x + TILE])
        num[y:y + TILE, x:x + TILE] += p * blend
        count[y:y + TILE, x:x + TILE] += 1.0
        wsum[y:y + TILE, x:x + TILE] += blend
    ref_bug = np.clip(np.divide(num, count, out=np.zeros_like(num), where=count != 0), 0, 1)
    ref_fix = np.clip(np.divide(num, wsum, out=np.zeros_like(num), where=wsum != 0), 0, 1)
    assert np.array_equal(bug, ref_bug)
    assert np.array_equal(fixed, ref_fix)


def test_label_smoothing_calibration_and_storage_emulation(tmp_path):
    vol = synthetic_volume(8)
    cfg = SweepConfig(depths=[0], orientations=("reverse",), mode="exact", reference_storage=True)
    raw, contract, _ = run_one(tmp_path, vol, fake_spec(calibration="label_smoothing_0.5", storage="u8_floor"), cfg,
                               0, "reverse")
    v = (np.clip(raw, 0, 1) * np.float32(255)).astype(np.uint8)        # villa: trunc(255 p)
    ref = np.clip((v.astype(np.float32) - 64.0) / 128.0, 0, 1)         # ensemble_maps.py read_map
    assert np.array_equal(contract, ref)
    cfg2 = SweepConfig(depths=[0], orientations=("reverse",), mode="exact", reference_storage=False)
    raw2, contract2, _ = run_one(tmp_path, vol, fake_spec(calibration="label_smoothing_0.5", storage="u8_floor"),
                                 cfg2, 0, "reverse", sub="noemu")
    assert np.allclose(contract2, np.clip((raw2 - 0.25) / 0.5, 0, 1), atol=1e-7)


def test_resume_markers_and_digest(tmp_path):
    vol = synthetic_volume(9)
    cfg = SweepConfig(depths=[0, 1], orientations=("reverse",), mode="exact")
    sweep = Sweep(surface_of(vol), tmp_path / "res", cfg, device="cpu")
    spec = fake_spec()
    recs = sweep.run_reader(spec, None)
    assert len([r for r in recs if not r.skipped]) == 2
    assert sweep.run_reader(spec, None) == []           # everything done
    other = SweepConfig(depths=[0, 1], orientations=("reverse",), mode="fast")
    with pytest.raises(RuntimeError):
        Sweep(surface_of(vol), tmp_path / "res", other, device="cpu").run_reader(spec, None)


class CountingDepthWeighted(DepthWeighted):
    """Records the batch size of every forward pass."""

    def __init__(self, depth):
        super().__init__(depth)
        self.batches = []

    def forward(self, x):
        self.batches.append(int(x.shape[0]))
        return super().forward(x)


def reference_batch_sizes(vol, d, orientation, batch):
    """villa: schedule blocks with data in any layer (level-0 occupancy), batch them, drop window-empty tiles."""
    n, h, w = vol.shape
    occ = (vol != 0).any(0)
    win = vol[window_indices(n, d, W_LAYERS, orientation)]
    blocks = [(y, x) for y in sliding_positions(h, TILE, STRIDE) for x in sliding_positions(w, TILE, STRIDE)
              if occ[y:y + TILE, x:x + TILE].any()]
    sizes = []
    for i in range(0, len(blocks), batch):
        k = sum(1 for (y, x) in blocks[i:i + batch] if win[:, y:y + TILE, x:x + TILE].any())
        if k:
            sizes.append(k)
    return sizes


def test_exact_mode_keeps_reference_batch_composition(tmp_path):
    vol = synthetic_volume(10)
    vol[3:, 34:70, 60:101] = 0                          # data only in layers 0-2: scheduled, but the d=0 window
                                                        # (layers 3-7) of tiles (38, 64) and (38, 69) is empty
    cfg = SweepConfig(depths=[0], orientations=("reverse",), mode="exact")
    spec = fake_spec()
    model = CountingDepthWeighted(W_LAYERS)
    sweep = Sweep(surface_of(vol), tmp_path / "bc", cfg, device="cpu")
    sweep.run_map(spec, model, 0, "reverse")
    assert model.batches == reference_batch_sizes(vol, 0, "reverse", spec.reference_batch)
    fast = SweepConfig(depths=[0], orientations=("reverse",), mode="fast", precision="fp32", batch=5)
    model2 = CountingDepthWeighted(W_LAYERS)
    Sweep(surface_of(vol), tmp_path / "bf", fast, device="cpu").run_map(spec, model2, 0, "reverse")
    assert sum(model2.batches) == sum(model.batches) and max(model2.batches) <= 5

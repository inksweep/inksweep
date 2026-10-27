"""The GPU robust-MAD normalisation must equal villa's CPU function bit for bit."""
import numpy as np
import pytest
import torch

from inksweep.readers.norm import RobustMadGPU, normalize_robust_reference


def villa_normalize_robust(image):
    """Literal transcription of villa normalize_robust(image) (mask=None, clip_values=True, float32)."""
    arr = np.asarray(image).astype(np.float32, copy=True)
    valid = arr.reshape(-1)
    if valid.size == 0:
        return arr
    std_preclip = float(np.std(valid))
    min_preclip = float(np.min(valid))
    max_preclip = float(np.max(valid))
    lower_val = float(np.percentile(valid, 1.0))
    upper_val = float(np.percentile(valid, 99.0))
    np.clip(arr, lower_val, upper_val, out=arr)
    valid = arr.reshape(-1)
    median = float(np.median(valid))
    mad = float(np.median(np.abs(valid - median)))
    scaled_mad = 1.4826 * mad
    if not np.isfinite(scaled_mad) or scaled_mad < 1e-6:
        percentile_span = upper_val - lower_val if lower_val is not None else max_preclip - min_preclip
        fallback_values = []
        if np.isfinite(std_preclip):
            fallback_values.append(abs(std_preclip))
        if percentile_span is not None and np.isfinite(percentile_span):
            fallback_values.append(abs(percentile_span) / 2.0)
        scaled_mad = next((c for c in fallback_values if c >= 1e-6), None)
        if scaled_mad is None or not np.isfinite(scaled_mad):
            scaled_mad = 1.0
    arr -= median
    arr /= scaled_mad
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def patches(rng, shape, kind):
    if kind == "uniform":
        return rng.integers(0, 256, size=shape, dtype=np.uint8)
    if kind == "ct":                                  # papyrus-like: a bright band over a dark background
        a = rng.normal(90, 25, size=shape)
        a[:, shape[1] // 3: shape[1] // 2] += 60
        return np.clip(a, 0, 255).astype(np.uint8)
    if kind == "mostly_zero":                         # > 50 % zero padding: numpy's MAD fallback branch
        a = rng.integers(40, 200, size=shape, dtype=np.uint8)
        a[:, :, : shape[2] * 2 // 3] = 0
        return a
    if kind == "constant":
        return np.full(shape, 77, np.uint8)
    if kind == "two_values":
        a = np.full(shape, 10, np.uint8)
        a[:, ::2] = 200
        return a
    if kind == "narrow":
        return rng.integers(100, 103, size=shape, dtype=np.uint8)
    raise ValueError(kind)


KINDS = ["uniform", "ct", "mostly_zero", "constant", "two_values", "narrow"]


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(17, 32, 32), (17, 128, 128), (5, 7, 9)])
def test_reference_transcription_matches_literal(kind, shape):
    rng = np.random.default_rng(1)
    p = patches(rng, shape, kind)
    assert np.array_equal(normalize_robust_reference(p), villa_normalize_robust(p))


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("shape", [(17, 32, 32), (17, 128, 128), (4, 9, 9), (5, 7, 9)])
def test_gpu_path_is_bit_identical(kind, shape):
    rng = np.random.default_rng(2)
    batch = np.stack([patches(rng, shape, kind) for _ in range(3)])
    norm = RobustMadGPU(int(np.prod(shape)))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ours = norm(torch.from_numpy(batch).to(dev)).cpu().numpy()
    for i in range(batch.shape[0]):
        ref = villa_normalize_robust(batch[i])
        assert ours[i].dtype == np.float32
        assert np.array_equal(ours[i], ref), f"{kind} {shape} patch {i}: max diff {np.abs(ours[i] - ref).max()}"


def test_mixed_batch_and_fallback_count():
    rng = np.random.default_rng(3)
    shape = (17, 64, 64)
    batch = np.stack([patches(rng, shape, k) for k in KINDS])
    norm = RobustMadGPU(int(np.prod(shape)))
    ours = norm(torch.from_numpy(batch)).numpy()
    for i in range(len(KINDS)):
        assert np.array_equal(ours[i], villa_normalize_robust(batch[i]))
    assert norm.n_fallback >= 2                       # mostly_zero and constant take the fallback branch


def test_against_installed_villa():
    norm_mod = pytest.importorskip("vesuvius.image_proc.intensity.normalization")
    rng = np.random.default_rng(4)
    for kind in KINDS:
        p = patches(rng, (17, 128, 128), kind)
        ref = norm_mod.normalize_robust(p.astype(np.float32))
        ours = RobustMadGPU(p.size)(torch.from_numpy(p[None])).numpy()[0]
        assert np.array_equal(ours, ref), kind

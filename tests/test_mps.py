"""Apple GPU (MPS) support: the conv3d-as-conv2d rewrite equals nn.Conv3d, and device resolution prefers CUDA, then MPS."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from inksweep.mps import Conv3dAs2d, conv3d_as_conv2d, prepare_for_device

CASES = [
    dict(in_channels=1, out_channels=8, kernel_size=3, padding=1, bias=False),
    dict(in_channels=4, out_channels=6, kernel_size=1, bias=False),
    dict(in_channels=3, out_channels=5, kernel_size=(7, 7, 7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
    dict(in_channels=2, out_channels=3, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), bias=True),
    dict(in_channels=2, out_channels=3, kernel_size=(3, 1, 3), dilation=(2, 1, 1), padding=(2, 0, 1), bias=True),
]


@pytest.mark.parametrize("kw", CASES)
def test_conv3d_as_conv2d_matches_conv3d(kw):
    torch.manual_seed(0)
    conv = nn.Conv3d(**kw).double()
    x = torch.randn(2, kw["in_channels"], 9, 17, 19, dtype=torch.float64)
    with torch.no_grad():
        assert torch.allclose(Conv3dAs2d(conv)(x), conv(x), rtol=1e-10, atol=1e-10)


def test_conversion_swaps_nested_modules_and_keeps_weights():
    model = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(), nn.Sequential(nn.Conv3d(4, 2, 1), nn.BatchNorm3d(2).eval()),
                          nn.Conv3d(2, 2, 3, groups=2, padding=1))
    x = torch.randn(1, 1, 5, 8, 8)
    with torch.no_grad():
        ref = model(x)
        assert conv3d_as_conv2d(model) == 2                # grouped conv3d is left alone
        assert torch.allclose(model(x), ref, atol=1e-5)
    assert sum(isinstance(m, Conv3dAs2d) for m in model.modules()) == 2


def test_prepare_only_rewrites_on_mps(monkeypatch):
    model = nn.Sequential(nn.Conv3d(1, 1, 3))
    prepare_for_device(model, "cpu")
    assert isinstance(model[0], nn.Conv3d)
    monkeypatch.setenv("INKSWEEP_MPS_NATIVE_CONV3D", "1")
    prepare_for_device(model, torch.device("mps"))
    assert isinstance(model[0], nn.Conv3d)
    monkeypatch.delenv("INKSWEEP_MPS_NATIVE_CONV3D")
    prepare_for_device(model, torch.device("mps"))
    assert isinstance(model[0], Conv3dAs2d)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an Apple GPU")
def test_conv3d_as_conv2d_on_mps_matches_cpu():
    torch.manual_seed(0)
    conv = nn.Conv3d(4, 8, 3, padding=1)
    x = torch.randn(2, 4, 8, 32, 32)
    with torch.no_grad():
        ref = conv(x)
        got = Conv3dAs2d(conv.to("mps"))(x.to("mps")).cpu()
    assert torch.allclose(got, ref, atol=1e-4)


def test_robust_mad_on_mps_is_bit_identical_to_cpu():
    if not torch.backends.mps.is_available():
        pytest.skip("needs an Apple GPU")
    import numpy as np

    from inksweep.readers.norm import RobustMadGPU

    rng = np.random.default_rng(0)
    patches = rng.integers(0, 256, (4, 17, 64, 64), dtype=np.uint8)
    patches[1] = 7                                       # constant patch: CPU fallback path
    norm = RobustMadGPU(17 * 64 * 64)
    cpu = norm(torch.from_numpy(patches))
    assert torch.equal(norm(torch.from_numpy(patches).to("mps")).cpu(), cpu)


def test_resolve_device(monkeypatch):
    from inksweep.sweep import resolve_device

    assert resolve_device("cpu") == "cpu"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"


@pytest.mark.parametrize("device", ["cpu"] + (["mps"] if torch.backends.mps.is_available() else []))
def test_histograms_match_numpy_for_skewed_patches(device):
    import numpy as np

    from inksweep.readers.norm import _histograms

    rng = np.random.default_rng(1)
    flat = rng.integers(0, 256, (3, 5000), dtype=np.uint8)
    flat[1] = 0                                          # one grey value: the slow case for atomic adds
    flat[2, :4000] = 17
    ref = np.stack([np.bincount(r, minlength=256) for r in flat])
    assert np.array_equal(_histograms(torch.from_numpy(flat).to(device)).cpu().numpy(), ref)


def test_unbatched_input_and_subclasses():
    torch.manual_seed(0)
    conv = nn.Conv3d(2, 3, 3, padding=1).double()
    x = torch.randn(2, 5, 6, 7, dtype=torch.float64)            # [C, D, H, W]
    with torch.no_grad():
        assert torch.allclose(Conv3dAs2d(conv)(x), conv(x), rtol=1e-10, atol=1e-10)

    class Custom(nn.Conv3d):                                     # its own forward must survive
        def forward(self, x):
            return super().forward(x) * 2

    model = nn.Sequential(Custom(1, 1, 3))
    assert conv3d_as_conv2d(model) == 0 and type(model[0]) is Custom


def test_auto_says_when_it_falls_back_to_the_cpu(monkeypatch, caplog):
    from inksweep.sweep import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with caplog.at_level("WARNING", logger="inksweep"):
        assert resolve_device("auto") == "cpu"
    assert "no CUDA or Apple GPU" in caplog.text

"""Apple GPU (MPS) helpers.

PyTorch's MPS backend runs ``nn.Conv3d`` as an explicit matrix product (``aten::mm``), which made Hecate's 3-D ResNet
only about 2x faster than the CPU on an M4 Pro. :func:`conv3d_as_conv2d` replaces every plain ``nn.Conv3d`` in a model
with :class:`Conv3dAs2d`, the same convolution written as a sum over the depth taps of 2-D convolutions, which MPS runs
natively: Hecate 9.6 µm then runs about 5x faster on MPS (batch 4), equal to the CPU float32 result within float32
rounding. Only applied on MPS; ``INKSWEEP_MPS_NATIVE_CONV3D=1`` keeps PyTorch's own conv3d.
"""
from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn


class Conv3dAs2d(nn.Module):
    """``nn.Conv3d`` (groups 1, zero padding) as ``sum_k conv2d(x[:, :, k::stride_d], w[:, :, k])`` over depth taps."""

    def __init__(self, conv: nn.Conv3d) -> None:
        super().__init__()
        self.kd, self.sd, self.pd, self.dd = conv.kernel_size[0], conv.stride[0], conv.padding[0], conv.dilation[0]
        self.stride2, self.padding2, self.dilation2 = conv.stride[1:], conv.padding[1:], conv.dilation[1:]
        self.weight = conv.weight
        self.bias = conv.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:  # unbatched [C, D, H, W], as nn.Conv3d accepts
            return self.forward(x[None])[0]
        b, c = x.shape[:2]
        if self.pd:
            x = F.pad(x, (0, 0, 0, 0, self.pd, self.pd))
        d_out = (x.shape[2] - self.dd * (self.kd - 1) - 1) // self.sd + 1
        out = None
        for k in range(self.kd):
            start = k * self.dd
            xs = x[:, :, start:start + self.sd * (d_out - 1) + 1:self.sd]          # [B, C, Do, H, W]
            xs = xs.transpose(1, 2).reshape(b * d_out, c, x.shape[3], x.shape[4])
            y = F.conv2d(xs, self.weight[:, :, k], None, self.stride2, self.padding2, self.dilation2)
            out = y if out is None else out + y
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out.reshape(b, d_out, out.shape[1], out.shape[2], out.shape[3]).transpose(1, 2)


def conv3d_as_conv2d(model: nn.Module) -> int:
    """Swap every eligible ``nn.Conv3d`` in ``model`` for :class:`Conv3dAs2d` in place; returns how many."""
    n = 0
    for name, child in list(model.named_children()):
        if (type(child) is nn.Conv3d and child.groups == 1 and child.padding_mode == "zeros"
                and isinstance(child.padding, tuple)):
            setattr(model, name, Conv3dAs2d(child))
            n += 1
        else:
            n += conv3d_as_conv2d(child)
    return n


def prepare_for_device(model: nn.Module, device: torch.device | str) -> nn.Module:
    """On MPS, run 3-D convolutions as 2-D ones (see the module docstring); elsewhere return ``model`` unchanged."""
    if torch.device(device).type == "mps" and os.environ.get("INKSWEEP_MPS_NATIVE_CONV3D") != "1":
        conv3d_as_conv2d(model)
    return model

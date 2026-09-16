"""Packed symmetric update quantization; no integer inference kernels.

Groups follow the sorted transmitted coordinates, so sparse tensors do not pay
for scales of empty blocks. Clipping is selected using only the local update.
"""
from __future__ import annotations

import math
import numpy as np
import torch


def pack_signed(values: torch.Tensor, bits: int) -> bytes:
    if bits not in (2, 4, 8):
        raise ValueError("bits must be 2, 4 or 8")
    limit = (1 << (bits - 1)) - 1
    if not bool(torch.isfinite(values).all()) or bool(
        ((values != values.round()) | (values.abs() > limit)).any()
    ):
        raise ValueError("Codes must be integral and within the symmetric range")
    codes = values.detach().cpu().reshape(-1).to(torch.int64).numpy()
    codes = (codes & ((1 << bits) - 1)).astype(np.uint8)
    per_byte = 8 // bits
    codes = np.pad(codes, (0, (-len(codes)) % per_byte))
    packed = np.zeros(len(codes) // per_byte, dtype=np.uint8)
    for offset in range(per_byte):
        packed |= codes[offset::per_byte] << (offset * bits)
    return packed.tobytes()


def unpack_signed(blob: bytes, count: int, bits: int) -> np.ndarray:
    if bits not in (2, 4, 8) or count < 0:
        raise ValueError("Invalid bit width or count")
    if len(blob) != (count * bits + 7) // 8:
        raise ValueError("Packed value length mismatch")
    raw = np.frombuffer(blob, dtype=np.uint8)
    per_byte = 8 // bits
    codes = np.empty(len(raw) * per_byte, dtype=np.int16)
    for offset in range(per_byte):
        codes[offset::per_byte] = (raw >> (offset * bits)) & ((1 << bits) - 1)
    if np.any(codes[count:] != 0):
        raise ValueError("Nonzero padding in packed values")
    codes = codes[:count]
    codes = np.where(codes >= (1 << (bits - 1)), codes - (1 << bits), codes)
    if np.any(codes == -(1 << (bits - 1))):
        raise ValueError("Reserved symmetric quantization code")
    return codes.astype(np.float32)


def quantize_groups(
    values: torch.Tensor, *, bits: int, group_size: int = 0,
    clipping: str = "max", stochastic: bool = False,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Return codes, FP32 scales and predicted squared reconstruction error.

    INT2 uses {-1, 0, 1}, with the fourth code reserved. MSE search includes
    max-abs, and hence cannot worsen the predicted distortion of that scale.
    With stochastic rounding, the objective includes rounding variance AND
    clipping bias. No validation or test labels enter this calculation.
    """
    if bits not in (2, 4, 8) or group_size < 0 or clipping not in ("max", "mse", "mse_refine"):
        raise ValueError("Invalid quantization options")
    x = values.detach().cpu().to(torch.float32).reshape(-1)
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("Nonfinite quantizer input")
    if not x.numel():
        return x, torch.ones(1), 0.0
    width = group_size or x.numel()
    groups = math.ceil(x.numel() / width)
    padded = torch.nn.functional.pad(x, (0, groups * width - x.numel()))
    matrix = padded.reshape(groups, width)
    limit = (1 << (bits - 1)) - 1
    maximum = matrix.abs().amax(dim=1, keepdim=True)
    ratios = (1.0,) if clipping == "max" else (1.0, .95, .9, .8, .7, .6, .5)
    best_error = torch.full((groups,), float("inf"), dtype=torch.float64)
    best_scale = torch.ones((groups, 1), dtype=torch.float32)
    for ratio in ratios:
        # Use exactly the FP32 scale that is sent over the wire, including
        # tiny inputs whose max/limit could otherwise underflow to zero.
        scale = (maximum * ratio / limit).clamp_min(torch.finfo(torch.float32).tiny)
        scale = torch.where(maximum == 0, torch.ones_like(scale), scale)
        scaled = (matrix / scale).clamp(-limit, limit)
        if stochastic:
            fraction = scaled - scaled.floor()
            error = ((scaled.double() * scale.double() - matrix.double()) ** 2
                     + fraction.double() * (1 - fraction.double()) * scale.double() ** 2)
        else:
            # Match the FP32 multiply performed by the actual wire decoder.
            error = ((scaled.round() * scale).double() - matrix.double()) ** 2
        error = error.sum(dim=1)
        improve = error < best_error
        best_scale[improve] = scale[improve]
        best_error = torch.minimum(best_error, error)
    if clipping == "mse_refine":
        if stochastic:
            raise ValueError("mse_refine is a deterministic INT quantizer")
        # Alternating least squares: for fixed signed codes, this is the exact
        # scale minimizing reconstruction SSE. Retain the old best if floating
        # rounding or a changed code assignment would make SSE worse.
        for _ in range(3):
            codes = (matrix / best_scale).clamp(-limit, limit).round().double()
            denominator = codes.square().sum(dim=1, keepdim=True)
            fitted = ((codes * matrix.double()).sum(dim=1, keepdim=True)
                      / denominator.clamp_min(1)).float()
            fitted = fitted.clamp_min(torch.finfo(torch.float32).tiny)
            fitted = torch.where(denominator > 0, fitted, best_scale)
            new_codes = (matrix / fitted).clamp(-limit, limit).round()
            error = ((new_codes * fitted).double() - matrix.double()).square().sum(dim=1)
            improve = error < best_error
            best_scale[improve] = fitted[improve]
            best_error = torch.minimum(best_error, error)
    scaled = (matrix / best_scale).clamp(-limit, limit)
    if stochastic:
        lower = scaled.floor()
        codes = lower + (torch.rand(scaled.shape, generator=generator) < scaled - lower)
    else:
        codes = scaled.round()
    return codes.flatten()[:x.numel()], best_scale.flatten(), float(best_error.sum())

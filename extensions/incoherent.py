"""Incoherence processing and a one-byte shared-exponent scale code.

Two independent wire-format economies for the sparse uplink message, both aimed
at the same measured bottleneck: at a fixed serialized budget the payload is
dominated by *addressing* coordinates, so the only way to transmit more signal
is to make each addressed coordinate cheaper. Block-structured Top-K already
amortizes one index over ``block_size`` coordinates; what remains is the cost
of the values and of the per-group scales.

``rotate_groups`` applies a randomized Hadamard transform (a seeded sign flip
followed by a normalized fast Walsh-Hadamard transform) inside each
quantization group. The transform is orthonormal, so it preserves the group's
L2 norm exactly, and it is its own inverse. Its purpose is the incoherence
property used by rotation-based distributed mean estimation (DRIVE, EDEN,
QUIC-FL) and by lattice-codebook weight quantizers (QuIP#, QTIP): after a
random rotation the coordinates of a group are close to identically
distributed, so a single scale describes all of them and every code in a
low-bit alphabet carries information. Without it, a block selected for holding
one large coordinate spends most of its codes representing that coordinate's
small neighbours as zero.

``encode_log8_scales`` replaces the FP32 per-group scale with one byte holding
a logarithmic offset from a per-tensor reference, in the spirit of the shared
block exponent of microscaling float formats. At INT2 with a group of 32 the
FP32 scale costs one bit per coordinate -- half of the value stream itself --
and the byte code removes three quarters of that. Rounding the scale is not
lossy in practice because the caller refits the integer codes against the
scale that will actually be sent.

Both sides derive the rotation seed from data already on the wire -- client id,
server round and tensor name -- so nothing is transmitted to describe it, and a
rerun of a campaign reproduces the same rotation.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch

# One byte spans this many octaves of scale at 1/8-octave resolution, which
# bounds the scale-rounding error at 4.5 percent and the dynamic range at
# 2^32. Both sides hard-code it; it is not negotiated on the wire.
LOG8_STEPS_PER_OCTAVE = 8.0
LOG8_MAX_CODE = 255


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def rotation_signs(
    *, client_id: int, server_round: int, name: str, groups: int, width: int
) -> torch.Tensor:
    """Return the shared +/-1 matrix for one tensor's quantization groups.

    Seeded from the payload's own identifiers so the decoder reconstructs it
    without any side channel. ``hashlib`` rather than ``hash`` because the
    built-in string hash is salted per process and would not survive a restart,
    let alone a different machine.
    """
    if groups < 0 or width <= 0:
        raise ValueError("Invalid rotation shape")
    material = f"{int(client_id)}:{int(server_round)}:{name}".encode("utf-8")
    digest = hashlib.blake2b(material, digest_size=8).digest()
    seed = int.from_bytes(digest, "little", signed=False)
    bits = np.random.Generator(np.random.Philox(key=seed)).integers(
        0, 2, size=(groups, width), dtype=np.uint8
    )
    return torch.from_numpy(bits.astype(np.float32) * 2.0 - 1.0)


def _walsh_hadamard(matrix: torch.Tensor) -> torch.Tensor:
    """In-place-style normalized fast Walsh-Hadamard transform along dim 1."""
    groups, width = matrix.shape
    if not is_power_of_two(width):
        raise ValueError("Hadamard width must be a power of two")
    working = matrix.clone()
    step = 1
    while step < width:
        view = working.reshape(groups, -1, 2, step)
        upper = view[:, :, 0, :].clone()
        lower = view[:, :, 1, :].clone()
        view[:, :, 0, :] = upper + lower
        view[:, :, 1, :] = upper - lower
        working = view.reshape(groups, width)
        step *= 2
    return working / math.sqrt(width)


def rotate_groups(
    values: torch.Tensor,
    *,
    group_size: int,
    client_id: int,
    server_round: int,
    name: str,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate whole groups of ``values``; leave any short tail untouched.

    A partial final group is not padded. Padding would have to be transmitted,
    because the rotation makes the padded positions carry signal, and at a few
    thousand coordinates per tensor the tail is worth less than the bytes that
    would cost. Encoder and decoder apply the identical rule, so the round trip
    stays exact.
    """
    if group_size <= 0 or not is_power_of_two(group_size):
        raise ValueError("Rotation requires a power-of-two group size")
    flat = values.detach().cpu().to(torch.float32).reshape(-1)
    count = int(flat.numel())
    groups = count // group_size
    if groups == 0:
        return flat.clone()
    full = groups * group_size
    matrix = flat[:full].reshape(groups, group_size)
    signs = rotation_signs(
        client_id=client_id, server_round=server_round, name=name,
        groups=groups, width=group_size,
    )
    # H is symmetric and H @ H = I once normalized, so the forward map is
    # (sign, then transform) and the inverse is (transform, then sign).
    if inverse:
        rotated = _walsh_hadamard(matrix) * signs
    else:
        rotated = _walsh_hadamard(matrix * signs)
    out = flat.clone()
    out[:full] = rotated.reshape(-1)
    return out


def encode_log8_scales(scales: torch.Tensor) -> tuple[float, np.ndarray, torch.Tensor]:
    """Return the FP32 reference, the byte codes, and the decoded scales.

    The decoded scales are what the caller must quantize against, so that the
    integer codes absorb the scale rounding instead of paying for it twice.
    """
    values = scales.detach().cpu().to(torch.float32).reshape(-1)
    if not bool(torch.isfinite(values).all()) or bool((values <= 0).any()):
        raise ValueError("Group scales must be finite and positive")
    reference = float(values.max())
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Scale reference must be finite and positive")
    octaves = torch.log2(values / reference).clamp_max(0.0)
    codes = torch.round(-octaves * LOG8_STEPS_PER_OCTAVE)
    codes = codes.clamp(0, LOG8_MAX_CODE).to(torch.int64)
    decoded = decode_log8_scales(reference, codes.numpy().astype(np.uint8))
    return reference, codes.numpy().astype(np.uint8), decoded


def decode_log8_scales(reference: float, codes: np.ndarray) -> torch.Tensor:
    if not math.isfinite(reference) or reference <= 0:
        raise ValueError("Scale reference must be finite and positive")
    exponent = -torch.from_numpy(codes.astype(np.float32)) / LOG8_STEPS_PER_OCTAVE
    decoded = float(reference) * torch.pow(torch.tensor(2.0), exponent)
    return decoded.to(torch.float32).clamp_min(torch.finfo(torch.float32).tiny)


def refit_codes(
    values: torch.Tensor, scales: torch.Tensor, *, group_size: int, bits: int
) -> torch.Tensor:
    """Nearest-code assignment against the scales that will be transmitted."""
    limit = (1 << (bits - 1)) - 1
    flat = values.detach().cpu().to(torch.float32).reshape(-1)
    if not flat.numel():
        return flat
    width = group_size or flat.numel()
    groups = math.ceil(flat.numel() / width)
    padded = torch.nn.functional.pad(flat, (0, groups * width - flat.numel()))
    matrix = padded.reshape(groups, width)
    column = scales.reshape(groups, 1)
    codes = (matrix / column).clamp(-limit, limit).round()
    return codes.flatten()[: flat.numel()]


# Lloyd-Max optimal four-level codebook for the standard normal. Two bits give
# four codes, but the existing symmetric integer quantizer spends them on
# {-1, 0, +1} and reserves the fourth, throwing away a quarter of the alphabet.
# A rotated group is close to normal by construction, so a fixed codebook fitted
# to N(0,1) is near-optimal for it and needs no per-group search.
GAUSSIAN4_CENTROIDS = torch.tensor(
    [-1.5104175806045532, -0.4527800381183624,
     0.4527800381183624, 1.5104175806045532], dtype=torch.float32
)
_GAUSSIAN4_BOUNDARIES = (GAUSSIAN4_CENTROIDS[:-1] + GAUSSIAN4_CENTROIDS[1:]) / 2


def quantize_gaussian4(
    values: torch.Tensor, *, group_size: int
) -> tuple[np.ndarray, torch.Tensor]:
    """Assign each value to one of four centroids; fit one scale per group.

    The scale is found by alternating least squares from several starts, the
    same shape of search the integer quantizer already uses, because a codebook
    is only as good as the scale that maps the group onto it.
    """
    flat = values.detach().cpu().to(torch.float32).reshape(-1)
    count = int(flat.numel())
    if count == 0:
        return np.zeros(0, dtype=np.uint8), torch.ones(1)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("Nonfinite quantizer input")
    width = group_size or count
    groups = math.ceil(count / width)
    padded = torch.nn.functional.pad(flat, (0, groups * width - count))
    matrix = padded.reshape(groups, width)
    live = (torch.arange(groups * width).reshape(groups, width) < count).float()
    counts = live.sum(dim=1, keepdim=True).clamp_min(1)
    rms = ((matrix.double().square() * live.double()).sum(dim=1, keepdim=True)
           / counts.double()).sqrt().float()
    peak = matrix.abs().amax(dim=1, keepdim=True) / GAUSSIAN4_CENTROIDS[-1]
    tiny = torch.finfo(torch.float32).tiny
    best_error = torch.full((groups,), float("inf"), dtype=torch.float64)
    best_scale = torch.ones((groups, 1))
    best_codes = torch.zeros((groups, width), dtype=torch.long)
    for start in (rms, peak, rms * 0.75, rms * 1.25):
        scale = start.clamp_min(tiny)
        for _ in range(5):
            codes = torch.bucketize(matrix / scale, _GAUSSIAN4_BOUNDARIES)
            levels = GAUSSIAN4_CENTROIDS[codes]
            error = (((levels * scale).double() - matrix.double()).square()
                     * live.double()).sum(dim=1)
            improve = error < best_error
            best_error[improve] = error[improve]
            best_scale[improve] = scale[improve]
            best_codes[improve] = codes[improve]
            denominator = (levels.double().square() * live.double()).sum(dim=1, keepdim=True)
            fitted = ((levels.double() * matrix.double() * live.double()).sum(dim=1, keepdim=True)
                      / denominator.clamp_min(1e-30)).float()
            scale = torch.where(denominator > 0, fitted, scale).clamp_min(tiny)
    return (best_codes.flatten()[:count].to(torch.uint8).numpy(),
            best_scale.flatten().clamp_min(tiny))


def dequantize_gaussian4(codes: np.ndarray, scales: torch.Tensor) -> torch.Tensor:
    index = torch.from_numpy(codes.astype(np.int64))
    if int(index.numel()) and (int(index.min()) < 0 or int(index.max()) > 3):
        raise ValueError("Gaussian codebook index is out of range")
    return GAUSSIAN4_CENTROIDS[index] * scales


def pack_codes2(codes: np.ndarray) -> bytes:
    raw = np.asarray(codes, dtype=np.uint8)
    if raw.size and int(raw.max()) > 3:
        raise ValueError("Two-bit codes must be in 0..3")
    padded = np.pad(raw, (0, (-len(raw)) % 4))
    packed = np.zeros(len(padded) // 4, dtype=np.uint8)
    for offset in range(4):
        packed |= padded[offset::4] << (2 * offset)
    return packed.tobytes()


def unpack_codes2(blob: bytes, count: int) -> np.ndarray:
    if count < 0 or len(blob) != (count * 2 + 7) // 8:
        raise ValueError("Packed codebook length mismatch")
    raw = np.frombuffer(blob, dtype=np.uint8)
    codes = np.empty(len(raw) * 4, dtype=np.uint8)
    for offset in range(4):
        codes[offset::4] = (raw >> (2 * offset)) & 3
    if np.any(codes[count:] != 0):
        raise ValueError("Nonzero padding in packed codes")
    return codes[:count]


def assign_gaussian4(
    values: torch.Tensor, scales: torch.Tensor, *, group_size: int
) -> np.ndarray:
    """Nearest-centroid assignment against scales that are already fixed.

    Used when the scale has been rounded for transmission: reassigning the
    codes absorbs that rounding instead of paying for it a second time.
    """
    flat = values.detach().cpu().to(torch.float32).reshape(-1)
    count = int(flat.numel())
    if count == 0:
        return np.zeros(0, dtype=np.uint8)
    width = group_size or count
    groups = math.ceil(count / width)
    padded = torch.nn.functional.pad(flat, (0, groups * width - count))
    matrix = padded.reshape(groups, width)
    codes = torch.bucketize(matrix / scales.reshape(groups, 1), _GAUSSIAN4_BOUNDARIES)
    return codes.flatten()[:count].to(torch.uint8).numpy()

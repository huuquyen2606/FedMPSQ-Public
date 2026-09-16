"""FedMPSQ global Top-K, symmetric INT8, error feedback, and wire codec."""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from extensions.lowbit import pack_signed, unpack_signed, quantize_groups
from extensions.mpsq_int4 import deterministic_topk_indices
from extensions.rare_budget import budgeted_block_indices, encode_runs, decode_runs
from extensions.incoherent import (
    assign_gaussian4,
    dequantize_gaussian4,
    pack_codes2,
    quantize_gaussian4,
    unpack_codes2,
    decode_log8_scales,
    encode_log8_scales,
    is_power_of_two,
    refit_codes,
    rotate_groups,
)


MAGIC = b"FMPSQ01!"
VERSION = 1
PREAMBLE = struct.Struct("<8sHHI")
METADATA = struct.Struct("<IIQQQff")
LAYOUT_PREFIX = struct.Struct("<HBBBBQQ")
CRC = struct.Struct("<I")

ENCODING_DENSE = 1
ENCODING_SPARSE_FLOAT32 = 2
ENCODING_SPARSE_INT8 = 3
ENCODING_SPARSE_INT4 = 4
ENCODING_SPARSE_INT2 = 5
GROUPED_ENCODINGS = {2: 6, 4: 7, 8: 8}
# The same grouped layout, except that the per-group scale is one byte of
# logarithmic offset from a per-tensor FP32 reference rather than a whole
# FP32 word. At INT2 with a group of 32 an FP32 scale costs one bit per
# coordinate, half as much again as the value stream it describes; the byte
# code removes three quarters of that.
LOG8_ENCODINGS = {2: 9, 4: 10, 8: 11}
# Two bits carry four codes, but the symmetric integer quantizer spends
# them on {-1, 0, +1} and reserves the fourth, discarding a quarter of the
# alphabet. These encodings spend all four on the Lloyd-Max codebook for a
# standard normal, which is what a rotated group looks like.
GAUSS4_ENCODINGS = {"fp32": 12, "log8": 13}
ENCODING_BITS = {3: 8, 4: 4, 5: 2, 6: 2, 7: 4, 8: 8, 9: 2, 10: 4, 11: 8, 12: 2, 13: 2}

# The layout's ``index_width`` byte names the index code, not only a width: 0 is
# a presence bitmap, 4 and 8 are explicit unsigned positions of that many bytes,
# and 255 is the Golomb-Rice gap stream below. Widths 4 and 8 keep their old
# meaning, so a payload written before the Rice code existed still decodes.
INDEX_CODEC_BITMAP = 0
INDEX_CODEC_RICE = 255
INDEX_CODEC_RUNS = 254
INDEX_CODEC_FULL = 253
COMPACT_LAYOUT_FLAG = 0x8000
# Values inside every quantization group were rotated by a seeded randomized
# Hadamard transform before quantization. The rotation is orthonormal and its
# seed comes from identifiers already on the wire, so it costs no bytes, and
# the decoder undoes it after rescaling.
ROTATION_FLAG = 0x4000
# Flags that say how the payload was coded, not what role it plays. A
# verifier checking that a message is the A5 client uplink must mask these
# out, or enabling a codec option would look like a protocol violation.
CODEC_FLAGS = COMPACT_LAYOUT_FLAG | ROTATION_FLAG
COMPACT_LAYOUT_HEADER = struct.Struct("<II")
# Rice parameter, then the byte length of the unary half of the stream.
_RICE_HEADER = struct.Struct("<BI")

DTYPE_TO_CODE = {
    torch.float32: 1,
    torch.float64: 2,
    torch.float16: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.bool: 6,
}
CODE_TO_NUMPY = {
    1: np.dtype("<f4"),
    2: np.dtype("<f8"),
    3: np.dtype("<f2"),
    4: np.dtype("<i8"),
    5: np.dtype("<i4"),
    6: np.dtype("u1"),
}
CODE_TO_TORCH = {
    1: torch.float32,
    2: torch.float64,
    3: torch.float16,
    4: torch.int64,
    5: torch.int32,
    6: torch.bool,
}


@dataclass(frozen=True)
class ByteBreakdown:
    header_bytes: int
    metadata_bytes: int
    layout_bytes: int
    index_bytes: int
    value_bytes: int
    scale_bytes: int

    @property
    def total_bytes(self) -> int:
        return (
            self.header_bytes
            + self.metadata_bytes
            + self.layout_bytes
            + self.index_bytes
            + self.value_bytes
            + self.scale_bytes
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "header_bytes": self.header_bytes,
            "metadata_bytes": self.metadata_bytes,
            "layout_bytes": self.layout_bytes,
            "index_bytes": self.index_bytes,
            "value_bytes": self.value_bytes,
            "scale_bytes": self.scale_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class EncodedPayload:
    data: bytes
    byte_breakdown: ByteBreakdown
    num_parameters: int
    stored_values: int
    theoretical_bytes: int
    theoretical_ideal_index_bytes: int

    @property
    def payload_bytes(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class DecodedPayload:
    state: dict[str, torch.Tensor]
    client_id: int
    server_round: int
    num_examples: int
    num_parameters: int
    stored_values: int
    declared_sparsity: float
    alpha_s: float
    flags: int


def address_payload(payload: EncodedPayload, client_id: int) -> EncodedPayload:
    """Reuse an encoded common downlink; change only recipient and CRC.

    Each recipient still pays the complete serialized message in the ledger.
    This caches computation, not network traffic or a previous round's state.
    """
    if not 0 <= client_id <= np.iinfo(np.uint32).max:
        raise ValueError("Client ID is outside the wire-format range")
    raw = bytearray(payload.data[:-CRC.size])
    struct.pack_into("<I", raw, PREAMBLE.size, client_id)
    data = bytes(raw) + CRC.pack(zlib.crc32(raw) & 0xFFFFFFFF)
    return replace(payload, data=data)


@dataclass(frozen=True)
class TopKResult:
    sparse_state: dict[str, torch.Tensor]
    selected_values: int
    num_parameters: int
    squared_error: float
    squared_norm: float


@dataclass(frozen=True)
class CompressionResult:
    payload: EncodedPayload
    decoded_update: dict[str, torch.Tensor]
    residual_state: dict[str, torch.Tensor]
    metrics: dict[str, float]


def require_finite_state(
    state: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    """Reject non-finite floating tensors instead of silently encoding them."""
    for name, tensor in state.items():
        if torch.is_floating_point(tensor) and not bool(torch.isfinite(tensor).all()):
            raise FloatingPointError(f"{label} tensor {name!r} contains NaN or Inf")


def _require_matching_float_layout(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    if reference.keys() != candidate.keys():
        raise ValueError(f"{label} tensor layout does not match the update")
    for name, tensor in reference.items():
        other = candidate[name]
        if tuple(tensor.shape) != tuple(other.shape):
            raise ValueError(f"{label} tensor shape differs for {name}")
        if not torch.is_floating_point(other):
            raise TypeError(f"{label} tensor {name!r} must be floating point")


def floating_state_delta(
    local_state: Mapping[str, torch.Tensor],
    global_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if local_state.keys() != global_state.keys():
        raise ValueError("Local and global states have different tensor layouts")
    return {
        name: (
            local_state[name].detach().cpu().to(torch.float32)
            - global_state[name].detach().cpu().to(torch.float32)
        )
        for name in local_state
        if torch.is_floating_point(local_state[name])
    }


def zeros_like_floating_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        name: torch.zeros_like(value, dtype=torch.float32, device="cpu")
        for name, value in state.items()
        if torch.is_floating_point(value)
    }


def clone_tensor_state(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in state.items()}


def state_squared_norm(state: Mapping[str, torch.Tensor]) -> float:
    return float(
        sum(
            torch.sum(value.detach().to(torch.float64) ** 2).item()
            for value in state.values()
        )
    )


def state_squared_error(
    first: Mapping[str, torch.Tensor],
    second: Mapping[str, torch.Tensor],
) -> float:
    if first.keys() != second.keys():
        raise ValueError("State layouts differ")
    return float(
        sum(
            torch.sum(
                (
                    first[name].detach().to(torch.float64)
                    - second[name].detach().to(torch.float64)
                )
                ** 2
            ).item()
            for name in first
        )
    )


def global_topk(
    update: Mapping[str, torch.Tensor],
    saliency: Mapping[str, torch.Tensor],
    *,
    sparsity: float,
    alpha_s: float,
    use_saliency: bool,
    epsilon: float,
    keep: int | None = None,
    block_size: int = 1,
    protected_groups: Sequence[Mapping[str, torch.Tensor]] = (),
    reserve_fraction: float = 0.0,
) -> TopKResult:
    """Apply one deterministic global mask using Eq. (7)--(8).

    ``keep`` overrides the coordinate budget implied by ``sparsity``. The
    compressor uses it when part of the payload is transmitted densely, so the
    density of the complete message still matches the declared ``sparsity``.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    if not 0.0 <= alpha_s <= 1.0:
        raise ValueError("alpha_s must be in [0, 1]")
    if epsilon <= 0.0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")
    names = list(update)
    if not names:
        raise ValueError("Update has no floating tensors")
    if any(not torch.is_floating_point(update[name]) for name in names):
        raise TypeError("Top-K update tensors must be floating point")
    require_finite_state(update, label="Top-K update")
    if use_saliency:
        for name in names:
            if (
                name in saliency
                and tuple(saliency[name].shape) != tuple(update[name].shape)
            ):
                raise ValueError(f"Saliency tensor shape differs for {name}")
        require_finite_state(saliency, label="Top-K saliency")
    flattened = [update[name].detach().cpu().to(torch.float32).reshape(-1) for name in names]
    magnitude = torch.cat(flattened).abs()
    num_parameters = int(magnitude.numel())
    if num_parameters == 0:
        raise ValueError("Update tensors contain no parameters")
    if keep is None:
        keep = max(1, int(math.ceil((1.0 - sparsity) * num_parameters)))
    else:
        keep = max(1, min(int(keep), num_parameters))
    magnitude_norm = magnitude / (float(magnitude.max()) + epsilon)
    if use_saliency:
        saliency_flat = torch.cat(
            [
                saliency.get(name, torch.zeros_like(update[name]))
                .detach()
                .cpu()
                .to(torch.float32)
                .reshape(-1)
                for name in names
            ]
        ).abs()
        saliency_norm = saliency_flat / (float(saliency_flat.max()) + epsilon)
        score = alpha_s * saliency_norm + (1.0 - alpha_s) * magnitude_norm
    else:
        score = magnitude_norm
    if block_size != 1 or (protected_groups and reserve_fraction > 0):
        selected, _ = budgeted_block_indices(
            score, {name: update[name].shape for name in names}, keep,
            block_size=block_size, protected_groups=protected_groups,
            reserve_fraction=reserve_fraction,
        )
    else:
        selected = deterministic_topk_indices(score, keep)
    sparse_flat = torch.zeros_like(magnitude)
    source_flat = torch.cat(flattened)
    sparse_flat[selected] = source_flat[selected]
    sparse: dict[str, torch.Tensor] = {}
    offset = 0
    for name in names:
        count = update[name].numel()
        sparse[name] = sparse_flat[offset : offset + count].reshape(update[name].shape)
        offset += count
    return TopKResult(
        sparse_state=sparse,
        selected_values=keep,
        num_parameters=num_parameters,
        squared_error=state_squared_error(update, sparse),
        squared_norm=state_squared_norm(update),
    )


def _numpy_dtype_code(tensor: torch.Tensor) -> int:
    if tensor.dtype not in DTYPE_TO_CODE:
        raise TypeError(f"Unsupported wire dtype: {tensor.dtype}")
    return DTYPE_TO_CODE[tensor.dtype]


def _tensor_bytes(tensor: torch.Tensor, dtype_code: int) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    if dtype_code == 6:
        return array.astype(np.uint8, copy=False).tobytes(order="C")
    return array.astype(CODE_TO_NUMPY[dtype_code], copy=False).tobytes(order="C")


def rice_parameter(count: int, numel: int) -> int:
    """Return the Golomb-Rice parameter for a mask of this density.

    Successive Top-K positions are a Bernoulli process, so the gaps between
    them are geometric with mean ``numel / count``. For a geometric source the
    optimal Golomb divisor is ``-1 / log2(1 - p)``; Rice restricts it to a
    power of two, and ``k = round(log2(ln 2 / p))`` is the standard choice.
    Both sides derive it from ``count`` and ``numel``, which the layout already
    carries, so the parameter never has to travel on the wire.
    """
    if count <= 0 or numel <= 0:
        return 0
    density = count / numel
    if density >= 1.0:
        return 0
    return int(min(24, max(0, round(math.log2(math.log(2.0) / density)))))


def _encode_rice_gaps(indices: np.ndarray, numel: int) -> bytes:
    """Rice-code the gaps between sorted sparse positions.

    The two halves of a Rice code are written as two separate streams rather
    than interleaved per value: every unary quotient first, then every
    fixed-width remainder. Interleaved codes have to be walked one symbol at a
    time to be read back; split this way the terminator of the ``i``-th unary
    run is the ``i``-th zero bit, so decoding both halves is a pair of vector
    operations instead of a Python loop over tens of thousands of symbols.
    """
    count = int(indices.size)
    if count == 0:
        return b""
    k = rice_parameter(count, numel)
    positions = indices.astype(np.int64, copy=False)
    gaps = np.empty(count, dtype=np.int64)
    gaps[0] = positions[0]
    if count > 1:
        gaps[1:] = positions[1:] - positions[:-1] - 1
    if np.any(gaps < 0):
        raise ValueError("Sparse positions must be strictly increasing")
    quotients = gaps >> k
    remainders = gaps & ((1 << k) - 1) if k else np.zeros(count, dtype=np.int64)

    unary_total = int(quotients.sum()) + count
    unary = np.ones(unary_total, dtype=np.uint8)
    lengths = quotients + 1
    ends = np.cumsum(lengths)
    unary[ends - 1] = 0
    unary_blob = np.packbits(unary, bitorder="little").tobytes()

    if k:
        shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
        remainder_bits = (
            (remainders[:, None] >> shifts[None, :]) & 1
        ).astype(np.uint8).reshape(-1)
        remainder_blob = np.packbits(remainder_bits, bitorder="little").tobytes()
    else:
        remainder_blob = b""
    return _RICE_HEADER.pack(k, len(unary_blob)) + unary_blob + remainder_blob


def _decode_rice_gaps(blob: bytes, count: int, numel: int) -> np.ndarray:
    if count == 0:
        if blob:
            raise ValueError("Empty Rice index stream carries trailing bytes")
        return np.zeros(0, dtype=np.int64)
    if len(blob) < _RICE_HEADER.size:
        raise ValueError("Rice index header is truncated")
    k, unary_bytes = _RICE_HEADER.unpack_from(blob, 0)
    if k > 24:
        raise ValueError("Rice parameter is out of range")
    if k != rice_parameter(count, numel):
        raise ValueError("Rice parameter disagrees with the declared density")
    start = _RICE_HEADER.size
    unary_end = start + unary_bytes
    remainder_bits_needed = count * k
    remainder_bytes = (remainder_bits_needed + 7) // 8
    if len(blob) != unary_end + remainder_bytes:
        raise ValueError("Rice index stream length does not match its header")
    unary = np.unpackbits(
        np.frombuffer(blob[start:unary_end], dtype=np.uint8), bitorder="little"
    )
    terminators = np.flatnonzero(unary == 0)
    if terminators.size < count:
        raise ValueError("Rice unary stream ends before the declared count")
    terminators = terminators[:count]
    # The i-th terminator sits after every quotient up to i plus the i earlier
    # terminators, so subtracting its own ordinal leaves the running total of
    # quotients; the individual quotients are its first difference.
    running = terminators - np.arange(count, dtype=np.int64)
    quotients = np.diff(running, prepend=np.int64(0))
    if np.any(quotients < 0):
        raise ValueError("Rice unary stream is malformed")
    if k:
        bits = np.unpackbits(
            np.frombuffer(blob[unary_end:], dtype=np.uint8), bitorder="little"
        )[:remainder_bits_needed].reshape(count, k)
        weights = (1 << np.arange(k - 1, -1, -1, dtype=np.int64))
        remainders = bits.astype(np.int64) @ weights
    else:
        remainders = np.zeros(count, dtype=np.int64)
    gaps = (quotients << k) + remainders
    positions = np.cumsum(gaps + 1) - 1
    if positions[-1] >= numel:
        raise ValueError("Decoded Rice positions fall outside the tensor")
    return positions


def _stochastic_round(
    scaled: torch.Tensor, generator: torch.Generator | None
) -> torch.Tensor:
    """Round to the neighbouring integer with probability given by the residue.

    This is unbiased for inputs inside the representable range. Clipping can
    introduce bias. Error feedback also compensates deterministic rounding;
    stochastic rounding is an ablation, not a prerequisite for error feedback.
    """
    lower = torch.floor(scaled)
    fraction = scaled - lower
    noise = torch.rand(
        scaled.shape, generator=generator, dtype=scaled.dtype, device=scaled.device
    )
    return lower + (noise < fraction).to(scaled.dtype)


def _pack_int4(values: torch.Tensor) -> bytes:
    """Pack signed nibbles in [-7, 7] two to a byte, low nibble first."""
    codes = values.to(torch.int64).numpy().astype(np.uint8) & 0x0F
    if codes.size % 2:
        codes = np.append(codes, np.zeros(1, dtype=np.uint8))
    return (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8).tobytes()


def _unpack_int4(blob: bytes, count: int) -> np.ndarray:
    raw = np.frombuffer(blob, dtype=np.uint8)
    codes = np.empty(raw.size * 2, dtype=np.uint8)
    codes[0::2] = raw & 0x0F
    codes[1::2] = raw >> 4
    codes = codes[:count].astype(np.int16)
    return np.where(codes >= 8, codes - 16, codes).astype(np.float32)


def _encode(
    state: Mapping[str, torch.Tensor],
    *,
    sparse: bool,
    quantized: bool,
    client_id: int,
    server_round: int,
    num_examples: int,
    declared_sparsity: float,
    alpha_s: float,
    flags: int,
    quant_bits: int = 8,
    index_codec: str = "rice",
    stochastic: bool = False,
    generator: torch.Generator | None = None,
    group_size: int = 0,
    clipping: str = "max",
    tensor_bits: Mapping[str, int] | None = None,
    raw_names: frozenset[str] = frozenset(),
    adaptive_error: float | None = None,
    compact_layout: bool = False,
    rotate: bool = False,
    scale_codec: str = "fp32",
    quantizer: str = "integer",
) -> EncodedPayload:
    if quant_bits not in {2, 4, 8}:
        raise ValueError("quant_bits must be 2, 4 or 8")
    if group_size < 0 or clipping not in {"max", "mse", "mse_refine"}:
        raise ValueError("Invalid group size or clipping mode")
    if adaptive_error is not None and (not math.isfinite(adaptive_error) or adaptive_error <= 0):
        raise ValueError("adaptive_error must be finite and positive")
    if any(bits not in {2, 4, 8} for bits in (tensor_bits or {}).values()):
        raise ValueError("Invalid per-tensor precision")
    if index_codec not in {"rice", "bitmap", "auto_runs"}:
        raise ValueError("index_codec must be rice, bitmap or auto_runs")
    if scale_codec not in {"fp32", "log8"}:
        raise ValueError("scale_codec must be fp32 or log8")
    if quantizer not in {"integer", "gaussian4"}:
        raise ValueError("quantizer must be integer or gaussian4")
    if quantizer == "gaussian4" and (quant_bits != 2 or not group_size):
        raise ValueError("The Gaussian codebook is a two-bit grouped quantizer")
    if quantizer == "gaussian4" and stochastic:
        raise ValueError("The Gaussian codebook is a deterministic quantizer")
    if rotate and not (group_size and is_power_of_two(group_size)):
        raise ValueError("Rotation requires a power-of-two quantization group")
    if scale_codec == "log8" and not group_size:
        raise ValueError("The byte scale code is defined for grouped scales only")
    if scale_codec == "log8" and stochastic:
        # Absorbing the byte code refits the integer codes by nearest
        # assignment, which would destroy the unbiasedness that stochastic
        # rounding exists to provide.
        raise ValueError("The byte scale code is a deterministic quantizer")
    if not 0 <= int(client_id) <= np.iinfo(np.uint32).max:
        raise ValueError("client_id is outside the wire-format range")
    if not 0 <= int(server_round) <= np.iinfo(np.uint32).max:
        raise ValueError("server_round is outside the wire-format range")
    if not 0 <= int(num_examples) <= np.iinfo(np.uint64).max:
        raise ValueError("num_examples is outside the wire-format range")
    if not 0 <= int(flags) <= np.iinfo(np.uint16).max:
        raise ValueError("flags is outside the wire-format range")
    if (
        not math.isfinite(declared_sparsity)
        or not 0.0 <= declared_sparsity < 1.0
    ):
        raise ValueError("declared_sparsity must be finite and in [0, 1)")
    if not math.isfinite(alpha_s) or not 0.0 <= alpha_s <= 1.0:
        raise ValueError("alpha_s must be finite and in [0, 1]")
    require_finite_state(state, label="Wire payload")
    layouts = bytearray()
    body = bytearray()
    layout_bytes = 0
    index_bytes = 0
    value_bytes = 0
    scale_bytes = 0
    total_parameters = 0
    stored_values = 0
    theoretical_ideal_index_bits = 0
    tensor_items = list(state.items())
    for name, raw_tensor in tensor_items:
        tensor = raw_tensor.detach().cpu().contiguous()
        encoded_name = name.encode("utf-8")
        if len(encoded_name) > 65535:
            raise ValueError("Tensor name is too long for the wire format")
        shape = tuple(int(value) for value in tensor.shape)
        ndim = len(shape)
        if ndim > 255:
            raise ValueError("Tensor rank exceeds the wire format")
        numel = int(tensor.numel())
        total_parameters += numel

        if sparse and name not in raw_names:
            if not torch.is_floating_point(tensor):
                raise TypeError("Sparse payloads support floating tensors only")
            logical = tensor.to(torch.float32).reshape(-1)
            indices = torch.nonzero(logical != 0, as_tuple=False).reshape(-1)
            count = int(indices.numel())
            # An explicit 32-bit position costs four bytes to address a single
            # INT8 value, so at rho=0.9 four fifths of the message is addressing.
            # A presence bitmap costs one bit per coordinate regardless of
            # density and is cheaper whenever the tensor is not extremely
            # sparse. Both codes are exact; the smaller one is chosen per
            # tensor, deterministically, and named in the layout.
            explicit_width = 4 if numel <= np.iinfo(np.uint32).max else 8
            bitmap_bytes = (numel + 7) // 8
            mask = np.zeros(numel, dtype=np.uint8)
            if count:
                mask[indices.numpy()] = 1
            candidates: list[tuple[int, int, bytes]] = [
                (
                    bitmap_bytes,
                    INDEX_CODEC_BITMAP,
                    np.packbits(mask, bitorder="little").tobytes(),
                ),
                (
                    count * explicit_width,
                    explicit_width,
                    indices.numpy()
                    .astype(
                        np.dtype("<u4") if explicit_width == 4 else np.dtype("<u8"),
                        copy=False,
                    )
                    .tobytes(),
                ),
            ]
            if index_codec in {"rice", "auto_runs"} and count:
                rice_blob = _encode_rice_gaps(indices.numpy(), numel)
                candidates.append((len(rice_blob), INDEX_CODEC_RICE, rice_blob))
            if index_codec == "auto_runs":
                if count == numel:
                    candidates.append((0, INDEX_CODEC_FULL, b""))
                elif count:
                    runs = encode_runs(indices.numpy())
                    run_blob = struct.pack("<I", len(runs)) + runs
                    candidates.append((len(run_blob), INDEX_CODEC_RUNS, run_blob))
            # Every candidate is exact, so picking the shortest can only shrink
            # the message. The choice is a pure function of the mask, so both
            # sides agree without negotiating and the ledger stays reproducible.
            _, index_width, index_blob = min(candidates, key=lambda item: item[0])
            if quantized:
                dtype_code = DTYPE_TO_CODE[torch.float32]
                selected = logical[indices]
                if rotate and count:
                    # Rotate before the scale is fitted, so that the scale
                    # describes the distribution actually being quantized.
                    selected = rotate_groups(
                        selected,
                        group_size=group_size,
                        client_id=client_id,
                        server_round=server_round,
                        name=name,
                    )
                bits = (tensor_bits or {}).get(name, quant_bits)
                if adaptive_error is not None:
                    norm = float(selected.double().square().sum())
                    # Search without consuming the actual rounding RNG stream.
                    for candidate in (b for b in (2, 4, 8) if b >= bits):
                        _, _, error = quantize_groups(
                            selected, bits=candidate, group_size=group_size,
                            clipping=clipping, stochastic=stochastic,
                            generator=torch.Generator().manual_seed(0),
                        )
                        bits = candidate
                        if error <= adaptive_error * max(norm, 1e-30):
                            break
                rounded, scales, _ = quantize_groups(
                    selected, bits=bits, group_size=group_size, clipping=clipping,
                    stochastic=stochastic, generator=generator,
                )
                if quantizer == "gaussian4":
                    codebook, scales = quantize_gaussian4(selected, group_size=group_size)
                    if scale_codec == "log8":
                        reference, scale_codes, scales = encode_log8_scales(scales)
                        codebook = assign_gaussian4(selected, scales, group_size=group_size)
                        scale_blob = (struct.pack("<I", group_size)
                                      + struct.pack("<f", reference) + scale_codes.tobytes())
                    else:
                        scale_blob = (struct.pack("<I", group_size)
                                      + scales.numpy().astype("<f4").tobytes())
                    encoding = GAUSS4_ENCODINGS[scale_codec]
                    values = pack_codes2(codebook)
                elif group_size and scale_codec == "log8":
                    reference, scale_codes, decoded_scales = encode_log8_scales(scales)
                    # Refit against the scale that will actually be sent, so
                    # the byte code costs only its own 1/8-octave grid.
                    rounded = refit_codes(
                        selected, decoded_scales, group_size=group_size, bits=bits
                    )
                    encoding = LOG8_ENCODINGS[bits]
                    scale_blob = (
                        struct.pack("<I", group_size)
                        + struct.pack("<f", reference)
                        + scale_codes.tobytes()
                    )
                else:
                    encoding = (
                        GROUPED_ENCODINGS[bits]
                        if group_size
                        else {2: 5, 4: 4, 8: 3}[bits]
                    )
                    scale_blob = scales.numpy().astype("<f4").tobytes()
                    if group_size:
                        scale_blob = struct.pack("<I", group_size) + scale_blob
                if quantizer != "gaussian4":
                    values = pack_signed(rounded, bits)
            else:
                encoding = ENCODING_SPARSE_FLOAT32
                dtype_code = DTYPE_TO_CODE[torch.float32]
                values = (
                    logical[indices]
                    .numpy()
                    .astype(np.dtype("<f4"), copy=False)
                    .tobytes()
                )
                scale_blob = b""
            if count:
                explicit_ideal = count * max(
                    1, int(math.ceil(math.log2(max(numel, 2))))
                )
                density = count / numel
                if 0.0 < density < 1.0:
                    entropy = -(
                        density * math.log2(density)
                        + (1.0 - density) * math.log2(1.0 - density)
                    )
                else:
                    entropy = 0.0
                mask_ideal = int(math.ceil(numel * entropy))
                theoretical_ideal_index_bits += min(explicit_ideal, mask_ideal)
        else:
            encoding = ENCODING_DENSE
            dtype_code = _numpy_dtype_code(tensor)
            index_width = 0
            count = numel
            index_blob = b""
            values = _tensor_bytes(tensor, dtype_code)
            scale_blob = b""

        prefix = LAYOUT_PREFIX.pack(
            len(encoded_name),
            ndim,
            encoding,
            dtype_code,
            index_width,
            numel,
            count,
        )
        shape_blob = b"".join(struct.pack("<Q", value) for value in shape)
        layouts.extend(prefix)
        layouts.extend(encoded_name)
        layouts.extend(shape_blob)
        body.extend(index_blob)
        body.extend(scale_blob)
        body.extend(values)
        layout_bytes += len(prefix) + len(encoded_name) + len(shape_blob)
        index_bytes += len(index_blob)
        scale_bytes += len(scale_blob)
        value_bytes += len(values)
        stored_values += count

    if flags & COMPACT_LAYOUT_FLAG:
        raise ValueError("Compact-layout transport flag is reserved for the encoder")
    if flags & ROTATION_FLAG:
        raise ValueError("Rotation flag is reserved for the encoder")
    if rotate and sparse and quantized:
        flags |= ROTATION_FLAG
    layout_blob = bytes(layouts)
    if compact_layout and layout_blob:
        packed = zlib.compress(layout_blob, level=9)
        candidate = COMPACT_LAYOUT_HEADER.pack(len(packed), len(layout_blob)) + packed
        if len(candidate) < len(layout_blob):
            layout_blob = candidate
            layout_bytes = len(candidate)
            flags |= COMPACT_LAYOUT_FLAG
    preamble = PREAMBLE.pack(MAGIC, VERSION, flags, len(tensor_items))
    metadata = METADATA.pack(
        int(client_id),
        int(server_round),
        int(num_examples),
        int(total_parameters),
        int(stored_values),
        float(declared_sparsity),
        float(alpha_s),
    )
    without_crc = preamble + metadata + layout_blob + bytes(body)
    crc = CRC.pack(zlib.crc32(without_crc) & 0xFFFFFFFF)
    payload = without_crc + crc
    breakdown = ByteBreakdown(
        header_bytes=PREAMBLE.size + CRC.size,
        metadata_bytes=METADATA.size,
        layout_bytes=layout_bytes,
        index_bytes=index_bytes,
        value_bytes=value_bytes,
        scale_bytes=scale_bytes,
    )
    if breakdown.total_bytes != len(payload):
        raise RuntimeError("Wire byte components do not sum to serialized payload")
    theoretical_bytes = breakdown.total_bytes
    ideal_index_bytes = int(math.ceil(theoretical_ideal_index_bits / 8.0))
    theoretical_ideal = (
        breakdown.header_bytes
        + breakdown.metadata_bytes
        + breakdown.layout_bytes
        + breakdown.scale_bytes
        + breakdown.value_bytes
        + ideal_index_bytes
    )
    return EncodedPayload(
        data=payload,
        byte_breakdown=breakdown,
        num_parameters=total_parameters,
        stored_values=stored_values,
        theoretical_bytes=theoretical_bytes,
        theoretical_ideal_index_bytes=theoretical_ideal,
    )


def serialize_dense_state(
    state: Mapping[str, torch.Tensor],
    *,
    client_id: int,
    server_round: int,
    num_examples: int,
    flags: int = 0,
) -> EncodedPayload:
    return _encode(
        state,
        sparse=False,
        quantized=False,
        client_id=client_id,
        server_round=server_round,
        num_examples=num_examples,
        declared_sparsity=0.0,
        alpha_s=0.0,
        flags=flags,
    )


def serialize_sparse_state(
    state: Mapping[str, torch.Tensor],
    *,
    quantized: bool,
    client_id: int,
    server_round: int,
    num_examples: int,
    declared_sparsity: float,
    alpha_s: float,
    flags: int = 1,
    quant_bits: int = 8,
    index_codec: str = "rice",
    stochastic: bool = False,
    generator: torch.Generator | None = None,
    group_size: int = 0,
    clipping: str = "max",
    tensor_bits: Mapping[str, int] | None = None,
    raw_names: frozenset[str] = frozenset(),
    adaptive_error: float | None = None,
    compact_layout: bool = False,
    rotate: bool = False,
    scale_codec: str = "fp32",
    quantizer: str = "integer",
) -> EncodedPayload:
    return _encode(
        state,
        sparse=True,
        quantized=quantized,
        client_id=client_id,
        server_round=server_round,
        num_examples=num_examples,
        declared_sparsity=declared_sparsity,
        alpha_s=alpha_s,
        flags=flags,
        quant_bits=quant_bits,
        index_codec=index_codec,
        stochastic=stochastic,
        generator=generator,
        group_size=group_size,
        clipping=clipping,
        tensor_bits=tensor_bits,
        raw_names=raw_names,
        adaptive_error=adaptive_error,
        compact_layout=compact_layout,
        rotate=rotate,
        scale_codec=scale_codec,
        quantizer=quantizer,
    )


def decode_payload(data: bytes) -> DecodedPayload:
    if len(data) < PREAMBLE.size + METADATA.size + CRC.size:
        raise ValueError("FedMPSQ payload is truncated")
    expected_crc = CRC.unpack_from(data, len(data) - CRC.size)[0]
    actual_crc = zlib.crc32(data[:-CRC.size]) & 0xFFFFFFFF
    if expected_crc != actual_crc:
        raise ValueError("FedMPSQ payload CRC mismatch")
    offset = 0
    magic, version, flags, tensor_count = PREAMBLE.unpack_from(data, offset)
    offset += PREAMBLE.size
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported FedMPSQ payload header")
    if flags & COMPACT_LAYOUT_FLAG:
        start = PREAMBLE.size + METADATA.size
        if start + COMPACT_LAYOUT_HEADER.size > len(data) - CRC.size:
            raise ValueError("Compact layout header is truncated")
        packed_size, raw_size = COMPACT_LAYOUT_HEADER.unpack_from(data, start)
        start += COMPACT_LAYOUT_HEADER.size
        end = start + packed_size
        if end > len(data) - CRC.size or not 0 < raw_size <= 16 * 1024 * 1024:
            raise ValueError("Invalid compact layout length")
        decompressor = zlib.decompressobj()
        try:
            layout = decompressor.decompress(data[start:end], raw_size + 1)
        except zlib.error as exc:
            raise ValueError("Invalid compressed layout") from exc
        if (len(layout) != raw_size or not decompressor.eof
                or decompressor.unused_data or decompressor.unconsumed_tail):
            raise ValueError("Compact layout length/stream mismatch")
        # Restore the canonical layout before the existing strict decoder.
        # No tensor values, indices or scales are changed by this transport.
        canonical = (PREAMBLE.pack(magic, version, flags & ~COMPACT_LAYOUT_FLAG, tensor_count)
                     + data[PREAMBLE.size:PREAMBLE.size + METADATA.size]
                     + layout + data[end:-CRC.size])
        return decode_payload(canonical + CRC.pack(zlib.crc32(canonical) & 0xFFFFFFFF))
    (
        client_id,
        server_round,
        num_examples,
        declared_parameters,
        declared_values,
        declared_sparsity,
        alpha_s,
    ) = METADATA.unpack_from(data, offset)
    offset += METADATA.size
    if (
        not math.isfinite(declared_sparsity)
        or not 0.0 <= declared_sparsity < 1.0
    ):
        raise ValueError("Payload declares an invalid sparsity")
    if not math.isfinite(alpha_s) or not 0.0 <= alpha_s <= 1.0:
        raise ValueError("Payload declares an invalid alpha_s")

    descriptors: list[dict[str, Any]] = []
    tensor_names: set[str] = set()
    for _ in range(tensor_count):
        if offset + LAYOUT_PREFIX.size > len(data) - CRC.size:
            raise ValueError("FedMPSQ layout is truncated")
        (
            name_length,
            ndim,
            encoding,
            dtype_code,
            index_width,
            numel,
            count,
        ) = LAYOUT_PREFIX.unpack_from(data, offset)
        offset += LAYOUT_PREFIX.size
        end_name = offset + name_length
        if end_name > len(data) - CRC.size:
            raise ValueError("FedMPSQ tensor name is truncated")
        name = data[offset:end_name].decode("utf-8")
        offset = end_name
        if not name or name in tensor_names:
            raise ValueError("Payload tensor names must be non-empty and unique")
        tensor_names.add(name)
        shape = []
        for _ in range(ndim):
            if offset + 8 > len(data) - CRC.size:
                raise ValueError("FedMPSQ tensor shape is truncated")
            shape.append(struct.unpack_from("<Q", data, offset)[0])
            offset += 8
        if int(np.prod(shape, dtype=np.int64)) != numel:
            raise ValueError(f"Invalid shape/numel for tensor {name}")
        if dtype_code not in CODE_TO_NUMPY:
            raise ValueError("Unsupported payload dtype code")
        if encoding not in {
            ENCODING_DENSE,
            ENCODING_SPARSE_FLOAT32,
            ENCODING_SPARSE_INT8,
            ENCODING_SPARSE_INT4,
            ENCODING_SPARSE_INT2,
            *GROUPED_ENCODINGS.values(),
            *LOG8_ENCODINGS.values(),
            *GAUSS4_ENCODINGS.values(),
        }:
            raise ValueError("Unsupported payload encoding")
        if count > numel:
            raise ValueError(f"Stored-value count exceeds tensor size for {name}")
        if (
            encoding != ENCODING_DENSE
            and dtype_code != DTYPE_TO_CODE[torch.float32]
        ):
            raise ValueError("Sparse payload tensors must use float32 logical dtype")
        descriptors.append(
            {
                "name": name,
                "shape": tuple(int(value) for value in shape),
                "encoding": encoding,
                "dtype_code": dtype_code,
                "index_width": index_width,
                "numel": int(numel),
                "count": int(count),
            }
        )

    state: dict[str, torch.Tensor] = {}
    stored_values = 0
    total_parameters = 0
    for descriptor in descriptors:
        name = descriptor["name"]
        encoding = descriptor["encoding"]
        numel = descriptor["numel"]
        count = descriptor["count"]
        dtype_code = descriptor["dtype_code"]
        total_parameters += numel
        stored_values += count
        if encoding == ENCODING_DENSE:
            if count != numel or descriptor["index_width"] != 0:
                raise ValueError("Invalid dense tensor descriptor")
            dtype = CODE_TO_NUMPY[dtype_code]
            size = count * dtype.itemsize
            end = offset + size
            if end > len(data) - CRC.size:
                raise ValueError("Dense tensor values are truncated")
            array = np.frombuffer(data[offset:end], dtype=dtype).copy()
            offset = end
            tensor = torch.from_numpy(array).to(CODE_TO_TORCH[dtype_code])
            if dtype_code == 6:
                tensor = tensor.to(torch.bool)
            state[name] = tensor.reshape(descriptor["shape"])
            continue

        index_width = descriptor["index_width"]
        if index_width not in {INDEX_CODEC_BITMAP, 4, 8, INDEX_CODEC_RICE, INDEX_CODEC_RUNS, INDEX_CODEC_FULL}:
            raise ValueError(
                "Sparse tensor requires a bitmap, 32/64-bit indices, or a Rice stream"
            )
        if index_width == INDEX_CODEC_FULL:
            if count != numel:
                raise ValueError("Full-support code requires count == numel")
            indices = np.arange(numel, dtype=np.int64)
        elif index_width == INDEX_CODEC_RUNS:
            if offset + 4 > len(data) - CRC.size:
                raise ValueError("Run index header is truncated")
            size = struct.unpack_from("<I", data, offset)[0]
            offset += 4
            if offset + size > len(data) - CRC.size:
                raise ValueError("Run index stream is truncated")
            indices = decode_runs(data[offset:offset+size], count, numel)
            offset += size
        elif index_width == INDEX_CODEC_RICE:
            if count == 0:
                indices = np.zeros(0, dtype=np.int64)
            else:
                k = rice_parameter(count, numel)
                if offset + _RICE_HEADER.size > len(data) - CRC.size:
                    raise ValueError("Rice index header is truncated")
                _, unary_bytes = _RICE_HEADER.unpack_from(data, offset)
                index_end = (
                    offset
                    + _RICE_HEADER.size
                    + unary_bytes
                    + (count * k + 7) // 8
                )
                if index_end > len(data) - CRC.size:
                    raise ValueError("Rice index stream is truncated")
                indices = _decode_rice_gaps(
                    data[offset:index_end], count, numel
                )
                offset = index_end
        elif index_width == INDEX_CODEC_BITMAP:
            index_end = offset + (numel + 7) // 8
            if index_end > len(data) - CRC.size:
                raise ValueError("Sparse index bitmap is truncated")
            mask = np.unpackbits(
                np.frombuffer(data[offset:index_end], dtype=np.uint8),
                count=numel,
                bitorder="little",
            )
            indices = np.flatnonzero(mask).astype(np.int64, copy=False)
            offset = index_end
            if indices.size != count:
                raise ValueError("Index bitmap does not match the declared count")
        else:
            index_dtype = np.dtype("<u4") if index_width == 4 else np.dtype("<u8")
            index_end = offset + count * index_width
            if index_end > len(data) - CRC.size:
                raise ValueError("Sparse indices are truncated")
            indices = np.frombuffer(
                data[offset:index_end], dtype=index_dtype
            ).astype(np.int64, copy=True)
            offset = index_end
            if indices.size and (
                int(indices.min()) < 0
                or int(indices.max()) >= numel
                or len(np.unique(indices)) != len(indices)
            ):
                raise ValueError("Sparse indices are invalid")
        if encoding in ENCODING_BITS:
            bits = ENCODING_BITS[encoding]
            group_size = 0
            codebook = encoding in GAUSS4_ENCODINGS.values()
            byte_scales = encoding in LOG8_ENCODINGS.values() or encoding == GAUSS4_ENCODINGS["log8"]
            if codebook or byte_scales or encoding in GROUPED_ENCODINGS.values():
                if offset + 4 > len(data) - CRC.size:
                    raise ValueError("Quantization group header is truncated")
                group_size = struct.unpack_from("<I", data, offset)[0]
                offset += 4
                if group_size == 0:
                    raise ValueError("Grouped encoding requires a positive group size")
            scale_count = max(1, (count + group_size - 1) // group_size) if group_size else 1
            if byte_scales:
                scale_end = offset + 4 + scale_count
                if scale_end > len(data) - CRC.size:
                    raise ValueError("Quantized scales are truncated")
                reference = struct.unpack_from("<f", data, offset)[0]
                if not math.isfinite(reference) or reference <= 0:
                    raise ValueError("Quantized scale must be finite and positive")
                scales = decode_log8_scales(
                    reference,
                    np.frombuffer(data[offset + 4:scale_end], dtype=np.uint8),
                ).numpy()
            else:
                scale_end = offset + 4 * scale_count
                if scale_end > len(data) - CRC.size:
                    raise ValueError("Quantized scales are truncated")
                scales = np.frombuffer(data[offset:scale_end], dtype="<f4")
            if not np.isfinite(scales).all() or np.any(scales <= 0):
                raise ValueError("Quantized scale must be finite and positive")
            offset = scale_end
            value_end = offset + (count * bits + 7) // 8
            if value_end > len(data) - CRC.size:
                raise ValueError("Quantized values are truncated")
            spread = np.repeat(scales, group_size)[:count] if group_size else scales[0]
            if codebook:
                values = dequantize_gaussian4(
                    unpack_codes2(data[offset:value_end], count),
                    torch.from_numpy(np.ascontiguousarray(spread, dtype=np.float32)),
                ).numpy()
            else:
                values = unpack_signed(data[offset:value_end], count, bits) * spread
            if flags & ROTATION_FLAG:
                if not group_size:
                    raise ValueError("Rotated payloads require a grouped scale")
                values = rotate_groups(
                    torch.from_numpy(np.ascontiguousarray(values, dtype=np.float32)),
                    group_size=group_size,
                    client_id=int(client_id),
                    server_round=int(server_round),
                    name=name,
                    inverse=True,
                ).numpy()
            offset = value_end
        elif encoding == ENCODING_SPARSE_FLOAT32:
            value_end = offset + count * 4
            if value_end > len(data) - CRC.size:
                raise ValueError("Sparse float32 values are truncated")
            values = np.frombuffer(data[offset:value_end], dtype="<f4").copy()
            offset = value_end
        else:
            raise ValueError("Unknown sparse encoding")
        flat = np.zeros(numel, dtype=np.float32)
        flat[indices] = values
        state[name] = torch.from_numpy(flat).reshape(descriptor["shape"])

    if offset != len(data) - CRC.size:
        raise ValueError("FedMPSQ payload has trailing or misaligned bytes")
    if total_parameters != declared_parameters or stored_values != declared_values:
        raise ValueError("FedMPSQ payload metadata counts do not match its body")
    require_finite_state(state, label="Decoded payload")
    return DecodedPayload(
        state=state,
        client_id=int(client_id),
        server_round=int(server_round),
        num_examples=int(num_examples),
        num_parameters=int(declared_parameters),
        stored_values=int(declared_values),
        declared_sparsity=float(declared_sparsity),
        alpha_s=float(alpha_s),
        flags=int(flags),
    )


def _compress_update_unbudgeted(
    update: Mapping[str, torch.Tensor],
    saliency: Mapping[str, torch.Tensor],
    residual: Mapping[str, torch.Tensor],
    *,
    uses_sparse: bool,
    uses_saliency: bool,
    uses_int8: bool,
    uses_error_feedback: bool,
    sparsity: float,
    alpha_s: float,
    epsilon: float,
    client_id: int,
    server_round: int,
    num_examples: int,
    dense_names: frozenset[str] | None = None,
    quant_bits: int = 8,
    index_codec: str = "rice",
    stochastic_rounding: bool = False,
    group_size: int = 0,
    clipping: str = "max",
    tensor_bits: Mapping[str, int] | None = None,
    adaptive_error: float | None = None,
    rounding_seed: int = 0,
    compact_layout: bool = False,
    block_size: int = 1,
    protected_groups: Sequence[Mapping[str, torch.Tensor]] = (),
    reserve_fraction: float = 0.0,
    rotate: bool = False,
    scale_codec: str = "fp32",
    quantizer: str = "integer",
) -> CompressionResult:
    """Execute the A0--A5 compressor and Eq. (11).

    ``dense_names`` lists tensors that are transmitted without sparsification.
    Non-gradient buffers such as the BatchNorm running statistics belong there:
    they are two orders of magnitude larger than a weight delta, so leaving
    them in the Top-K pool makes a single buffer coordinate set
    ``magnitude.max()`` in Eq. (7) and drives every weight score to about
    1e-2, which turns the ``alpha_s`` blend into pure saliency selection. Their
    coordinate count is subtracted from the Top-K budget, so the density of the
    complete payload still matches ``sparsity``.
    """
    if uses_int8 and not uses_sparse:
        raise ValueError("MVP INT8 is defined only for sparse A4/A5 payloads")
    if uses_error_feedback and not uses_sparse:
        raise ValueError("Error feedback requires sparse compression")
    if quant_bits not in {2, 4, 8, 32}:
        raise ValueError("quant_bits must be 2, 4, 8 or 32")
    if quant_bits == 32 and uses_int8:
        raise ValueError("FP32 transport must disable integer quantization")
    # One stream per (client, round) so a rerun of the campaign reproduces the
    # same rounding decisions, and two clients in the same round never share
    # one. Seeding from the pair rather than a global counter also keeps the
    # result independent of how many clients ran in parallel.
    generator: torch.Generator | None = None
    if stochastic_rounding:
        generator = torch.Generator()
        # CPU generators can alias seeds that differ only in high 32 bits.
        # Mix all identifiers into the low bits, including the experiment seed.
        generator.manual_seed(
            (int(rounding_seed) * 1000003 + int(server_round) * 9176
             + int(client_id) * 6361) % (2**32)
        )
    require_finite_state(update, label="Client update")
    require_finite_state(saliency, label="Client saliency")
    require_finite_state(residual, label="Client residual")
    _require_matching_float_layout(update, residual, label="Residual")
    if uses_saliency:
        _require_matching_float_layout(update, saliency, label="Saliency")
    update_cpu = {
        name: value.detach().cpu().to(torch.float32)
        for name, value in update.items()
    }
    dense_reference = serialize_dense_state(
        update_cpu,
        client_id=client_id,
        server_round=server_round,
        num_examples=num_examples,
        flags=1,
    )
    reserved = frozenset(dense_names or frozenset()) & set(update_cpu)

    # Eq. (11): the memory carries whatever the message could not express, so
    # it is added *before* the mask is chosen. Compensating after Top-K leaves
    # the sparsification error -- about 18 percent of the update norm at
    # sparsity 0.9 -- permanently uncorrected, and injects the memory's own
    # support into the payload, which pushed the transmitted density from the
    # declared 0.10 up to 0.20 over twenty rounds.
    if uses_error_feedback:
        compensated_input = {
            name: value + residual[name].detach().cpu().to(torch.float32)
            for name, value in update_cpu.items()
        }
    else:
        compensated_input = clone_tensor_state(update_cpu)

    if uses_sparse:
        total_parameters = sum(value.numel() for value in compensated_input.values())
        reserved_parameters = sum(
            compensated_input[name].numel() for name in reserved
        )
        pool = {
            name: value
            for name, value in compensated_input.items()
            if name not in reserved
        }
        if not pool:
            raise ValueError("Every update tensor was reserved for dense transport")
        budget = max(1, int(math.ceil((1.0 - sparsity) * total_parameters)))
        if budget - reserved_parameters < 1:
            raise ValueError(
                "Densely transmitted buffers fill the whole Top-K budget: "
                f"{reserved_parameters} reserved coordinates against a budget "
                f"of {budget} at sparsity {sparsity}. Lower the sparsity or "
                "use a buffer-free normalisation."
            )
        pool_topk = global_topk(
            pool,
            saliency,
            sparsity=sparsity,
            alpha_s=alpha_s,
            use_saliency=uses_saliency,
            epsilon=epsilon,
            keep=budget - reserved_parameters,
            block_size=block_size,
            protected_groups=protected_groups,
            reserve_fraction=reserve_fraction,
        )
        sparse_state = dict(pool_topk.sparse_state)
        for name in reserved:
            sparse_state[name] = compensated_input[name].clone()
        topk = TopKResult(
            sparse_state=sparse_state,
            selected_values=pool_topk.selected_values + reserved_parameters,
            num_parameters=total_parameters,
            squared_error=state_squared_error(compensated_input, sparse_state),
            squared_norm=state_squared_norm(compensated_input),
        )
        compensated = topk.sparse_state
    else:
        topk = TopKResult(
            sparse_state=clone_tensor_state(compensated_input),
            selected_values=sum(value.numel() for value in compensated_input.values()),
            num_parameters=sum(value.numel() for value in compensated_input.values()),
            squared_error=0.0,
            squared_norm=state_squared_norm(compensated_input),
        )
        compensated = topk.sparse_state
    sparse_update = topk.sparse_state

    if uses_sparse:
        payload = serialize_sparse_state(
            compensated,
            quantized=uses_int8,
            client_id=client_id,
            server_round=server_round,
            num_examples=num_examples,
            declared_sparsity=sparsity,
            alpha_s=alpha_s if uses_saliency else 0.0,
            flags=1,
            quant_bits=8 if quant_bits == 32 else quant_bits,
            index_codec=index_codec,
            stochastic=stochastic_rounding,
            generator=generator,
            group_size=group_size,
            clipping=clipping,
            tensor_bits=tensor_bits,
            raw_names=reserved,
            adaptive_error=adaptive_error,
            compact_layout=compact_layout,
            rotate=rotate,
            scale_codec=scale_codec,
            quantizer=quantizer,
        )
    else:
        payload = serialize_dense_state(
            compensated,
            client_id=client_id,
            server_round=server_round,
            num_examples=num_examples,
            flags=1,
        )
    decoded = decode_payload(payload.data).state
    quantization_error = state_squared_error(compensated, decoded)
    quantization_norm = state_squared_norm(compensated)
    if uses_error_feedback:
        residual_new = {
            name: compensated_input[name] - decoded[name]
            for name in compensated_input
        }
    else:
        residual_new = {
            name: torch.zeros_like(value)
            for name, value in compensated.items()
        }
    require_finite_state(residual_new, label="Updated residual")
    transmitted = payload.stored_values
    update_sparsity = 1.0 - (
        sum(int(torch.count_nonzero(value)) for value in sparse_update.values())
        / max(topk.num_parameters, 1)
    )
    payload_sparsity = 1.0 - transmitted / max(topk.num_parameters, 1)
    sparsification_l2_error = math.sqrt(topk.squared_error)
    sparsification_l2_norm = math.sqrt(topk.squared_norm)
    quantization_l2_error = math.sqrt(quantization_error)
    quantization_l2_norm = math.sqrt(quantization_norm)
    metrics = {
        "num_parameters": float(topk.num_parameters),
        "topk_selected_values": float(topk.selected_values),
        "transmitted_values": float(transmitted),
        "update_sparsity": float(update_sparsity),
        "payload_sparsity": float(payload_sparsity),
        "sparsification_squared_error": float(topk.squared_error),
        "sparsification_squared_norm": float(topk.squared_norm),
        "sparsification_l2_error": float(sparsification_l2_error),
        "sparsification_l2_norm": float(sparsification_l2_norm),
        "sparsification_relative_error": float(
            topk.squared_error / (topk.squared_norm + epsilon)
        ),
        "sparsification_relative_l2_error": float(
            sparsification_l2_error / (sparsification_l2_norm + epsilon)
        ),
        "quantization_squared_error": float(quantization_error),
        "quantization_squared_norm": float(quantization_norm),
        "quantization_l2_error": float(quantization_l2_error),
        "quantization_l2_norm": float(quantization_l2_norm),
        "quantization_relative_error": float(
            quantization_error / (quantization_norm + epsilon)
        ),
        "quantization_relative_l2_error": float(
            quantization_l2_error / (quantization_l2_norm + epsilon)
        ),
        "residual_l2_norm": float(math.sqrt(state_squared_norm(residual_new))),
        "serialized_uplink_bytes": float(payload.payload_bytes),
        "theoretical_uplink_bytes": float(payload.theoretical_bytes),
        "theoretical_ideal_index_uplink_bytes": float(
            payload.theoretical_ideal_index_bytes
        ),
        "dense_reference_serialized_bytes": float(dense_reference.payload_bytes),
        "dense_reference_raw_bytes": float(
            sum(value.numel() * value.element_size() for value in update_cpu.values())
        ),
        "compression_ratio_vs_serialized_dense": float(
            dense_reference.payload_bytes / max(payload.payload_bytes, 1)
        ),
    }
    metrics.update(
        {
            f"uplink_{key}": float(value)
            for key, value in payload.byte_breakdown.as_dict().items()
        }
    )
    return CompressionResult(
        payload=payload,
        decoded_update=decoded,
        residual_state=residual_new,
        metrics=metrics,
    )


def compress_update(update, saliency, residual, *, uplink_budget_bytes=None, **kwargs):
    """Bound the complete serialized client message, retaining original EF.

    Search K within the requested density ceiling; every candidate uses the
    same input residual. Only the selected payload/residual is committed.
    Codec switches make byte size non-monotone, so this bounded search is a
    heuristic, not a proof of globally optimal K. Feasibility is exact.
    """
    if uplink_budget_bytes is None:
        return _compress_update_unbudgeted(update, saliency, residual, **kwargs)
    if type(uplink_budget_bytes) is not int or uplink_budget_bytes <= 0:
        raise ValueError("Invalid serialized uplink budget")
    if not kwargs.get("uses_sparse"):
        raise ValueError("Byte budget requires sparse updates")
    total = sum(v.numel() for v in update.values())
    dense = set(kwargs.get("dense_names") or ()) & set(update)
    minimum = 1 + sum(update[name].numel() for name in dense)
    maximum = max(1, math.ceil((1 - kwargs["sparsity"]) * total))
    if minimum > maximum:
        raise ValueError("Density ceiling cannot accommodate dense buffers")
    attempted = {}
    best = None
    best_error = float("inf")
    def evaluate(keep):
        nonlocal best, best_error
        if keep in attempted:
            return attempted[keep]
        options = dict(kwargs)
        # Interior of the ceil interval avoids floating-point K+1 artifacts.
        options["sparsity"] = 1 - (keep - .25) / total
        result = _compress_update_unbudgeted(update, saliency, residual, **options)
        assert int(result.metrics["topk_selected_values"]) == keep
        size = len(result.payload.data)
        attempted[keep] = size
        if size <= uplink_budget_bytes:
            error = state_squared_norm(result.residual_state) if kwargs.get("uses_error_feedback") else state_squared_error(update, result.decoded_update)
            if error < best_error:
                best, best_error = result, error
                best.metrics["effective_sparsity"] = options["sparsity"]
        return size
    if evaluate(maximum) > uplink_budget_bytes:
        if evaluate(minimum) > uplink_budget_bytes:
            raise ValueError("Uplink budget is smaller than the minimum complete payload")
        lo, hi = minimum, maximum - 1
        # Ten bisections plus one proportional starting point bound CPU cost.
        probe = max(lo, min(hi, int(maximum * uplink_budget_bytes / attempted[maximum])))
        for _ in range(11):
            if lo > hi:
                break
            size = evaluate(probe)
            if size <= uplink_budget_bytes:
                lo = probe + 1
            else:
                hi = probe - 1
            probe = (lo + hi) // 2
    assert best is not None and len(best.payload.data) <= uplink_budget_bytes
    best.metrics.update(uplink_budget_bytes=float(uplink_budget_bytes),
                        uplink_budget_utilization=len(best.payload.data)/uplink_budget_bytes,
                        budget_codec_trials=float(len(attempted)))
    return best


def aggregate_payloads(
    global_state: Mapping[str, torch.Tensor],
    payloads: list[bytes],
) -> tuple[dict[str, torch.Tensor], list[DecodedPayload]]:
    """Sample-weighted FedAvg/FedProx update aggregation from wire bytes."""
    if not payloads:
        raise ValueError("No client payloads to aggregate")
    decoded = [decode_payload(payload) for payload in payloads]
    return aggregate_decoded_payloads(global_state, decoded), decoded


def aggregate_decoded_payloads(
    global_state: Mapping[str, torch.Tensor],
    decoded: list[DecodedPayload],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Aggregate already-decoded messages so decode time remains auditable.

    ``weights`` overrides sample weighting. Uniform weights give clients equal
    influence; this is not class-balanced weighting and does not optimize F1.
    """
    if not decoded:
        raise ValueError("No decoded client payloads to aggregate")
    if any(item.num_examples <= 0 for item in decoded):
        raise ValueError("Each aggregated client payload must have positive examples")
    client_ids = [item.client_id for item in decoded]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError("Aggregated payloads contain duplicate client IDs")
    server_rounds = {item.server_round for item in decoded}
    if len(server_rounds) != 1:
        raise ValueError("Aggregated payloads come from different server rounds")
    total_examples = sum(item.num_examples for item in decoded)
    if total_examples <= 0:
        raise ValueError("Cannot aggregate payloads with zero examples")
    if weights is None:
        shares = [item.num_examples / total_examples for item in decoded]
    else:
        shares = [float(value) for value in weights]
        if len(shares) != len(decoded):
            raise ValueError("One aggregation weight is required per payload")
        if any(not math.isfinite(value) or value < 0.0 for value in shares):
            raise ValueError("Aggregation weights must be finite and non-negative")
        total_weight = sum(shares)
        if total_weight <= 0.0:
            raise ValueError("Aggregation weights must sum to a positive value")
        shares = [value / total_weight for value in shares]
    communicated_names = set(decoded[0].state)
    if any(set(item.state) != communicated_names for item in decoded[1:]):
        raise ValueError("Client payload layouts differ")
    for item in decoded:
        require_finite_state(item.state, label=f"Client {item.client_id} payload")
    result = clone_tensor_state(global_state)
    for name in communicated_names:
        if name not in global_state:
            raise ValueError(f"Payload contains unknown tensor: {name}")
        reference_shape = tuple(global_state[name].shape)
        if any(tuple(item.state[name].shape) != reference_shape for item in decoded):
            raise ValueError(f"Payload tensor shape differs for {name}")
        mean_update = sum(
            item.state[name].to(torch.float64) * share
            for item, share in zip(decoded, shares, strict=True)
        )
        result[name] = (
            global_state[name].detach().cpu().to(torch.float64) + mean_update
        ).to(global_state[name].dtype)
    return result

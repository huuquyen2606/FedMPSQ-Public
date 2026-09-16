"""FedPAQ and DAdaQuant stochastic update quantization."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass

import numpy as np
import torch


StateDict = dict[str, torch.Tensor]

_PAYLOAD_MAGIC = b"QSGD"
_PAYLOAD_VERSION = 1
_ENCODING_FIXED_WIDTH = 0
_ENCODING_ELIAS_OMEGA = 1
_LAYOUT_FORMAT = "torch-state-update-v1"

# magic, version, encoding, magnitude width, reserved, levels, client id,
# server round, client example count (n_k), number of quantized float values,
# encoded bit length, layout byte length, raw auxiliary byte length. Explicit
# little-endian, standard-size fields avoid native alignment and make the
# header byte count deterministic.
_PAYLOAD_HEADER = struct.Struct("<4sBBBBIIIQQQIQ")
_NORM_FIELD = struct.Struct("<f")
_DENSE_PAYLOAD_MAGIC = b"FDNS"
_DENSE_PAYLOAD_VERSION = 1
# magic, version, reserved, flags, client id, server round, n_k, raw value
# length, layout length.
_DENSE_PAYLOAD_HEADER = struct.Struct("<4sBBHIIQQI")


@dataclass(frozen=True)
class QuantizedTensorLayout:
    """One floating-point tensor's position in the flattened update vector."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    num_values: int


@dataclass(frozen=True)
class RawTensorLayout:
    """One non-floating state tensor serialized verbatim after the bitstream."""

    name: str
    shape: tuple[int, ...]
    dtype: str
    wire_dtype: str
    num_values: int
    byte_offset: int
    byte_length: int


@dataclass(frozen=True)
class QuantizedPayloadByteCounts:
    """Mutually exclusive byte components of one complete wire payload."""

    header_bytes: int
    norm_bytes: int
    layout_bytes: int
    quantized_value_bytes: int
    index_bytes: int
    auxiliary_value_bytes: int

    @property
    def value_bytes(self) -> int:
        """Primary encoded value bytes; auxiliary tensor bytes are separate."""
        return self.quantized_value_bytes

    @property
    def total_bytes(self) -> int:
        return (
            self.header_bytes
            + self.norm_bytes
            + self.layout_bytes
            + self.quantized_value_bytes
            + self.index_bytes
            + self.auxiliary_value_bytes
        )

    def as_dict(self) -> dict[str, int]:
        """Return JSON-ready, mutually exclusive component counts and total."""
        return {
            "header_bytes": self.header_bytes,
            "norm_bytes": self.norm_bytes,
            "layout_bytes": self.layout_bytes,
            "value_bytes": self.quantized_value_bytes,
            "index_bytes": self.index_bytes,
            "auxiliary_bytes": self.auxiliary_value_bytes,
            "total_bytes": self.total_bytes,
        }


@dataclass(frozen=True)
class DecodedQuantizedPayload:
    """Self-contained result of decoding a serialized QSGD client update."""

    values: np.ndarray
    auxiliary_state: StateDict
    levels: int
    norm: float
    num_values: int
    encoding: str
    magnitude_width: int
    tensor_layout: tuple[QuantizedTensorLayout, ...]
    raw_tensor_layout: tuple[RawTensorLayout, ...]
    byte_counts: QuantizedPayloadByteCounts
    client_id: int
    server_round: int
    num_examples: int


@dataclass(frozen=True)
class SerializedDenseState:
    """A dense state dict serialized with the same accounting contract."""

    state: StateDict
    payload: bytes
    byte_counts: QuantizedPayloadByteCounts
    client_id: int
    server_round: int
    num_examples: int

    @property
    def payload_bytes(self) -> int:
        if len(self.payload) != self.byte_counts.total_bytes:
            raise ValueError("Dense payload component counts do not sum to payload size")
        return self.byte_counts.total_bytes


@dataclass(frozen=True)
class QuantizedStateUpdate:
    """A decoded update plus its complete, self-describing wire payload."""

    state: StateDict
    payload: bytes
    levels: int
    norm: float
    num_values: int
    nonzero_values: int
    squared_error: float
    squared_norm: float
    encoding: str
    magnitude_width: int
    tensor_layout: tuple[QuantizedTensorLayout, ...]
    raw_tensor_layout: tuple[RawTensorLayout, ...]
    byte_counts: QuantizedPayloadByteCounts
    client_id: int
    server_round: int
    num_examples: int

    @property
    def payload_bytes(self) -> int:
        if len(self.payload) != self.byte_counts.total_bytes:
            raise ValueError("Quantized payload component counts do not sum to payload size")
        return self.byte_counts.total_bytes

    @property
    def effective_bits_per_value(self) -> float:
        return (8.0 * len(self.payload)) / max(self.num_values, 1)

    @property
    def scale(self) -> float:
        return self.norm / float(self.levels)


def _state_layout(
    state: StateDict,
    quantized_names: set[str] | None = None,
) -> tuple[tuple[QuantizedTensorLayout, ...], int]:
    layout: list[QuantizedTensorLayout] = []
    total = 0
    for name, tensor in state.items():
        if torch.is_floating_point(tensor) and (
            quantized_names is None or name in quantized_names
        ):
            count = tensor.numel()
            layout.append(
                QuantizedTensorLayout(
                    name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    num_values=count,
                )
            )
            total += count
    return tuple(layout), total


def _serialize_tensor_state(
    state: StateDict,
    *,
    include_floating: bool,
    excluded_names: set[str] | None = None,
) -> tuple[tuple[RawTensorLayout, ...], bytes]:
    layout: list[RawTensorLayout] = []
    chunks: list[bytes] = []
    offset = 0
    for name, tensor in state.items():
        if excluded_names is not None and name in excluded_names:
            continue
        if torch.is_floating_point(tensor) and not include_floating:
            continue
        try:
            array = tensor.detach().cpu().contiguous().numpy()
        except (TypeError, RuntimeError) as exc:
            raise TypeError(
                f"Cannot serialize non-floating state tensor {name!r} with dtype "
                f"{tensor.dtype}"
            ) from exc
        wire_dtype = array.dtype.newbyteorder("<")
        wire_array = array.astype(wire_dtype, copy=False)
        chunk = wire_array.tobytes(order="C")
        spec = RawTensorLayout(
            name=name,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            wire_dtype=wire_dtype.str,
            num_values=tensor.numel(),
            byte_offset=offset,
            byte_length=len(chunk),
        )
        layout.append(spec)
        chunks.append(chunk)
        offset += len(chunk)
    return tuple(layout), b"".join(chunks)


def _serialize_raw_state(
    state: StateDict,
    *,
    quantized_names: set[str] | None = None,
) -> tuple[tuple[RawTensorLayout, ...], bytes]:
    return _serialize_tensor_state(
        state,
        include_floating=quantized_names is not None,
        excluded_names=quantized_names,
    )


def _serialize_layout(
    tensor_layout: tuple[QuantizedTensorLayout, ...],
    raw_tensor_layout: tuple[RawTensorLayout, ...],
) -> bytes:
    document = {
        "float_tensors": [
            {
                "dtype": spec.dtype,
                "name": spec.name,
                "num_values": spec.num_values,
                "shape": list(spec.shape),
            }
            for spec in tensor_layout
        ],
        "format": _LAYOUT_FORMAT,
        "raw_tensors": [
            {
                "byte_length": spec.byte_length,
                "byte_offset": spec.byte_offset,
                "dtype": spec.dtype,
                "name": spec.name,
                "num_values": spec.num_values,
                "shape": list(spec.shape),
                "wire_dtype": spec.wire_dtype,
            }
            for spec in raw_tensor_layout
        ],
    }
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _valid_shape(value: object) -> bool:
    return isinstance(value, list) and all(
        isinstance(dimension, int)
        and not isinstance(dimension, bool)
        and dimension >= 0
        for dimension in value
    )


def _parse_layout(
    payload: bytes,
    *,
    num_values: int,
    auxiliary_bytes: int,
) -> tuple[tuple[QuantizedTensorLayout, ...], tuple[RawTensorLayout, ...]]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid quantized payload layout metadata") from exc
    if not isinstance(document, dict) or document.get("format") != _LAYOUT_FORMAT:
        raise ValueError("Unsupported quantized payload layout format")
    float_entries = document.get("float_tensors")
    raw_entries = document.get("raw_tensors")
    if not isinstance(float_entries, list) or not isinstance(raw_entries, list):
        raise ValueError("Quantized payload layout must contain tensor lists")

    tensor_layout: list[QuantizedTensorLayout] = []
    seen_names: set[str] = set()
    float_total = 0
    for entry in float_entries:
        if not isinstance(entry, dict) or not _valid_shape(entry.get("shape")):
            raise ValueError("Invalid floating tensor layout entry")
        name = entry.get("name")
        dtype = entry.get("dtype")
        count = entry.get("num_values")
        if (
            not isinstance(name, str)
            or name in seen_names
            or not isinstance(dtype, str)
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError("Invalid floating tensor layout fields")
        shape = tuple(entry["shape"])
        if math.prod(shape) != count:
            raise ValueError("Floating tensor shape does not match num_values")
        seen_names.add(name)
        float_total += count
        tensor_layout.append(QuantizedTensorLayout(name, shape, dtype, count))
    if float_total != num_values:
        raise ValueError("Tensor layout does not match quantized vector length")

    raw_tensor_layout: list[RawTensorLayout] = []
    next_offset = 0
    for entry in raw_entries:
        if not isinstance(entry, dict) or not _valid_shape(entry.get("shape")):
            raise ValueError("Invalid raw tensor layout entry")
        name = entry.get("name")
        dtype = entry.get("dtype")
        wire_dtype = entry.get("wire_dtype")
        count = entry.get("num_values")
        offset = entry.get("byte_offset")
        length = entry.get("byte_length")
        integer_fields = (count, offset, length)
        if (
            not isinstance(name, str)
            or name in seen_names
            or not isinstance(dtype, str)
            or not isinstance(wire_dtype, str)
            or any(
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < 0
                for value in integer_fields
            )
            or offset != next_offset
        ):
            raise ValueError("Invalid raw tensor layout fields")
        shape = tuple(entry["shape"])
        try:
            numpy_dtype = np.dtype(wire_dtype)
        except TypeError as exc:
            raise ValueError("Unsupported raw tensor wire dtype") from exc
        if math.prod(shape) != count or count * numpy_dtype.itemsize != length:
            raise ValueError("Raw tensor shape or dtype does not match byte length")
        seen_names.add(name)
        next_offset += length
        raw_tensor_layout.append(
            RawTensorLayout(name, shape, dtype, wire_dtype, count, offset, length)
        )
    if next_offset != auxiliary_bytes:
        raise ValueError("Raw tensor layout does not consume auxiliary payload")
    return tuple(tensor_layout), tuple(raw_tensor_layout)


def _flatten_float_update(
    local_state: StateDict,
    global_state: StateDict,
    quantized_names: set[str] | None = None,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for name, global_tensor in global_state.items():
        if torch.is_floating_point(global_tensor) and (
            quantized_names is None or name in quantized_names
        ):
            delta = local_state[name].detach().cpu() - global_tensor.detach().cpu()
            values.append(delta.reshape(-1).to(torch.float32).numpy())
    if not values:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)


def _elias_omega_bits(value: int) -> list[int]:
    bits = [0]
    while value > 0:
        value_bits = [int(bit) for bit in bin(value)[2:]]
        bits = value_bits + bits
        value = len(value_bits) - 1
    return bits


def _encode_fixed_width(
    magnitudes: np.ndarray,
    signs: np.ndarray,
    levels: int,
) -> tuple[bytes, int, int]:
    magnitude_width = max(1, int(levels).bit_length())
    bits: list[int] = []
    for magnitude, sign in zip(magnitudes.tolist(), signs.tolist(), strict=True):
        bits.extend(int(bit) for bit in f"{int(magnitude):0{magnitude_width}b}")
        bits.append(int(sign))
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()
    return packed, len(bits), magnitude_width


def _decode_fixed_width(
    payload: bytes,
    *,
    bit_length: int,
    levels: int,
    norm: float,
    num_values: int,
    magnitude_width: int,
) -> np.ndarray:
    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="big",
    )[:bit_length]
    stride = magnitude_width + 1
    if bit_length != stride * num_values:
        raise ValueError("Invalid fixed-width QSGD payload length")
    decoded = np.empty(num_values, dtype=np.float32)
    for index in range(num_values):
        start = index * stride
        magnitude = 0
        for bit in bits[start : start + magnitude_width]:
            magnitude = (magnitude << 1) | int(bit)
        sign = int(bits[start + magnitude_width])
        decoded[index] = magnitude * (1.0 - 2.0 * sign) * norm / float(levels)
    return decoded


def _encode_qsgd_lossless(
    magnitudes: np.ndarray,
    signs: np.ndarray,
) -> tuple[bytes, int]:
    bits: list[int] = []
    zero_run = 0
    last_index = len(magnitudes) - 1
    for index, (magnitude, sign) in enumerate(
        zip(magnitudes.tolist(), signs.tolist(), strict=True)
    ):
        if magnitude == 0 and index < last_index:
            zero_run += 1
            continue
        bits.extend(_elias_omega_bits(zero_run))
        zero_run = 0
        bits.extend(_elias_omega_bits(int(magnitude)))
        bits.append(int(sign))
    packed = np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="big").tobytes()
    return packed, len(bits)


def _decode_elias_omega(bits: np.ndarray, index: int) -> tuple[int, int]:
    value = 0
    while True:
        if index >= len(bits):
            raise ValueError("Truncated Elias omega code")
        if int(bits[index]) == 0:
            return value, index + 1
        width = value + 1
        encoded = bits[index : index + width]
        if len(encoded) != width:
            raise ValueError("Truncated Elias omega code")
        value = 0
        for bit in encoded:
            value = (value << 1) | int(bit)
        index += width


def _decode_qsgd_lossless(
    payload: bytes,
    *,
    bit_length: int,
    levels: int,
    norm: float,
    num_values: int,
) -> np.ndarray:
    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8),
        bitorder="big",
    )[:bit_length]
    values: list[float] = []
    index = 0
    while index < bit_length:
        zero_run, index = _decode_elias_omega(bits, index)
        values.extend([0.0] * zero_run)
        magnitude, index = _decode_elias_omega(bits, index)
        if index >= bit_length:
            raise ValueError("Missing QSGD sign bit")
        sign = int(bits[index])
        index += 1
        values.append(magnitude * (1.0 - 2.0 * sign) * norm / float(levels))
    if len(values) != num_values:
        raise ValueError("Decoded QSGD payload has the wrong vector length")
    return np.asarray(values, dtype=np.float32)


def _validate_unsigned(value: int, *, name: str, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
        or value > maximum
    ):
        raise ValueError(f"{name} must be an integer in [0, {maximum}]")


def _decode_raw_state(
    payload: bytes,
    layout: tuple[RawTensorLayout, ...],
) -> StateDict:
    state: StateDict = {}
    for spec in layout:
        chunk = payload[spec.byte_offset : spec.byte_offset + spec.byte_length]
        array = np.frombuffer(chunk, dtype=np.dtype(spec.wire_dtype), count=spec.num_values)
        tensor = torch.from_numpy(array.reshape(spec.shape).copy())
        if str(tensor.dtype) != spec.dtype:
            raise ValueError(
                f"Raw tensor {spec.name!r} decoded as {tensor.dtype}, expected {spec.dtype}"
            )
        state[spec.name] = tensor
    return state


def _build_quantized_payload(
    *,
    norm: float,
    magnitudes: np.ndarray,
    signs: np.ndarray,
    levels: int,
    lossless_qsgd_encoding: bool,
    tensor_layout: tuple[QuantizedTensorLayout, ...],
    raw_tensor_layout: tuple[RawTensorLayout, ...],
    raw_state_payload: bytes,
    client_id: int,
    server_round: int,
    num_examples: int,
) -> tuple[bytes, QuantizedPayloadByteCounts]:
    _validate_unsigned(levels, name="levels", maximum=0xFFFFFFFF)
    if levels < 1:
        raise ValueError("QSGD levels must be >= 1")
    _validate_unsigned(client_id, name="client_id", maximum=0xFFFFFFFF)
    _validate_unsigned(server_round, name="server_round", maximum=0xFFFFFFFF)
    _validate_unsigned(num_examples, name="num_examples", maximum=0xFFFFFFFFFFFFFFFF)
    _validate_unsigned(
        int(magnitudes.size),
        name="num_values",
        maximum=0xFFFFFFFFFFFFFFFF,
    )
    if lossless_qsgd_encoding:
        value_payload, bit_length = _encode_qsgd_lossless(magnitudes, signs)
        encoding = _ENCODING_ELIAS_OMEGA
        magnitude_width = 0
    else:
        value_payload, bit_length, magnitude_width = _encode_fixed_width(
            magnitudes,
            signs,
            levels,
        )
        encoding = _ENCODING_FIXED_WIDTH
    layout_payload = _serialize_layout(tensor_layout, raw_tensor_layout)
    _validate_unsigned(bit_length, name="bit_length", maximum=0xFFFFFFFFFFFFFFFF)
    _validate_unsigned(
        len(layout_payload),
        name="layout_bytes",
        maximum=0xFFFFFFFF,
    )
    _validate_unsigned(
        len(raw_state_payload),
        name="auxiliary_bytes",
        maximum=0xFFFFFFFFFFFFFFFF,
    )
    header = _PAYLOAD_HEADER.pack(
        _PAYLOAD_MAGIC,
        _PAYLOAD_VERSION,
        encoding,
        magnitude_width,
        0,
        levels,
        client_id,
        server_round,
        num_examples,
        int(magnitudes.size),
        bit_length,
        len(layout_payload),
        len(raw_state_payload),
    )
    norm_payload = _NORM_FIELD.pack(float(norm))
    payload = header + norm_payload + layout_payload + value_payload + raw_state_payload
    byte_counts = QuantizedPayloadByteCounts(
        header_bytes=len(header),
        norm_bytes=len(norm_payload),
        layout_bytes=len(layout_payload),
        quantized_value_bytes=len(value_payload),
        # QSGD is a dense vector codec. Position is implicit in the serialized
        # tensor layout, so there is no sparse index array for the quantized baselines.
        index_bytes=0,
        auxiliary_value_bytes=len(raw_state_payload),
    )
    if byte_counts.total_bytes != len(payload):
        raise AssertionError("Internal QSGD payload byte accounting mismatch")
    return payload, byte_counts


def decode_quantized_payload(payload: bytes) -> DecodedQuantizedPayload:
    """Decode one QSGD message without external levels, length, or layout state."""
    minimum_size = _PAYLOAD_HEADER.size + _NORM_FIELD.size
    if len(payload) < minimum_size:
        raise ValueError("Truncated quantized payload header")
    (
        magic,
        version,
        encoding_id,
        magnitude_width,
        reserved,
        levels,
        client_id,
        server_round,
        num_examples,
        num_values,
        bit_length,
        layout_length,
        auxiliary_length,
    ) = _PAYLOAD_HEADER.unpack_from(payload, 0)
    if magic != _PAYLOAD_MAGIC or version != _PAYLOAD_VERSION:
        raise ValueError("Unsupported quantized payload magic or version")
    if reserved != 0:
        raise ValueError("Quantized payload reserved header byte must be zero")
    if levels < 1:
        raise ValueError("Quantized payload levels must be >= 1")
    norm = float(_NORM_FIELD.unpack_from(payload, _PAYLOAD_HEADER.size)[0])
    if not math.isfinite(norm) or norm < 0.0:
        raise ValueError("Quantized payload norm must be finite and non-negative")
    value_length = (bit_length + 7) // 8
    expected_length = minimum_size + layout_length + value_length + auxiliary_length
    if expected_length != len(payload):
        raise ValueError("Quantized payload length does not match its header")

    layout_start = minimum_size
    value_start = layout_start + layout_length
    auxiliary_start = value_start + value_length
    layout_payload = payload[layout_start:value_start]
    value_payload = payload[value_start:auxiliary_start]
    auxiliary_payload = payload[auxiliary_start:]
    tensor_layout, raw_tensor_layout = _parse_layout(
        layout_payload,
        num_values=num_values,
        auxiliary_bytes=auxiliary_length,
    )

    if encoding_id == _ENCODING_FIXED_WIDTH:
        expected_width = max(1, int(levels).bit_length())
        if magnitude_width != expected_width:
            raise ValueError("Fixed-width QSGD magnitude width does not match levels")
        values = _decode_fixed_width(
            value_payload,
            bit_length=bit_length,
            levels=levels,
            norm=norm,
            num_values=num_values,
            magnitude_width=magnitude_width,
        )
        encoding = "fixed_width"
    elif encoding_id == _ENCODING_ELIAS_OMEGA:
        if magnitude_width != 0:
            raise ValueError("Elias-omega QSGD must use zero fixed magnitude width")
        values = _decode_qsgd_lossless(
            value_payload,
            bit_length=bit_length,
            levels=levels,
            norm=norm,
            num_values=num_values,
        )
        encoding = "zero_rle_elias_omega"
    else:
        raise ValueError("Unsupported QSGD payload encoding")

    byte_counts = QuantizedPayloadByteCounts(
        header_bytes=_PAYLOAD_HEADER.size,
        norm_bytes=_NORM_FIELD.size,
        layout_bytes=layout_length,
        quantized_value_bytes=value_length,
        index_bytes=0,
        auxiliary_value_bytes=auxiliary_length,
    )
    return DecodedQuantizedPayload(
        values=values,
        auxiliary_state=_decode_raw_state(auxiliary_payload, raw_tensor_layout),
        levels=levels,
        norm=norm,
        num_values=num_values,
        encoding=encoding,
        magnitude_width=magnitude_width,
        tensor_layout=tensor_layout,
        raw_tensor_layout=raw_tensor_layout,
        byte_counts=byte_counts,
        client_id=client_id,
        server_round=server_round,
        num_examples=num_examples,
    )


def decode_dense_state_payload(payload: bytes) -> SerializedDenseState:
    """Decode a dense client-state message and validate every byte component."""
    if len(payload) < _DENSE_PAYLOAD_HEADER.size:
        raise ValueError("Truncated dense payload header")
    (
        magic,
        version,
        reserved,
        flags,
        client_id,
        server_round,
        num_examples,
        value_length,
        layout_length,
    ) = _DENSE_PAYLOAD_HEADER.unpack_from(payload, 0)
    if magic != _DENSE_PAYLOAD_MAGIC or version != _DENSE_PAYLOAD_VERSION:
        raise ValueError("Unsupported dense payload magic or version")
    if reserved != 0 or flags != 0:
        raise ValueError("Dense payload reserved header fields must be zero")
    expected_length = _DENSE_PAYLOAD_HEADER.size + layout_length + value_length
    if expected_length != len(payload):
        raise ValueError("Dense payload length does not match its header")
    layout_start = _DENSE_PAYLOAD_HEADER.size
    value_start = layout_start + layout_length
    tensor_layout, raw_tensor_layout = _parse_layout(
        payload[layout_start:value_start],
        num_values=0,
        auxiliary_bytes=value_length,
    )
    if tensor_layout:
        raise ValueError("Dense payload must encode all tensors as raw values")
    state = _decode_raw_state(payload[value_start:], raw_tensor_layout)
    byte_counts = QuantizedPayloadByteCounts(
        header_bytes=_DENSE_PAYLOAD_HEADER.size,
        norm_bytes=0,
        layout_bytes=layout_length,
        quantized_value_bytes=value_length,
        index_bytes=0,
        auxiliary_value_bytes=0,
    )
    return SerializedDenseState(
        state=state,
        payload=payload,
        byte_counts=byte_counts,
        client_id=client_id,
        server_round=server_round,
        num_examples=num_examples,
    )


def serialize_dense_state(
    state: StateDict,
    *,
    client_id: int,
    server_round: int,
    num_examples: int,
) -> SerializedDenseState:
    """Serialize a dense baseline upload, including layout and message metadata."""
    _validate_unsigned(client_id, name="client_id", maximum=0xFFFFFFFF)
    _validate_unsigned(server_round, name="server_round", maximum=0xFFFFFFFF)
    _validate_unsigned(num_examples, name="num_examples", maximum=0xFFFFFFFFFFFFFFFF)
    raw_layout, value_payload = _serialize_tensor_state(
        state,
        include_floating=True,
    )
    layout_payload = _serialize_layout((), raw_layout)
    _validate_unsigned(
        len(layout_payload),
        name="layout_bytes",
        maximum=0xFFFFFFFF,
    )
    _validate_unsigned(
        len(value_payload),
        name="value_bytes",
        maximum=0xFFFFFFFFFFFFFFFF,
    )
    header = _DENSE_PAYLOAD_HEADER.pack(
        _DENSE_PAYLOAD_MAGIC,
        _DENSE_PAYLOAD_VERSION,
        0,
        0,
        client_id,
        server_round,
        num_examples,
        len(value_payload),
        len(layout_payload),
    )
    payload = header + layout_payload + value_payload
    decoded = decode_dense_state_payload(payload)
    if tuple(decoded.state) != tuple(state):
        raise ValueError("Decoded dense state layout differs from the encoded layout")
    for name, tensor in state.items():
        if not torch.equal(decoded.state[name], tensor.detach().cpu()):
            raise ValueError(f"Dense tensor {name!r} changed during serialization")
    return decoded


def _stochastic_qsgd(
    vector: np.ndarray,
    levels: int,
    generator: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if levels < 1:
        raise ValueError("QSGD levels must be >= 1")
    norm = float(np.linalg.norm(vector))
    if norm == 0.0 or vector.size == 0:
        magnitudes = np.zeros(vector.size, dtype=np.int64)
        signs = np.zeros(vector.size, dtype=np.uint8)
        return vector.copy(), magnitudes, signs, norm
    scaled = np.abs(vector) * float(levels) / norm
    lower = np.floor(scaled).astype(np.int64)
    probabilities = scaled - lower
    rounded_up = generator.random(vector.size) <= probabilities
    magnitudes = np.minimum(lower + rounded_up.astype(np.int64), levels)
    signs = (vector < 0.0).astype(np.uint8)
    quantized = (
        norm
        * (1.0 - 2.0 * signs.astype(np.float32))
        * magnitudes.astype(np.float32)
        / float(levels)
    ).astype(np.float32)
    return quantized, magnitudes, signs, norm


def quantize_state_update(
    local_state: StateDict,
    global_state: StateDict,
    *,
    levels: int,
    seed: int,
    lossless_qsgd_encoding: bool,
    client_id: int = 0,
    server_round: int = 0,
    num_examples: int = 0,
    quantized_names: set[str] | None = None,
) -> QuantizedStateUpdate:
    """Quantize and serialize one complete FedPAQ/QSGD client message."""
    if quantized_names is not None:
        unknown_names = quantized_names - set(global_state)
        if unknown_names:
            raise ValueError(f"Unknown quantized state names: {sorted(unknown_names)}")
        nonfloating_names = {
            name for name in quantized_names
            if not torch.is_floating_point(global_state[name])
        }
        if nonfloating_names:
            raise ValueError(
                f"Quantized state names must be floating point: {sorted(nonfloating_names)}"
            )
    vector = _flatten_float_update(local_state, global_state, quantized_names)
    generator = np.random.default_rng(seed)
    _, magnitudes, signs, norm = _stochastic_qsgd(vector, levels, generator)
    tensor_layout, layout_values = _state_layout(global_state, quantized_names)
    if layout_values != vector.size:
        raise ValueError("Global state layout does not match flattened update length")
    raw_tensor_layout, raw_state_payload = _serialize_raw_state(
        local_state,
        quantized_names=quantized_names,
    )
    payload, expected_byte_counts = _build_quantized_payload(
        norm=norm,
        magnitudes=magnitudes,
        signs=signs,
        levels=levels,
        lossless_qsgd_encoding=lossless_qsgd_encoding,
        tensor_layout=tensor_layout,
        raw_tensor_layout=raw_tensor_layout,
        raw_state_payload=raw_state_payload,
        client_id=client_id,
        server_round=server_round,
        num_examples=num_examples,
    )
    decoded_payload = decode_quantized_payload(payload)
    if decoded_payload.tensor_layout != tensor_layout:
        raise ValueError("Decoded QSGD tensor layout differs from the encoded layout")
    if decoded_payload.raw_tensor_layout != raw_tensor_layout:
        raise ValueError("Decoded QSGD raw tensor layout differs from the encoded layout")
    if decoded_payload.byte_counts != expected_byte_counts:
        raise ValueError("Decoded QSGD byte counts differ from the encoded payload")
    decoded = decoded_payload.values

    reconstructed: StateDict = {}
    offset = 0
    float_specs = {spec.name: spec for spec in tensor_layout}
    for name, global_tensor in global_state.items():
        if name in float_specs:
            spec = float_specs[name]
            delta = torch.from_numpy(
                decoded[offset : offset + spec.num_values].reshape(spec.shape)
            )
            reconstructed[name] = delta.to(dtype=global_tensor.dtype)
            offset += spec.num_values
        else:
            try:
                reconstructed[name] = decoded_payload.auxiliary_state[name]
            except KeyError as exc:
                raise ValueError(f"Missing serialized raw state tensor {name!r}") from exc
    if offset != len(decoded):
        raise ValueError("Quantized update layout did not consume the complete vector")
    error = decoded.astype(np.float64) - vector.astype(np.float64)
    return QuantizedStateUpdate(
        state=reconstructed,
        payload=payload,
        levels=decoded_payload.levels,
        norm=decoded_payload.norm,
        num_values=decoded_payload.num_values,
        nonzero_values=int(np.count_nonzero(magnitudes)),
        squared_error=float(np.dot(error, error)),
        squared_norm=float(np.dot(vector.astype(np.float64), vector.astype(np.float64))),
        encoding=decoded_payload.encoding,
        magnitude_width=decoded_payload.magnitude_width,
        tensor_layout=decoded_payload.tensor_layout,
        raw_tensor_layout=decoded_payload.raw_tensor_layout,
        byte_counts=decoded_payload.byte_counts,
        client_id=decoded_payload.client_id,
        server_round=decoded_payload.server_round,
        num_examples=decoded_payload.num_examples,
    )


class DAdaQuantController:
    """DAdaQuant time- and client-adaptive level assignment."""

    def __init__(self, config) -> None:
        self.config = config
        self.base_level = int(config.min_level)
        self.moving_losses: list[float] = []
        self.last_increase_index = 0

    def observe_weighted_loss(self, loss: float) -> None:
        if not math.isfinite(float(loss)):
            raise FloatingPointError("DAdaQuant weighted loss must be finite")
        if not self.moving_losses:
            moving = float(loss)
        else:
            psi = float(self.config.moving_average)
            moving = psi * self.moving_losses[-1] + (1.0 - psi) * float(loss)
        self.moving_losses.append(moving)

    def level_for_next_round(self) -> int:
        interval = int(self.config.convergence_interval)
        num_losses = len(self.moving_losses)
        if (
            num_losses >= self.last_increase_index + interval
            and self.moving_losses[-1] > self.moving_losses[-interval]
        ):
            increased = min(self.base_level * 2, int(self.config.max_level))
            if increased > self.base_level:
                self.base_level = increased
                self.last_increase_index = num_losses
        return self.base_level

    @staticmethod
    def client_levels(
        base_level: int,
        example_counts: dict[int, int],
        *,
        min_level: int = 1,
        max_level: int | None = None,
    ) -> dict[int, int]:
        if (
            base_level < 1
            or min_level < 1
            or (max_level is not None and max_level < min_level)
        ):
            raise ValueError("DAdaQuant levels must satisfy 1 <= min <= max")
        if any(count <= 0 for count in example_counts.values()):
            raise ValueError("DAdaQuant requires positive client example counts")
        total = sum(example_counts.values())
        if total <= 0:
            raise ValueError("DAdaQuant requires positive client example counts")
        weights = {client_id: count / total for client_id, count in example_counts.items()}
        sum_two_thirds = sum(weight ** (2.0 / 3.0) for weight in weights.values())
        sum_squares = sum(weight**2 for weight in weights.values())
        scale = base_level * math.sqrt(sum_two_thirds / sum_squares)
        ceiling = max_level if max_level is not None else math.inf
        return {
            client_id: int(
                min(
                    ceiling,
                    max(min_level, round(scale * weight ** (2.0 / 3.0))),
                )
            )
            for client_id, weight in weights.items()
        }

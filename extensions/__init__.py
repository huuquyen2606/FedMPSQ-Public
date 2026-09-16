"""Paper-faithful federated extension primitives."""

from extensions.distillation import bidirectional_decoupled_losses
from extensions.quantization import (
    DAdaQuantController,
    DecodedQuantizedPayload,
    QuantizedPayloadByteCounts,
    QuantizedStateUpdate,
    SerializedDenseState,
    decode_dense_state_payload,
    decode_quantized_payload,
    quantize_state_update,
    serialize_dense_state,
)

__all__ = [
    "DAdaQuantController",
    "DecodedQuantizedPayload",
    "QuantizedPayloadByteCounts",
    "QuantizedStateUpdate",
    "SerializedDenseState",
    "bidirectional_decoupled_losses",
    "decode_dense_state_payload",
    "decode_quantized_payload",
    "quantize_state_update",
    "serialize_dense_state",
]

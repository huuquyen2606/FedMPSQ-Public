"""Send the client update as one flat vector instead of 24 named tensors.

Why this exists
---------------
`compress_update` is currently handed the model's `state_dict`, so every uplink
message re-sends a layout entry and a scale header for each of the 24 tensors.
That cost is fixed: it does not shrink when the byte budget shrinks. Measured on
the real 34-class model (`packet_budget_study.py`):

    budget 3334 B  ->  fixed metadata 602 B (18.1 %),  k =  9,342 coordinates
    budget 1716 B  ->  fixed metadata 582 B (34.0 %),  k =  3,784 coordinates
    budget  858 B  ->  fixed metadata 558 B (65.1 %),  k =  1,038 coordinates

The tensor layout is static and both sides already hold it (the server keeps
`transmitted_global_state`, the client receives it). Registering it once instead
of per message removes the whole cost:

    budget 1716 B  ->  fixed metadata 125 B (7.3 %),   k =  5,392 coordinates

That is +42.5 % coordinates at the 1/3 budget for zero loss in fidelity, because
nothing about the quantizer, the rotation or the selection changes -- only which
bytes of description ride along.

Guards
------
Flattening is only sound when no per-tensor mechanism is in play:
  * `dense_names` (non-trainable float buffers, i.e. BatchNorm statistics) are
    exempted from Top-K by name, which a flat vector cannot express;
  * `protected_groups` / `minority_head_reserve` address coordinates by tensor;
  * `tensor_bits` / `protect_small_tensors` sets precision per tensor.
Each of those raises rather than silently changing meaning.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

FLAT_KEY = "update"


def wire_layout(state: Mapping[str, torch.Tensor]) -> tuple[tuple[str, torch.Size], ...]:
    """Ordered (name, shape) of the floating tensors, in state_dict order.

    Both endpoints derive this from the model they already hold, so it costs no
    bytes. Ordering is the state_dict ordering, which PyTorch keeps stable for a
    fixed module definition.
    """
    return tuple((name, value.shape) for name, value in state.items()
                 if torch.is_floating_point(value))


def flatten_state(state: Mapping[str, torch.Tensor],
                  layout: Sequence[tuple[str, torch.Size]]) -> dict[str, torch.Tensor]:
    """Concatenate the floating tensors of `state` into a single wire tensor."""
    missing = [name for name, _ in layout if name not in state]
    if missing:
        raise ValueError(f"Flat uplink is missing tensors: {missing[:4]}")
    return {FLAT_KEY: torch.cat([state[name].reshape(-1) for name, _ in layout])}


def unflatten_state(flat: Mapping[str, torch.Tensor],
                    layout: Sequence[tuple[str, torch.Size]]) -> dict[str, torch.Tensor]:
    """Split the wire tensor back into the per-tensor state it came from."""
    if FLAT_KEY not in flat:
        raise ValueError(f"Flat uplink payload has no {FLAT_KEY!r} tensor")
    vector = flat[FLAT_KEY].reshape(-1)
    expected = sum(int(torch.Size(shape).numel()) for _, shape in layout)
    if vector.numel() != expected:
        raise ValueError(f"Flat uplink carries {vector.numel()} values, layout expects {expected}")
    out, offset = {}, 0
    for name, shape in layout:
        size = int(torch.Size(shape).numel())
        out[name] = vector[offset:offset + size].reshape(shape).clone()
        offset += size
    return out


def check_flat_uplink_is_sound(*, dense_names, protected_groups, tensor_bits) -> None:
    """Refuse to flatten when a per-tensor mechanism would lose its meaning."""
    if dense_names:
        raise ValueError(
            "Flat uplink cannot express dense_names (non-trainable float buffers). "
            "Use model.norm=layer, which has none, or keep the per-tensor wire.")
    if protected_groups:
        raise ValueError(
            "Flat uplink needs protected_groups remapped to flat offsets; "
            "set method.minority_head_reserve=0.0 or extend this helper.")
    if tensor_bits:
        raise ValueError(
            "Flat uplink cannot express per-tensor precision; "
            "set method.protect_small_tensors=false.")

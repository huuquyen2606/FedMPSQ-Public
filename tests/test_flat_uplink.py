"""The flat wire format must move bytes, not meaning.

Sending the update as one vector removes the per-tensor layout and scale
headers, which at a one-third packet budget are a third of the packet. These checks pin
the two properties that make that safe: the reconstruction is exact, and the
error-feedback identity the codec relies on is untouched.
"""
import unittest

import torch

from extensions.fedmpsq import compress_update, decode_payload
from extensions.flat_uplink import (
    FLAT_KEY,
    check_flat_uplink_is_sound,
    flatten_state,
    unflatten_state,
    wire_layout,
)
from fl.fedmpsq_config import load_fedmpsq_config
from models.dcnn_bilstm import DCNNBiLSTM

CODEC = dict(uses_sparse=True, uses_saliency=False, uses_int8=True,
             uses_error_feedback=True, sparsity=0.9, alpha_s=0.0, epsilon=1e-12,
             index_codec="auto_runs", stochastic_rounding=False, clipping="mse_refine",
             compact_layout=True, block_size=32, protected_groups=(), reserve_fraction=0.0,
             rotate=True, scale_codec="log8", quantizer="gaussian4", quant_bits=2,
             group_size=256)


def model_state():
    torch.manual_seed(0)
    model = DCNNBiLSTM(input_dim=46, num_classes=34, conv_channels=(64, 128, 128),
                       kernel_size=3, lstm_hidden_size=128, lstm_layers=1,
                       dropout=0.2, norm="layer")
    return model.state_dict()


class FlatUplinkTests(unittest.TestCase):
    def test_round_trip_restores_every_tensor_exactly(self):
        state = model_state()
        layout = wire_layout(state)
        delta = {name: torch.randn_like(value) * 1e-3
                 for name, value in state.items() if torch.is_floating_point(value)}
        restored = unflatten_state(flatten_state(delta, layout), layout)
        self.assertEqual(set(restored), set(delta))
        for name, value in delta.items():
            self.assertTrue(torch.equal(restored[name], value), name)
            self.assertEqual(restored[name].shape, value.shape)

    def test_flat_packet_decodes_into_the_named_state_and_keeps_error_feedback(self):
        state = model_state()
        layout = wire_layout(state)
        delta = {name: torch.randn_like(value) * 1e-3
                 for name, value in state.items() if torch.is_floating_point(value)}
        flat = flatten_state(delta, layout)
        result = compress_update(flat, {}, {k: torch.zeros_like(v) for k, v in flat.items()},
                                 client_id=3, server_round=2, num_examples=1024,
                                 uplink_budget_bytes=1715, **CODEC)
        wire = result.payload.data
        self.assertLessEqual(len(wire), 1715)
        decoded = decode_payload(wire).state
        # Eq. residual = input - decoded has to survive the flat layout, or the
        # next round's selection starts from the wrong backlog.
        residual = result.residual_state[FLAT_KEY]
        self.assertEqual(float((flat[FLAT_KEY] - decoded[FLAT_KEY] - residual).abs().max()), 0.0)
        restored = unflatten_state(decoded, layout)
        self.assertEqual(set(restored), set(delta))
        for name, value in delta.items():
            self.assertEqual(restored[name].shape, value.shape)

    def test_flat_packet_spends_less_on_description_than_the_per_tensor_packet(self):
        state = model_state()
        layout = wire_layout(state)
        delta = {name: torch.randn_like(value) * 1e-3
                 for name, value in state.items() if torch.is_floating_point(value)}
        flat = flatten_state(delta, layout)
        per_tensor = compress_update(delta, {}, {k: torch.zeros_like(v) for k, v in delta.items()},
                                     client_id=0, server_round=1, num_examples=64,
                                     uplink_budget_bytes=1715, **CODEC)
        one_vector = compress_update(flat, {}, {k: torch.zeros_like(v) for k, v in flat.items()},
                                     client_id=0, server_round=1, num_examples=64,
                                     uplink_budget_bytes=1715, **CODEC)
        described = lambda payload: payload.byte_breakdown.layout_bytes + payload.byte_breakdown.scale_bytes
        self.assertLess(described(one_vector.payload), described(per_tensor.payload))
        self.assertGreater(one_vector.payload.stored_values, per_tensor.payload.stored_values)

    def test_guards_refuse_what_a_flat_vector_cannot_express(self):
        with self.assertRaises(ValueError):
            check_flat_uplink_is_sound(dense_names=frozenset({"bn.running_mean"}),
                                       protected_groups=(), tensor_bits=None)
        with self.assertRaises(ValueError):
            check_flat_uplink_is_sound(dense_names=frozenset(),
                                       protected_groups=({"head": torch.arange(4)},),
                                       tensor_bits=None)
        with self.assertRaises(ValueError):
            check_flat_uplink_is_sound(dense_names=frozenset(), protected_groups=(),
                                       tensor_bits={"head": 8})

    def test_unflatten_rejects_a_payload_of_the_wrong_length(self):
        state = model_state()
        layout = wire_layout(state)
        with self.assertRaises(ValueError):
            unflatten_state({FLAT_KEY: torch.zeros(17)}, layout)
        with self.assertRaises(ValueError):
            unflatten_state({"other": torch.zeros(4)}, layout)

    def test_config_rejects_flat_uplink_with_per_tensor_mechanisms(self):
        from pathlib import Path
        base = Path(__file__).resolve().parents[1] / "configs/paper34_v1/fedmpsq.yaml"
        if not base.is_file():
            self.skipTest("uplink-v4 base config is not present")
        ok = load_fedmpsq_config(base, {"method.flat_uplink": True})
        self.assertTrue(ok.method.flat_uplink)
        with self.assertRaises(ValueError):
            load_fedmpsq_config(base, {"method.flat_uplink": True,
                                       "method.minority_head_reserve": 0.1})
        with self.assertRaises(ValueError):
            load_fedmpsq_config(base, {"method.flat_uplink": True,
                                       "method.protect_small_tensors": True})


if __name__ == "__main__":
    unittest.main()

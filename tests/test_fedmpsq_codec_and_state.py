'''Focused codec, aggregation, residual, and checkpoint tests for FedMPSQ.'''

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from extensions.fedmpsq import (
    CRC,
    LAYOUT_PREFIX,
    METADATA,
    PREAMBLE,
    aggregate_payloads,
    aggregate_decoded_payloads,
    compress_update,
    decode_payload,
    global_topk,
    serialize_dense_state,
    serialize_sparse_state,
    _decode_rice_gaps,
    _encode_rice_gaps,
    _stochastic_round,
    zeros_like_floating_state,
)
from fl.fedmpsq_config import load_fedmpsq_config
from scripts.train_fedmpsq import checkpoint_payload, empty_traffic
from scripts.train_fedmpsq import validate_resume_checkpoint

ROOT = Path(__file__).resolve().parents[1]


class TestFedMPSQCodecAndState(unittest.TestCase):
    def test_sparse_int8_roundtrip_crc_and_exact_byte_components(self) -> None:
        state = {
            'w': torch.tensor([0.0, -2.0, 1.0, 0.0], dtype=torch.float32),
            'b': torch.tensor([0.0, 3.0], dtype=torch.float32),
        }
        encoded = serialize_sparse_state(
            state,
            quantized=True,
            client_id=7,
            server_round=4,
            num_examples=123,
            declared_sparsity=0.5,
            alpha_s=0.25,
            flags=5,
        )
        decoded = decode_payload(encoded.data)

        self.assertEqual(decoded.client_id, 7)
        self.assertEqual(decoded.server_round, 4)
        self.assertEqual(decoded.num_examples, 123)
        self.assertEqual(decoded.num_parameters, 6)
        self.assertEqual(decoded.stored_values, 3)
        self.assertAlmostEqual(decoded.declared_sparsity, 0.5)
        self.assertAlmostEqual(decoded.alpha_s, 0.25)
        self.assertEqual(decoded.flags, 5)
        torch.testing.assert_close(
            decoded.state['w'],
            state['w'],
            atol=2.0 / 127.0,
            rtol=0,
        )
        torch.testing.assert_close(
            decoded.state['b'],
            state['b'],
            atol=3.0 / 127.0,
            rtol=0,
        )

        # Two rank-one tensors: prefix + one-byte name + one uint64 shape each.
        expected_layout = 2 * (LAYOUT_PREFIX.size + 1 + 8)
        breakdown = encoded.byte_breakdown
        self.assertEqual(breakdown.header_bytes, PREAMBLE.size + CRC.size)
        self.assertEqual(breakdown.metadata_bytes, METADATA.size)
        self.assertEqual(breakdown.layout_bytes, expected_layout)
        # Three coordinates across a 4- and a 2-element tensor: a presence
        # bitmap is one byte each, against twelve for explicit uint32 offsets,
        # so the encoder picks the bitmap and the layout records index_width 0.
        self.assertEqual(breakdown.index_bytes, 1 + 1)
        self.assertEqual(breakdown.value_bytes, 3)
        self.assertEqual(breakdown.scale_bytes, 2 * 4)
        self.assertEqual(breakdown.total_bytes, len(encoded.data))
        self.assertEqual(len(encoded.data), encoded.payload_bytes)
        self.assertEqual(encoded.theoretical_bytes, encoded.payload_bytes)
        components = (
            'header_bytes',
            'metadata_bytes',
            'layout_bytes',
            'index_bytes',
            'value_bytes',
            'scale_bytes',
        )
        self.assertEqual(
            sum(breakdown.as_dict()[key] for key in components),
            breakdown.as_dict()['total_bytes'],
        )

        corrupted = bytearray(encoded.data)
        corrupted[-CRC.size - 1] ^= 0x01
        with self.assertRaisesRegex(ValueError, 'CRC mismatch'):
            decode_payload(bytes(corrupted))

    def _sparse_state(self, density: float, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        state = {}
        for name, size in (('encoder.weight', 40000), ('head.bias', 512)):
            values = torch.randn(size, generator=generator)
            keep = torch.rand(size, generator=generator) < density
            state[name] = torch.where(keep, values, torch.zeros_like(values))
        return state

    def test_rice_gap_code_is_exact_and_beats_a_bitmap_below_ten_percent(self) -> None:
        """The gap code must reconstruct the mask and be worth choosing.

        Exactness is what lets it replace the bitmap without changing a single
        decoded update; the size comparison is what makes replacing it useful.
        A presence bitmap spends one bit per coordinate whatever the density,
        so the two cross over near the point where the mask stops being sparse.
        """
        numel = 200000
        generator = np.random.default_rng(11)
        for density in (0.002, 0.01, 0.02, 0.05, 0.1, 0.5):
            with self.subTest(density=density):
                count = max(1, int(numel * density))
                indices = np.sort(
                    generator.choice(numel, size=count, replace=False)
                ).astype(np.int64)

                blob = _encode_rice_gaps(indices, numel)
                np.testing.assert_array_equal(
                    _decode_rice_gaps(blob, count, numel), indices
                )

                bitmap_bytes = (numel + 7) // 8
                if density <= 0.1:
                    self.assertLess(len(blob), bitmap_bytes)
                # Rice is within a small constant of the entropy of the mask,
                # which is the floor any exact index code has to respect.
                entropy_bits = numel * (
                    -density * math.log2(density)
                    - (1.0 - density) * math.log2(1.0 - density)
                )
                self.assertGreater(len(blob) * 8, 0.9 * entropy_bits)
                self.assertLess(len(blob) * 8, 1.25 * entropy_bits)

    def test_rice_index_leaves_every_decoded_value_bit_identical(self) -> None:
        """Swapping the index code must not move the model by one bit."""
        state = self._sparse_state(0.1)
        common = dict(
            quantized=True,
            client_id=2,
            server_round=3,
            num_examples=64,
            declared_sparsity=0.9,
            alpha_s=0.0,
        )
        bitmap = serialize_sparse_state(state, index_codec='bitmap', **common)
        rice = serialize_sparse_state(state, index_codec='rice', **common)

        decoded_bitmap = decode_payload(bitmap.data).state
        decoded_rice = decode_payload(rice.data).state
        for name in state:
            self.assertTrue(torch.equal(decoded_bitmap[name], decoded_rice[name]))
        self.assertLess(rice.byte_breakdown.index_bytes, bitmap.byte_breakdown.index_bytes)
        self.assertEqual(rice.byte_breakdown.value_bytes, bitmap.byte_breakdown.value_bytes)

    def test_int4_halves_the_value_bytes_and_survives_the_round_trip(self) -> None:
        state = self._sparse_state(0.02, seed=5)
        common = dict(
            quantized=True,
            client_id=1,
            server_round=1,
            num_examples=32,
            declared_sparsity=0.98,
            alpha_s=0.0,
            index_codec='rice',
        )
        eight = serialize_sparse_state(state, quant_bits=8, **common)
        four = serialize_sparse_state(state, quant_bits=4, **common)

        stored = eight.stored_values
        self.assertEqual(eight.byte_breakdown.value_bytes, stored)
        self.assertEqual(four.byte_breakdown.value_bytes, (stored + 1) // 2)

        decoded = decode_payload(four.data).state
        for name, tensor in state.items():
            support = tensor != 0
            self.assertTrue(bool((decoded[name][~support] == 0).all()))
            error = float(((decoded[name] - tensor) ** 2).sum())
            norm = float((tensor ** 2).sum())
            # Sixteen levels over the observed maximum: the residual is bounded
            # by the grid, not merely small on this draw.
            self.assertLess(error / norm, 0.05)

    def test_stochastic_rounding_removes_the_quantizer_bias(self) -> None:
        """Nearest rounding repeats one sign; the random rule averages it out.

        Error feedback can cancel a zero-mean residual over rounds but not a
        systematic one, so an unbiased quantizer is what makes a coarse grid
        safe to pair with it.
        """
        values = torch.linspace(-1.0, 1.0, 100000)
        scale = float(values.abs().max()) / 7.0
        deterministic = torch.round(values / scale).clamp(-7, 7) * scale

        generator = torch.Generator().manual_seed(3)
        total = torch.zeros_like(values)
        draws = 32
        for _ in range(draws):
            total += _stochastic_round(values / scale, generator).clamp(-7, 7) * scale
        stochastic = total / draws

        self.assertLess(
            float((stochastic - values).pow(2).mean()),
            float((deterministic - values).pow(2).mean()) / 4.0,
        )

    def test_compressor_honours_the_codec_settings_it_is_given(self) -> None:
        update = self._sparse_state(1.0, seed=9)
        saliency = {name: value.abs() for name, value in update.items()}
        residual = zeros_like_floating_state(update)
        common = dict(
            uses_sparse=True,
            uses_saliency=False,
            uses_int8=True,
            uses_error_feedback=True,
            alpha_s=0.0,
            epsilon=1e-12,
            client_id=0,
            server_round=1,
            num_examples=128,
        )
        baseline = compress_update(
            update, saliency, residual, sparsity=0.9,
            quant_bits=8, index_codec='bitmap', **common
        )
        tuned = compress_update(
            update, saliency, residual, sparsity=0.98,
            quant_bits=4, index_codec='rice', stochastic_rounding=True, **common
        )

        self.assertLess(
            tuned.metrics['serialized_uplink_bytes'],
            baseline.metrics['serialized_uplink_bytes'] / 4.0,
        )
        self.assertAlmostEqual(tuned.metrics['payload_sparsity'], 0.98, places=3)
        with self.assertRaisesRegex(ValueError, 'quant_bits must be 2, 4, 8 or 32'):
            compress_update(
                update, saliency, residual, sparsity=0.9, quant_bits=6, **common
            )


    def test_dense_codec_roundtrips_supported_tensor_dtypes_and_layout(self) -> None:
        state = {
            'weights': torch.tensor(
                [[1.5, -2.0], [0.0, 4.25]],
                dtype=torch.float32,
            ),
            'counter': torch.tensor([3, 8], dtype=torch.int64),
            'mask': torch.tensor([True, False], dtype=torch.bool),
        }
        encoded = serialize_dense_state(
            state,
            client_id=1,
            server_round=2,
            num_examples=9,
        )
        decoded = decode_payload(encoded.data)

        self.assertEqual(list(decoded.state), list(state))
        for name, value in state.items():
            with self.subTest(tensor=name):
                self.assertEqual(decoded.state[name].dtype, value.dtype)
                torch.testing.assert_close(decoded.state[name], value)
        self.assertEqual(encoded.byte_breakdown.index_bytes, 0)
        self.assertEqual(encoded.byte_breakdown.scale_bytes, 0)

    def test_global_topk_is_global_and_saliency_can_change_selection(self) -> None:
        update = {
            'first': torch.tensor([1.0, 4.0]),
            'second': torch.tensor([-3.0, 2.0]),
        }
        saliency = {
            'first': torch.tensor([10.0, 0.0]),
            'second': torch.tensor([0.0, 9.0]),
        }

        magnitude = global_topk(
            update,
            saliency,
            sparsity=0.5,
            alpha_s=0.5,
            use_saliency=False,
            epsilon=1.0e-12,
        )
        torch.testing.assert_close(
            magnitude.sparse_state['first'],
            torch.tensor([0.0, 4.0]),
        )
        torch.testing.assert_close(
            magnitude.sparse_state['second'],
            torch.tensor([-3.0, 0.0]),
        )
        self.assertEqual(magnitude.selected_values, 2)
        self.assertEqual(magnitude.num_parameters, 4)

        class_aware = global_topk(
            update,
            saliency,
            sparsity=0.5,
            alpha_s=1.0,
            use_saliency=True,
            epsilon=1.0e-12,
        )
        torch.testing.assert_close(
            class_aware.sparse_state['first'],
            torch.tensor([1.0, 0.0]),
        )
        torch.testing.assert_close(
            class_aware.sparse_state['second'],
            torch.tensor([0.0, 2.0]),
        )

    def test_global_topk_ties_are_stable_and_keep_exact_ceiling(self) -> None:
        update = {'w': torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0])}
        saliency = {'w': torch.ones(5)}

        first = global_topk(
            update,
            saliency,
            sparsity=0.5,
            alpha_s=0.5,
            use_saliency=True,
            epsilon=1.0e-12,
        )
        second = global_topk(
            update,
            saliency,
            sparsity=0.5,
            alpha_s=0.5,
            use_saliency=True,
            epsilon=1.0e-12,
        )

        self.assertEqual(first.selected_values, 3)
        torch.testing.assert_close(
            first.sparse_state['w'],
            torch.tensor([1.0, -1.0, 1.0, 0.0, 0.0]),
        )
        torch.testing.assert_close(
            first.sparse_state['w'],
            second.sparse_state['w'],
        )

    def test_codec_rejects_nonfinite_updates_and_residuals(self) -> None:
        common = {
            'saliency': {'w': torch.tensor([0.0, 1.0])},
            'uses_sparse': True,
            'uses_saliency': True,
            'uses_int8': True,
            'uses_error_feedback': True,
            'sparsity': 0.5,
            'alpha_s': 0.5,
            'epsilon': 1.0e-12,
            'client_id': 0,
            'server_round': 1,
            'num_examples': 8,
        }
        with self.assertRaisesRegex(FloatingPointError, 'Client update'):
            compress_update(
                {'w': torch.tensor([float('nan'), 1.0])},
                residual={'w': torch.zeros(2)},
                **common,
            )
        with self.assertRaisesRegex(FloatingPointError, 'Client residual'):
            compress_update(
                {'w': torch.tensor([0.5, 1.0])},
                residual={'w': torch.tensor([float('inf'), 0.0])},
                **common,
            )

    def test_serialized_byte_counter_matches_file_size_on_disk(self) -> None:
        encoded = serialize_sparse_state(
            {'w': torch.tensor([0.0, 2.0, -1.0])},
            quantized=True,
            client_id=0,
            server_round=1,
            num_examples=3,
            declared_sparsity=1.0 / 3.0,
            alpha_s=0.5,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / 'payload.bin'
            path.write_bytes(encoded.data)
            self.assertEqual(path.stat().st_size, encoded.payload_bytes)
            self.assertEqual(
                path.stat().st_size,
                encoded.byte_breakdown.total_bytes,
            )

    def test_aggregation_is_sample_weighted_and_preserves_other_state(self) -> None:
        global_state = {
            'w': torch.tensor([10.0], dtype=torch.float32),
            'batch_counter': torch.tensor(11, dtype=torch.int64),
        }
        first = serialize_dense_state(
            {'w': torch.tensor([1.0])},
            client_id=0,
            server_round=1,
            num_examples=1,
        )
        second = serialize_dense_state(
            {'w': torch.tensor([3.0])},
            client_id=1,
            server_round=1,
            num_examples=3,
        )

        aggregated, decoded = aggregate_payloads(
            global_state,
            [first.data, second.data],
        )

        torch.testing.assert_close(aggregated['w'], torch.tensor([12.5]))
        torch.testing.assert_close(
            aggregated['batch_counter'],
            torch.tensor(11),
        )
        self.assertEqual([item.num_examples for item in decoded], [1, 3])

    def test_uniform_aggregation_overrides_the_sample_count_rule(self) -> None:
        """Eq. (12) can be re-weighted without touching the payloads.

        Under this partition the client holding a collapsed class can carry a
        tenth of the say of the clients that never saw it, so the choice of
        weights is a method decision, not an implementation detail.
        """
        global_state = {'w': torch.zeros(2)}
        payloads = [
            serialize_dense_state(
                {'w': torch.tensor([1.0, 0.0])},
                client_id=0, server_round=1, num_examples=9_000,
            ).data,
            serialize_dense_state(
                {'w': torch.tensor([0.0, 1.0])},
                client_id=1, server_round=1, num_examples=1_000,
            ).data,
        ]
        decoded = [decode_payload(item) for item in payloads]

        by_samples = aggregate_decoded_payloads(global_state, decoded)
        torch.testing.assert_close(by_samples['w'], torch.tensor([0.9, 0.1]))

        uniform = aggregate_decoded_payloads(global_state, decoded, [1.0, 1.0])
        torch.testing.assert_close(uniform['w'], torch.tensor([0.5, 0.5]))

        # Weights are normalised, so their scale cannot change the step size.
        rescaled = aggregate_decoded_payloads(global_state, decoded, [7.0, 7.0])
        torch.testing.assert_close(rescaled['w'], uniform['w'])

        with self.assertRaisesRegex(ValueError, 'One aggregation weight'):
            aggregate_decoded_payloads(global_state, decoded, [1.0])
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            aggregate_decoded_payloads(global_state, decoded, [1.0, -1.0])

    def test_a5_persistent_error_feedback_obeys_equation_two_rounds(self) -> None:
        saliency = {'w': torch.tensor([0.0, 1.0, 2.0, 3.0])}
        residual_0 = {'w': torch.zeros(4)}
        update_1 = {'w': torch.tensor([0.2, -1.5, 3.0, 0.7])}

        first_sparse = global_topk(
            {'w': update_1['w'] + residual_0['w']},
            saliency,
            sparsity=0.5,
            alpha_s=0.5,
            use_saliency=True,
            epsilon=1.0e-12,
        ).sparse_state
        first = compress_update(
            update_1,
            saliency,
            residual_0,
            uses_sparse=True,
            uses_saliency=True,
            uses_int8=True,
            uses_error_feedback=True,
            sparsity=0.5,
            alpha_s=0.5,
            epsilon=1.0e-12,
            client_id=0,
            server_round=1,
            num_examples=8,
        )
        # Eq. (11): the memory is added before the mask, so what it carries is
        # everything the message dropped -- the sparsification error included.
        u_1 = {'w': update_1['w'] + residual_0['w']}
        torch.testing.assert_close(first.decoded_update['w'].nonzero(), first_sparse['w'].nonzero())
        torch.testing.assert_close(
            first.residual_state['w'],
            u_1['w'] - first.decoded_update['w'],
        )
        self.assertAlmostEqual(
            first.metrics['sparsification_l2_error'] ** 2,
            first.metrics['sparsification_squared_error'],
            places=6,
        )
        self.assertAlmostEqual(
            first.metrics['quantization_l2_error'] ** 2,
            first.metrics['quantization_squared_error'],
            places=6,
        )
        self.assertGreaterEqual(
            first.metrics['sparsification_relative_l2_error'],
            0.0,
        )
        self.assertGreaterEqual(
            first.metrics['quantization_relative_l2_error'],
            0.0,
        )

        update_2 = {'w': torch.tensor([-2.0, 0.6, 0.1, 1.2])}
        second_sparse = global_topk(
            {'w': update_2['w'] + first.residual_state['w']},
            saliency,
            sparsity=0.5,
            alpha_s=0.5,
            use_saliency=True,
            epsilon=1.0e-12,
        ).sparse_state
        second = compress_update(
            update_2,
            saliency,
            first.residual_state,
            uses_sparse=True,
            uses_saliency=True,
            uses_int8=True,
            uses_error_feedback=True,
            sparsity=0.5,
            alpha_s=0.5,
            epsilon=1.0e-12,
            client_id=0,
            server_round=2,
            num_examples=8,
        )
        u_2 = {'w': update_2['w'] + first.residual_state['w']}
        torch.testing.assert_close(second.decoded_update['w'].nonzero(), second_sparse['w'].nonzero())
        torch.testing.assert_close(
            second.residual_state['w'],
            u_2['w'] - second.decoded_update['w'],
        )
        # The memory never widens the payload: the transmitted support stays at
        # the Top-K budget instead of growing with the residual's own support.
        self.assertEqual(second.metrics['transmitted_values'], 2.0)
        self.assertEqual(second.metrics['topk_selected_values'], 2.0)
        self.assertAlmostEqual(
            second.metrics['residual_l2_norm'],
            torch.linalg.vector_norm(second.residual_state['w']).item(),
        )

    def test_full_checkpoint_contains_resume_state_and_hashes(self) -> None:
        settings = load_fedmpsq_config(
            ROOT / 'configs' / 'paper34_v1' / 'fedmpsq.yaml'
        )
        global_state = {'w': torch.tensor([1.0, 2.0])}
        saliency = {
            client_id: {'w': torch.tensor([0.1, 0.2])}
            for client_id in range(10)
        }
        residual = {
            client_id: {'w': torch.tensor([0.01, -0.01])}
            for client_id in range(10)
        }
        traffic = empty_traffic()

        payload = checkpoint_payload(
            settings=settings,
            server_round=3,
            global_state=global_state,
            saliency_states=saliency,
            residual_states=residual,
            cumulative_traffic=traffic,
            client_cumulative_traffic={
                client_id: traffic.copy()
                for client_id in range(10)
            },
            best_round=2,
            best_macro_f1=0.75,
            best_model_state=global_state,
            best_validation={'macro_f1': 0.75},
            round_records=[
                {'round': round_id}
                for round_id in range(4)
            ],
            client_records=[
                {'round': round_id, 'client_id': 0}
                for round_id in range(1, 4)
            ],
            partition_hash='partition-sha256',
            task_contract_hash='task-sha256',
            target_contract_hash='target-sha256',
            config_hash='config-sha256',
            target_state={'reached': False},
        )

        required = {
            'schema_version',
            'protocol_version',
            'round',
            'global_state',
            'global_state_sha256',
            'saliency_ema_by_client',
            'error_feedback_residual_by_client',
            'cumulative_traffic',
            'client_cumulative_traffic',
            'best_round',
            'best_validation_macro_f1',
            'best_model_state',
            'best_validation',
            'round_records',
            'client_records',
            'partition_hash',
            'task_contract_sha256',
            'target_contract_sha256',
            'scientific_config_sha256',
            'training_seed',
            'data_split_seed',
            'target_state',
            'rng_state',
        }
        self.assertLessEqual(required, payload.keys())
        self.assertEqual(payload['round'], 3)
        self.assertEqual(payload['training_seed'], 42)
        self.assertEqual(payload['data_split_seed'], 42)
        self.assertEqual(payload['partition_hash'], 'partition-sha256')
        self.assertEqual(payload['task_contract_sha256'], 'task-sha256')
        self.assertEqual(payload['target_contract_sha256'], 'target-sha256')
        self.assertEqual(payload['scientific_config_sha256'], 'config-sha256')
        torch.testing.assert_close(
            payload['saliency_ema_by_client'][0]['w'],
            saliency[0]['w'],
        )
        torch.testing.assert_close(
            payload['error_feedback_residual_by_client'][0]['w'],
            residual[0]['w'],
        )
        self.assertLessEqual(
            {'python', 'numpy', 'torch_cpu', 'torch_cuda'},
            payload['rng_state'].keys(),
        )

        # Checkpoints own independent snapshots, not live training references.
        global_state['w'][0] = -999.0
        residual[0]['w'][0] = -999.0
        self.assertEqual(payload['global_state']['w'][0].item(), 1.0)
        self.assertAlmostEqual(
            payload['error_feedback_residual_by_client'][0]['w'][0].item(),
            0.01,
        )

        validate_resume_checkpoint(
            payload,
            settings=settings,
            reference_global_state={'w': torch.tensor([0.0, 0.0])},
            partition_hash='partition-sha256',
            task_contract_hash='task-sha256',
            target_contract_hash='target-sha256',
            config_hash='config-sha256',
        )
        corrupted = copy.deepcopy(payload)
        corrupted['global_state']['w'][0] += 1.0
        with self.assertRaisesRegex(ValueError, 'global-state hash'):
            validate_resume_checkpoint(
                corrupted,
                settings=settings,
                reference_global_state={'w': torch.tensor([0.0, 0.0])},
                partition_hash='partition-sha256',
                task_contract_hash='task-sha256',
                target_contract_hash='target-sha256',
                config_hash='config-sha256',
            )
        missing_rng = copy.deepcopy(payload)
        del missing_rng['rng_state']['numpy']
        with self.assertRaisesRegex(ValueError, 'RNG state is incomplete'):
            validate_resume_checkpoint(
                missing_rng,
                settings=settings,
                reference_global_state={'w': torch.tensor([0.0, 0.0])},
                partition_hash='partition-sha256',
                task_contract_hash='task-sha256',
                target_contract_hash='target-sha256',
                config_hash='config-sha256',
            )


if __name__ == '__main__':
    unittest.main()

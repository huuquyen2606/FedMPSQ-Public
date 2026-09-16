'''Rotation and byte-scale-code tests for the FedMPSQ sparse uplink codec.'''

from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from extensions.fedmpsq import (
    GAUSS4_ENCODINGS,
    LOG8_ENCODINGS,
    ROTATION_FLAG,
    compress_update,
    decode_payload,
    serialize_sparse_state,
    zeros_like_floating_state,
)
from extensions.incoherent import (
    _walsh_hadamard,
    decode_log8_scales,
    encode_log8_scales,
    rotate_groups,
    rotation_signs,
)


class TestIncoherentPrimitives(unittest.TestCase):
    def test_transform_is_the_orthonormal_hadamard_matrix_and_its_own_inverse(self) -> None:
        """A rotation that changed the norm would change the update it carries.

        The encoder fits a quantization scale to the rotated group and the
        decoder inverts the rotation after rescaling, so anything other than an
        exact orthonormal involution would leak error into every coordinate.
        """
        for width in (2, 4, 8, 16, 32, 64):
            with self.subTest(width=width):
                explicit = torch.ones(1, 1)
                while explicit.shape[0] < width:
                    explicit = torch.cat(
                        (
                            torch.cat((explicit, explicit), dim=1),
                            torch.cat((explicit, -explicit), dim=1),
                        ),
                        dim=0,
                    )
                explicit = explicit / math.sqrt(width)
                sample = torch.randn(6, width, generator=torch.Generator().manual_seed(width))

                transformed = _walsh_hadamard(sample)
                torch.testing.assert_close(transformed, sample @ explicit.T, atol=1e-5, rtol=0)
                torch.testing.assert_close(_walsh_hadamard(transformed), sample, atol=1e-5, rtol=0)
                self.assertAlmostEqual(
                    float(transformed.norm()), float(sample.norm()), delta=1e-4
                )

    def test_rotation_round_trips_exactly_and_leaves_a_short_tail_alone(self) -> None:
        """A partial final group is not padded, because padding would be sent.

        The rotation makes padded positions carry signal, so they could not be
        dropped again; at a few thousand coordinates per tensor the tail is
        worth less than the bytes. Both sides must apply the identical rule.
        """
        for count, width in ((100, 32), (128, 32), (3, 32), (1024, 16), (37, 8)):
            with self.subTest(count=count, width=width):
                sample = torch.randn(count, generator=torch.Generator().manual_seed(count))
                keys = dict(group_size=width, client_id=3, server_round=7, name='block.weight')

                rotated = rotate_groups(sample, **keys)
                restored = rotate_groups(rotated, **keys, inverse=True)

                torch.testing.assert_close(restored, sample, atol=1e-5, rtol=0)
                self.assertAlmostEqual(float(rotated.norm()), float(sample.norm()), delta=1e-4)
                tail = count % width
                if tail:
                    torch.testing.assert_close(rotated[count - tail:], sample[count - tail:])

    def test_rotation_seed_is_reproducible_and_separates_clients_rounds_tensors(self) -> None:
        """Nothing about the rotation is transmitted, so both sides derive it.

        The seed uses hashlib rather than the built-in string hash, which is
        salted per process and would not survive a restart, let alone the
        server and the client running on different machines.
        """
        shape = dict(groups=4, width=8)
        base = rotation_signs(client_id=1, server_round=2, name='w', **shape)

        self.assertTrue(torch.equal(base, rotation_signs(client_id=1, server_round=2, name='w', **shape)))
        self.assertFalse(torch.equal(base, rotation_signs(client_id=1, server_round=3, name='w', **shape)))
        self.assertFalse(torch.equal(base, rotation_signs(client_id=2, server_round=2, name='w', **shape)))
        self.assertFalse(torch.equal(base, rotation_signs(client_id=1, server_round=2, name='b', **shape)))
        self.assertCountEqual(torch.unique(base).tolist(), [-1.0, 1.0])

    def test_byte_scale_code_holds_a_wide_range_at_an_eighth_of_an_octave(self) -> None:
        scales = torch.tensor([1.0, 0.5, 0.013, 7e-5, 3.3, 2.0e-9])
        reference, codes, decoded = encode_log8_scales(scales)

        self.assertEqual(codes.dtype, np.uint8)
        self.assertEqual(len(codes), len(scales))
        torch.testing.assert_close(decode_log8_scales(reference, codes), decoded)
        self.assertLess(float((decoded / scales - 1).abs().max()), 0.05)


class TestIncoherentWireFormat(unittest.TestCase):
    def _sparse_state(self, density: float, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        state = {}
        for name, size in (('encoder.weight', 40000), ('head.bias', 512)):
            values = torch.randn(size, generator=generator)
            keep = torch.rand(size, generator=generator) < density
            state[name] = torch.where(keep, values, torch.zeros_like(values))
        return state

    def _common(self):
        return dict(
            quantized=True,
            client_id=4,
            server_round=9,
            num_examples=64,
            declared_sparsity=0.95,
            alpha_s=0.0,
            index_codec='auto_runs',
            clipping='mse_refine',
        )

    def test_rotated_payload_declares_the_flag_and_survives_the_round_trip(self) -> None:
        state = self._sparse_state(0.05, seed=3)

        encoded = serialize_sparse_state(
            state, quant_bits=4, group_size=32, rotate=True, **self._common()
        )
        decoded = decode_payload(encoded.data)

        self.assertTrue(decoded.flags & ROTATION_FLAG)
        for name, tensor in state.items():
            support = tensor != 0
            # The rotation must not move mass onto coordinates that were never
            # selected: the sparsity contract is what the index stream promises.
            self.assertTrue(bool((decoded.state[name][~support] == 0).all()))
            error = float(((decoded.state[name] - tensor) ** 2).sum())
            self.assertLess(error / float((tensor ** 2).sum()), 0.05)

    def test_byte_scale_code_shrinks_the_scale_stream_by_about_four(self) -> None:
        """The scale stream is not a rounding detail at INT2 with small groups.

        One FP32 scale per group of 32 costs a full bit per coordinate, half
        again as much as the two-bit value it describes, and the byte code is
        what makes a small group affordable at all.
        """
        state = self._sparse_state(0.05, seed=7)
        common = dict(quant_bits=2, group_size=32, **self._common())

        wide = serialize_sparse_state(state, **common)
        narrow = serialize_sparse_state(state, scale_codec='log8', **common)

        groups = sum(
            math.ceil(int((tensor != 0).sum()) / 32) for tensor in state.values()
        )
        self.assertEqual(wide.byte_breakdown.scale_bytes, 4 * len(state) + 4 * groups)
        self.assertEqual(narrow.byte_breakdown.scale_bytes, 8 * len(state) + groups)
        self.assertLess(narrow.byte_breakdown.scale_bytes, wide.byte_breakdown.scale_bytes / 3)

        decoded = decode_payload(narrow.data).state
        for name, tensor in state.items():
            self.assertTrue(bool((decoded[name][tensor == 0] == 0).all()))
            error = float(((decoded[name] - tensor) ** 2).sum())
            wide_error = float(
                ((decode_payload(wide.data).state[name] - tensor) ** 2).sum()
            )
            # Refitting the codes against the transmitted scale keeps the byte
            # code from costing accuracy on top of the bytes it saves.
            self.assertLess(error, wide_error * 1.25)

    def test_new_encodings_are_distinct_and_leave_the_old_ones_decodable(self) -> None:
        state = self._sparse_state(0.05, seed=11)
        legacy = serialize_sparse_state(state, quant_bits=4, group_size=128, **self._common())
        modern = serialize_sparse_state(
            state, quant_bits=4, group_size=128, scale_codec='log8', rotate=True, **self._common()
        )

        self.assertEqual(sorted(LOG8_ENCODINGS.values()), [9, 10, 11])
        self.assertFalse(decode_payload(legacy.data).flags & ROTATION_FLAG)
        self.assertTrue(decode_payload(modern.data).flags & ROTATION_FLAG)
        for name, tensor in state.items():
            support = tensor != 0
            self.assertTrue(bool((decode_payload(legacy.data).state[name][~support] == 0).all()))

    def test_codec_rejects_options_it_cannot_honour(self) -> None:
        state = self._sparse_state(0.05, seed=13)
        with self.assertRaisesRegex(ValueError, 'power-of-two'):
            serialize_sparse_state(state, quant_bits=4, group_size=100, rotate=True, **self._common())
        with self.assertRaisesRegex(ValueError, 'power-of-two'):
            serialize_sparse_state(state, quant_bits=4, group_size=0, rotate=True, **self._common())
        with self.assertRaisesRegex(ValueError, 'grouped scales only'):
            serialize_sparse_state(state, quant_bits=4, group_size=0, scale_codec='log8', **self._common())
        with self.assertRaisesRegex(ValueError, 'scale_codec must be'):
            serialize_sparse_state(state, quant_bits=4, group_size=32, scale_codec='fp16', **self._common())

    def test_four_level_codebook_beats_ternary_on_the_same_bytes(self) -> None:
        """Two bits hold four codes; the integer quantizer reserves one of them.

        An exact round trip proves nothing here: the encoder and the decoder can
        agree perfectly on a byte stream that carries the wrong values, which is
        exactly what happens if the packed codebook is overwritten by the
        integer packing. Only the reconstruction error catches that, so this
        test asserts quality, not just agreement.
        """
        generator = torch.Generator().manual_seed(17)
        update = {"block.weight": torch.randn(8192, generator=generator),
                  "head.bias": torch.randn(96, generator=generator)}
        saliency = {name: value.abs() for name, value in update.items()}
        shared = dict(
            uses_sparse=True, uses_saliency=True, uses_int8=True,
            uses_error_feedback=True, sparsity=0.9, alpha_s=0.25, epsilon=1e-12,
            client_id=2, server_round=5, num_examples=128, quant_bits=2,
            group_size=256, clipping="mse_refine", index_codec="auto_runs",
            compact_layout=True, block_size=32, rotate=True, scale_codec="log8",
            uplink_budget_bytes=4000,
        )
        results = {}
        for quantizer in ("integer", "gaussian4"):
            results[quantizer] = compress_update(
                update, saliency, zeros_like_floating_state(update),
                quantizer=quantizer, **shared
            )

        ternary, codebook = results["integer"], results["gaussian4"]
        self.assertEqual(len(ternary.payload.data), len(codebook.payload.data))
        self.assertEqual(
            int(ternary.metrics["topk_selected_values"]),
            int(codebook.metrics["topk_selected_values"]),
        )
        self.assertLess(
            codebook.metrics["quantization_relative_error"],
            0.75 * ternary.metrics["quantization_relative_error"],
        )
        decoded = decode_payload(codebook.payload.data)
        self.assertIn(decoded.state["block.weight"].dtype, (torch.float32,))
        for name, tensor in update.items():
            torch.testing.assert_close(
                decoded.state[name], codebook.decoded_update[name], atol=0, rtol=0
            )
            torch.testing.assert_close(
                codebook.residual_state[name] + codebook.decoded_update[name],
                tensor, atol=1e-5, rtol=0,
            )

    def test_codebook_encodings_are_distinct_and_options_are_checked(self) -> None:
        self.assertEqual(sorted(GAUSS4_ENCODINGS.values()), [12, 13])
        state = self._sparse_state(0.05, seed=23)
        with self.assertRaisesRegex(ValueError, "two-bit grouped quantizer"):
            serialize_sparse_state(state, quant_bits=4, group_size=256,
                                   quantizer="gaussian4", **self._common())
        with self.assertRaisesRegex(ValueError, "two-bit grouped quantizer"):
            serialize_sparse_state(state, quant_bits=2, group_size=0,
                                   quantizer="gaussian4", **self._common())
        with self.assertRaisesRegex(ValueError, "quantizer must be"):
            serialize_sparse_state(state, quant_bits=2, group_size=256,
                                   quantizer="lattice8", **self._common())

    def test_error_feedback_identity_survives_rotation_and_the_byte_scale(self) -> None:
        """Eq. (11) must still hold exactly: residual = input - what was decoded.

        Error feedback is the mechanism that lets an aggressive compressor
        converge at all, and it is only sound if the memory is the exact
        complement of the transmitted message on the wire, not an estimate of it.
        """
        generator = torch.Generator().manual_seed(21)
        update = {'block.weight': torch.randn(4096, generator=generator),
                  'head.bias': torch.randn(64, generator=generator)}
        saliency = {name: value.abs() for name, value in update.items()}
        residual = zeros_like_floating_state(update)

        for _ in range(3):
            result = compress_update(
                update, saliency, residual,
                uses_sparse=True, uses_saliency=True, uses_int8=True,
                uses_error_feedback=True, sparsity=0.9, alpha_s=0.25, epsilon=1e-12,
                client_id=2, server_round=5, num_examples=128,
                quant_bits=2, group_size=32, clipping='mse_refine',
                index_codec='auto_runs', compact_layout=True, block_size=32,
                rotate=True, scale_codec='log8',
            )
            wire = decode_payload(result.payload.data).state
            for name in update:
                torch.testing.assert_close(wire[name], result.decoded_update[name], atol=0, rtol=0)
                torch.testing.assert_close(
                    result.residual_state[name] + result.decoded_update[name],
                    update[name] + residual[name], atol=1e-5, rtol=0,
                )
            residual = result.residual_state


if __name__ == '__main__':
    unittest.main()

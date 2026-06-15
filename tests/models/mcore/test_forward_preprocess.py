"""Tests for preprocess/postprocess functions in axon.models.mcore.forward.util.

Covers:
  - preprocess_bshd / postprocess_bshd  (left-padding removal and recovery)
  - preprocess_packed_seqs / postprocess_packed_seqs  (sequence packing / unpacking)

All tests run on CPU with TP=1, CP=1 via a mock of megatron.core.parallel_state.

Usage:
    pytest tests/models/mcore/test_forward_preprocess.py -v
"""

from unittest.mock import MagicMock, patch

import torch

# ---------------------------------------------------------------------------
# Mock helper
# ---------------------------------------------------------------------------


def _mock_mpu(tp_size=1, cp_size=1, cp_rank=0):
    """Return a context manager that mocks megatron.core.parallel_state."""
    mock = MagicMock()
    mock.get_tensor_model_parallel_world_size.return_value = tp_size
    mock.get_context_parallel_world_size.return_value = cp_size
    mock.get_context_parallel_rank.return_value = cp_rank
    return patch("axon.models.mcore.forward.util.mpu", mock)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bool_mask(*values):
    """Create a boolean attention mask tensor from 0/1 integer values."""
    return torch.tensor(values, dtype=torch.bool)


def _bool_mask_2d(rows):
    """Create a 2-D boolean attention mask from a list of lists of 0/1."""
    return torch.tensor(rows, dtype=torch.bool)


# ===========================================================================
# TestPreprocessBshd
# ===========================================================================


class TestPreprocessBshd:
    def test_removes_left_padding(self):
        """Input with left padding [0,0,1,1,1] compacts to length 3."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_bshd

            input_ids = torch.tensor([[0, 0, 10, 20, 30]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[False, False, True, True, True]])
            position_ids = torch.tensor([[0, 0, 0, 1, 2]], dtype=torch.long)

            new_ids, new_mask, new_pos = preprocess_bshd(input_ids, attention_mask, position_ids)

            assert new_ids.shape == (1, 3)
            assert new_ids[0].tolist() == [10, 20, 30]
            assert new_mask[0].tolist() == [True, True, True]
            assert new_pos[0].tolist() == [0, 1, 2]

    def test_preserves_right_padded(self):
        """Already right-padded input: valid tokens stay in order."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_bshd

            input_ids = torch.tensor([[10, 20, 30, 0, 0]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[True, True, True, False, False]])
            position_ids = torch.tensor([[0, 1, 2, 0, 0]], dtype=torch.long)

            new_ids, new_mask, new_pos = preprocess_bshd(input_ids, attention_mask, position_ids)

            assert new_ids.shape == (1, 3)
            assert new_ids[0].tolist() == [10, 20, 30]

    def test_batch_mixed_padding(self):
        """Batch of 2 with different left-pad amounts."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_bshd

            # seq0: 2 pad + 4 valid, seq1: 3 pad + 3 valid  => max_valid = 4
            input_ids = torch.tensor(
                [
                    [0, 0, 1, 2, 3, 4],
                    [0, 0, 0, 5, 6, 7],
                ],
                dtype=torch.long,
            )
            attention_mask = _bool_mask_2d(
                [
                    [False, False, True, True, True, True],
                    [False, False, False, True, True, True],
                ]
            )
            position_ids = torch.tensor(
                [
                    [0, 0, 0, 1, 2, 3],
                    [0, 0, 0, 0, 1, 2],
                ],
                dtype=torch.long,
            )

            new_ids, new_mask, new_pos = preprocess_bshd(input_ids, attention_mask, position_ids)

            # Output seq_len = max(4, 3) = 4
            assert new_ids.shape == (2, 4)
            assert new_ids[0].tolist() == [1, 2, 3, 4]
            assert new_ids[1, :3].tolist() == [5, 6, 7]
            assert new_ids[1, 3].item() == 0  # zero-padded on the right

    def test_pre_process_false_skips_input_ids(self):
        """With pre_process=False, input_ids returned as-is."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_bshd

            input_ids = torch.tensor([[0, 0, 10, 20, 30]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[False, False, True, True, True]])
            position_ids = torch.tensor([[0, 0, 0, 1, 2]], dtype=torch.long)

            new_ids, new_mask, new_pos = preprocess_bshd(input_ids, attention_mask, position_ids, pre_process=False)

            # input_ids is returned unchanged
            assert torch.equal(new_ids, input_ids)
            # But attention_mask and position_ids are still compacted
            assert new_mask.shape[1] == 3
            assert new_pos.shape[1] == 3

    def test_output_seqlen_matches_max_valid(self):
        """Output seq_len equals max(valid_lengths) across batch."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_bshd

            input_ids = torch.tensor(
                [
                    [0, 0, 0, 1, 2],
                    [0, 1, 2, 3, 4],
                ],
                dtype=torch.long,
            )
            attention_mask = _bool_mask_2d(
                [
                    [False, False, False, True, True],
                    [False, True, True, True, True],
                ]
            )
            position_ids = torch.tensor(
                [
                    [0, 0, 0, 0, 1],
                    [0, 0, 1, 2, 3],
                ],
                dtype=torch.long,
            )

            new_ids, new_mask, new_pos = preprocess_bshd(input_ids, attention_mask, position_ids)

            # max valid = max(2, 4) = 4
            assert new_ids.shape[1] == 4


# ===========================================================================
# TestPostprocessBshd
# ===========================================================================


class TestPostprocessBshd:
    def test_recovers_left_padding(self):
        """Compacted result placed back into original left-padded positions."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_bshd

            # Compacted result: shape [1, 3, 4] (batch=1, seq=3, hidden=4)
            result = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]])
            # Compacted attention mask (after preprocess): all valid
            attention_mask = torch.tensor([[True, True, True]])
            # Original attention mask with left padding
            original_attention_mask = torch.tensor([[False, False, True, True, True]])
            origin_seqlen = 5

            recovered = postprocess_bshd(result, attention_mask, original_attention_mask, origin_seqlen)

            assert recovered.shape == (1, 5, 4)
            # Padded positions should be zero
            assert recovered[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0]
            assert recovered[0, 1].tolist() == [0.0, 0.0, 0.0, 0.0]
            # Valid positions should have the values
            assert recovered[0, 2].tolist() == [1.0, 2.0, 3.0, 4.0]
            assert recovered[0, 3].tolist() == [5.0, 6.0, 7.0, 8.0]
            assert recovered[0, 4].tolist() == [9.0, 10.0, 11.0, 12.0]

    def test_roundtrip(self):
        """preprocess -> identity forward -> postprocess recovers original layout."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_bshd, preprocess_bshd

            # Original: batch=1, seq=6, left-padded with 2 padding tokens
            input_ids = torch.tensor([[0, 0, 10, 20, 30, 40]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[False, False, True, True, True, True]])
            position_ids = torch.tensor([[0, 0, 0, 1, 2, 3]], dtype=torch.long)
            origin_seqlen = 6

            new_ids, new_mask, new_pos = preprocess_bshd(input_ids, attention_mask, position_ids)

            # Simulate forward: just expand input_ids to have a hidden dim
            # new_ids shape: [1, 4]
            forward_out = new_ids.unsqueeze(-1).float()  # [1, 4, 1]

            recovered = postprocess_bshd(forward_out, new_mask, attention_mask, origin_seqlen)

            assert recovered.shape == (1, 6, 1)
            # Padding positions should be zero
            assert recovered[0, 0, 0].item() == 0.0
            assert recovered[0, 1, 0].item() == 0.0
            # Valid positions should match original valid tokens
            assert recovered[0, 2, 0].item() == 10.0
            assert recovered[0, 3, 0].item() == 20.0
            assert recovered[0, 4, 0].item() == 30.0
            assert recovered[0, 5, 0].item() == 40.0

    def test_post_process_false_returns_input(self):
        """With post_process=False, returns input unchanged."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_bshd

            result = torch.randn(1, 3, 8)
            attention_mask = torch.ones(1, 3, dtype=torch.bool)
            original_attention_mask = torch.tensor([[False, False, True, True, True]])

            out = postprocess_bshd(result, attention_mask, original_attention_mask, origin_seqlen=5, post_process=False)
            assert torch.equal(out, result)


# ===========================================================================
# TestPreprocessPackedSeqs
# ===========================================================================


class TestPreprocessPackedSeqs:
    def test_packs_single_sequence(self):
        """One sequence: packed output has the right length."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_packed_seqs

            input_ids = torch.tensor([[10, 20, 30, 0, 0]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[True, True, True, False, False]])

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            # With TP=1, align_size=1, so padded_len = 3
            # Output shape: [1, total_packed_len]
            assert packed.dim() == 2
            assert packed.shape[0] == 1
            # The packed length should be >= 3 (the valid length)
            assert packed.shape[1] >= 3
            # First 3 values should be the valid tokens
            assert packed[0, :3].tolist() == [10, 20, 30]

    def test_packs_batch_removes_padding(self):
        """Batch of 2 with different valid lengths: total packed = sum of valid (with TP alignment)."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_packed_seqs

            input_ids = torch.tensor(
                [
                    [10, 20, 30, 0, 0],
                    [40, 50, 0, 0, 0],
                ],
                dtype=torch.long,
            )
            attention_mask = _bool_mask_2d(
                [
                    [True, True, True, False, False],
                    [True, True, False, False, False],
                ]
            )

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            # TP=1, align=1, so padded lengths are 3 and 2, total = 5
            assert packed.shape[0] == 1
            assert packed.shape[1] == 5

    def test_packed_seq_params_valid(self):
        """Verify cu_seqlens and max_seqlen are correct."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_packed_seqs

            input_ids = torch.tensor(
                [
                    [10, 20, 30, 0, 0],
                    [40, 50, 0, 0, 0],
                ],
                dtype=torch.long,
            )
            attention_mask = _bool_mask_2d(
                [
                    [True, True, True, False, False],
                    [True, True, False, False, False],
                ]
            )

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            # cu_seqlens_q should be [0, 3, 5] for TP=1
            cu = params.cu_seqlens_q.tolist()
            assert cu[0] == 0
            assert cu[1] == 3  # first seq has 3 valid tokens
            assert cu[2] == 5  # cumulative: 3 + 2 = 5
            assert params.max_seqlen_q == 3

    def test_pre_process_false_returns_original(self):
        """With pre_process=False, input_ids is unchanged."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_packed_seqs

            input_ids = torch.tensor([[10, 20, 30, 0, 0]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[True, True, True, False, False]])

            out, params = preprocess_packed_seqs(input_ids, attention_mask, pre_process=False)

            assert torch.equal(out, input_ids)
            # PackedSeqParams should still be populated
            assert params.cu_seqlens_q is not None

    def test_content_preserved(self):
        """Packed values match the valid tokens from the original sequences."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import preprocess_packed_seqs

            input_ids = torch.tensor(
                [
                    [0, 0, 10, 20, 30],  # left-padded, 3 valid
                    [0, 40, 50, 60, 70],  # left-padded, 4 valid
                ],
                dtype=torch.long,
            )
            attention_mask = _bool_mask_2d(
                [
                    [False, False, True, True, True],
                    [False, True, True, True, True],
                ]
            )

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            cu = params.cu_seqlens_q.tolist()
            # First sequence's valid tokens
            seq0 = packed[0, cu[0] : cu[0] + 3].tolist()
            assert seq0 == [10, 20, 30]
            # Second sequence's valid tokens
            seq1 = packed[0, cu[1] : cu[1] + 4].tolist()
            assert seq1 == [40, 50, 60, 70]


# ===========================================================================
# TestPostprocessPackedSeqs
# ===========================================================================


class TestPostprocessPackedSeqs:
    def test_unpacks_single_sequence(self):
        """Unpack single sequence back to [batch, seq_len, hidden] shape."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_packed_seqs, preprocess_packed_seqs

            input_ids = torch.tensor([[10, 20, 30, 0, 0]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[True, True, True, False, False]])

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            # Simulate forward: add hidden dim
            hidden = packed.unsqueeze(-1).float()  # [1, packed_len, 1]

            output = postprocess_packed_seqs(hidden, params, attention_mask, batch_size=1, seq_len=5)

            assert output.shape == (1, 5, 1)
            assert output[0, 0, 0].item() == 10.0
            assert output[0, 1, 0].item() == 20.0
            assert output[0, 2, 0].item() == 30.0
            # Padding positions should be zero
            assert output[0, 3, 0].item() == 0.0
            assert output[0, 4, 0].item() == 0.0

    def test_roundtrip_preserves_values(self):
        """preprocess -> identity -> postprocess recovers original valid token positions."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_packed_seqs, preprocess_packed_seqs

            input_ids = torch.tensor([[0, 0, 10, 20, 30]], dtype=torch.long)
            attention_mask = _bool_mask_2d([[False, False, True, True, True]])

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            # Identity forward with hidden dim
            hidden = packed.unsqueeze(-1).float()

            output = postprocess_packed_seqs(hidden, params, attention_mask, batch_size=1, seq_len=5)

            assert output.shape == (1, 5, 1)
            # Left-padding positions are zero
            assert output[0, 0, 0].item() == 0.0
            assert output[0, 1, 0].item() == 0.0
            # Valid positions recover their values
            assert output[0, 2, 0].item() == 10.0
            assert output[0, 3, 0].item() == 20.0
            assert output[0, 4, 0].item() == 30.0

    def test_post_process_false_returns_input(self):
        """Passthrough when disabled."""
        with _mock_mpu():
            from megatron.core.packed_seq_params import PackedSeqParams

            from axon.models.mcore.forward.util import postprocess_packed_seqs

            dummy_output = torch.randn(1, 10, 4)
            dummy_params = PackedSeqParams()
            dummy_mask = torch.ones(2, 5, dtype=torch.bool)

            out = postprocess_packed_seqs(
                dummy_output,
                dummy_params,
                dummy_mask,
                batch_size=2,
                seq_len=5,
                post_process=False,
            )
            assert torch.equal(out, dummy_output)

    def test_batch_roundtrip(self):
        """Multiple sequences with different lengths round-trip correctly."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_packed_seqs, preprocess_packed_seqs

            input_ids = torch.tensor(
                [
                    [0, 0, 10, 20, 30],
                    [0, 40, 50, 60, 70],
                ],
                dtype=torch.long,
            )
            attention_mask = _bool_mask_2d(
                [
                    [False, False, True, True, True],
                    [False, True, True, True, True],
                ]
            )

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            # Identity forward with hidden dim
            hidden = packed.unsqueeze(-1).float()

            output = postprocess_packed_seqs(hidden, params, attention_mask, batch_size=2, seq_len=5)

            assert output.shape == (2, 5, 1)
            # Sequence 0: left-pad positions zero, valid positions recover
            assert output[0, 0, 0].item() == 0.0
            assert output[0, 1, 0].item() == 0.0
            assert output[0, 2, 0].item() == 10.0
            assert output[0, 3, 0].item() == 20.0
            assert output[0, 4, 0].item() == 30.0
            # Sequence 1
            assert output[1, 0, 0].item() == 0.0
            assert output[1, 1, 0].item() == 40.0
            assert output[1, 2, 0].item() == 50.0
            assert output[1, 3, 0].item() == 60.0
            assert output[1, 4, 0].item() == 70.0


# ===========================================================================
# TestPreprocessPostprocessRoundtrip (integration)
# ===========================================================================


class TestPreprocessPostprocessRoundtrip:
    def test_bshd_roundtrip_with_hidden_dim(self):
        """Full bshd pipeline with [batch, seq_len, hidden] tensors and left padding."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_bshd, preprocess_bshd

            batch_size, seq_len, hidden_dim = 2, 8, 16
            # Build input_ids with a hidden dim (simulating embeddings)
            # seq0: 3 left-pad + 5 valid, seq1: 5 left-pad + 3 valid
            torch.manual_seed(42)
            full_embeddings = torch.randn(batch_size, seq_len, hidden_dim)

            attention_mask = _bool_mask_2d(
                [
                    [False, False, False, True, True, True, True, True],
                    [False, False, False, False, False, True, True, True],
                ]
            )
            position_ids = torch.tensor(
                [
                    [0, 0, 0, 0, 1, 2, 3, 4],
                    [0, 0, 0, 0, 0, 0, 1, 2],
                ],
                dtype=torch.long,
            )

            origin_seqlen = seq_len

            new_emb, new_mask, new_pos = preprocess_bshd(full_embeddings, attention_mask, position_ids)

            # max valid = max(5, 3) = 5
            assert new_emb.shape == (2, 5, hidden_dim)

            # Simulate model forward (identity): output = new_emb
            model_output = new_emb.clone()

            recovered = postprocess_bshd(model_output, new_mask, attention_mask, origin_seqlen)

            assert recovered.shape == (batch_size, seq_len, hidden_dim)

            # Verify valid positions match original
            for b in range(batch_size):
                valid_positions = attention_mask[b].bool()
                torch.testing.assert_close(
                    recovered[b][valid_positions],
                    full_embeddings[b][valid_positions],
                )
                # Padding positions should be zero
                pad_positions = ~valid_positions
                assert (recovered[b][pad_positions] == 0).all()

    def test_packed_seqs_roundtrip_with_hidden_dim(self):
        """Full packed-seqs pipeline with [batch, seq_len, hidden] style data."""
        with _mock_mpu():
            from axon.models.mcore.forward.util import postprocess_packed_seqs, preprocess_packed_seqs

            batch_size, seq_len, _hidden_dim = 3, 10, 8

            # Create input_ids that we'll use as proxy for token indices
            torch.manual_seed(123)
            # Use unique values so we can verify content preservation
            input_ids = torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len)

            # Different valid lengths: 7, 4, 10
            attention_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
            valid_lengths = [7, 4, 10]
            for b, vl in enumerate(valid_lengths):
                attention_mask[b, seq_len - vl :] = True  # left-padded

            packed, params = preprocess_packed_seqs(input_ids, attention_mask)

            # Simulate forward: use packed token ids as a 1-d "hidden" representation
            hidden = packed.unsqueeze(-1).float()  # [1, total_packed, 1]

            output = postprocess_packed_seqs(
                hidden,
                params,
                attention_mask,
                batch_size=batch_size,
                seq_len=seq_len,
            )

            assert output.shape == (batch_size, seq_len, 1)

            # Verify each sequence
            for b in range(batch_size):
                valid_pos = attention_mask[b].bool()
                expected = input_ids[b][valid_pos].float().unsqueeze(-1)
                torch.testing.assert_close(output[b][valid_pos], expected)

                # Padding positions should be zero
                pad_pos = ~valid_pos
                if pad_pos.any():
                    assert (output[b][pad_pos] == 0).all()

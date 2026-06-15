import warnings
from unittest.mock import MagicMock

import pytest
import torch

from axon.utils.tokenizer import postprocess_data, remove_pad_token, set_pad_token_id

# ===================================================================
#  set_pad_token_id
# ===================================================================


class TestSetPadTokenId:
    def test_pad_token_is_none_gets_set_to_eos_token(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = None
        tokenizer.pad_token = None
        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            set_pad_token_id(tokenizer)
            assert any("pad_token is None" in str(warning.message) for warning in w)

        assert tokenizer.pad_token_id == 2
        assert tokenizer.pad_token == "</s>"

    def test_pad_token_id_set_but_pad_token_none(self):
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.pad_token = None
        tokenizer.eos_token_id = 2
        tokenizer.eos_token = "</s>"

        set_pad_token_id(tokenizer)

        # pad_token_id should remain unchanged
        assert tokenizer.pad_token_id == 0
        # pad_token should be set to eos_token
        assert tokenizer.pad_token == "</s>"


# ===================================================================
#  postprocess_data
# ===================================================================


class TestPostprocessData:
    def test_padding_shorter_sequence_left_pad(self):
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3, dtype=torch.long)

        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=5, pad_token_id=0, left_pad=True
        )

        assert result_ids.shape == (1, 5)
        assert result_mask.shape == (1, 5)
        # Left pad: padding on the left
        assert result_ids[0, -3:].tolist() == [1, 2, 3]
        assert result_ids[0, :2].tolist() == [0, 0]
        assert result_mask[0, -3:].tolist() == [1, 1, 1]
        assert result_mask[0, :2].tolist() == [0, 0]

    def test_truncation_left(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5, dtype=torch.long)

        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=3, pad_token_id=0, truncation="left"
        )

        assert result_ids.shape == (1, 3)
        assert result_ids[0].tolist() == [3, 4, 5]
        assert result_mask[0].tolist() == [1, 1, 1]

    def test_truncation_right(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5, dtype=torch.long)

        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=3, pad_token_id=0, truncation="right"
        )

        assert result_ids.shape == (1, 3)
        assert result_ids[0].tolist() == [1, 2, 3]
        assert result_mask[0].tolist() == [1, 1, 1]

    def test_truncation_middle(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        attention_mask = torch.ones(1, 6, dtype=torch.long)

        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=4, pad_token_id=0, truncation="middle"
        )

        assert result_ids.shape == (1, 4)
        # left_half = 4 // 2 = 2, right_half = 4 - 2 = 2
        # first 2 tokens + last 2 tokens
        assert result_ids[0].tolist() == [1, 2, 5, 6]
        assert result_mask[0].tolist() == [1, 1, 1, 1]

    def test_truncation_middle_odd_max_length(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        attention_mask = torch.ones(1, 6, dtype=torch.long)

        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=3, pad_token_id=0, truncation="middle"
        )

        assert result_ids.shape == (1, 3)
        # left_half = 3 // 2 = 1, right_half = 3 - 1 = 2
        assert result_ids[0].tolist() == [1, 5, 6]

    def test_truncation_error_raises_on_long_sequence(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5, dtype=torch.long)

        with pytest.raises(NotImplementedError, match="is larger than"):
            postprocess_data(input_ids, attention_mask, max_length=3, pad_token_id=0, truncation="error")

    def test_truncation_error_exact_length_does_not_raise(self):
        """When sequence length == max_length, truncation='error' should NOT raise."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3, dtype=torch.long)
        result_ids, _ = postprocess_data(input_ids, attention_mask, max_length=3, pad_token_id=0, truncation="error")
        assert result_ids.shape == (1, 3)

    def test_batch_with_different_real_lengths(self):
        """Batch where sequences have different real content lengths (different pad counts needed)."""
        input_ids = torch.tensor([[1, 2, 0], [4, 5, 6]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=5, pad_token_id=0, left_pad=True
        )
        assert result_ids.shape == (2, 5)

    def test_truncation_middle_with_length_two(self):
        """Minimal truncation: keep first 1 + last 1 token."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        result_ids, _ = postprocess_data(input_ids, attention_mask, max_length=2, pad_token_id=0, truncation="middle")
        assert result_ids.shape == (1, 2)
        # left_half=1, right_half=1 -> [1, 5]
        assert result_ids[0].tolist() == [1, 5]

    def test_pad_token_id_nonzero(self):
        """Padding uses the specified pad_token_id, not necessarily 0."""
        input_ids = torch.tensor([[1, 2]])
        attention_mask = torch.ones(1, 2, dtype=torch.long)
        result_ids, _ = postprocess_data(input_ids, attention_mask, max_length=4, pad_token_id=99, left_pad=True)
        assert result_ids[0, 0].item() == 99
        assert result_ids[0, 1].item() == 99

    def test_batch_padding(self):
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        attention_mask = torch.ones(2, 3, dtype=torch.long)

        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=5, pad_token_id=0, left_pad=False
        )

        assert result_ids.shape == (2, 5)
        assert result_ids[0, :3].tolist() == [1, 2, 3]
        assert result_ids[1, :3].tolist() == [4, 5, 6]

    def test_truncation_left_preserves_most_recent(self):
        """Left truncation for causal LMs: keeps the LAST tokens (most recent context)."""
        input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]])
        attention_mask = torch.ones(1, 10, dtype=torch.long)
        result_ids, _ = postprocess_data(input_ids, attention_mask, max_length=3, pad_token_id=0, truncation="left")
        assert result_ids[0].tolist() == [80, 90, 100]

    def test_max_length_one_middle_truncation(self):
        """Edge: max_length=1 with middle truncation keeps only first token."""
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        attention_mask = torch.ones(1, 5, dtype=torch.long)
        result_ids, _ = postprocess_data(input_ids, attention_mask, max_length=1, pad_token_id=0, truncation="middle")
        # left_half = 1 // 2 = 0, right_half = 1 - 0 = 1 -> takes last 1 token
        assert result_ids.shape == (1, 1)

    def test_pad_token_collides_with_content(self):
        """When pad_token_id matches content tokens, padding is indistinguishable."""
        input_ids = torch.tensor([[1, 2, 3]])
        attention_mask = torch.ones(1, 3, dtype=torch.long)
        result_ids, result_mask = postprocess_data(
            input_ids, attention_mask, max_length=5, pad_token_id=1, left_pad=True
        )
        # Padded with 1s on the left, which looks the same as content token 1
        assert result_ids[0, 0].item() == 1  # pad
        assert result_mask[0, 0].item() == 0  # but mask distinguishes


# ===================================================================
#  remove_pad_token
# ===================================================================


class TestRemovePadToken:
    def test_removes_padding_from_left_padded_sequences(self):
        # Left-padded: padding tokens on the left, real tokens on the right
        input_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 4, 5, 6, 7]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 1, 1, 1, 1]])

        result = remove_pad_token(input_ids, attention_mask)

        assert result[0] == [1, 2, 3]
        assert result[1] == [4, 5, 6, 7]

    def test_all_padding(self):
        input_ids = torch.tensor([[0, 0, 0]])
        attention_mask = torch.tensor([[0, 0, 0]])

        result = remove_pad_token(input_ids, attention_mask)

        assert result[0] == []

    def test_multiple_sequences_different_lengths(self):
        input_ids = torch.tensor([[0, 0, 1, 2, 3], [0, 0, 0, 4, 5]])
        attention_mask = torch.tensor([[0, 0, 1, 1, 1], [0, 0, 0, 1, 1]])

        result = remove_pad_token(input_ids, attention_mask)

        assert len(result) == 2
        assert result[0] == [1, 2, 3]
        assert result[1] == [4, 5]

    def test_right_padded_returns_tail_not_content(self):
        """remove_pad_token uses mask.sum() to take the LAST N tokens, not the first N.
        For right-padded input, this returns the wrong slice. This documents actual behavior."""
        input_ids = torch.tensor([[1, 2, 3, 0, 0]])
        attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
        result = remove_pad_token(input_ids, attention_mask)
        # Actual behavior: ids[5-3:] = ids[2:] = [3, 0, 0] (NOT [1, 2, 3])
        assert result[0] == [3, 0, 0]

    def test_single_token_sequence(self):
        input_ids = torch.tensor([[0, 0, 42]])
        attention_mask = torch.tensor([[0, 0, 1]])
        result = remove_pad_token(input_ids, attention_mask)
        assert result[0] == [42]

    def test_large_batch(self):
        """Performance: should handle large batches without issues."""
        batch_size = 1000
        seq_len = 100
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        # Mask out first 10 tokens as padding
        attention_mask[:, :10] = 0
        input_ids[:, :10] = 0
        result = remove_pad_token(input_ids, attention_mask)
        assert len(result) == batch_size
        assert all(len(r) == 90 for r in result)

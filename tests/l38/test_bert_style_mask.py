"""Tests for the 80/10/10 BERT-style masking corruption (train_phage_esm_v2.py),
using a fake tokenizer so no real ESM-2 weights/network access are needed."""
import torch

from src.l38.train_phage_esm_v2 import bert_style_mask


class FakeTokenizer:
    """Minimal stand-in matching the subset of the HF tokenizer API used by
    bert_style_mask: all_special_ids, mask_token_id, convert_tokens_to_ids."""

    mask_token_id = 32
    all_special_ids = [0, 1, 2, 3, 32]  # pad, cls, eos, unk, mask
    _aa_to_id = {aa: 4 + i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}

    def convert_tokens_to_ids(self, token):
        return self._aa_to_id[token]


def test_bert_style_mask_labels_only_selected_positions():
    tok = FakeTokenizer()
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 2]])  # cls, 4 residues, eos
    gen = torch.Generator().manual_seed(0)

    masked_ids, labels = bert_style_mask(input_ids, tok, mask_prob=1.0, seed_generator=gen)

    # special tokens (cls=1, eos=2) never selected regardless of mask_prob
    assert labels[0, 0].item() == -100
    assert labels[0, 5].item() == -100
    # with mask_prob=1.0, all non-special positions are selected -> labeled with original id
    for pos in [1, 2, 3, 4]:
        assert labels[0, pos].item() == input_ids[0, pos].item()


def test_bert_style_mask_zero_prob_selects_nothing():
    tok = FakeTokenizer()
    input_ids = torch.tensor([[1, 5, 6, 7, 8, 2]])
    gen = torch.Generator().manual_seed(0)

    masked_ids, labels = bert_style_mask(input_ids, tok, mask_prob=0.0, seed_generator=gen)

    assert (labels == -100).all()
    assert torch.equal(masked_ids, input_ids)


def test_bert_style_mask_produces_mix_of_mask_random_and_unchanged():
    tok = FakeTokenizer()
    torch.manual_seed(0)
    # Large batch so all three corruption branches (80/10/10) are exercised
    input_ids = torch.randint(4, 24, (1, 2000))
    gen = torch.Generator().manual_seed(1)

    masked_ids, labels = bert_style_mask(input_ids, tok, mask_prob=1.0, seed_generator=gen)

    selected = labels != -100
    n_selected = selected.sum().item()
    n_mask_token = (masked_ids[selected] == tok.mask_token_id).sum().item()
    n_unchanged = (masked_ids[selected] == input_ids[selected]).sum().item()

    # roughly 80% mask, 10% random substitution, 10% unchanged -- allow generous tolerance
    assert n_selected > 1900
    frac_mask = n_mask_token / n_selected
    frac_unchanged = n_unchanged / n_selected
    assert 0.7 < frac_mask < 0.9, f"expected ~80% masked, got {frac_mask:.2f}"
    assert 0.03 < frac_unchanged < 0.2, f"expected ~10% unchanged, got {frac_unchanged:.2f}"


def test_bert_style_mask_never_masks_special_tokens():
    tok = FakeTokenizer()
    input_ids = torch.tensor([[1, 5, 6, 0, 0, 2]])  # includes pad (0)
    gen = torch.Generator().manual_seed(0)

    masked_ids, labels = bert_style_mask(input_ids, tok, mask_prob=1.0, seed_generator=gen)

    for pos, tid in enumerate(input_ids[0].tolist()):
        if tid in tok.all_special_ids:
            assert labels[0, pos].item() == -100
            assert masked_ids[0, pos].item() == tid

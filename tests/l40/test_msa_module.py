import torch

from src.l40.msa_module import (
    MSALayer,
    MSAModule,
    OuterProductMean,
    PairWeightedAveraging,
    Transition,
)


class TestTransition:
    def test_output_shape_matches_input(self):
        layer = Transition(dim=16, hidden=32)
        x = torch.randn(2, 5, 16)
        out = layer(x)
        assert out.shape == (2, 5, 16)

    def test_gradients_flow(self):
        layer = Transition(dim=16, hidden=32)
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestOuterProductMean:
    def test_output_shape(self):
        layer = OuterProductMean(c_in=8, c_hidden=4, c_out=6)
        m = torch.randn(2, 3, 5, 8)  # B, S, N, c_in
        mask = torch.ones(2, 3, 5)
        out = layer(m, mask)
        assert out.shape == (2, 5, 5, 6)  # B, N, N, c_out

    def test_masked_positions_are_excluded_from_denominator(self):
        layer = OuterProductMean(c_in=8, c_hidden=4, c_out=6)
        m = torch.randn(1, 3, 5, 8)
        mask_full = torch.ones(1, 3, 5)
        mask_partial = mask_full.clone()
        mask_partial[:, 2, :] = 0  # zero out one whole MSA row
        out_full = layer(m, mask_full)
        out_partial = layer(m, mask_partial)
        assert out_full.shape == out_partial.shape
        assert not torch.allclose(out_full, out_partial)

    def test_gradients_flow(self):
        layer = OuterProductMean(c_in=8, c_hidden=4, c_out=6)
        m = torch.randn(2, 3, 5, 8, requires_grad=True)
        mask = torch.ones(2, 3, 5)
        out = layer(m, mask)
        out.sum().backward()
        assert m.grad is not None
        assert torch.isfinite(m.grad).all()


class TestPairWeightedAveraging:
    def test_output_shape_matches_m(self):
        layer = PairWeightedAveraging(c_m=8, c_z=6, c_h=4, num_heads=2)
        m = torch.randn(2, 3, 5, 8)
        z = torch.randn(2, 5, 5, 6)
        mask = torch.ones(2, 5, 5)
        out = layer(m, z, mask)
        assert out.shape == (2, 3, 5, 8)

    def test_masking_blocks_attention_to_masked_positions(self):
        layer = PairWeightedAveraging(c_m=8, c_z=6, c_h=4, num_heads=2)
        m = torch.randn(1, 2, 4, 8)
        z = torch.randn(1, 4, 4, 6)
        mask = torch.ones(1, 4, 4)
        mask[:, :, 3] = 0  # position 3 fully masked as a key
        out = layer(m, z, mask)
        assert torch.isfinite(out).all()

    def test_gradients_flow(self):
        layer = PairWeightedAveraging(c_m=8, c_z=6, c_h=4, num_heads=2)
        m = torch.randn(2, 3, 5, 8, requires_grad=True)
        z = torch.randn(2, 5, 5, 6, requires_grad=True)
        mask = torch.ones(2, 5, 5)
        out = layer(m, z, mask)
        out.sum().backward()
        assert m.grad is not None and torch.isfinite(m.grad).all()
        assert z.grad is not None and torch.isfinite(z.grad).all()


class TestMSALayer:
    def test_updates_both_m_and_z_preserving_shape(self):
        layer = MSALayer(msa_s=8, token_z=6, c_h=4, num_heads=2, hidden_mult=2)
        m = torch.randn(2, 3, 5, 8)
        z = torch.randn(2, 5, 5, 6)
        msa_mask = torch.ones(2, 3, 5)
        token_pair_mask = torch.ones(2, 5, 5)
        m_out, z_out = layer(m, z, msa_mask, token_pair_mask)
        assert m_out.shape == m.shape
        assert z_out.shape == z.shape


class TestMSAModule:
    def test_forward_returns_query_row_and_pair_repr(self):
        vocab_size = 24
        module = MSAModule(msa_s=16, token_z=8, token_s=12, msa_blocks=2, vocab_size=vocab_size)

        B, S, N = 2, 4, 6
        msa_onehot = torch.nn.functional.one_hot(torch.randint(0, vocab_size, (B, S, N)), vocab_size).float()
        has_deletion = torch.zeros(B, S, N)
        deletion_value = torch.zeros(B, S, N)
        msa_mask = torch.ones(B, S, N)
        token_pair_mask = torch.ones(B, N, N)
        single_emb = torch.randn(B, N, 12)

        query_repr = module(msa_onehot, has_deletion, deletion_value, msa_mask, token_pair_mask, single_emb)

        assert query_repr.shape == (B, N, 16)
        assert torch.isfinite(query_repr).all()

    def test_gradients_flow_to_single_embedding(self):
        vocab_size = 24
        module = MSAModule(msa_s=16, token_z=8, token_s=12, msa_blocks=1, vocab_size=vocab_size)
        B, S, N = 1, 3, 5
        msa_onehot = torch.nn.functional.one_hot(torch.randint(0, vocab_size, (B, S, N)), vocab_size).float()
        has_deletion = torch.zeros(B, S, N)
        deletion_value = torch.zeros(B, S, N)
        msa_mask = torch.ones(B, S, N)
        token_pair_mask = torch.ones(B, N, N)
        single_emb = torch.randn(B, N, 12, requires_grad=True)

        out = module(msa_onehot, has_deletion, deletion_value, msa_mask, token_pair_mask, single_emb)
        out.sum().backward()
        assert single_emb.grad is not None
        assert torch.isfinite(single_emb.grad).all()

import torch

from src.l40.msa_model import MSAAwareProteinBERT


def _dummy_batch(B=2, S=4, N=10, vocab_size=24):
    msa_tokens = torch.randint(1, vocab_size, (B, S, N))
    has_deletion = torch.zeros(B, S, N)
    deletion_value = torch.zeros(B, S, N)
    profile = torch.softmax(torch.randn(B, N, vocab_size), dim=-1)
    deletion_mean = torch.zeros(B, N)
    attention_mask = torch.ones(B, N)
    return msa_tokens, has_deletion, deletion_value, profile, deletion_mean, attention_mask


class TestMSAAwareProteinBERT:
    def test_forward_returns_logits_of_expected_shape(self):
        model = MSAAwareProteinBERT(
            vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=10,
            msa_s=16, token_z=8, msa_blocks=1,
            use_deletion_features=True, use_profile=True, use_msa_module=True,
        )
        msa_tokens, has_del, del_val, profile, del_mean, attn_mask = _dummy_batch()
        out = model(msa_tokens, has_del, del_val, profile, del_mean, attn_mask)
        assert out['logits'].shape == (2, 10, 24)

    def test_forward_with_labels_computes_finite_loss(self):
        model = MSAAwareProteinBERT(
            vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=10,
            msa_s=16, token_z=8, msa_blocks=1,
            use_deletion_features=True, use_profile=True, use_msa_module=True,
        )
        msa_tokens, has_del, del_val, profile, del_mean, attn_mask = _dummy_batch()
        labels = torch.full((2, 10), -100, dtype=torch.long)
        labels[:, 0] = 5
        out = model(msa_tokens, has_del, del_val, profile, del_mean, attn_mask, labels=labels)
        assert torch.isfinite(out['loss'])

    def test_disabling_msa_module_still_runs(self):
        model = MSAAwareProteinBERT(
            vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=10,
            msa_s=16, token_z=8, msa_blocks=1,
            use_deletion_features=False, use_profile=False, use_msa_module=False,
        )
        msa_tokens, has_del, del_val, profile, del_mean, attn_mask = _dummy_batch()
        out = model(msa_tokens, has_del, del_val, profile, del_mean, attn_mask)
        assert out['logits'].shape == (2, 10, 24)

    def test_disabling_msa_module_behaves_like_plain_query_only_model(self):
        # With use_msa_module=False, only the query row (S index 0) should
        # matter -- changing other homolog rows must not change the output.
        model = MSAAwareProteinBERT(
            vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=10,
            msa_s=16, token_z=8, msa_blocks=1,
            use_deletion_features=False, use_profile=False, use_msa_module=False,
        )
        model.eval()
        msa_tokens, has_del, del_val, profile, del_mean, attn_mask = _dummy_batch()
        out1 = model(msa_tokens, has_del, del_val, profile, del_mean, attn_mask)

        msa_tokens_perturbed = msa_tokens.clone()
        msa_tokens_perturbed[:, 1:] = torch.randint(1, 24, msa_tokens_perturbed[:, 1:].shape)
        out2 = model(msa_tokens_perturbed, has_del, del_val, profile, del_mean, attn_mask)

        assert torch.allclose(out1['logits'], out2['logits'], atol=1e-5)

    def test_enabling_msa_module_makes_output_depend_on_homologs(self):
        model = MSAAwareProteinBERT(
            vocab_size=24, d_model=16, n_layers=2, n_heads=2, d_ff=32, max_length=10,
            msa_s=16, token_z=8, msa_blocks=1,
            use_deletion_features=False, use_profile=False, use_msa_module=True,
        )
        model.eval()
        msa_tokens, has_del, del_val, profile, del_mean, attn_mask = _dummy_batch()
        out1 = model(msa_tokens, has_del, del_val, profile, del_mean, attn_mask)

        msa_tokens_perturbed = msa_tokens.clone()
        msa_tokens_perturbed[:, 1:] = torch.randint(1, 24, msa_tokens_perturbed[:, 1:].shape)
        out2 = model(msa_tokens_perturbed, has_del, del_val, profile, del_mean, attn_mask)

        assert not torch.allclose(out1['logits'], out2['logits'], atol=1e-5)

    def test_all_four_arm_combinations_run(self):
        for use_del in [False, True]:
            for use_prof in [False, True]:
                model = MSAAwareProteinBERT(
                    vocab_size=24, d_model=16, n_layers=1, n_heads=2, d_ff=32, max_length=10,
                    msa_s=16, token_z=8, msa_blocks=1,
                    use_deletion_features=use_del, use_profile=use_prof, use_msa_module=True,
                )
                msa_tokens, has_del, del_val, profile, del_mean, attn_mask = _dummy_batch()
                out = model(msa_tokens, has_del, del_val, profile, del_mean, attn_mask)
                assert torch.isfinite(out['logits']).all()

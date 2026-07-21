import numpy as np
import torch

from src.l40.msa_aware_data import MSAAwareDataset, load_msa_aware_protein_data
from src.l40.vocab import PAD_TOKEN


def _write_npz(path, seq_len, n_homologs, taxonomy_ids=None):
    """Writes a Boltz-style .npz with a query (taxonomy=-1) plus n_homologs
    aligned sequences, each seq_len long, no deletions, distinct taxa."""
    if taxonomy_ids is None:
        taxonomy_ids = list(range(100, 100 + n_homologs))
    n_seqs = 1 + n_homologs
    starts = [i * seq_len for i in range(n_seqs)]
    ends = [(i + 1) * seq_len for i in range(n_seqs)]
    taxa = [-1] + taxonomy_ids
    sequences = np.array(
        list(zip(range(n_seqs), taxa, starts, ends, [0] * n_seqs, [0] * n_seqs)),
        dtype=[('seq_idx', 'i2'), ('taxonomy', 'i4'), ('res_start', 'i4'),
               ('res_end', 'i4'), ('del_start', 'i4'), ('del_end', 'i4')],
    )
    deletions = np.array([], dtype=[('res_idx', 'i2'), ('deletion', 'i2')])
    res_types = (np.arange(n_seqs * seq_len, dtype='i4') % 20 + 1)
    residues = np.array(list(zip(res_types)), dtype=[('res_type', 'i4')])
    np.savez(path, sequences=sequences, residues=residues, deletions=deletions)


class TestLoadMsaAwareProteinData:
    def test_indexes_npz_files_one_record_per_file(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_len=40, n_homologs=3)
        _write_npz(tmp_path / "b.npz", seq_len=40, n_homologs=3)

        file_sequences = load_msa_aware_protein_data(str(tmp_path), max_files=10)

        assert set(file_sequences.keys()) == {"a.npz", "b.npz"}
        assert len(file_sequences["a.npz"]) == 1


class TestMSAAwareDataset:
    def test_returns_expected_tensor_shapes(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_len=40, n_homologs=4)
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz'}
        dataset = MSAAwareDataset(records=[record], max_length=48, msa_depth=5, fixed_seed=0)

        item = dataset[0]

        assert item['msa_tokens'].shape == (5, 48)
        assert item['has_deletion'].shape == (5, 48)
        assert item['deletion_value'].shape == (5, 48)
        assert item['profile'].shape == (48, 24)
        assert item['deletion_mean'].shape == (48,)
        assert item['attention_mask'].shape == (48,)
        assert item['labels'].shape == (48,)

    def test_query_row_is_masked_but_homolog_rows_are_not(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_len=40, n_homologs=4)
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz'}
        dataset = MSAAwareDataset(records=[record], max_length=48, msa_depth=5,
                                   mask_prob=1.0, fixed_seed=0)

        item = dataset[0]

        # With mask_prob=1.0, every query position should be labeled (masked).
        assert (item['labels'][:40] != -100).all()
        # Homolog rows (1..4) must retain their real, un-corrupted token ids --
        # they were never passed through masking.
        for row in range(1, 5):
            assert (item['msa_tokens'][row, :40] >= 1).all()
            assert (item['msa_tokens'][row, :40] <= 20).all()

    def test_padding_beyond_query_length_uses_pad_token(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_len=20, n_homologs=3)
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz'}
        dataset = MSAAwareDataset(records=[record], max_length=32, msa_depth=4, fixed_seed=0)

        item = dataset[0]

        assert (item['msa_tokens'][:, 20:] == PAD_TOKEN).all()
        assert (item['attention_mask'][20:] == 0).all()

    def test_pads_msa_depth_when_fewer_homologs_available(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_len=20, n_homologs=2)  # only 2 homologs
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz'}
        dataset = MSAAwareDataset(records=[record], max_length=32, msa_depth=6, fixed_seed=0)

        item = dataset[0]

        assert item['msa_tokens'].shape[0] == 6  # padded up to msa_depth via repeats

    def test_deterministic_given_fixed_seed(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_len=40, n_homologs=4)
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz'}
        ds1 = MSAAwareDataset(records=[record], max_length=48, msa_depth=5, fixed_seed=42)
        ds2 = MSAAwareDataset(records=[record], max_length=48, msa_depth=5, fixed_seed=42)

        item1, item2 = ds1[0], ds2[0]

        assert torch.equal(item1['msa_tokens'], item2['msa_tokens'])
        assert torch.equal(item1['labels'], item2['labels'])

    def test_profile_computed_from_unmasked_original_sequences(self, tmp_path):
        # All homologs + query share identical residue content at every position
        # in this fixture (see _write_npz's res_types % 20 + 1 pattern is NOT
        # actually identical across rows -- use a custom npz where every row is
        # identical to make the profile assertion unambiguous).
        path = tmp_path / "a.npz"
        seq_len, n_seqs = 20, 3  # keep residue values (1..seq_len) within vocab_size=24
        starts = [i * seq_len for i in range(n_seqs)]
        ends = [(i + 1) * seq_len for i in range(n_seqs)]
        sequences = np.array(
            list(zip(range(n_seqs), [-1, 100, 200], starts, ends, [0] * n_seqs, [0] * n_seqs)),
            dtype=[('seq_idx', 'i2'), ('taxonomy', 'i4'), ('res_start', 'i4'),
                   ('res_end', 'i4'), ('del_start', 'i4'), ('del_end', 'i4')],
        )
        deletions = np.array([], dtype=[('res_idx', 'i2'), ('deletion', 'i2')])
        res_types = np.tile(np.arange(1, seq_len + 1, dtype='i4'), n_seqs)  # identical across rows
        residues = np.array(list(zip(res_types)), dtype=[('res_type', 'i4')])
        np.savez(path, sequences=sequences, residues=residues, deletions=deletions)

        record = {'data_dir': str(tmp_path), 'filename': 'a.npz'}
        dataset = MSAAwareDataset(records=[record], max_length=32, msa_depth=3, mask_prob=1.0, fixed_seed=0)
        item = dataset[0]

        # Position 0 has value 1 in every row -> profile[0, 1] should be 1.0
        # regardless of query masking (profile uses the ORIGINAL query, not the
        # corrupted training input).
        assert torch.isclose(item['profile'][0, 1], torch.tensor(1.0))

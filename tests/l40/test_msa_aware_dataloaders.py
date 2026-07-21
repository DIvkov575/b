import numpy as np

from src.l40.msa_aware_data import create_msa_aware_dataloaders


def _write_npz(path, seq_len, n_homologs):
    n_seqs = 1 + n_homologs
    starts = [i * seq_len for i in range(n_seqs)]
    ends = [(i + 1) * seq_len for i in range(n_seqs)]
    taxa = [-1] + list(range(100, 100 + n_homologs))
    sequences = np.array(
        list(zip(range(n_seqs), taxa, starts, ends, [0] * n_seqs, [0] * n_seqs)),
        dtype=[('seq_idx', 'i2'), ('taxonomy', 'i4'), ('res_start', 'i4'),
               ('res_end', 'i4'), ('del_start', 'i4'), ('del_end', 'i4')],
    )
    deletions = np.array([], dtype=[('res_idx', 'i2'), ('deletion', 'i2')])
    res_types = (np.arange(n_seqs * seq_len, dtype='i4') % 20 + 1)
    residues = np.array(list(zip(res_types)), dtype=[('res_type', 'i4')])
    np.savez(path, sequences=sequences, residues=residues, deletions=deletions)


class TestCreateMsaAwareDataloaders:
    def test_returns_loaders_with_expected_batch_shape(self, tmp_path):
        for i in range(10):
            _write_npz(tmp_path / f"s{i}.npz", seq_len=40, n_homologs=4)

        train_loader, val_loader, test_loader = create_msa_aware_dataloaders(
            str(tmp_path), batch_size=4, max_length=48, max_files=10, msa_depth=5,
        )

        batch = next(iter(train_loader))
        assert batch['msa_tokens'].shape[1:] == (5, 48)
        assert batch['profile'].shape[1:] == (48, 24)

    def test_split_matches_plain_msa_data_split(self, tmp_path):
        # create_msa_aware_dataloaders must use the SAME create_diverse_splits
        # seed as msa_data.py, so the boltz and msa-aware ablation arms compare
        # apples-to-apples on file-level split.
        from src.l40.msa_data import create_diverse_splits, load_protein_data
        from src.l40.msa_aware_data import load_msa_aware_protein_data

        for i in range(20):
            _write_npz(tmp_path / f"s{i}.npz", seq_len=40, n_homologs=4)

        plain_fs = load_protein_data(str(tmp_path), max_files=20, sequences_per_file=1)
        plain_train, plain_val, plain_test = create_diverse_splits(plain_fs)
        plain_train_files = {s['filename'] for s in plain_train}

        aware_fs = load_msa_aware_protein_data(str(tmp_path), max_files=20)
        aware_train, aware_val, aware_test = create_diverse_splits(
            aware_fs, max_seqs_per_file_train=1, max_seqs_per_file_val=1,
        )
        aware_train_files = {s['filename'] for s in aware_train}

        assert plain_train_files == aware_train_files

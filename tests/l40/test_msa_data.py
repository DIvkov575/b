import numpy as np
import torch

from src.l40.msa_data import (
    MSADataset,
    create_diverse_splits,
    create_dataloaders,
    load_protein_data,
)
from src.l40.vocab import MASK_TOKEN, PAD_TOKEN


def _write_npz(path, seq_lengths, residue_fill=None):
    """Writes a Boltz-style MSA .npz shard with one sequence per entry in seq_lengths."""
    starts = []
    ends = []
    cursor = 0
    for length in seq_lengths:
        starts.append(cursor)
        cursor += length
        ends.append(cursor)

    sequences = np.array(
        list(zip(starts, ends)), dtype=[('res_start', 'i4'), ('res_end', 'i4')]
    )

    total_residues = cursor
    if residue_fill is None:
        res_types = np.arange(total_residues, dtype='i4') % 20 + 1  # avoid PAD/MASK/GAP
    else:
        res_types = np.array(residue_fill, dtype='i4')
    residues = np.array(list(zip(res_types)), dtype=[('res_type', 'i4')])

    np.savez(path, sequences=sequences, residues=residues)


class TestLoadSequence:
    def test_loads_correct_slice_for_seq_idx(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_lengths=[5, 8])
        dataset = MSADataset(records=[])
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz', 'seq_idx': 1, 'length': None}
        sequence = dataset._load_sequence(record)
        assert len(sequence) == 8

    def test_out_of_range_seq_idx_returns_empty(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_lengths=[5])
        dataset = MSADataset(records=[])
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz', 'seq_idx': 5, 'length': None}
        sequence = dataset._load_sequence(record)
        assert len(sequence) == 0


class TestGetItem:
    def test_short_sequence_returns_zero_loss_dummy(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_lengths=[10])
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz', 'seq_idx': 0, 'length': None}
        dataset = MSADataset(records=[record], max_length=32)

        item = dataset[0]

        assert item['seq_length'] == 0
        assert torch.equal(item['labels'], torch.full((32,), -100, dtype=torch.long))
        assert torch.equal(item['attention_mask'], torch.zeros(32))

    def test_pads_to_max_length(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_lengths=[40])
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz', 'seq_idx': 0, 'length': None}
        dataset = MSADataset(records=[record], max_length=64, fixed_seed=0)

        item = dataset[0]

        assert item['input_ids'].shape == (64,)
        assert item['seq_length'] == 40
        assert (item['input_ids'][40:] == PAD_TOKEN).all()

    def test_truncates_beyond_max_length(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_lengths=[100])
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz', 'seq_idx': 0, 'length': None}
        dataset = MSADataset(records=[record], max_length=32, fixed_seed=0)

        item = dataset[0]

        assert item['input_ids'].shape == (32,)

    def test_fixed_seed_gives_deterministic_masking(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_lengths=[40])
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz', 'seq_idx': 0, 'length': None}

        ds1 = MSADataset(records=[record], max_length=64, fixed_seed=42)
        ds2 = MSADataset(records=[record], max_length=64, fixed_seed=42)

        item1 = ds1[0]
        item2 = ds2[0]

        assert torch.equal(item1['input_ids'], item2['input_ids'])
        assert torch.equal(item1['labels'], item2['labels'])

    def test_masked_positions_use_mask_token_or_random_or_original(self, tmp_path):
        _write_npz(tmp_path / "a.npz", seq_lengths=[100])
        record = {'data_dir': str(tmp_path), 'filename': 'a.npz', 'seq_idx': 0, 'length': None}
        dataset = MSADataset(records=[record], max_length=100, mask_prob=1.0, fixed_seed=0)

        item = dataset[0]

        masked_positions = (item['labels'] != -100)
        assert masked_positions.all()
        # Every masked position's token is either MASK, a random AA (1-21), or unchanged original.
        assert ((item['input_ids'] == MASK_TOKEN) | (item['input_ids'] >= 1)).all()


class TestLoadProteinData:
    def test_indexes_npz_files_without_opening_them(self, tmp_path):
        (tmp_path / "a.npz").write_bytes(b"not a real npz")
        (tmp_path / "b.npz").write_bytes(b"not a real npz")
        (tmp_path / "c.txt").write_text("ignored")

        file_sequences = load_protein_data(str(tmp_path), max_files=10, sequences_per_file=3)

        assert set(file_sequences.keys()) == {"a.npz", "b.npz"}
        assert len(file_sequences["a.npz"]) == 3
        assert file_sequences["a.npz"][0]['data_dir'] == str(tmp_path)

    def test_respects_max_files(self, tmp_path):
        for name in ["a.npz", "b.npz", "c.npz"]:
            (tmp_path / name).write_bytes(b"")

        file_sequences = load_protein_data(str(tmp_path), max_files=2, sequences_per_file=1)

        assert len(file_sequences) == 2


class TestCreateDiverseSplits:
    def _make_file_sequences(self, n_files=20, seqs_per_file=5):
        return {
            f"file_{i}.npz": [
                {'data_dir': '/x', 'filename': f'file_{i}.npz', 'seq_idx': j, 'length': None}
                for j in range(seqs_per_file)
            ]
            for i in range(n_files)
        }

    def test_split_ratios_partition_all_files(self):
        file_sequences = self._make_file_sequences(n_files=20)

        train, val, test = create_diverse_splits(file_sequences, train_file_ratio=0.7, val_file_ratio=0.15)

        train_files = {s['filename'] for s in train}
        val_files = {s['filename'] for s in val}
        test_files = {s['filename'] for s in test}

        assert train_files.isdisjoint(val_files)
        assert train_files.isdisjoint(test_files)
        assert val_files.isdisjoint(test_files)
        assert len(train_files) + len(val_files) + len(test_files) == 20

    def test_deterministic_given_fixed_internal_seed(self):
        file_sequences = self._make_file_sequences(n_files=20)

        train1, val1, test1 = create_diverse_splits(file_sequences)
        train2, val2, test2 = create_diverse_splits(file_sequences)

        assert [s['filename'] for s in train1] == [s['filename'] for s in train2]
        assert [s['filename'] for s in val1] == [s['filename'] for s in val2]
        assert [s['filename'] for s in test1] == [s['filename'] for s in test2]

    def test_caps_sequences_per_file(self):
        file_sequences = self._make_file_sequences(n_files=20, seqs_per_file=10)

        train, val, _ = create_diverse_splits(
            file_sequences, max_seqs_per_file_train=5, max_seqs_per_file_val=2
        )

        from collections import Counter
        train_counts = Counter(s['filename'] for s in train)
        val_counts = Counter(s['filename'] for s in val)
        assert all(c <= 5 for c in train_counts.values())
        assert all(c <= 2 for c in val_counts.values())

    def test_test_set_gets_single_sequence_per_file(self):
        file_sequences = self._make_file_sequences(n_files=20)

        _, _, test = create_diverse_splits(file_sequences, train_file_ratio=0.5, val_file_ratio=0.2)

        from collections import Counter
        test_counts = Counter(s['filename'] for s in test)
        assert all(c == 1 for c in test_counts.values())


class TestCreateDataloaders:
    def test_returns_loaders_with_expected_batch_shapes(self, tmp_path):
        for i in range(10):
            _write_npz(tmp_path / f"f{i}.npz", seq_lengths=[40])

        train_loader, val_loader, test_loader = create_dataloaders(
            str(tmp_path), batch_size=4, max_length=48, max_files=10, sequences_per_file=1
        )

        batch = next(iter(train_loader))
        assert batch['input_ids'].shape[1] == 48
        assert batch['input_ids'].shape[0] <= 4
        assert 'labels' in batch and 'attention_mask' in batch

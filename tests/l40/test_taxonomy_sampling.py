import numpy as np

from src.l40.taxonomy_sampling import sample_diverse_homologs


def _make_sequences(taxonomy_ids):
    return np.array(
        [(i, tax, 0, 10, 0, 0) for i, tax in enumerate(taxonomy_ids)],
        dtype=[('seq_idx', 'i2'), ('taxonomy', 'i4'), ('res_start', 'i4'),
               ('res_end', 'i4'), ('del_start', 'i4'), ('del_end', 'i4')],
    )


class TestSampleDiverseHomologs:
    def test_always_includes_the_query_sequence(self):
        sequences = _make_sequences([-1, 100, 200, 300])
        rng = np.random.RandomState(0)
        selected = sample_diverse_homologs(sequences, n=2, rng=rng)
        assert selected[0]['seq_idx'] == 0

    def test_picks_distinct_taxa_before_repeating_any(self):
        sequences = _make_sequences([-1, 100, 100, 200, 300])
        rng = np.random.RandomState(0)
        selected = sample_diverse_homologs(sequences, n=3, rng=rng)
        taxa_selected = [s['taxonomy'] for s in selected[1:]]  # exclude query
        assert len(set(taxa_selected)) == len(taxa_selected)

    def test_falls_back_to_repeats_when_fewer_distinct_taxa_than_n(self):
        sequences = _make_sequences([-1, 100, 200])  # only 2 distinct homolog taxa
        rng = np.random.RandomState(0)
        selected = sample_diverse_homologs(sequences, n=4, rng=rng)
        assert len(selected) == 4  # query + 3 homolog picks, even though only 2 distinct taxa exist

    def test_returns_at_most_n_sequences(self):
        sequences = _make_sequences([-1, 100, 200, 300, 400, 500])
        rng = np.random.RandomState(0)
        selected = sample_diverse_homologs(sequences, n=3, rng=rng)
        assert len(selected) == 3

    def test_deterministic_given_same_rng_state(self):
        sequences = _make_sequences([-1, 100, 100, 200, 200, 300])
        selected1 = sample_diverse_homologs(sequences, n=3, rng=np.random.RandomState(7))
        selected2 = sample_diverse_homologs(sequences, n=3, rng=np.random.RandomState(7))
        assert [s['seq_idx'] for s in selected1] == [s['seq_idx'] for s in selected2]

    def test_single_sequence_returns_just_the_query(self):
        sequences = _make_sequences([-1])
        rng = np.random.RandomState(0)
        selected = sample_diverse_homologs(sequences, n=5, rng=rng)
        assert len(selected) == 1
        assert selected[0]['seq_idx'] == 0

import numpy as np

from src.l40.train_msa_ablation import run_msa_ablation_training


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


def test_run_msa_ablation_training_produces_finite_metrics_for_each_arm(tmp_path):
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    for i in range(12):
        _write_npz(npz_dir / f"s{i}.npz", seq_len=40, n_homologs=4)

    for use_del in [False, True]:
        for use_prof in [False, True]:
            out_path = tmp_path / f"arm_del{use_del}_prof{use_prof}.json"
            result = run_msa_ablation_training(
                data_path=str(npz_dir), out_path=str(out_path),
                max_files=12, max_length=48, batch_size=4, epochs=1, msa_depth=5,
                d_model=16, n_layers=1, n_heads=2, d_ff=32, msa_s=16, token_z=8, msa_blocks=1,
                use_deletion_features=use_del, use_profile=use_prof, use_msa_module=True,
                seed=0,
            )
            assert np.isfinite(result["epochs"][0]["train_loss"])
            assert np.isfinite(result["epochs"][0]["val_loss"])
            assert 0.0 <= result["epochs"][0]["val_accuracy"] <= 1.0

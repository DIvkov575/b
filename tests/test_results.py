import json
from pathlib import Path
from src.results import write_results_csv, write_summary, rank_designs


def test_write_results_csv(tmp_path):
    results = [
        {"path": "a.pdb", "contacts_a": 10, "contacts_b": 8, "n_residues": 120,
         "pae_a": 7.5, "pae_b": 8.2, "n_clashes": 2, "passes": True},
        {"path": "b.pdb", "contacts_a": 3, "contacts_b": 1, "n_residues": 110,
         "pae_a": 15.0, "pae_b": 20.0, "n_clashes": 0, "passes": False},
    ]
    out = str(tmp_path / "results.csv")
    write_results_csv(results, out)
    assert Path(out).exists()
    content = Path(out).read_text()
    assert "design_id" in content
    assert "a.pdb" in content


def test_write_results_csv_empty(tmp_path):
    out = str(tmp_path / "empty.csv")
    write_results_csv([], out)


def test_write_summary(tmp_path):
    results = [
        {"contacts_a": 10, "contacts_b": 8, "passes": True},
        {"contacts_a": 2, "contacts_b": 1, "passes": False},
        {"contacts_a": 7, "contacts_b": 6, "passes": True},
    ]
    out = str(tmp_path / "summary.json")
    write_summary(results, out)
    data = json.loads(Path(out).read_text())
    assert data["total_designs"] == 3
    assert data["pass_all_filters"] == 2


def test_rank_designs():
    results = [
        {"path": "a.pdb", "pae_a": 15, "pae_b": 15, "contacts_a": 5, "contacts_b": 5},
        {"path": "b.pdb", "pae_a": 5, "pae_b": 5, "contacts_a": 10, "contacts_b": 10},
        {"path": "c.pdb", "pae_a": 8, "pae_b": 8, "contacts_a": 8, "contacts_b": 8},
    ]
    ranked = rank_designs(results)
    assert ranked[0]["path"] == "b.pdb"  # best: low pae + high contacts
    assert ranked[-1]["path"] == "a.pdb"  # worst: high pae + low contacts

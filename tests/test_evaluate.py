import numpy as np
import json
from src.evaluate import check_ternary_clash, evaluate_design, build_af2_command, parse_af2_pae


def test_no_clash():
    target_a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    target_b = np.array([[50, 0, 0], [51, 0, 0], [52, 0, 0]], dtype=float)
    n_clashes = check_ternary_clash(target_a, target_b, clash_dist=2.0)
    assert n_clashes == 0


def test_clash_detected():
    target_a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    target_b = np.array([[0.5, 0, 0], [1.5, 0, 0]], dtype=float)
    n_clashes = check_ternary_clash(target_a, target_b, clash_dist=2.0)
    assert n_clashes > 0


def test_evaluate_design_passes():
    result = evaluate_design(
        design_path="mock.pdb",
        pae_a=8.5,
        pae_b=9.2,
        n_clashes=3,
        pae_threshold=10.0,
        max_clashes=5
    )
    assert result["passes"] == True


def test_evaluate_design_fails_pae():
    result = evaluate_design(
        design_path="mock.pdb",
        pae_a=12.0,
        pae_b=9.0,
        n_clashes=2
    )
    assert result["passes"] == False


def test_evaluate_design_fails_clashes():
    result = evaluate_design(
        design_path="mock.pdb",
        pae_a=5.0,
        pae_b=5.0,
        n_clashes=10,
        max_clashes=5
    )
    assert result["passes"] == False


def test_build_af2_command():
    cmd = build_af2_command("input.fasta", "output/")
    assert "colabfold_batch" in cmd
    assert "alphafold2_multimer_v3" in cmd


def test_parse_af2_pae(tmp_path):
    pae_matrix = np.random.uniform(5, 15, (20, 20)).tolist()
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"pae": pae_matrix}))
    pae = parse_af2_pae(str(result_file), chain_a_len=10)
    assert 5.0 < pae < 15.0

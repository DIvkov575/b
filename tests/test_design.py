from pathlib import Path
from src.design import (build_proteinmpnn_cmd, parse_mpnn_fastas,
                        pdb_to_fasta, write_fasta, create_af2_input)


def test_build_proteinmpnn_cmd(tmp_path):
    cmd = build_proteinmpnn_cmd(
        input_pdb="design.pdb",
        output_dir=str(tmp_path),
        chains_to_design="C",
        num_sequences=4
    )
    assert "protein_mpnn_run.py" in cmd
    assert "--designed_chain C" in cmd
    assert "--fixed_chain A B" in cmd
    assert "--num_seq_per_target 4" in cmd


def test_parse_mpnn_fastas(tmp_path):
    fasta = tmp_path / "seqs" / "design.fa"
    fasta.parent.mkdir(parents=True)
    fasta.write_text(">seq1\nMKLLVVAA\n>seq2\nMKLLGGAA\n")
    results = parse_mpnn_fastas(str(tmp_path))
    assert len(results) == 2
    assert results[0]["sequence"] == "MKLLVVAA"


def test_pdb_to_fasta():
    seq = pdb_to_fasta("tests/fixtures/ubiquitin.pdb", "A")
    assert len(seq) > 50
    assert all(c.isalpha() for c in seq)


def test_write_fasta(tmp_path):
    out = str(tmp_path / "test.fasta")
    write_fasta({"protein_A": "MKLLAAA", "protein_B": "GGGVVV"}, out)
    content = Path(out).read_text()
    assert ">protein_A" in content
    assert "MKLLAAA" in content


def test_create_af2_input(tmp_path):
    out = str(tmp_path / "af2_input.fasta")
    result = create_af2_input(
        binder_seq="MKLLAAA",
        target_pdb="tests/fixtures/ubiquitin.pdb",
        target_chain="A",
        output_path=out
    )
    assert Path(result).exists()
    content = Path(result).read_text()
    assert ">binder" in content
    assert ">target" in content

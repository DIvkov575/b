import numpy as np
from src.prepare import place_targets, generate_contig, format_hotspots, select_hotspots_auto


def test_place_targets():
    result = place_targets(
        pdb_a="tests/fixtures/ubiquitin.pdb",
        pdb_b="tests/fixtures/sumo.pdb",
        separation=60.0
    )
    assert "combined_pdb" in result
    assert "chain_a_id" in result
    assert "chain_b_id" in result
    com_dist = np.linalg.norm(
        np.array(result["com_a"]) - np.array(result["com_b"])
    )
    assert 55.0 < com_dist < 65.0


def test_generate_contig():
    contig = generate_contig(
        chain_a_length=76,
        chain_b_length=97,
        binder_min=100,
        binder_max=150
    )
    assert "A1-76" in contig
    assert "B1-97" in contig
    assert "100-150" in contig


def test_format_hotspots():
    result = format_hotspots([8, 44, 48], [35, 37, 39])
    assert "A8" in result
    assert "B35" in result
    assert result.startswith("[")
    assert result.endswith("]")


def test_select_hotspots_auto():
    hotspots = select_hotspots_auto(
        pdb_path="tests/fixtures/ubiquitin.pdb",
        chain_id="A",
        n_hotspots=5
    )
    assert len(hotspots) == 5
    assert all(isinstance(h, int) for h in hotspots)

import numpy as np
from pathlib import Path
from src.prepare import clean_pdb, sample_orientations, place_targets_multi, validate_hotspots


def test_clean_pdb(tmp_path):
    output = str(tmp_path / "clean.pdb")
    result = clean_pdb("tests/fixtures/ubiquitin.pdb", output)
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_sample_orientations():
    orients = sample_orientations(5)
    assert len(orients) == 5
    assert np.allclose(orients[0], np.eye(3))
    for rot in orients:
        assert rot.shape == (3, 3)


def test_place_targets_multi(tmp_path):
    placements = place_targets_multi(
        "tests/fixtures/ubiquitin.pdb",
        "tests/fixtures/sumo.pdb",
        separation=60.0,
        n_orientations=3,
        output_base=str(tmp_path)
    )
    assert len(placements) == 3
    for p in placements:
        assert Path(p["combined_pdb"]).exists()


def test_validate_hotspots():
    valid = validate_hotspots("tests/fixtures/ubiquitin.pdb", "A", [1, 8, 44, 9999])
    assert 8 in valid
    assert 44 in valid
    assert 9999 not in valid

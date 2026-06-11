import os
from pathlib import Path
from src.pipeline import run_pipeline


def test_pipeline_mock_mode():
    """Full pipeline runs without error in mock mode and returns a list."""
    results = run_pipeline(
        config_path="configs/targets.yaml",
        defaults_path="configs/defaults.yaml",
        mock=True
    )
    assert isinstance(results, list)


def test_pipeline_dry_run():
    """Dry run mode prints command and returns empty list."""
    results = run_pipeline(
        config_path="configs/targets.yaml",
        defaults_path="configs/defaults.yaml",
        dry_run=True
    )
    assert results == []


def test_outputs_created():
    """Mock mode creates output directory with PDB files."""
    run_pipeline(
        config_path="configs/targets.yaml",
        defaults_path="configs/defaults.yaml",
        mock=True
    )
    output_dir = Path("outputs/ubiquitin_sumo/designs")
    assert output_dir.exists()
    pdbs = list(output_dir.glob("*.pdb"))
    assert len(pdbs) > 0


def test_combined_target_created():
    """Pipeline creates the combined target PDB."""
    run_pipeline(
        config_path="configs/targets.yaml",
        defaults_path="configs/defaults.yaml",
        mock=True
    )
    combined = Path("outputs/combined_target.pdb")
    assert combined.exists()
    assert combined.stat().st_size > 0

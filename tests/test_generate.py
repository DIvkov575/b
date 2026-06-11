from src.generate import build_rfdiffusion_cmd, parse_output_pdbs, generate_mock_designs


def test_build_rfdiffusion_cmd():
    cmd = build_rfdiffusion_cmd(
        input_pdb="outputs/combined_target.pdb",
        contig="A1-76/0 B1-97/0 100-150",
        hotspots="[A8,A44,A48,B35,B37,B39]",
        output_dir="outputs/designs",
        num_designs=100,
        diffusion_steps=50
    )
    assert "run_inference.py" in cmd
    assert "A1-76/0 B1-97/0 100-150" in cmd
    assert "A8,A44,A48,B35,B37,B39" in cmd
    assert "num_designs=100" in cmd


def test_parse_output_pdbs(tmp_path):
    for i in range(3):
        (tmp_path / f"design_{i}.pdb").write_text("ATOM mock")
    pdbs = parse_output_pdbs(str(tmp_path))
    assert len(pdbs) == 3


def test_generate_mock_designs(tmp_path):
    output_dir = str(tmp_path / "designs")
    pdbs = generate_mock_designs(output_dir, "dummy.pdb", num_designs=5)
    assert len(pdbs) == 5
    # Verify PDBs are loadable
    from src.utils import load_structure, get_ca_coords
    struct = load_structure(pdbs[0])
    coords = get_ca_coords(struct)
    assert coords.shape == (120, 3)

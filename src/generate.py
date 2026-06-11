from pathlib import Path
import subprocess
import logging
import numpy as np

logger = logging.getLogger(__name__)


def build_rfdiffusion_cmd(
    input_pdb: str,
    contig: str,
    hotspots: str,
    output_dir: str,
    num_designs: int = 100,
    diffusion_steps: int = 50,
    rfdiffusion_path: str = "RFdiffusion/scripts/run_inference.py"
) -> str:
    cmd = (
        f"python {rfdiffusion_path} "
        f"inference.input_pdb={input_pdb} "
        f"'contigmap.contigs=[{contig}]' "
        f"'ppi.hotspot_res={hotspots}' "
        f"inference.output_prefix={output_dir}/design "
        f"inference.num_designs={num_designs} "
        f"diffuser.T={diffusion_steps}"
    )
    return cmd


def run_rfdiffusion(cmd: str, dry_run: bool = False) -> int:
    logger.info(f"Running: {cmd}")
    if dry_run:
        logger.info("DRY RUN — skipping actual execution")
        return 0
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"RFdiffusion failed: {result.stderr}")
    return result.returncode


def parse_output_pdbs(output_dir: str) -> list:
    output_path = Path(output_dir)
    pdbs = sorted(output_path.glob("*.pdb"))
    return [str(p) for p in pdbs]


def generate_mock_designs(output_dir: str, combined_pdb: str, num_designs: int = 10) -> list:
    from Bio.PDB import Structure, Model, Chain, Residue, Atom

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_designs):
        struct = Structure.Structure(f"design_{i}")
        model = Model.Model(0)
        struct.add(model)
        chain = Chain.Chain("C")
        model.add(chain)

        n_res = 120
        coords = np.zeros((n_res, 3))
        for j in range(1, n_res):
            direction = np.random.randn(3)
            direction = direction / np.linalg.norm(direction) * 3.8
            coords[j] = coords[j - 1] + direction

        for j in range(n_res):
            res = Residue.Residue((" ", j + 1, " "), "ALA", " ")
            atom = Atom.Atom("CA", coords[j], 1.0, 1.0, " ", "CA", j + 1, "C")
            res.add(atom)
            chain.add(res)

        from Bio.PDB import PDBIO
        io = PDBIO()
        io.set_structure(struct)
        io.save(str(output_path / f"design_{i}.pdb"))

    return parse_output_pdbs(str(output_path))

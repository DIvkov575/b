from pathlib import Path
import subprocess
import logging

logger = logging.getLogger(__name__)


def build_proteinmpnn_cmd(
    input_pdb: str,
    output_dir: str,
    chains_to_design: str = "C",
    num_sequences: int = 4,
    sampling_temp: float = 0.1,
    proteinmpnn_path: str = "ProteinMPNN/protein_mpnn_run.py"
) -> str:
    """Build ProteinMPNN command for multi-chain sequence design."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = (
        f"python {proteinmpnn_path} "
        f"--pdb_path {input_pdb} "
        f"--out_folder {output_dir} "
        f"--num_seq_per_target {num_sequences} "
        f"--sampling_temp {sampling_temp} "
        f"--batch_size 1 "
        f"--designed_chain {chains_to_design} "
        f"--fixed_chain A B"
    )
    return cmd


def run_proteinmpnn(cmd: str, dry_run: bool = False) -> int:
    logger.info(f"Running ProteinMPNN: {cmd}")
    if dry_run:
        logger.info("DRY RUN — skipping")
        return 0
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ProteinMPNN failed: {result.stderr}")
    return result.returncode


def parse_mpnn_fastas(output_dir: str) -> list:
    """Parse ProteinMPNN output FASTA files."""
    output_path = Path(output_dir)
    fastas = sorted(output_path.rglob("*.fa")) + sorted(output_path.rglob("*.fasta"))
    sequences = []
    for fasta in fastas:
        with open(fasta) as f:
            lines = f.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 < len(lines) and lines[i].startswith(">"):
                sequences.append({
                    "header": lines[i].strip(),
                    "sequence": lines[i + 1].strip(),
                    "source_file": str(fasta),
                })
    return sequences


def pdb_to_fasta(pdb_path: str, chain_id: str = "C") -> str:
    """Extract sequence from a PDB chain. Returns single-letter amino acid string."""
    from Bio.PDB import PDBParser
    from Bio.Data.IUPACData import protein_letters_3to1

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("protein", pdb_path)
    model = struct[0]

    sequence = []
    for residue in model[chain_id]:
        if residue.get_id()[0] != " ":
            continue
        resname = residue.get_resname()
        one_letter = protein_letters_3to1.get(resname.capitalize(), "X")
        sequence.append(one_letter)

    return "".join(sequence)


def write_fasta(sequences: dict, output_path: str):
    """Write a FASTA file. sequences = {header: sequence}"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for header, seq in sequences.items():
            f.write(f">{header}\n{seq}\n")
    return output_path


def create_af2_input(binder_seq: str, target_pdb: str, target_chain: str,
                     output_path: str) -> str:
    """Create a FASTA file with binder + target sequences for AF2-Multimer."""
    target_seq = pdb_to_fasta(target_pdb, target_chain)
    sequences = {
        "binder": binder_seq,
        "target": target_seq,
    }
    return write_fasta(sequences, output_path)

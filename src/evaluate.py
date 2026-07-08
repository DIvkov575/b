import numpy as np
from src.utils import check_steric_clash
import json
import logging

logger = logging.getLogger(__name__)


def check_ternary_clash(target_a_coords: np.ndarray, target_b_coords: np.ndarray,
                        clash_dist: float = 2.0) -> int:
    return check_steric_clash(target_a_coords, target_b_coords, clash_dist)


def evaluate_design(design_path: str, pae_a: float, pae_b: float,
                    n_clashes: int, pae_threshold: float = 10.0,
                    max_clashes: int = 5) -> dict:
    passes = (pae_a < pae_threshold and
              pae_b < pae_threshold and
              n_clashes <= max_clashes)
    return {
        "path": design_path,
        "pae_a": pae_a,
        "pae_b": pae_b,
        "n_clashes": n_clashes,
        "passes": passes,
    }


def build_af2_command(fasta_path: str, output_dir: str) -> str:
    return (
        f"colabfold_batch {fasta_path} {output_dir} "
        f"--model-type alphafold2_multimer_v3 "
        f"--num-models 1 --num-recycle 3"
    )


def parse_af2_pae(result_json: str, chain_a_len: int) -> float:
    with open(result_json) as f:
        data = json.load(f)
    pae = np.array(data["pae"])
    inter_pae = pae[:chain_a_len, chain_a_len:].mean()
    return float(inter_pae)

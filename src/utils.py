import numpy as np
from Bio.PDB import PDBParser
from pathlib import Path


def load_structure(pdb_path: str):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("protein", pdb_path)


def get_ca_coords(structure) -> np.ndarray:
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
        break
    return np.array(coords)


def get_residue_coords(structure, chain_id: str) -> dict:
    coords = {}
    model = structure[0]
    chain = model[chain_id]
    for residue in chain:
        if "CA" in residue:
            coords[residue.get_id()[1]] = residue["CA"].get_vector().get_array()
    return coords


def get_surface_residues(structure, neighbor_threshold: int = 12) -> list:
    ca_coords = get_ca_coords(structure)
    n_residues = len(ca_coords)
    surface = []
    for i in range(n_residues):
        distances = np.linalg.norm(ca_coords - ca_coords[i], axis=1)
        n_neighbors = np.sum((distances > 0) & (distances < 10.0))
        if n_neighbors < neighbor_threshold:
            surface.append(i + 1)
    return surface


def compute_contacts(coords_a: np.ndarray, coords_b: np.ndarray, threshold: float = 4.0) -> int:
    from scipy.spatial.distance import cdist
    dists = cdist(coords_a, coords_b)
    return int(np.sum(dists < threshold))


def check_steric_clash(coords_a: np.ndarray, coords_b: np.ndarray, clash_dist: float = 2.0) -> int:
    from scipy.spatial.distance import cdist
    dists = cdist(coords_a, coords_b)
    return int(np.sum(dists < clash_dist))

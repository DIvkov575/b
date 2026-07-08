import numpy as np
from src.utils import load_structure, get_ca_coords, compute_contacts
import logging

logger = logging.getLogger(__name__)


def count_target_contacts(binder_coords: np.ndarray, target_coords: np.ndarray,
                          threshold: float = 8.0) -> int:
    return compute_contacts(binder_coords, target_coords, threshold)


def analyze_design(design_pdb: str, target_pdb: str,
                   chain_a_id: str = "A", chain_b_id: str = "B",
                   contact_threshold: float = 8.0) -> dict:
    design_struct = load_structure(design_pdb)
    target_struct = load_structure(target_pdb)

    binder_coords = get_ca_coords(design_struct)

    model = target_struct[0]
    coords_a = np.array([r["CA"].get_vector().get_array()
                         for r in model[chain_a_id] if "CA" in r])
    coords_b = np.array([r["CA"].get_vector().get_array()
                         for r in model[chain_b_id] if "CA" in r])

    contacts_a = count_target_contacts(binder_coords, coords_a, contact_threshold)
    contacts_b = count_target_contacts(binder_coords, coords_b, contact_threshold)

    return {
        "path": design_pdb,
        "contacts_a": contacts_a,
        "contacts_b": contacts_b,
        "n_residues": len(binder_coords),
    }


def filter_designs_by_contacts(designs: list, min_contacts: int = 5) -> list:
    return [d for d in designs if d["contacts_a"] >= min_contacts
            and d["contacts_b"] >= min_contacts]

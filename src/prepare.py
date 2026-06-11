import numpy as np
from Bio.PDB import PDBParser, PDBIO, Structure, Model, Chain
from pathlib import Path
from src.utils import get_ca_coords, get_surface_residues, load_structure


def place_targets(pdb_a: str, pdb_b: str, separation: float = 60.0) -> dict:
    struct_a = load_structure(pdb_a)
    struct_b = load_structure(pdb_b)

    coords_a = get_ca_coords(struct_a)
    coords_b = get_ca_coords(struct_b)

    com_a = coords_a.mean(axis=0)
    com_b = coords_b.mean(axis=0)

    shift_a = -com_a
    shift_b = np.array([separation, 0.0, 0.0]) - com_b

    for atom in struct_a.get_atoms():
        atom.set_coord(atom.get_vector().get_array() + shift_a)
    for atom in struct_b.get_atoms():
        atom.set_coord(atom.get_vector().get_array() + shift_b)

    output_path = Path("outputs") / "combined_target.pdb"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined = Structure.Structure("combined")
    new_model = Model.Model(0)
    combined.add(new_model)

    chain_a = list(struct_a[0].get_chains())[0]
    chain_a.id = "A"
    chain_a.detach_parent()
    new_model.add(chain_a)

    chain_b = list(struct_b[0].get_chains())[0]
    chain_b.id = "B"
    chain_b.detach_parent()
    new_model.add(chain_b)

    io = PDBIO()
    io.set_structure(combined)
    io.save(str(output_path))

    new_com_a = np.zeros(3)
    new_com_b = np.array([separation, 0.0, 0.0])

    return {
        "combined_pdb": str(output_path),
        "chain_a_id": "A",
        "chain_b_id": "B",
        "com_a": new_com_a.tolist(),
        "com_b": new_com_b.tolist(),
        "chain_a_length": len(coords_a),
        "chain_b_length": len(coords_b),
    }


def generate_contig(chain_a_length: int, chain_b_length: int,
                    binder_min: int = 100, binder_max: int = 150) -> str:
    return f"A1-{chain_a_length}/0 B1-{chain_b_length}/0 {binder_min}-{binder_max}"


def format_hotspots(hotspots_a: list, hotspots_b: list) -> str:
    all_hotspots = [f"A{r}" for r in hotspots_a] + [f"B{r}" for r in hotspots_b]
    return "[" + ",".join(all_hotspots) + "]"


def select_hotspots_auto(pdb_path: str, chain_id: str, n_hotspots: int = 5) -> list:
    struct = load_structure(pdb_path)
    surface = get_surface_residues(struct)
    if len(surface) <= n_hotspots:
        return surface
    indices = np.linspace(0, len(surface) - 1, n_hotspots, dtype=int)
    return [surface[i] for i in indices]

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


def clean_pdb(input_pdb: str, output_pdb: str) -> str:
    """Remove waters, ligands, and alternate conformations. Keep only protein ATOM records."""
    from Bio.PDB import PDBParser, PDBIO, Select

    class ProteinSelect(Select):
        def accept_residue(self, residue):
            return residue.get_id()[0] == " "  # exclude HETATM (non-blank het flag)

        def accept_atom(self, atom):
            return atom.get_altloc() in (" ", "A")  # keep blank or first alt conf

    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("protein", input_pdb)
    io = PDBIO()
    io.set_structure(struct)
    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    io.save(output_pdb, ProteinSelect())
    return output_pdb


def sample_orientations(n_orientations: int = 5) -> list:
    """Generate rotation matrices for sampling relative target orientations."""
    from scipy.spatial.transform import Rotation
    if n_orientations == 1:
        return [np.eye(3)]
    rotations = Rotation.random(n_orientations - 1, random_state=42)
    matrices = [np.eye(3)] + [r.as_matrix() for r in rotations]
    return matrices


def place_targets_multi(pdb_a: str, pdb_b: str, separation: float = 60.0,
                        n_orientations: int = 5, output_base: str = "outputs") -> list:
    """Place targets in multiple relative orientations. Returns list of placement dicts."""
    orientations = sample_orientations(n_orientations)
    placements = []

    for idx, rot_matrix in enumerate(orientations):
        struct_a = load_structure(pdb_a)
        struct_b = load_structure(pdb_b)

        coords_a = get_ca_coords(struct_a)
        coords_b = get_ca_coords(struct_b)

        com_a = coords_a.mean(axis=0)
        com_b = coords_b.mean(axis=0)

        shift_a = -com_a
        # Apply rotation to target B's offset direction
        offset = rot_matrix @ np.array([separation, 0.0, 0.0])
        shift_b = offset - com_b

        for atom in struct_a.get_atoms():
            atom.set_coord(atom.get_vector().get_array() + shift_a)
        for atom in struct_b.get_atoms():
            new_coord = rot_matrix @ (atom.get_vector().get_array() - com_b) + offset
            atom.set_coord(new_coord)

        from Bio.PDB import Structure, Model, Chain, PDBIO
        combined = Structure.Structure(f"combined_{idx}")
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

        output_path = Path(output_base) / f"combined_target_ori{idx}.pdb"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        io = PDBIO()
        io.set_structure(combined)
        io.save(str(output_path))

        placements.append({
            "combined_pdb": str(output_path),
            "orientation_idx": idx,
            "chain_a_id": "A",
            "chain_b_id": "B",
            "chain_a_length": len(coords_a),
            "chain_b_length": len(coords_b),
        })

    return placements


def validate_hotspots(pdb_path: str, chain_id: str, hotspots: list) -> list:
    """Check that hotspot residue numbers exist in the PDB. Return valid ones."""
    struct = load_structure(pdb_path)
    model = struct[0]
    valid = []
    for chain in model:
        if chain.id == chain_id:
            residue_ids = {r.get_id()[1] for r in chain if r.get_id()[0] == " "}
            for h in hotspots:
                if h in residue_ids:
                    valid.append(h)
            break
    return valid

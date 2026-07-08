import numpy as np
from src.utils import load_structure, get_ca_coords, get_surface_residues, compute_contacts, check_steric_clash


def test_load_structure():
    struct = load_structure("tests/fixtures/ubiquitin.pdb")
    assert struct is not None
    residues = list(struct.get_residues())
    assert len(residues) > 50


def test_get_ca_coords():
    struct = load_structure("tests/fixtures/ubiquitin.pdb")
    coords = get_ca_coords(struct)
    assert isinstance(coords, np.ndarray)
    assert coords.shape[1] == 3
    assert coords.shape[0] > 50


def test_get_surface_residues():
    struct = load_structure("tests/fixtures/ubiquitin.pdb")
    surface = get_surface_residues(struct)
    assert len(surface) > 10
    assert all(isinstance(r, int) for r in surface)


def test_compute_contacts_close():
    a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    b = np.array([[0, 3, 0], [1, 3, 0]], dtype=float)
    assert compute_contacts(a, b, threshold=4.0) > 0


def test_compute_contacts_far():
    a = np.array([[0, 0, 0]], dtype=float)
    b = np.array([[0, 50, 0]], dtype=float)
    assert compute_contacts(a, b, threshold=4.0) == 0


def test_check_steric_clash():
    a = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    b = np.array([[0.5, 0, 0]], dtype=float)
    assert check_steric_clash(a, b, clash_dist=2.0) > 0


def test_no_steric_clash():
    a = np.array([[0, 0, 0]], dtype=float)
    b = np.array([[50, 0, 0]], dtype=float)
    assert check_steric_clash(a, b, clash_dist=2.0) == 0

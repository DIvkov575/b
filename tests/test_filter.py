import numpy as np
from src.filter import count_target_contacts, filter_designs_by_contacts


def test_count_target_contacts_close():
    binder = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    target = np.array([[0, 3, 0], [1, 3, 0]], dtype=float)
    contacts = count_target_contacts(binder, target, threshold=4.0)
    assert contacts > 0


def test_count_target_contacts_far():
    binder = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
    target = np.array([[0, 50, 0]], dtype=float)
    contacts = count_target_contacts(binder, target, threshold=4.0)
    assert contacts == 0


def test_filter_designs_both_pass():
    designs = [
        {"path": "a.pdb", "contacts_a": 10, "contacts_b": 8},
        {"path": "b.pdb", "contacts_a": 10, "contacts_b": 2},
        {"path": "c.pdb", "contacts_a": 1, "contacts_b": 8},
    ]
    passed = filter_designs_by_contacts(designs, min_contacts=5)
    assert len(passed) == 1
    assert passed[0]["path"] == "a.pdb"


def test_filter_empty():
    designs = [
        {"path": "x.pdb", "contacts_a": 0, "contacts_b": 0},
    ]
    passed = filter_designs_by_contacts(designs, min_contacts=5)
    assert len(passed) == 0

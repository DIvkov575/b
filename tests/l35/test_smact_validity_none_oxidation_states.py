"""third_party/diffcsp/scripts/eval_utils.py's smact_validity() crashes with
TypeError ("object of type 'NoneType' has no len()") on any composition
containing an element whose smact.oxidation_states is None -- confirmed real
elements in this smact version's database: He, Ne, Ar, Pm, At, Rn, Fr, Ra,
Es, Fm, Md, No, Lr. Pm (Promethium) genuinely appears in real MP-20 ground-
truth structures (e.g. mp-1183241, mp-863742 -- both crashed a real 50-
structure eval run on the actual EC2 instance). Not a version-mismatch bug
(unlike the earlier smact_compat_shim, which was deleted once the real cause
turned out to be a stale smact==4.0.0 install) -- this is DiffCSP's own
vendored eval_utils.py never handling this real case, likely because the
paper's own eval never happened to sample Pm.
"""
import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _diffcsp_on_path():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    diffcsp_path = os.path.join(repo_root, "third_party", "diffcsp")
    scripts_path = os.path.join(diffcsp_path, "scripts")
    for p in (diffcsp_path, scripts_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("PROJECT_ROOT", diffcsp_path)
    original_cwd = os.getcwd()
    yield
    os.chdir(original_cwd)


def test_smact_validity_none_oxidation_states_shim_returns_false_for_promethium():
    import src.l35.smact_validity_none_oxidation_states_shim  # noqa: F401
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    from eval_utils import smact_validity

    # Should not raise; an element with no known oxidation states can't be
    # confirmed charge-balanced, so the conservative answer is invalid.
    result = smact_validity((89, 61), (1, 1))

    assert result is False


def test_shim_does_not_change_behavior_for_normal_compositions():
    import src.l35.smact_validity_none_oxidation_states_shim  # noqa: F401
    import src.l35.torch_scatter_compat_shim  # noqa: F401
    from eval_utils import smact_validity

    # NaCl: Na=11, Cl=17 -- real, well-known, definitely smact_validity()-
    # valid composition with no None-oxidation-state elements -- the shim
    # must be a no-op here.
    result = smact_validity((11, 17), (1, 1))

    assert result is True

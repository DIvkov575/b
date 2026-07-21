"""third_party/diffcsp/scripts/eval_utils.py's smact_validity() crashes
(TypeError: object of type 'NoneType' has no len()) on any composition
containing an element whose smact.oxidation_states is None -- confirmed real
in this smact version's own database: He, Ne, Ar, Pm, At, Rn, Fr, Ra, Es, Fm,
Md, No, Lr. Pm (Promethium) genuinely appears in real MP-20 ground-truth
structures (mp-1183241, mp-863742 both crashed a real 50-structure eval on
the actual EC2 instance) -- not a hypothetical edge case.

Wraps eval_utils.smact_validity to return False immediately for any
composition containing such an element (an element with no known oxidation
states can't be confirmed charge-balanced, so the conservative answer is
"not smact-valid"), leaving every other composition's behavior unchanged.
"""
import sys

_diffcsp_scripts_module = sys.modules.get("eval_utils")
if _diffcsp_scripts_module is None:
    import eval_utils as _diffcsp_scripts_module

import smact

from diffcsp.common.data_utils import chemical_symbols

_real_smact_validity = _diffcsp_scripts_module.smact_validity


def _smact_validity_compat(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple(chemical_symbols[elem] for elem in comp)
    space = smact.element_dictionary(elem_symbols)
    if any(el.oxidation_states is None for el in space.values()):
        return False
    return _real_smact_validity(comp, count, use_pauling_test=use_pauling_test, include_alloys=include_alloys)


_diffcsp_scripts_module.smact_validity = _smact_validity_compat

"""Import this before any vendored third_party/diffcsp module to satisfy its
`from torch_scatter import scatter` / `from torch_scatter.composite import
scatter_softmax` / `from torch_scatter import segment_coo, segment_csr` imports
without the real torch_scatter package, which fails to build on this machine.

Only `scatter(..., reduce='mean')` is exercised by DiffCSP's default config
(edge_style='fc'); scatter_softmax/segment_coo/segment_csr are wired to raise
if actually called, so a silent wrong-answer on an unexercised path is impossible.
"""
import sys
import types

from src.l35.torch_scatter_compat import scatter


def _unsupported(name):
    def _raise(*args, **kwargs):
        raise NotImplementedError(
            f"torch_scatter_compat_shim: '{name}' has no native-torch replacement "
            "here; DiffCSP's default (edge_style='fc') config never calls it -- "
            "if you're hitting this, edge_style='knn' or a code path other than "
            "the default is in use and needs its own verified compat shim."
        )
    return _raise


_scatter_module = types.ModuleType("torch_scatter")
_scatter_module.scatter = scatter
_scatter_module.segment_coo = _unsupported("segment_coo")
_scatter_module.segment_csr = _unsupported("segment_csr")

_composite_module = types.ModuleType("torch_scatter.composite")
_composite_module.scatter_softmax = _unsupported("scatter_softmax")

sys.modules.setdefault("torch_scatter", _scatter_module)
sys.modules.setdefault("torch_scatter.composite", _composite_module)

import os

# nn.TransformerEncoder's padding-mask fast path calls an op MPS doesn't
# implement (aten::_nested_tensor_from_mask_left_aligned). Must be set before
# torch is imported by ANY test module in this package, not just train_pilot —
# pytest's collection order otherwise imports torch (via test_baseline_data,
# test_msa_data, etc.) before train_pilot.py's own module-level guard runs.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

"""L47: generic activation-patching harness for ESM2, reused across Task B
(validate patching against L42's already-trusted thermostability harness)
and Task A (redo Vig et al.'s 2021 correlational attention-head finding as
a real causal test -- the literature gap confirmed via a dedicated search,
see docs/L47_ACTIVATION_PATCHING.md).

Patching, unlike L42/L45's steering (ADD a vector to activations), works by
SUBSTITUTION: run the model on a "clean" input, run it again on a
"corrupted"/counterfactual input, cache both, then splice one specific
component's activation from one run into the other run and measure how
much the output shifts. Sweeping this substitution across every
layer/component gives a causal-necessity map, not just a correlational one.

Feasibility confirmed directly (2026-07-31) before writing this module:
ESM2's EsmSelfAttention.forward() merges all attention heads into a single
[batch, seq, hidden_size] tensor via `.reshape(*input_shape, -1)` before
returning -- but this reshape is a plain view over head-ordered memory
(verified: reshaping a captured real hook output back to
[batch, seq, num_heads, head_dim] cleanly recovers per-head slices), so a
single forward hook on EsmSelfAttention is sufficient for head-level
patching. No need for output_attentions=True or lower-level surgery.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class PatchTarget:
    """Identifies exactly what to patch: a layer index, a component
    ('attention_output', 'mlp_output', or 'residual_stream'), and
    optionally a specific attention head index (only meaningful when
    component='attention_output'; None means patch all heads' combined
    output, i.e. the whole attention-block output).
    """
    layer: int
    component: str  # 'attention_output' | 'mlp_output' | 'residual_stream'
    head: Optional[int] = None

    def __post_init__(self):
        if self.component not in ("attention_output", "mlp_output", "residual_stream"):
            raise ValueError(f"unknown component: {self.component}")
        if self.head is not None and self.component != "attention_output":
            raise ValueError("head index only meaningful for component='attention_output'")


def get_module_for_target(model, target: PatchTarget):
    layer = model.esm.encoder.layer[target.layer]
    if target.component == "attention_output":
        return layer.attention.self
    if target.component == "mlp_output":
        return layer.output
    if target.component == "residual_stream":
        return layer
    raise ValueError(f"unknown component: {target.component}")


class ActivationCache:
    """Captures a module's output on a forward pass. Registered as a
    forward hook; call remove() when done. Stores a detached CPU-agnostic
    clone (kept on-device; caller decides whether to move to cpu) so later
    patching doesn't accidentally alias into a live computation graph.
    """

    def __init__(self):
        self.value = None

    def __call__(self, module, inputs, output):
        raw = output[0] if isinstance(output, tuple) else output
        self.value = raw.detach().clone()
        return output


def register_cache_hook(model, target: PatchTarget) -> tuple[ActivationCache, "torch.utils.hooks.RemovableHandle"]:
    cache = ActivationCache()
    module = get_module_for_target(model, target)
    handle = module.register_forward_hook(cache)
    return cache, handle


class PatchHook:
    """Substitutes a cached activation (optionally restricted to one
    attention head's slice) into the current forward pass, in place of
    whatever this run would have produced at that point. num_heads/head_dim
    only needed when target.head is not None.
    """

    def __init__(self, target: PatchTarget, patch_value: torch.Tensor, num_heads: Optional[int] = None, head_dim: Optional[int] = None):
        self.target = target
        self.patch_value = patch_value
        self.num_heads = num_heads
        self.head_dim = head_dim

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        current = output[0] if is_tuple else output

        if self.target.head is None:
            patched = self.patch_value.to(current.device, current.dtype)
        else:
            if self.num_heads is None or self.head_dim is None:
                raise ValueError("num_heads and head_dim required for single-head patching")
            current_per_head = current.view(*current.shape[:-1], self.num_heads, self.head_dim)
            patch_per_head = self.patch_value.to(current.device, current.dtype).view(
                *self.patch_value.shape[:-1], self.num_heads, self.head_dim
            )
            patched_per_head = current_per_head.clone()
            patched_per_head[..., self.target.head, :] = patch_per_head[..., self.target.head, :]
            patched = patched_per_head.view(*current.shape)

        if is_tuple:
            return (patched,) + output[1:]
        return patched


def run_with_cache(model, tokenizer, sequence: str, target: PatchTarget, device: str, max_len: int = 400):
    """Run a clean forward pass, capturing the activation at `target`.
    Returns (cached_activation, full_output_logits)."""
    cache, handle = register_cache_hook(model, target)
    try:
        enc = tokenizer(sequence[:max_len], return_tensors="pt", truncation=True, max_length=max_len + 2).to(device)
        with torch.no_grad():
            out = model(**enc)
    finally:
        handle.remove()
    return cache.value, out.logits, enc


def run_with_patch(model, tokenizer, sequence: str, target: PatchTarget, patch_value: torch.Tensor, device: str, max_len: int = 400):
    """Run a forward pass on `sequence`, but with `target`'s activation
    replaced by `patch_value` (typically cached from a DIFFERENT sequence).
    Returns full_output_logits."""
    num_heads = model.config.num_attention_heads if target.head is not None else None
    head_dim = (model.config.hidden_size // model.config.num_attention_heads) if target.head is not None else None
    hook = PatchHook(target, patch_value, num_heads=num_heads, head_dim=head_dim)
    module = get_module_for_target(model, target)
    handle = module.register_forward_hook(hook)
    try:
        enc = tokenizer(sequence[:max_len], return_tensors="pt", truncation=True, max_length=max_len + 2).to(device)
        with torch.no_grad():
            out = model(**enc)
    finally:
        handle.remove()
    return out.logits, enc

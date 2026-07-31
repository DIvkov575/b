"""L46: unsupervised feature discovery on ESM2-650M via InterPLM's published,
pretrained sparse autoencoders (SAEs) -- Simon & Zou, "InterPLM," arXiv:2412.12101,
Nature Methods 2025. Trained checkpoints hosted at huggingface.co/Elana/
InterPLM-esm2-650m (verified downloadable via a real fetch before writing this).

This is the "algorithmic feature discovery, no target metric hardcoded ahead
of time" method requested as a follow-up to L42/L43/L45 (all of which
required specifying a property -- thermostability, solubility -- upfront).
An SAE decomposes real activations into a large, sparse, overcomplete basis
of individually-interpretable directions WITHOUT ever being told what
property to look for; each learned feature can then be inspected/correlated
against real sequences AFTER the fact to see what it responds to.

Architecture reimplemented from scratch (NOT vendored via git clone, per
project convention) -- confirmed by direct inspection of a downloaded
checkpoint's state_dict keys (bias, encoder.weight, encoder.bias,
decoder.weight -- no decoder bias, no stored normalization buffer) that
this is a plain pre-bias + linear encoder + ReLU + linear decoder SAE, small
enough to reimplement exactly rather than depend on the full `interplm`
package (which requires interplm.train.configs.TrainingRunConfig and a
YAML config file not needed for pure inference).
"""
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download

MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
HF_REPO = "Elana/InterPLM-esm2-650m"
VALID_LAYERS = [1, 9, 18, 24, 30, 33]
FEATURE_DIM = 10240
ESM_DIM = 1280


class ReLUSAE(nn.Module):
    """Minimal reimplementation of InterPLM's ReLUSAE inference path
    (encode/decode only -- no training-time ghost-mode or loss logic,
    since this is used purely for feature discovery on frozen weights).
    Verified against a real downloaded checkpoint's state_dict keys and
    shapes before trusting this matches the actual saved architecture.
    """

    def __init__(self, activation_dim: int = ESM_DIM, dict_size: int = FEATURE_DIM):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.bias))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return self.decoder(f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_interplm_sae(layer: int, device: str = "cpu") -> ReLUSAE:
    if layer not in VALID_LAYERS:
        raise ValueError(f"layer must be one of {VALID_LAYERS} for {HF_REPO}, got {layer}")
    ckpt_path = hf_hub_download(repo_id=HF_REPO, filename=f"layer_{layer}/ae_normalized.pt")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    sae = ReLUSAE()
    sae.load_state_dict(state_dict)
    sae.to(device).eval()
    return sae


@torch.no_grad()
def extract_per_residue_activations(model, tokenizer, sequence: str, layer: int, device: str, max_len: int = 1020):
    """Per-RESIDUE (not mean-pooled) activations at a given layer -- SAE
    features are often sparse and localized to specific motifs/residues
    (per InterPLM's own findings), so mean-pooling across a sequence would
    dilute exactly the kind of feature this method is meant to discover.
    ESM-2 requires sequence length < 1022 (2 special tokens + residues).
    """
    seq = sequence[:max_len]
    enc = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_len + 2).to(device)
    out = model(**enc, output_hidden_states=True)
    hidden = out.hidden_states[layer].squeeze(0).float()  # [seq_len+2, d_model], includes cls/eos
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"].squeeze(0).tolist())
    return hidden, tokens

# L46 — Unsupervised SAE Feature Discovery (No Target Property Specified)

## Why this exists

L42/L43/L45 all required choosing a target property (thermostability,
solubility) before running anything. This asks a different question: can
we discover structure/features WITHOUT specifying any target metric ahead
of time, then characterize what was found afterward? Sparse Autoencoders
(SAEs) are the field's standard tool for this — an SAE decomposes real
activations into a large, sparse, overcomplete basis of directions with
zero biological label involved in training.

## Method

Checked whether pretrained SAE weights already exist for ESM2-650M before
considering training one from scratch (expensive). Confirmed via real
download: InterPLM (Simon & Zou, Nature Methods 2025) publishes trained
checkpoints on HuggingFace (`Elana/InterPLM-esm2-650m`) for 6 layers (1, 9,
18, 24, 30, 33) — three of which (18, 24, 30) overlap with layers L45's
causal steering sweep flagged as significant.

The full `interplm` package requires `interplm.train.configs.TrainingRunConfig`
and a YAML config file not needed for pure inference — rather than vendor
the whole package (against project convention: no `git clone`-based
vendoring), inspected a downloaded checkpoint's raw `state_dict` keys/shapes
directly and reimplemented the ~15-line architecture from scratch
(`ReLUSAE` in `src/l38/l46_sae_feature_discovery.py`): pre-encoding bias,
linear encoder + ReLU, linear decoder with no bias. Verified this loads the
real checkpoint correctly (4 tests, `tests/l38/test_l46_sae_feature_discovery.py`)
and produces sane reconstructions on real ESM2-650M activations before
trusting it (mean reconstruction error 85 vs. mean original activation norm
355 — meaningfully below the original scale, not garbage).

Ran the SAE over 100 real sequences (meltome dataset, Tm labels never
loaded during discovery) at layer 24, extracting PER-RESIDUE (not
mean-pooled) activations — SAE features are often sparse/localized to
specific motifs, so pooling would dilute exactly what this method is meant
to find. Ranked all 10,240 features by a label-free selectivity score
(`max_activation * (1 - firing_fraction)`): fires strong but rare = a real
candidate pattern; fires everywhere or never = not interesting, regardless
of any biological annotation.

## Results

Only 15 of 10,240 features never fired at all across ~24,000 real residues
— this SAE is genuinely well-utilized on real protein sequences, not
degenerate. Top 15 most-selective features each tied to a specific residue
context (e.g. feature 877 fires strongest on glutamate in a
`CCDAD[E]LNNKG` context; feature 9730 on valine in `DADPW[V]LGGVV`) — full
list in `src/l38/l46_discovery_out.json`.

**Post-hoc characterization only (labels used AFTER discovery, not during):**
checked the top 15 features' mean-pooled per-sequence activation against
real Tm labels on the same 100 sequences. Two features showed a weak
correlation (feature 567: r=0.260, p=0.009; feature 9730: r=0.212, p=0.034)
— suggestive but NOT statistically robust after correcting for testing 15
features at once (Bonferroni-corrected threshold ≈0.0033, neither clears
it). Treat as a lead worth a larger follow-up sample, not a confirmed
finding.

## What's NOT done

- No published feature-to-annotation table exists for these checkpoints
  (checked directly: InterPLM's repo only ships scripts to compute this
  yourself against Swiss-Prot; the live interplm.ai browser was down (502)
  when checked). The Tm-correlation check above is the only
  characterization that currently exists for these specific feature IDs.
- Only ran at layer 24, only 100 sequences, only one property checked
  post-hoc. Scaling to more sequences/layers, or running InterPLM's own
  Swiss-Prot annotation scripts for a proper characterization, are both
  live options not yet taken.

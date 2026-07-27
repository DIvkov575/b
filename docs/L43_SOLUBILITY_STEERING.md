# L43 — Extend the L42 Steering Harness to a Second Target: Solubility

**Pre-registered protocol.** Locked 2026-07-26, before any run.

## STATUS (2026-07-27): code complete, GPU run BLOCKED — not yet executed

Everything below this line was written and locked BEFORE any run, per this
project's usual discipline. As of 2026-07-27, implementation is done and
tested, but **no GPU run has happened yet** — AWS credentials for the EC2
instance (`i-0659e54e8adc759d3`, see memory `l35-ec2-remote-host-reference`
for the tunnel/SSH details) expired mid-session (`aws sts get-caller-
identity` / `describe-instances` both fail with `AuthFailure`). The user
chose to skip the GPU run for now rather than walk through re-auth live.

**What's done and verified (all local, no GPU needed):**
- `src/l38/l43_solubility_steering.py` — GRAVY (Kyte & Doolittle 1982)
  solubility proxy, `solubility_proxy_excluding()` for the leucine-exclusion-
  style robustness check. Sanity-checked against real ubiquitin (soluble,
  GRAVY < 0) and homopolymer extremes.
- `tests/l38/test_l43_solubility_steering.py` — 9 tests, all passing.
- `src/l38/l43_run_repro.py` — full run script, structurally identical to
  L42's (same model, same hook class, same degeneracy filter, same paired-
  bootstrap verdict logic), pointed at a new dataset and scorer. Compiles
  clean; not yet run end-to-end (needs GPU).
- Real dataset already downloaded and length-filtered: `src/l38/data_cache/
  solubility/{train,test}.csv` (`hazemessam/solubility`, HuggingFace, same
  author/format convention as L42's meltome dataset) — 49,583 usable
  (length<=400) sequences, 28,522 insoluble / 21,061 soluble by real
  experimental label, confirmed via direct pandas inspection.
- Full local test suite (92 tests, everything except the 3 torch-dependent
  files this sandbox can't import) passes.

**What's NOT done:**
- The actual GPU run (`python -m src.l38.l43_run_repro` on the EC2 A10G,
  inside `.venv-l38`) has never been executed. `src/l38/l43_repro_out/`
  does not exist yet.
- No results, no verdict, no PASS/KILL/AMBIGUOUS call — this doc's rule
  below is unapplied.

**Next agent / next session: to resume,**
1. Refresh AWS credentials (`aws sso login` or whatever this account's flow
   is — check with the user first, this session did not attempt it after
   the user chose to defer).
2. Reconnect via the EC2 Instance Connect Endpoint tunnel (see memory
   `l35-ec2-remote-host-reference` for the exact commands: tunnel to
   `eice-0695d0c190091d2a0`, then `ssh -p <PORT> -i ~/.ssh/biostat-l35-key
   ubuntu@localhost`).
3. `scp` `src/l38/l43_run_repro.py`, `src/l38/l43_solubility_steering.py`,
   `tests/l38/test_l43_solubility_steering.py`, and the `data_cache/
   solubility/` CSVs to `~/biostat/src/l38/...` (data_cache is gitignored/
   not synced automatically — verify it's present or re-download the same
   HuggingFace URLs used locally: `https://huggingface.co/datasets/
   hazemessam/solubility/resolve/main/{train,test}.csv`).
4. Run `.venv-l38/bin/python -m pytest tests/l38/test_l43_solubility_steering.py
   -q` remotely first (sanity check), then `.venv-l38/bin/python -m
   src.l38.l43_run_repro`.
5. Apply the PASS/KILL/AMBIGUOUS rule below to whatever `l43_repro_out/
   results.json` says — do not skip the residue-exclusion robustness check
   if a significant effect appears, per the AMBIGUOUS clause (this is
   exactly the check that caught L42's false positive).

## Why this exists

L42 (docs/L42_STEERING_REPRO.md) reproduced Huang et al.'s difference-of-
means activation steering on ESM2-650M for thermostability, after fixing a
real decoding-collapse artifact. That result validated the harness for
thermostability specifically, but doesn't establish whether the harness
generalizes to a different target property or whether the thermostability
result was itself a fluke tied to that particular dataset/split. A follow-
up literature survey (2026-07-24) also surfaced that continuous,
compositionally-grounded properties are the one repeatedly-successful
recipe for single-direction steering in the field, and solubility is the
most direct next test of that recipe using infrastructure that's already
built and validated.

## The claim under test

**H1.** A difference-of-means steering vector built from ESM2-650M
activations of soluble vs. insoluble sequences (real experimental labels,
hazemessam/solubility dataset) increases a downstream solubility proxy
score when added to the residual stream during masked-marginal generation,
relative to both an unsteered baseline and a matched-norm random-direction
control.

This is a generalization check, not a new scientific claim on its own:
success buys confidence that L42's reproduction wasn't thermostability-
specific; failure would suggest the harness (or the difference-of-means
technique at this scale) is narrower than L42 alone implied.

## Method (reuses L42's validated recipe unchanged except where noted)

- **Model:** `facebook/esm2_t33_650M_UR50D` — identical to L42.
- **Target property:** solubility. **Dataset:** `hazemessam/solubility`
  (HuggingFace, same author/format convention as L42's `hazemessam/
  meltome`), real experimental soluble/insoluble binary labels, ~62K
  sequences. Unlike L42's continuous Tm labels (requiring a percentile
  split), this dataset's ground truth is already binary — used directly as
  the high/low group split.
- **Steering vector construction:** difference-of-means, identical to L42 —
  mean per-layer activation (averaged across tokens) over the soluble group
  minus the insoluble group, across all 33 transformer layers.
- **Generation:** identical to L42 — single-shot masked-marginal fill,
  `mask_fraction=0.3`, `alpha` in `[0.0, 0.1, 0.25, 0.5, 1.0, 2.0]`. Both
  values are REUSED, not re-derived, from L42's empirical tuning on this
  same model/generation setup — there's no a priori reason the collapse
  point would be property-specific rather than model/generation-specific,
  but this is an assumption, not a re-verified fact. If the results below
  show degeneracy behaving very differently from L42's pattern (e.g.
  collapse starting much earlier or not at all through alpha=2.0), that
  itself is worth flagging rather than silently trusted.
- **Scoring:** GRAVY (Grand Average of hYdropathy, Kyte & Doolittle 1982) —
  a purely compositional, model-free hydrophobicity index, negated so
  higher = more soluble-like. Chosen for the same reason as L42's
  `ivywrel_fraction`: cannot be confounded by the generation model's own
  fluency judgments, and is NOT derived from anything observed in this
  project's own generated sequences (it's a 1982 literature-standard scale).
  See `src/l38/l43_solubility_steering.py`.
- **Degeneracy filter:** identical to L42's `is_degenerate_sequence`
  (>25% single-residue frequency) — applied BEFORE scoring, same
  `MIN_NONDEGENERATE_PAIRS=30` trust guard on the bootstrap.
- **Significance test:** identical to L42's direct real-vs-random paired
  bootstrap (`paired_bootstrap_mean_diff`), not two separate vs.-baseline
  tests.
- **Leucine-exclusion-style robustness check:** if a significant effect is
  found, rerun the score with whichever residue(s) turn out to dominate the
  steering direction's shift excluded (`solubility_proxy_excluding`), to
  rule out "same collapse artifact, different name" — the exact check that
  saved L42 from a false positive.

## PASS / KILL / AMBIGUOUS rule (pre-registered, identical structure to L42)

- **PASS:** real-direction steering significantly beats the random-direction
  control (bootstrap CI excludes 0) at at least one alpha with
  `>= MIN_NONDEGENERATE_PAIRS` surviving non-degenerate sequence pairs, AND
  the effect survives a residue-exclusion robustness check if applicable.
- **KILL:** no alpha shows a significant real-vs-random effect with enough
  surviving non-degenerate pairs.
- **AMBIGUOUS:** a significant effect is found but does NOT survive the
  residue-exclusion check (i.e., it's an artifact of the same kind that
  produced L42's false positive) — treat as informative but not sufficient
  to claim generalization.

## What this is NOT

Not a new scientific claim about solubility biology on its own — a
successful PASS here means "the L42 harness generalizes to a second
property," which is valuable for deciding whether to trust this pipeline on
a genuinely novel target, not a publishable solubility-design result by
itself.

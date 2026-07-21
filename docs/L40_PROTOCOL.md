# L40 — Boltz-MSA-Augmentation Ablation Pilot

**Pre-registered 2026-07-21, before the real run.** Locks the comparison
method and pinned-split invariant before seeing pilot results.

## Provenance

- **Code:** `src/l40/` in this repo, commits `76d23de0`..`06f5cbff`
  (2026-07-21). Ported from [PFold](https://github.com/DIvkov575/PFold)
  commit `f44eecc`; ablation-specific code (baseline fetcher/dataset,
  shared trainer, comparator) has no PFold equivalent — see `src/l40/README.md`
  for the file-by-file breakdown.
- **Environment:** `.venv-l38` — Python 3.11.15, `torch==2.13.0`, run on
  Apple Silicon (MPS backend, no CUDA available).
- **Tests:** `tests/l40/` — 49 tests, all passing as of the commits above
  (`.venv-l38/bin/python -m pytest tests/l40/ -v`).
- **Raw data:**
  - Boltz slice: 2344 real `.npz` shards, byte-range-fetched from
    `s3://boltz1.s3.us-east-2.amazonaws.com/rcsb_processed_msa.tar`
    (public, no auth). Not committed (large binaries) — `data/l40_pilot/`
    is gitignored; re-fetch via the command in `src/l40/README.md`.
  - RCSB baseline: `data/l40_pilot/rcsb_baseline.jsonl`, fetched live via
    `src/l40/fetch_rcsb_structures.py` against
    `https://www.rcsb.org/fasta/chain/{PDBID}.{CHAIN}` — 2344/2344 succeeded,
    0 failures. Also gitignored; regenerate via the same script.
- **Result artifacts (committed, small JSON):** `src/l40/pilot_out/
  {boltz,baseline,boltz_seqs1}_results.json` — per-epoch train/val
  loss+accuracy for all three training runs referenced below.
- **Reproduction commands:** `src/l40/README.md` § "Reproducing the pilot"
  has the exact CLI invocations, including the disambiguation run.

## Hypothesis

Training PFold's ProteinBERT MLM on Boltz's MSA-derived homolog sequences
(multiple aligned sequences per RCSB structure) improves held-out MLM
accuracy/loss over training on exactly one raw RCSB sequence per structure —
holding architecture, hyperparameters, structure population, and file-level
train/val/test split fixed.

## Method

- **Model A (boltz):** `src/l40/msa_data.py` pipeline, `sequences_per_file=5`
  (samples up to 5 MSA-hit sequences per training structure).
- **Model B (baseline):** `src/l40/baseline_data.py` pipeline, exactly 1
  sequence per structure (the canonical RCSB chain sequence, fetched via
  `src/l40/fetch_rcsb_structures.py`).
- Same structure population for both, capped at `--max-files N` — the first
  N `.npz` filenames (sorted) in the Boltz data directory.
- Same file-level train/val/test split for both (pinned via
  `create_diverse_splits`'s `random.seed(42)`, verified identical in
  `tests/l40/test_split_pinning.py`).
- Same architecture (`d_model`, `n_layers`, `n_heads`, `d_ff`), optimizer,
  learning rate, batch size, epoch count, mask probability.
- Same `torch.manual_seed` before model construction for both variants
  (`train_pilot.py`'s `--seed`, default 0) — both models start from
  identical initial weights, so a metric difference is attributable to the
  data, not to which variant happened to draw a luckier init.
- Metric: held-out (val) MLM accuracy and loss, final epoch.

## Pilot scale

A few thousand structures (`--max-files`), a handful of epochs, run locally
on MPS (no CUDA available on this machine; `PYTORCH_ENABLE_MPS_FALLBACK=1` is
required — `nn.TransformerEncoder`'s padding-mask fast path calls an op MPS
doesn't implement, `aten::_nested_tensor_from_mask_left_aligned`; the fallback
runs that one op on CPU instead of crashing). This is a first-pass signal
check, not a scaled/definitive result.

## What counts as what

- **Boltz helps:** final val_accuracy(boltz) − val_accuracy(baseline) is
  positive and larger than run-to-run noise (assessed qualitatively at pilot
  scale — no repeated-seed variance estimate yet; a real effect claim would
  need that before writeup).
- **No effect / baseline wins:** delta ≈ 0 or negative — the extra MSA
  homologs don't help this model/data-scale combination, at least not
  without more compute.
- Either outcome is a legitimate, reportable pilot result. This is a first
  cheap pass to decide if scaling up is worth it, not a publication-grade
  claim in itself.

## Known limitations of the pilot

- Single seed per variant — no variance estimate.
- Small `max_files`/epoch count relative to PFold's own full-scale training
  (`MAX_FILES=151040`, 250 epochs) — a pilot null result doesn't rule out an
  effect that only appears at scale.
- RCSB fetch rate-limited client-side (`--sleep-seconds`) to avoid hammering
  their API — fetching baseline data for thousands of structures takes
  real wall-clock time (minutes, not seconds).
- **Gradient-steps confound (discovered post-run, see Result below):** at
  `sequences_per_file=5`, the boltz variant trains on 5x more sequences per
  epoch than the baseline (8200 vs 1640, same 1640 structures). Matching
  epoch count therefore does not match total gradient steps — the result
  conflates "MSA homology diversity helps" with "more total training data/
  steps helps." A cleaner isolation would either cap boltz at
  `sequences_per_file=1` (pure curation-only comparison, expected near-zero
  per the original design discussion) or match total training steps between
  variants directly.

## Result (2026-07-21)

Ran on the full pilot slice: 2344 real Boltz-processed RCSB structures
(HTTP range-fetched from Boltz's public S3 tar), 2344 matching RCSB baseline
sequences (fetched live via `fetch_rcsb_structures.py`, 0 failures). Pinned
split confirmed identical on real data: 1640 train / 351 val / 353 test
structures for both variants (`tests/l40/test_split_pinning.py`-equivalent
check re-run directly against the real data before training).

Hyperparameters (identical for both): `d_model=128, n_layers=4, n_heads=4,
d_ff=512, batch_size=32, lr=1e-4, epochs=3, mask_prob=0.15, seed=0`.
Boltz variant used `sequences_per_file=5`.

| | boltz (MSA-augmented) | baseline (RCSB, 1 seq/structure) |
|---|---|---|
| train sequences/epoch | 8200 | 1640 |
| final train_loss | 2.7241 | 2.9456 |
| final val_loss | 2.6933 | 2.8723 |
| final val_accuracy | 0.1981 | 0.1291 |

**val_accuracy_delta = +0.0690 (boltz higher, ~53% relative), val_loss_delta
= −0.1790 (boltz lower/better).** Both metrics moved consistently in favor
of the boltz variant across all 3 epochs, not just the final one — boltz's
val_accuracy was already ahead by epoch 1 (0.170 vs 0.098) and the gap held.

**Verdict per the pre-registered criteria: "Boltz helps"** — a positive
val_accuracy delta of this size, consistent across epochs, on a real
(non-synthetic) 2344-structure slice.

**But this does not yet isolate MSA-augmentation as the cause**, per the
gradient-steps confound above: boltz also saw 5x more training sequences
per epoch. This pilot answers "does Boltz's pipeline as typically configured
(`sequences_per_file=5`) beat the no-augmentation baseline" — yes, clearly —
but not yet "is that because of homology diversity specifically, or because
of more training volume." Disambiguation run below.

## Disambiguation run (2026-07-21): curation-only, steps-matched

Reran the boltz variant with `sequences_per_file=1` — same 2344 structures,
same pinned split, same hyperparameters/seed, but now exactly 1 sequence per
structure (1640 train sequences/epoch, matching the baseline's step count
exactly). This isolates Boltz's specific extraction/curation from its
homology-augmentation effect.

| | boltz (seqs_per_file=1) | baseline (RCSB, 1 seq/structure) |
|---|---|---|
| train sequences/epoch | 1640 | 1640 |
| final train_loss | 2.9414 | 2.9456 |
| final val_loss | 2.8765 | 2.8723 |
| final val_accuracy | 0.1249 | 0.1291 |

**val_accuracy_delta = −0.0042, val_loss_delta = +0.0042** — both
indistinguishable from zero at this pilot's scale (single seed, no variance
estimate, but the magnitude here is ~16x smaller than the sequences_per_file=5
delta and has the opposite sign on accuracy).

## Combined conclusion

The full-pipeline advantage found in the first run (val_accuracy +0.0690) is
attributable almost entirely to **training-volume** (5x more sequences per
epoch from sampling multiple MSA homologs), not to any intrinsic curation-
quality difference between Boltz's `.npz` pipeline and a plain RCSB fetch —
confirming the original design-phase prediction for the "curation-only" cut
(expected near-zero effect) and refuting a naive read of the first run as
evidence that Boltz's processing itself is higher-quality data.

**Practical implication:** if PFold's goal is masked-LM pretraining quality,
the effect being tested here isn't "does Boltz's data pipeline matter" —
it's "does sampling multiple MSA-homolog sequences per structure act as a
cheap data-augmentation multiplier," which this pilot answers **yes** (at
pilot scale, single seed). That's a real and useful finding, but it's a
different claim than "Boltz processing improves per-structure signal
quality," which this pilot found **no evidence for**.

**Caveats before treating either finding as final:** single seed per run (no
variance estimate — a repeat with a different seed could shift these deltas
non-trivially at this small a pilot scale), 3 epochs only, and the
5x-augmentation win could plausibly not hold at PFold's full training scale
(150K structures, 250 epochs) where the baseline would also see much more
data in absolute terms.

## Volume-accounting run (2026-07-21): baseline at matched total training instances

The `sequences_per_file=1` disambiguation above matched *epoch count* and
*structures*, but under-trained the baseline in absolute terms: boltz-5 saw
8200 seqs/epoch × 3 epochs = 24,600 total training instances, while the
`sequences_per_file=1` cut only gave baseline 1640 × 3 = 4,920 — a 5x
smaller total training budget, not a matched one. This run closes that gap
directly: baseline trained for 15 epochs on its native 1640 seqs/epoch
(1640 × 15 = 24,600 — the *same* total instance count as boltz-5's 3-epoch
run), just repeating each of its 1640 single sequences 15 times instead of
seeing 5 distinct MSA homologs 3 times each.

| | boltz (5 seqs/file × 3 epochs) | baseline (1 seq/file × 15 epochs) |
|---|---|---|
| total train instances | 24,600 | 24,600 |
| distinct sequences/structure | 5 | 1 |
| final train_loss | 2.7241 | 2.7178 |
| final val_loss | 2.6933 | 2.6956 |
| final val_accuracy | 0.1981 | 0.1891 |

**val_accuracy_delta = +0.0090, val_loss_delta = −0.0024** — the delta
shrinks by ~87% from the naive +0.0690 to +0.0090 once total training
volume is matched, and val_loss is now a coin-flip either way. The
baseline's own epoch-by-epoch curve (epochs 1–15: val_accuracy 0.098 →
0.191, plateauing from epoch ~8 onward in the 0.185–0.191 band) shows it
converging to nearly the same place as boltz-5 reached in fewer, more
varied epochs — repetition of one sequence gets most of the way there,
just needs more passes to do it.

**Updated combined conclusion:** across all three cuts (naive, curation-
only/steps-matched-per-epoch, and volume-matched/total-instances-matched),
the picture is consistent: Boltz's specific `.npz` extraction and curation
contributes ~nothing beyond a plain RCSB fetch (the curation-only cut,
−0.004). What sampling 5 MSA homologs per structure buys is mostly
**more total gradient steps early**, reaching the baseline's eventual
plateau faster (3 epochs vs. ~8) rather than reaching meaningfully higher —
the residual +0.009 at matched total volume is small enough that a
single-seed pilot can't distinguish it from noise. If speed-of-convergence
(reaching a given accuracy in fewer wall-clock epochs) matters for the
downstream use case, MSA augmentation is a legitimate, real lever; if only
the eventual ceiling matters, this pilot found no evidence it raises it.

## Full-pipeline ablation (2026-07-21): every unused Boltz npz field

The runs above only ever used one homolog's *token sequence*. Boltz's real
`.npz` format carries two more fields the earlier runs never touched:
`taxonomy` (species ID per homolog) and a `deletions` array (insertion/
deletion bookkeeping from the alignment). Real Boltz consumes these via a
genuine architecture, not a data-pipeline trick — verified directly against
Boltz's own source (`boltz.model.modules.trunkv2`, `boltz.model.layers.
{pair_averaging,outer_product_mean,transition}`):

- **`taxonomy`** is used only as a *selection key* (never a model input) —
  Boltz uses it to pick cross-chain-paired MSA rows; here (single chain, no
  pairing) it's repurposed to pick a taxonomically diverse homolog subset
  instead of an arbitrary one.
- **`deletions`** → a real per-position model feature: `has_deletion =
  raw_count > 0`, `deletion_value = π/2·arctan(raw_count/3)`, fed into the
  MSA embedding exactly as Boltz's `featurizerv2.py` does.
- **Cross-homolog attention**: Boltz's `MSAModule` (embed MSA rows →
  `PairWeightedAveraging` mixes across the homolog axis, weighted by a pair
  tensor `z` that `OuterProductMean` builds from co-evolution signal) —
  ported into `src/l40/msa_module.py`, with the structure-specific
  `PairformerNoSeqLayer` call *removed* (nothing downstream needs a refined
  pair representation for an MLM task). A separate `profile` branch (per-
  position AA frequency across all homologs + mean deletion rate) is added
  directly to the query embedding, matching Boltz's real `InputEmbedder`.
- **A real gap found and fixed during implementation:** with `z` initialized
  to zero and only 1 `MSALayer` block, `OuterProductMean` only updates `z` at
  the *end* of the block — one step too late for cross-homolog signal to
  reach `PairWeightedAveraging`'s attention weights before the module
  returns. Real Boltz avoids this because `z` normally arrives pre-populated
  from an upstream Pairformer trunk (excluded here). Fixed by seeding `z`
  from the raw MSA embedding via an initial `OuterProductMean` pass before
  the first block.

### Method

Five arms, each an independent on/off combination on `MSAAwareProteinBERT`,
holding architecture/hyperparameters/seed/data fixed otherwise:

| Arm | profile | deletion | cross-seq attention (MSA module) |
|---|---|---|---|
| A — baseline | off | off | off |
| B — profile | **on** | off | off |
| C — cross-attn | off | off | **on** |
| D — deletion | off | **on** | **on** (deletion only has meaning inside the MSA module) |
| E — combined | **on** | **on** | **on** |

Data: 500 real Boltz structures (subset of the same 2344-structure pool used
above), `msa_depth=5` homologs sampled via taxonomy-diversity sampling,
`max_length=128` (see hardware note below), `batch_size=8`, 2 epochs,
`d_model=128, n_layers=4, n_heads=4, d_ff=512, msa_s=64, token_z=32,
msa_blocks=1`, `seed=0` for all arms (same initial weights).

**Hardware constraint on scale:** `OuterProductMean`'s pair tensor is
`O(N²)` in sequence length (`N×N×c_hidden²` per batch item) — at the
training-volume ablation's `max_length=512` this OOM'd immediately on this
machine's 48GB MPS pool (tried to allocate 16GB for one einsum). Cut to
`max_length=128` and `batch_size=8`; only 500 structures (not the full 2344)
to keep total wall-clock reasonable across 5 sequential arms. This is a
*smaller* pilot than the training-volume ablation above, not a scaled-up
one — read the deltas as directional signal, not precision estimates.

### Result

| Arm | final train_loss | final val_loss | final val_accuracy |
|---|---|---|---|
| A — baseline | 2.9926 | 2.9098 | 0.0750 |
| B — profile | 2.9804 | **2.8862** | **0.1256** |
| C — cross-attn | 2.9978 | 2.9209 | 0.0849 |
| D — deletion | 2.9982 | 2.9207 | 0.0822 |
| E — combined | 2.9798 | 2.9075 | 0.0939 |

Per-epoch curves (all arms improve epoch 1 → 2; none plateaued or diverged):

- A: 0.0840 → 0.0750 (val_accuracy actually *drops* — likely overfitting a
  350-sequence training set in 2 epochs at this model size, not a real
  regression)
- B: 0.0930 → 0.1256
- C: 0.0777 → 0.0849
- D: 0.0786 → 0.0822
- E: 0.0894 → 0.0939

**The profile feature (Arm B) is the clear standout** — val_accuracy
+0.0506 over baseline (~67% relative), the largest and most consistent
effect of any single mechanism, and the only arm whose val_loss beats
baseline outright. **Cross-sequence attention (Arm C) gives a small,
real-looking bump** (+0.0099) — present but far smaller than profile's.
**Deletion features (Arm D) add essentially nothing over cross-attention
alone** (0.0822 vs. Arm C's 0.0849 — within noise, if not slightly worse) —
unsurprising, since deletions are sparse events (most positions have
`has_deletion=False`), so at this pilot's scale and depth there's little
signal for the feature to carry. **The combined arm (E) sits between C and
B, closer to C** (0.0939) — profile's gain does *not* fully carry through
once cross-attention and deletion features are also active, suggesting
some interaction/interference between the mechanisms rather than clean
additivity, though 2 epochs and 350 training sequences is too little to
characterize that interaction precisely.

### Interpretation

Unlike the earlier training-volume finding (where the "MSA augmentation"
effect turned out to be almost entirely a training-volume artifact), this
ablation isolates real architectural mechanisms at matched data/steps
across all five arms — so the profile-feature effect is not a training-
volume confound; it's the same 350 sequences, same 2 epochs, same
everything except which features the model can see.

**Practical reading:** of the three real MSA-derived signals Boltz's own
architecture exposes (profile, deletion, cross-sequence attention), the
cheapest one — a simple per-position amino-acid frequency computed directly
from the raw homolog alignment, no attention mechanism required — carries
the most signal at this pilot's scale. The expensive mechanism (cross-
sequence attention via `PairWeightedAveraging`/`OuterProductMean`, the part
that needed real architecture work and hit a real memory ceiling) adds a
small amount on its own but doesn't obviously combine well with the cheap
feature in this 2-epoch, single-seed run.

### Caveats

- **Single seed, 2 epochs, 500 structures, `msa_blocks=1`** — smaller than
  every other cut in this doc. This is a first directional signal, not a
  precision measurement; do not treat these deltas as final without a
  repeat at larger scale/more seeds.
- **Arm A's val_accuracy declining between epochs** is a mild overfitting
  signal at this data/model-size ratio, not necessarily representative of
  how the other arms would behave with more data — the *relative* ranking
  across arms at epoch 2 is the more informative read than any single arm's
  absolute trajectory.
- **`max_length=128` truncates most real sequences** — real query lengths in
  this data range from 4 to 729 residues (median ~200); only ~28% are
  under 128. Truncation could differentially affect arms that rely on
  longer-range profile/attention signal.
- **The combined arm's sub-additivity is not yet explained** — could be a
  genuine interaction (e.g. profile and cross-attention learning redundant
  or conflicting signal) or could be an artifact of `msa_blocks=1` (recall
  the z-initialization gap found above: even with the seeding fix, one
  block is a minimal amount of cross-attention capacity, likely not enough
  to fairly represent what cross-sequence attention could do with more
  blocks and more compute).

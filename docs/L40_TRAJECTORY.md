# L40 — Trajectory: motivations, pivots, confounders, conclusions

A narrative trace of how the Boltz-MSA-augmentation ablation was actually
reasoned through, kept separate from `docs/L40_PROTOCOL.md` (the
pre-registered method + results record). This doc exists to make the
*thinking* auditable — where the framing was wrong, where a result was
initially misread, and what corrected it — not just the final numbers.

Timeline: 2026-07-21, commits `76d23de0`..`2a7438e6` in this repo. Code was
later extracted to a standalone repo (see "Pivot 4" below), which is
expected to be overwritten — this doc is the durable record of the
reasoning, independent of that repo's fate.

## Motivation

The user asked to look at their own [PFold](https://github.com/DIvkov575/PFold)
repo (a from-scratch ProteinBERT-style MLM, plateaued ~0.21 val accuracy
after months of iteration) and copy its data import/processing into a new
project. That request, by itself, was just a port. The actual research
question came from the very next message: benchmark two PLMs, one trained
on Boltz-processed data and one without — i.e., does Boltz's specific RCSB
MSA data pipeline produce meaningfully better pretraining data than a
plain fetch?

This was framed from the start as **causal inference about a data
pipeline**, not a model-quality horse race: the two models exist to let a
single variable (which pipeline produced the training data) be isolated
and its effect measured, everything else held fixed. That framing is what
made the later confounder-hunting (see Pivots 1–2) the obviously-correct
next move rather than an afterthought — once you're asking "what does the
pipeline *cause*", any uncontrolled variable that also differs between the
two arms is a threat to the whole exercise, not a minor caveat.

## Design decisions made before running anything

Three choices, each resolved by asking the user directly rather than
guessing, because each one changed what the result would mean:

1. **What "without Boltz processing" means.** Boltz's MSA `.npz` files
   aren't themselves a coevolution signal — they're Boltz's particular
   extraction/curation of RCSB structures into per-structure homolog sets.
   The live options were: (a) same architecture, different generic corpus
   (UniProt), or (b) same underlying RCSB structures, skip the MSA-specific
   slicing. Went with a same-structures cut, sourced fresh from RCSB
   directly — this holds the *structure population* fixed, isolating the
   pipeline step rather than confounding it with a totally different data
   source.
2. **What the ablation axis actually is.** PFold's own config defaults to
   `sequences_per_file=1` — one sequence per structure, same as the plain
   fetch would give. If both arms used that, the comparison would be
   cosmetic (same content, fetched two ways) and Boltz's actual
   distinguishing feature — multiple aligned homologs per structure —
   would never be exercised. Chose to set the Boltz arm to
   `sequences_per_file>1` specifically so the comparison would touch the
   pipeline's real value-add, not just its packaging.
3. **Reuse an existing PFold checkpoint or retrain from scratch?** PFold's
   `models_frozen/*.pt` checkpoints came from a training history with
   several undocumented architecture changes (per its own `readme.txt`).
   Reusing one would confound "data source" with "whatever config happened
   to produce that checkpoint." Retrained both arms from scratch, identical
   architecture/hyperparameters/seed, varying only the data.

## Pivot 1: the naive result looked like a real finding — and wasn't fully investigated before being reported

First real run: Boltz variant (`sequences_per_file=5`) vs. baseline
(1 seq/structure), 3 epochs each, 2344 real structures. Result:
val_accuracy 0.198 vs. 0.129, a +0.069 delta — a large, consistent,
same-direction-every-epoch effect. This was initially written up as
**"Boltz helps"** per the pre-registered decision rule (a positive,
epoch-consistent delta on real data).

That verdict was correct *by the letter of the pre-registered rule*, but
the rule itself had a blind spot: it only checked the *direction and
consistency* of the delta, not whether the two arms were otherwise
comparable. They weren't. **Confounder found immediately upon writing up
the result, not before running it:** at `sequences_per_file=5`, the Boltz
arm trains on 8200 sequences/epoch against the baseline's 1640 — a 5x
difference in training volume per epoch, from the same 1640 structures.
Matching *epoch count* is not the same as matching *gradient steps*. The
result as first reported conflated two different possible causes: "Boltz's
specific curation produces better training examples" and "there's simply
5x more of them." This is the exact failure mode the causal framing was
supposed to prevent, and it slipped through anyway because the
pre-registered rule checked the wrong invariant.

**Lesson, generalized:** when one ablation arm has a configuration knob
that also changes dataset size (here, `sequences_per_file`), a
pre-registered "is the delta positive and consistent" rule is not
sufficient to attribute the delta to the intended variable. The rule
needed a second clause: are gradient-steps-per-epoch actually matched? It
wasn't checked until after the result was already in hand.

## Pivot 2: curation-only disambiguation — isolate the pipeline from volume

Reran the Boltz arm at `sequences_per_file=1`, matching the baseline's
step count exactly (same 1640 structures, same 1640 sequences/epoch, same
everything else). This isolates Boltz's specific extraction/curation
*alone*, with the volume confound removed.

Result: val_accuracy delta collapsed to **−0.0042** (baseline slightly
ahead) — indistinguishable from noise, and the opposite sign from the
naive run. **Boltz's specific `.npz` processing, on its own, contributes
nothing measurable beyond a plain RCSB fetch.** This directly falsified
the naive "Boltz helps" reading and confirmed the confounder identified in
Pivot 1 was the actual cause, not a hypothetical risk.

## Pivot 3: the disambiguation itself was incomplete — total volume, not just per-epoch volume, needed matching

Even after Pivot 2, one more asymmetry remained, caught on a second look
at the numbers rather than at design time: matching *sequences-per-epoch*
still left total training volume unmatched across the *original* 3-epoch
comparison. Boltz-5 (naive run) saw 8200 × 3 = 24,600 total training
instances; the baseline's own 3-epoch run only saw 1640 × 3 = 4,920 — a 5x
smaller total budget, not a matched one. Pivot 2 answered "is Boltz's
curation good, independent of volume" (no) but left open "was the original
+0.069 win *entirely* explained by volume, or does some of it survive
under a volume-matched (not just steps-per-epoch-matched) comparison?"

Reran the baseline for **15 epochs** instead of 3 — 1640 × 15 = 24,600,
now matching Boltz-5's total instance count exactly, just via repetition
of the same 1640 sequences instead of seeing 5 distinct homologs 3 times
each. Result: val_accuracy delta shrank to **+0.0090** (from +0.0690) —
an 87% reduction. The baseline's own epoch curve (0.098 → 0.191 over 15
epochs, plateauing from ~epoch 8) showed it converging to nearly the same
place Boltz-5 reached in 3 epochs, just needing more passes to get there.

**This is the pivot that changed the final claim from "no effect" to
"real but narrow effect."** Pivot 2 alone would have supported "Boltz's
pipeline doesn't matter at all." Pivot 3 showed that's an overstatement —
there's a small residual (+0.009) that Pivot 2's steps-matched cut doesn't
explain, and the honest characterization of what Boltz's `sequences_per_file>1`
actually buys is **faster convergence from free extra training volume**,
not zero benefit and not a higher ceiling either.

## Pivot 4: repo extraction, and a framing correction that arrived one turn too late to prevent being asked to justify itself

After the three-cut result was written up in biostat's `docs/L40_PROTOCOL.md`,
the user asked directly: **"did you compare an msa model to non msa?!"**
The honest answer was no — and should have been stated as a caveat in the
original writeup, not only in response to being asked. Both models are the
identical single-sequence `ProteinBERT` (plain `nn.TransformerEncoder`, no
cross-sequence attention, no coevolution features, no MSA-Transformer- or
Evoformer-style row/column attention). What was actually varied is the
*training data source* — Boltz's MSA-derived `.npz` shards, consumed one
sequence at a time as independent training examples, vs. a plain RCSB
fetch. "MSA-augmentation ablation" is accurate as a data-pipeline
description but invites a natural misreading as an architecture
comparison, which the original writeup didn't pre-empt.

The user then reframed the project's actual purpose precisely: *"the
purpose of this repo is boltz data processing pipeline ablation evaluated
on downstream plm"* — i.e., the downstream PLM is the *measurement
instrument* for judging the data pipeline, not the object of study itself.
That's the correct framing, and it's a meaningfully different sentence
than "MSA model vs. non-MSA model." The corrected framing was written as
an explicit "What this is NOT" section leading both the new repo's
`README.md` and `docs/PROTOCOL.md`, specifically to prevent the same
misreading for anyone encountering the result without this conversation's
context.

The code, tests, docs, and result artifacts were then extracted into a
standalone public repo
([`DIvkov575/boltz-msa-ablation`](https://github.com/DIvkov575/boltz-msa-ablation))
so the experiment could be read, cited, and rerun independent of the
larger private `biostat` repo — verified to pass all 49 tests standalone
before pushing. The user has since indicated they expect to overwrite that
repo soon, which is why this trajectory record lives here in `biostat`
instead: the *result* (`docs/L40_PROTOCOL.md`, still in this repo, plus its
copy in the extracted repo) and the *reasoning that produced it* (this
doc) are meant to survive independently of any one repo's lifecycle.

## Final conclusion, stated precisely

Across three progressively more controlled comparisons:

| Cut | What it isolates | val_accuracy delta |
|---|---|---|
| Naive (matched epochs, unmatched steps) | Nothing cleanly — confounded | +0.0690 |
| Curation-only (matched steps/epoch) | Boltz's specific extraction/curation, alone | −0.0042 |
| Volume-matched (matched total instances) | Whether any residual survives full volume-matching | +0.0090 |

**Boltz's data-processing pipeline's value is almost entirely a training-
volume effect** (sampling 5 homologs per structure is a free 5x
data-multiplier), **not evidence of higher intrinsic per-example curation
quality** — the curation-only cut found ~zero effect from the pipeline's
extraction/cleanup step in isolation. The practical decision this
supports: adopt the pipeline if faster convergence / fewer wall-clock
epochs matters for your training budget; don't adopt it expecting a higher
accuracy ceiling than the same structures' canonical sequences would
eventually reach with enough passes.

## What this trajectory illustrates about the process, not just the result

- **A pre-registered decision rule can still let a confounded result
  through** if it only checks direction/consistency and not whether the
  compared conditions are otherwise matched. The fix wasn't "don't
  pre-register" — it was noticing the confound immediately upon write-up
  rather than treating the pre-registered verdict as final.
- **One disambiguation run is not automatically enough.** Pivot 2 (steps-
  matched per epoch) felt like a complete fix but left a second, subtler
  asymmetry (total volume across the original epoch counts) unaddressed.
  Pivot 3 only happened because the numbers were looked at again with the
  question "what exactly does this rule out, and what does it not," not
  because it was planned from the start.
- **A precise, corrected framing question from the user ("did you compare
  an msa model to non msa?!") surfaced a real gap in the original writeup**
  — the docs described the data-side variable accurately but didn't
  pre-empt the architecture-side misreading. The fix was structural (an
  explicit "what this is not" section), not just a one-off clarification.
- **Caveats and limitations named honestly throughout (single seed, no
  variance estimate, pilot scale, single architecture) are themselves part
  of the rigor record** — the results in `docs/L40_PROTOCOL.md` are stated
  as a pilot signal, not a definitive claim, deliberately.

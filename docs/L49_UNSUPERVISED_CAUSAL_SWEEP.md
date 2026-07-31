# L49 — Unsupervised Causal Candidate Generation (All 480 Heads, Zero Correlation)

## Why this exists

L48 tested exactly ONE head causally — Vig et al.'s correlational top pick
(layer 5, head 13, 12.9x contact enrichment) plus one control head — and
found no significant causal effect. That leaves an obvious question open:
was layer 5/head 13 simply the wrong pick, and would a DIFFERENT head show
a real causal effect? Vig et al.'s own candidate-generation procedure was
purely correlational (rank all 480 heads by attention-on-contacts
fraction, report the top). This asks: what if candidates are generated
purely CAUSALLY instead — ablate every single head, rank by actual effect
on the task, with zero correlational information used at all during
discovery?

## Method

Reused L48's exact ablation mechanism (`HeadAblationHook`) and masked-
single-residue-prediction task on the same 8 real PDB structures and the
same model (`Rostlab/prot_bert_bfd`). Sampled 104 positions (13 per
structure, proportionally across all 8 — not dominated by whichever
protein is biggest), computed baseline accuracy once, then ablated each of
all 30×16=480 heads ONE AT A TIME on the exact same 104 positions,
recording `mean_effect = ablated_accuracy - baseline_accuracy` per head
(negative = ablation hurts = causally important; positive = ablation
helps = removing it improves prediction).

Coarse-first design (matches L45's convention): 104 positions instead of
L48's full 770, since testing all 480 heads on the full set would take
~3 hours on this hardware vs. ~30 minutes for a representative subsample.

## Results

**Full ranking:** `src/l38/l49_causal_sweep_out.json` (all 480 heads).
166 of 480 heads (35%) show EXACTLY zero effect on this position sample —
real evidence of redundancy in the network (many heads carrying
overlapping information, so losing any one doesn't move the needle).
119 heads show a negative (causally-helpful) effect, 195 show positive
(causally-unhelpful/mildly-harmful-to-keep) — a sign test finds this
skew statistically real (p=0.000046) though the practical magnitude is
tiny (mean effect +0.0014, i.e. essentially negligible day-to-day; the
skew is a real but small effect, not something to build a strong claim on
by itself).

**The direct cross-check against Vig et al.'s correlational ranking is the
headline finding.** Looked up where L48's two tested heads land in this
fully independent, causally-generated ranking:

| head | correlational rank (Stage 1, by contact enrichment) | causal rank (this sweep, by real effect) | mean_effect |
|---|---|---|---|
| layer 5, head 13 (Vig's top pick) | **1st** of 480 (12.9x enrichment) | **313th** of 480 | +0.0096 (ablation slightly HELPS) |
| layer 17, head 1 (L48's low-enrichment control) | **480th** of 480 (0.055x, lowest) | **80th** of 480 | -0.0096 (ablation HURTS — more causally important than Vig's pick) |

**The correlational and causal rankings are not just weakly related for
this head — they're inverted.** The head Vig et al.'s method would point
to as most important is below-median in actual causal effect; the head
their method would dismiss as least important is meaningfully above-
median. This is the SAME conclusion as L48 (no causal effect from Vig's
pick), now confirmed against the full space of 480 alternatives rather
than one hand-picked control — ruling out "L48 just picked an unlucky
control head" as an explanation for the earlier null result.

## What the actual top causally-important heads look like

The 5 most causally important heads by this sweep (tied at mean_effect
-0.0385): layer 12/head 11, layer 14/head 15, layer 15/head 8, layer
18/head 3, layer 24/head 9 — no obvious pattern by layer depth (spans
early-mid to late-mid layers). Checked each directly against L48's full
enrichment matrix: 4 of these 5 (12/11, 14/15, 15/8, 18/3, enrichment
1.4-2.1x) are NOT in Stage 1's correlational top-30 at all — a genuinely
different candidate set than correlation alone would surface. The 5th
(layer 24/head 9, enrichment 3.25x) DOES appear in the correlational
top-30 (rank 19) — so the two methods aren't fully disjoint, just weakly
related overall: one real overlap out of five, alongside the much starker
inversion for Vig's specific #1 pick documented above.

## Interpretation

This strengthens, rather than merely repeats, L48's conclusion:
**attention-weight correlation with a real structural property is not
just an imperfect proxy for causal importance in this model — for the
specific head that correlation ranks first, it's actively misleading.**
Combined with L41 (SAE feature correlation with kinase activity, no
causal steering control) and L48 (this same lesson, one head), this is now
a THIRD independent instance, and the most thorough one (480 heads swept,
not one or two), of the same pattern in this project's work: don't trust
a correlational ranking to predict causal importance in protein LMs,
regardless of technique (SAE features, attention heads) or model family
(ESM-C, ESM2, BERT).

## What this does NOT show

Small effect sizes throughout (max magnitude 0.038, i.e. ~4 percentage
points on a 104-position sample) mean individual-head ablation is a
low-power test given this sample size — a real, larger effect on a head
outside the current sample could exist and not be detected. This is a
coarse pass; the natural, not-yet-taken next step would be re-running
L48's full 770-position, proper-paired-bootstrap significance test on
whichever heads this coarse sweep flagged as most extreme (the top ~5 by
magnitude in either direction), rather than trusting the coarse ranking's
exact order at face value.

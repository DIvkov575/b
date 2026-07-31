# L48 — Task A: Redoing Vig et al.'s Contact-Map Attention Head, Causally

## Why this exists

Vig et al., "BERTology Meets Biology" (ICLR 2021, arXiv:2006.15222) found
specific attention heads in protein LMs whose attention weights strongly
correlate with real 3D structure (contact maps, binding sites). Their
strongest result: one head in ProtBert-BFD puts 63.2% of its attention on
residue pairs in physical contact vs. a low background rate. The authors
state directly: "all of the above analyses are purely associative and do
not attempt to establish a causal link." A dedicated literature search
(2026-07-30) confirmed nobody has gone back and tested this causally in
the 5+ years since — this fills that specific, real gap (L47's Task A).

## Stage 1: replicate the correlational finding on real data

Used `Rostlab/prot_bert_bfd` — the exact model Vig et al. found their
strongest result on — and 8 real, diverse PDB structures (ubiquitin,
crambin, lysozyme, myoglobin, T4 lysozyme, protein G, fibronectin domain,
SH3 domain), downloaded directly from RCSB. Extracted real Cα-Cα contact
maps (8Å threshold, minimum sequence separation 6 — standard contact-
prediction convention, confirmed necessary: without excluding near-
diagonal pairs, background contact rate on a compact protein like
ubiquitin is 3-4x inflated vs. the pooled multi-protein rate).

**Result: replicated cleanly.** Pooled across all 8 structures, the
single most contact-enriched head (**layer 5, head 13**) puts 36.1% of its
attention on real contacts, a **12.9x enrichment** over the pooled
background rate (2.8%). A cluster of heads in layer 29 (the second-to-last
layer) also shows strong enrichment (8.8-10.4x), echoing Vig et al.'s own
finding that high-level structural concepts concentrate in deeper layers.
Full per-head matrix: `src/l38/l48_replication_out.json`.

## Stage 2: the actual causal test

**Task:** single-position masked-residue prediction — mask exactly ONE
residue at a time (every other position, including distant residues in
real 3D contact, stays visible) and predict it. A model genuinely using
non-local structural context should do measurably better on
contact-bearing positions than a purely local-sequence model would.

**Intervention:** zero out one attention head's contribution entirely
(`HeadAblationHook`, same head-slicing mechanism verified in L47's Phase 0
feasibility check, applied to `BertSelfAttention`). Compared three
conditions, pooled across all 770 real residues from the 8 structures:
1. Baseline (no ablation)
2. **Top contact head ablated** (layer 5, head 13 — the 12.9x-enriched
   head from Stage 1)
3. **Control head ablated** (layer 17, head 1 — the LOWEST-enrichment
   real head found in Stage 1, 0.055x background — a genuine "barely
   touches contacts at all" control, not an arbitrary/random pick)

**Real bug caught before the full run:** `BertForMaskedLM` nests its
encoder under `.bert.encoder`, unlike the plain `BertModel` used in Stage 1
(`.encoder` directly) — a smoke test on one small structure caught this
immediately (`AttributeError`) before wasting the full run.

**Result: no significant causal effect, on either head.** Paired
bootstrap (10,000 resamples, same method as L42/L43/L45/L47) on 603
contact-bearing and 167 non-contact-bearing masked positions:

| comparison | contact positions (n=603) | non-contact positions (n=167) |
|---|---|---|
| top-head-ablated vs. baseline | +0.0033 [-0.0050, 0.0116], not sig | -0.0060 [-0.0180, 0.0000], not sig |
| control-head-ablated vs. baseline | +0.0017 [-0.0182, 0.0199], not sig | -0.0060 [-0.0539, 0.0419], not sig |
| top-head vs. control-head ablation | +0.0017 [-0.0182, 0.0216], not sig | 0.0000 [-0.0479, 0.0479], not sig |

Every single confidence interval crosses zero. Ablating the 12.9x-contact-
enriched head is statistically indistinguishable from ablating a head that
barely attends to contacts at all — and neither ablation measurably hurts
prediction, even specifically at the positions with a real long-range
structural contact.

## Interpretation

**This is a real, clean negative result on the causal question, five
years after the correlational one was published.** The attention pattern
Vig et al. found is real (replicated cleanly in Stage 1) but does not
appear to be causally NECESSARY for the model's masked-residue prediction
behavior at contact-bearing positions — removing it entirely doesn't hurt
performance more than removing an arbitrary head that barely touches
contacts. This is consistent with — not contradicting — this project's own
prior findings: L41 found a raw SAE feature's high correlation with kinase
activity didn't translate to causal steering control; the field's own
antibody-SAE literature (cited in this project's L41 postmortem) found the
same "correlation without causal control" pattern. This is now a second,
independently-collected data point for the same broader lesson, on a
completely different technique (attention-head ablation, not SAE steering)
and a completely different model family (BERT, not ESM).

## What this does NOT show

Does not show attention is never causally important in protein LMs — only
that THIS SPECIFIC head, on THIS SPECIFIC task (single-position masked
prediction), does not show a detectable effect at this sample size (n=603
contact positions). A model may be highly REDUNDANT (many heads carrying
similar information, so ablating one doesn't matter) rather than not using
structural attention causally at all — this test cannot distinguish those
two explanations, and a full ablation of ALL contact-enriched heads
simultaneously (not tested here) would be the natural next check if this
result needed to be pushed further.

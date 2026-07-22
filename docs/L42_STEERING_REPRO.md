# L42 — Reproduce Huang et al.'s Activation Steering (Sanity Check Before Any New Claim)

**Pre-registered protocol.** Locked 2026-07-22, before any run.

## Why this exists

L41 (docs/L41_PROTOCOL.md, docs/L41_PAPER_ANALYSIS.md) tried activation
steering on ESM-C-300M toward a discrete function (kinase activity) using a
raw SAE decoder direction, and got a weak/null result. A dedicated
literature check afterward found this was the *expected* outcome, not an
anomaly: every published steering success targets a **continuous**
biophysical property (not a discrete function class) using a
**difference-of-means** vector (not a raw SAE feature), on a **mid-size
model** (650M+, not 300M). The one directly comparable small-scale attempt
(ProGen3-112M, arXiv:2606.16044) also failed, with the authors attributing
it to scale.

Before trying a new target, the responsible move is to confirm the
steering *harness itself* is trustworthy by reproducing a result that is
already known to work: Huang et al., "Steering Protein Language Models"
(arXiv:2509.07983, ICML 2025) — ESM2-650M, thermostability, difference-of-
means vectors. If this reproduces, we know our pipeline is sound and any
future new-target experiment's result (positive or negative) is
trustworthy. If it doesn't reproduce, we've found a real problem in our
own harness, not in the technique.

## The claim under test

**H1.** Adding a difference-of-means steering vector (built from ESM2-650M
activations of high- vs. low-thermostability sequences) to the model's
residual stream during masked-marginal scoring/generation increases a
downstream thermostability proxy score, relative to both an unsteered
baseline and a matched-norm random-direction control — replicating Huang
et al.'s qualitative finding (steered thermostability fitness ≫ unsteered).

This is explicitly a **reproduction**, not a novel claim — success here
buys confidence in the harness, not a new publishable result on its own.

## Method (mirrors Huang et al.'s description, adapted to what's checkable here)

- **Model:** `facebook/esm2_t33_650M_UR50D` (ESM2-650M, official HF
  checkpoint — matches Huang et al.'s exact model).
- **Target property:** thermostability. Proxy dataset: FLIP's meltome/
  thermostability split, or (fallback if not cleanly downloadable without
  an external repo) a UniProt-derived proxy — e.g. sequences from
  thermophilic vs. psychrophilic/mesophilic source organisms (taxonomy-based
  proxy for stability, acknowledged as weaker than a real ΔG/Tm assay but
  usable for a qualitative reproduction check).
- **Steering vector construction:** difference-of-means, per Huang et al.:
  mean per-layer activation (averaged across all tokens) over a
  high-thermostability sequence set minus the same over a low set. NOT an
  SAE feature — this is the specific fix flagged by the L42 literature
  check (raw SAE features are documented as less reliably causal).
- **Layer(s):** apply across all transformer layers except the input
  embedding layer, per Huang et al.'s description (h̃_l = h_l + α·v_l,
  renormalized to original activation norm) — not a single arbitrarily
  chosen layer (the mistake in L41).
- **Alpha sweep:** small range bracketing where Huang et al. report peak
  effect before over-steering collapse; determine empirically via a dose-
  response check (their own methodology), not by copying L41's arbitrary
  [5, 10, 20].
- **Evaluation:** score generated/steered sequences with an independent
  thermostability proxy (not the same mechanism used to build the steering
  vector) — e.g. a simple sequence-composition-based stability heuristic or
  a frozen-embedding probe trained on a disjoint split, mirroring L41's
  Gate 3 independent-classifier discipline.
- **Control:** matched-norm random direction, same as L41 Gate 2 — this
  control survives from L41 unchanged, it was never the flawed part.

## PASS / KILL rule (pre-registered)

- **PASS (harness confirmed trustworthy):** real-direction steering beats
  both the unsteered baseline and the random-direction control, with a
  clear dose-response pattern (effect grows with alpha up to a collapse
  point, per Huang et al.'s own reported shape) — not just a one-off
  numeric edge.
- **KILL (harness has a real problem, independent of target choice):** no
  clear separation from the random control, or no dose-response pattern.
  If this happens, STOP — do not proceed to any new-target experiment
  until the harness bug is found, since a broken reproduction of a known
  result means any new result (positive or negative) from the same
  pipeline is unfalsifiable.
- **AMBIGUOUS:** partial reproduction (e.g. direction beats baseline but
  not convincingly beats the random control, or a weak/noisy dose-response).
  Treat as informative but not sufficient to trust the harness for a new
  claim — investigate further before building on it.

## Compute

ESM2-650M easily fits on the A10G (23GB free, confirmed idle before this
run). Expect comparable or slightly higher cost than L41's 300M runs given
~2x the parameter count, still well within a single-session budget.

## What this is NOT

Not a new scientific claim, not a paper-quality result on its own, and not
guaranteed to fully match Huang et al.'s exact numbers (different proxy
dataset for thermostability, since their exact training/eval sequences
aren't independently confirmed to be trivially downloadable here). The bar
is qualitative reproduction of the *pattern* (steering > random > and
dose-responsive), not numerical exactness.

---

## RESULTS (2026-07-22) — NOT a clean reproduction; found a real artifact, twice

Real generation + real steering pipeline ran end-to-end on the A10G with
ESM2-650M, 300 sequences for vector-building, 60 held-out low-Tm eval
sequences, 6-point alpha sweep. Two real bugs were caught and fixed along
the way (both before trusting any number):

1. **Alpha range and mask fraction, both wrong on the first attempt.**
   Inherited L41's mask_fraction=0.8 and an alpha range of [2, 16] without
   re-checking either against this model/task. Direct inspection found (a)
   even the UNSTEERED baseline degenerates at mask_fraction=0.8 (single-shot
   80%-masked infilling is simply too hard for this model in one forward
   pass — independent of steering), and (b) alpha=2 already fully saturates
   generation into near-pure poly-leucine, making every alpha from 2-16
   produce byte-identical output and byte-identical scores (confirmed:
   15+ decimal places identical) — not a "no effect" finding, a "already
   maxed out before the sweep starts" finding. Fixed: mask_fraction=0.3
   (confirmed non-degenerate baseline), alpha grid re-derived empirically
   from a manual sweep: [0.1, 0.25, 0.5, 1.0, 2.0].

2. **Scoring metric was confounded with generation fluency, not
   thermostability.** First version scored generated sequences with the
   model's OWN self-likelihood. Once alpha got large enough to push toward
   the poly-leucine collapse, likelihood dropped sharply — but that's
   because poly-leucine looks unusual to the model, not necessarily because
   it's less thermostable. Self-likelihood cannot distinguish "less stable"
   from "doesn't look like normal protein text anymore." **Fixed by
   switching to the Guruprasad et al. (1990) instability index** — a
   purely compositional formula (no model involved at all), verified against
   a known reference (hen egg-white lysozyme scores 16.1, correctly under
   the classical 40-point "stable" threshold) before trusting it.

**After both fixes, the result initially looked like a clean PASS** (real
direction's effect grows with alpha, consistently larger than a matched-norm
random control, automated verdict = PASS) — until the actual generated
sequences were inspected directly rather than trusting the aggregate score:

```
baseline:        MAQTTPIAEQMAALNNSSDTSFAADSSSSLLNATCPARRQNSVDQRKISRSFSDDSSSS...
real, alpha=1.0: MAQTLPIAEQMALLNNSLDTLFAADLSLRLLNATCPARLQNSVDQRKILRSFLDLLLSL...
real, alpha=2.0: MAQTLPIAEQMALLNNSLDTLFAADLSLLLLNATCPARLQNSVDQRKILRSFLDLLLSL...
```

**This is the SAME poly-leucine collapse found during the mask-fraction/
alpha diagnostic, just at a lower alpha threshold than first suspected.**
The instability-index metric happens to score leucine-heavy sequences as
artificially *low*-instability (i.e. "stable") because leucine's dipeptide
interactions are mostly neutral in the DIWV table — so the "PASS" result is
measuring "the model collapsed toward a residue this specific formula
rewards," not a genuine, biologically meaningful thermostability shift.
**Two different, methodologically-independent scoring metrics (self-
likelihood, then instability index) have now both been confounded by the
exact same underlying generation artifact** — which is itself informative
(the steering vector's dominant, robust effect at this alpha range really is
"push toward leucine," not "push toward whatever high-Tm sequences actually
look like"), but it means neither number should be trusted as a
reproduction of Huang et al.'s result.

**Lower alphas (0.1–0.5) show small, non-collapsed changes** (1-3 residue
substitutions per sequence, no visible leucine bias) — this is the only
regime that might contain a real, uncontaminated signal, but the effect
size there is small and hasn't been checked against a proper significance
test (no bootstrap CI computed yet, unlike L39's rigor).

## Honest verdict: reproduction inconclusive, not confirmed

**This has NOT reproduced Huang et al.'s result cleanly.** What it has done:
found and fixed two real, load-bearing methodology bugs (wrong alpha/mask
scale; a confounded scoring metric), and discovered that the specific
difference-of-means direction constructed here has a dominant failure mode
(collapse toward leucine) that contaminates evaluation at the alpha range
where the effect is large enough to matter. Two honest paths forward, not
yet taken: (a) restrict analysis to the low-alpha (0.1–0.5), non-collapsed
regime and properly bootstrap it for significance, accepting a possibly
small/inconclusive effect size, or (b) diagnose why leucine-collapse
specifically dominates this direction (composition mismatch between the
low/high Tm groups used to build it?) before trusting any alpha range.
Neither has been done. **Do not build a new-target experiment on this
harness yet** — the harness itself just produced a false-positive-looking
PASS that only fell apart on manual inspection of the raw generated text,
which is exactly the failure mode this reproduction check was meant to
catch before it could contaminate a new claim.

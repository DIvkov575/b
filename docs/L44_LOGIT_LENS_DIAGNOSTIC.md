# L44 — Logit-Lens Prediction of Steering Collapse Residue (Falsified)

**Not pre-registered** — this was a same-session diagnostic idea, tested
quickly and cleanly falsified. Documented for completeness, matching this
project's discipline of recording real negative results, not just wins.

## The hypothesis

In both L42 (thermostability) and L43 (solubility), pushing steering alpha
too high didn't collapse generation into random garbage — it collapsed
into ONE SPECIFIC amino acid (leucine for L42, alanine/glycine for L43),
and that residue was over-represented in whichever group built the
respective steering vector. Hypothesis: over-steering collapse is
predictable in advance, before ever running the expensive generation sweep,
by checking which vocabulary token a steering vector points toward most
strongly in the model's own embedding space (a logit-lens-style check).

## Method (with two real bugs found and fixed before trusting the result)

1. **First attempt: route the vector through the full `lm_head`**
   (`dense -> layer_norm -> decoder`). Failed a self-identity smoke test:
   projecting leucine's OWN embedding row through the full head does not
   recover leucine as the top match (returns K/M/`<unk>` instead) — the
   `dense`+`layer_norm` transform is fit for real hidden states passing
   through the full forward pass, not for an arbitrary difference vector
   injected directly.
2. **Second attempt: raw dot product against the tied embedding matrix
   directly** (skip `lm_head` entirely). Passed the self-identity smoke
   test (leucine -> leucine, score 8.69, next-closest 1.93) but failed on
   the real steering vectors: top-1 tokens were B/U/Z/O (non-standard
   amino-acid ambiguity codes) at nearly every layer for both targets.
   Root cause, checked directly: these tokens have outlier embedding
   NORMS (up to 6.15 vs. ~2.4–3.4 for the 20 standard residues) that
   dominate a raw dot product regardless of direction.
3. **Final method: cosine similarity, restricted to the 20 standard amino
   acids.** Passes the self-identity smoke test cleanly and removes both
   confounds above (`src/l38/l44_logit_lens_diagnostic.py`,
   `logit_lens_top_tokens()`).

## Result: hypothesis falsified

Even with the corrected method, neither target's collapse residue is the
dominant top-1 direction at most layers:

- **Thermostability**: aggregated top-1 across 33 layers is dominated by V
  and T (14 and 8 layers respectively); leucine appears as top-1 in only 5
  of 33 layers, never with a cosine similarity above ~0.03 (out of a max
  possible 1.0).
- **Solubility**: aggregated top-1 is dominated by T and R (16 and 6
  layers); alanine appears as top-1 in only 3 of 33 layers (cosine
  similarity 0.024–0.029), glycine never appears as top-1 at any layer.

Full per-layer detail: `src/l38/l44_logit_lens_out.json`.

## Why this makes sense in hindsight

A difference-of-means vector is built by averaging real activations across
150 real sequences per group — there's no reason it should point cleanly
at one token's embedding the way a hand-picked "inject this concept"
vector might. The observed collapse is more likely a downstream DECODING
DYNAMICS effect: iterative argmax mask-filling compounds a small,
distributed bias across all 33 layers over many masked positions, and
whichever residue is even slightly favored by the aggregate perturbation
gets locked in and then self-reinforces (once a few positions fill with L,
neighboring positions become more likely to also fill with L, since L-rich
local context shifts the model's own predictions). That's a property of
the GENERATION PROCESS, not something visible in a single static
projection of the vector itself.

## What this rules out

Cheap pre-screening for collapse risk via a static vector projection,
before running the actual (expensive) generation sweep, does not work with
this method. If collapse risk needs to be predicted in advance for a
future target, the honest answer is: it doesn't look predictable this way,
and you have to run the empirical alpha sweep as L42/L43 already do
(check degeneracy rate directly on generated output) rather than trying to
shortcut it via the vector's own geometry.

# L37 — Math Spec v2: Closing the MD-Realism Gap (Local Boundedness + Anti-Trapping)

**Date:** 2026-07-15 · **Status:** Draft spec (pre-proof) · **Resolves the open item in:**
`docs/L37_MATH_SOLUTION.md` §6 ("MD realism check") · **Depends on:** Theorem 2, Theorem 3,
No-go Theorem B, and §8.2 of `L37_MATH_SOLUTION.md` (not restated in full here)

This is a plan for the next proof attempt, not a completed result. It states precisely what
must be shown, the machinery to attempt it with, and — up front — the gate most likely to
block it, so a partial result doesn't get overclaimed the way the original spec's headline
theorem did. Nothing here requires compute; GATE-A is a model-fact check against the MDGen
paper/code, everything else is pure math.

---

## 0. Exactly what gap this closes

`L37_MATH_SOLUTION.md` Theorem 2 proves `D_χ² ≤ 2B·KL(ρ̂‖ρ)` **conditional on both normalized
joint densities `q, q̂` being bounded by `B` a.e.** Its own "MD realism check" shows this
hypothesis is **not free**: a stationary Ornstein–Uhlenbeck process — about the simplest
possible confined (mean-reverting, quadratic-potential) system — already has `q(x,y) → ∞`
along the diagonal. If the *nicest* linear-Gaussian confined system fails global boundedness,
**global boundedness should be expected to fail generically** on any noncompact state space
with a continuous, everywhere-supported equilibrium density. Chasing a global bound is very
likely a dead end.

The solution doc's own §7 "Consequence for the draft's local-sublevel route (C3)" already
shows the *naive* fix — restrict to a compact sublevel set and bound the excluded mass —is
**insufficient**: No-go Theorem B exhibits a rare region that is simultaneously small in
`μ`-mass, small in its contribution to bulk `D_χ`/TV, and yet hosts a spurious slow mode that
drags a nontrivial eigenvalue to 1. Their diagnosis: a valid local route needs *"a uniform
conductance, escape-probability, or return-time bound for the excluded region"* — not just a
mass bound.

**This spec's target:** replace global boundedness with (a) a **local** boundedness lemma on a
compact sublevel set, PLUS (b) an explicit **anti-trapping** condition of the kind their
diagnosis calls for, stated precisely enough to attempt for MDGen's actual reference dynamics
and — separately, and this is the likely blocker — for its distilled student.

---

## 1. The target object, pinned to MDGen

| Object | MDGen specifics |
|---|---|
| Per-residue coordinate | rigid-body pose (rotation ∈ SO(3), translation) + up to 7 torsion angles (backbone + sidechain) |
| Torsion component | compact (flat torus) — **not** the concern |
| Rotation component | compact (SO(3)) — **not** the concern |
| Translation component | ℝ³ per residue — **the concern**, see below |
| `μ` | canonical (NVT) Boltzmann, `dμ ∝ exp(−βU(x))dx`, `U` = physical force field (bonded + nonbonded) |
| `τ` | ~10 ps coarse lag (MDGen's reported step) |

**GATE-A (model fact, not a proof — resolve first, cheap).** MDGen's per-residue frames must be
checked against the paper/code (arXiv:2409.17808) for **which of two cases holds**:
- **(A-local)** frames are chain-relative (each residue's pose expressed relative to a moving
  reference threaded along the backbone). Then translation is *intrinsically* confined by the
  representation itself — adjacent frames can't drift apart in the coordinate system, independent
  of physics. Local boundedness (§2) becomes easy on the *whole* space, not just a sublevel set.
- **(A-global)** frames are in one fixed lab/global frame (the AlphaFold2/RFdiffusion convention).
  Then translation is **not** confined by construction; confinement is purely a fact about the
  physical potential `U` (covalent bond/angle terms), and everything below is load-bearing.

This single fact changes how much of §2–3 is even necessary. **Not yet checked this session —
first action before investing in the harder gates.**

Separately: confirm MDGen's reference MD trajectories were generated under thermostatted
(Langevin/Nosé–Hoover, NVT) rather than microcanonical (NVE) dynamics — `L37_MATH_SOLUTION.md`
§2.1 already flags that the canonical density `Z⁻¹e^{−βU}` describes NVT, not NVE, and that
underdamped dynamics is reversible only after momentum reversal, not in the ordinary sense used
by Theorem 1–3. This is a precondition for every reversibility assumption downstream.

---

## 2. Local boundedness lemma (should be the easy half)

**Claim.** Let `Ω_R = {U ≤ R}`. If (i) `μ` has a continuous, strictly positive density on a
neighborhood of `Ω_R`, and (ii) the lag-`τ` transition kernel `p(y|x)` has a jointly continuous,
everywhere-positive density on `Ω_R × Ω_R`, then `q(x,y) = p(y|x)/μ(y)` is bounded on
`Ω_R × Ω_R` by some finite `B_R`, by continuity + compactness (extreme value theorem). This
direction is close to free *given* (i) and (ii).

**The actual content is (ii).** For underdamped Langevin dynamics (noise only on velocity, not
position), existence of a smooth, everywhere-positive transition density at any fixed `τ > 0` is
a **hypoellipticity** fact — the generator satisfies Hörmander's bracket condition because the
velocity noise, combined with the position-velocity coupling, spans the full tangent space after
one Lie bracket. This is classical (Hörmander's theorem; made quantitative via Malliavin calculus,
e.g. Norris's lemma) but **must be cited and checked against MDGen's actual reference dynamics**,
not asserted. One point in our favor: a *finite, positive* lag `τ` (MDGen's ~10 ps, not an
infinitesimal step) is the *easy* regime for hypoelliptic smoothing — heat kernels of hypoelliptic
generators become smoother, not more singular, away from `t=0`. This should make (ii) **more**
plausible at MDGen's coarse lag than at an infinitesimal one, not less — but this needs an
explicit citation (Aronson-type two-sided Gaussian bounds, or the hypoelliptic analog), not just
the intuition stated here.

**GATE-B:** does a citable hypoellipticity/heat-kernel result actually deliver (ii) for the
reference Langevin dynamics at MDGen's lag? *(Standard machinery, needs the right citation —
flagged for fetch-verification, not yet done.)*

---

## 3. Anti-trapping condition (the hard half)

The goal is a **quantitative** version of "the excluded region `Ω_R^c` cannot hide a slow mode,"
precise enough to combine with §2's local bound into a *global* spectral-gap statement.

**Candidate machinery:**
- **Cheeger's inequality** for reversible Markov chains: `Φ²/2 ≤ 1−λ₁ ≤ 2Φ`, where `Φ` is the
  conductance (bottleneck ratio). This directly relates the *global* spectral gap to a
  *geometric/flow* quantity — exactly the kind of object the solution doc's diagnosis calls for
  ("uniform conductance"). *(Canonical statement — flagged for exact-form fetch-verification.)*
- **Foster–Lyapunov geometric drift** (Meyn–Tweedie): a condition `𝔼[V(X_{t+τ})|X_t=x] ≤ γV(x) + b·1_{Ω_R}(x)`
  for a Lyapunov function `V` (e.g., `V = U` or a power of it) gives **geometric ergodicity** —
  a quantitative, uniform-in-tail spectral gap lower bound — and is the standard tool for proving
  exactly this for Langevin-type dynamics under a dissipativity condition on the force field.
  *(Reference literature on ergodicity of Langevin integrators under dissipativity — flagged for
  fetch-verification; not yet independently confirmed this session.)*

**GATE-C (reference side — plausible, needs the citation nailed down):** does a drift condition
of this form hold for confined molecular force fields (covalent + native-state stability), giving
a *reference-side* uniform escape-rate bound outside any `Ω_R` with small excluded mass? This is
the piece that would rule out a real No-go-Theorem-B-style trap **in the true dynamics** — i.e.
that real confined MD doesn't have a hidden slow trap the reference measure fails to see.

**GATE-D (student side — the likely blocker).** Does an *analogous* drift/conductance bound hold
for the **distilled few-step generator**? This is almost certainly **not free**. A learned
generative sampler has no architectural guarantee of satisfying a drift condition; its tail
behavior — what it does in low-density regions it rarely trains on — is exactly where generative
models are known to misbehave (mode-seeking collapse, ungrounded extrapolation far from the data
manifold). GATE-D is where I expect this to actually block, not GATE-B/C.

**The design implication, if GATE-D doesn't close for free:** the anti-trapping condition has to
be *enforced*, not derived — e.g. a minimum-noise floor guaranteeing a uniform (Doeblin-style)
minorization everywhere, an explicit energy-based rejection step, or a plausibility/designability
gate on generated frames. That last option is worth flagging directly: it is structurally the same
kind of guardrail this session's earlier **L18/L32 diagnostics** (a designability-correlated
distance that catches silently-implausible structures) were built around. If GATE-D can only be
*enforced* rather than proved, the honest framing is: **the spectral theorem is conditional on the
method design including such a safeguard** — it is not a free-standing statement about generic
few-step distillation. That is a materially weaker (but still real) claim than the original
unconditional theorem, and it should be stated as conditional from the start rather than
discovered as a gap later.

---

## 4. Gate table (cheapest / most-likely-true first)

| Gate | What it checks | Type | Expected outcome |
|---|---|---|---|
| **A** | MDGen frames: chain-relative or global? (+ NVT vs NVE reference data) | model fact, read the paper/code | unresolved — do first |
| **B** | Hypoellipticity ⇒ local density boundedness on `Ω_R` | citable lemma | plausible, needs citation nailed down |
| **C** | Geometric drift for the *reference* Langevin dynamics ⇒ anti-trap bound | citable + adapt | plausible, needs citation nailed down |
| **D** | Matching drift/conductance bound for the **student** | open design question | **likely blocked** without an explicit enforced safeguard |
| **E** | Assemble B+C(+D) into a non-vacuous, MDGen-realistic timescale bound | synthesis | conditional on D — either a real (conditional) theorem, or fall back to §8.2 |

**Order of attack:** A (fastest — settles how much of B is even needed) → B (low difficulty given
the right citation) → C (medium — literature search + adaptation to a confined molecular
potential) → D (the genuine open question — resolve as "requires an explicit safeguard," not as a
free proof) → E (assembly, low difficulty once A–D are settled).

---

## 5. Fallback, pre-registered now

If GATE-D does not close — even with an enforced safeguard — the standing theoretical result
remains `L37_MATH_SOLUTION.md` §8.2: a Pinsker-based bound on a single bounded, fixed teacher-mode
autocorrelation, requiring no density-ratio assumption, but **not identifying a student
eigenvalue**. That should be written up as the paper's theory contribution rather than continuing
to chase the full spectral bound. Pre-registering this now is deliberate — it's the same
discipline (state the kill condition before you have a result to be attached to) that this
project's earlier post-mortems named as the fix for over-building past a weak premise.

## 6. Citations used above, verification status

All of the following are cited from standard-literature memory in this spec and **have not been
fetch-verified this session** — do so before any write-up relies on them:
Hörmander's hypoellipticity theorem; Norris's lemma (Malliavin-calculus route to smooth
hypoelliptic transition densities); Aronson-type two-sided Gaussian heat-kernel bounds; the
classical Markov-chain Cheeger inequality (`Φ²/2 ≤ 1−λ₁ ≤ 2Φ`); Meyn & Tweedie, *Markov Chains and
Stochastic Stability* (Foster–Lyapunov geometric ergodicity); geometric-ergodicity-under-
dissipativity results for Langevin dynamics/integrators (exact reference to be pinned down).

# L37 — Math Spec: Kinetic-Observable Error Bounds for Few-Step Distilled Trajectory Generators

**Date:** 2026-07-13 · **Status:** Proof-audited 2026-07-14; end-to-end theorem blocked · Target model: MDGen (arXiv:2409.17808)

> **Resolution:** See [`L37_MATH_SOLUTION.md`](L37_MATH_SOLUTION.md). GATE-0 passes.
> GATE-1's registered C1 condition is disproved: bounding only `p̂/μ` does not make
> KL control `D_χ`. No registered PASS/KILL branch exactly fits the result, and the
> headline MD theorem remains blocked. The corrected bridge proved in the companion
> assumes both normalized joint densities are bounded; this is a conditional
> sufficient condition, not a necessary one or an established MDGen property. The
> proposed TV marginal correction is also insufficient in a reference-uniform
> theorem. GATE-2 is superseded: once common-space operator-norm control is
> available, self-adjointness of the reference removes any need for a global
> student `κ(V)` factor.
>
> The scalar first-order mode formula requires a simple eigenvalue; repeated modes
> have only cluster/subspace guarantees. A two-time pair law alone does not
> establish Markov-consistent multi-time kinetics. The corrected MFPT/committor
> result is a target-specific killed-kernel `L²` bound, not a pointwise bound and
> not a consequence of the full-chain slow gap. Actual path claims require
> infinite-horizon Markov consistency only when they are identified with that
> pair-kernel resolvent; non-Markov path laws can still define hitting quantities
> directly. Stationarity, absolute continuity, MD-specific density constants, the
> exact loss-to-segment-KL premise, empirical GATE-3 calibration, and novelty
> remain open.
>
> The body below preserves the pre-proof registration with audit annotations and
> contains claims now proved false or too broad. The claim-by-claim table in the
> companion solution is authoritative.

This document states the mathematics precisely enough to (a) attempt the proofs, or (b) show a
step is blocked. It is the pre-registered theory kill-gate for L37. The single load-bearing
uncertainty — the bridge from an achievable distillation loss to the χ² distance the spectral
bound needs — is isolated in §6 and gated in §8. Nothing here requires compute.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| `𝒳` | configuration space of one frame (MDGen: SE(3) pose × torsion angles per residue; treat as a Polish space with reference measure) |
| `μ` | equilibrium (Boltzmann) distribution on `𝒳`, `dμ = Z⁻¹e^{−βU}dx` |
| `τ` | lag time (one MDGen coarse step ≈ 10 ps) |
| `p(y∣x)` | true lag-`τ` transition kernel of the reference MD |
| `p̂(y∣x)` | lag-`τ` transition kernel *induced by the few-step distilled generator* |
| `ρ(x,y) = μ(x)p(y∣x)` | true two-time joint law at lag `τ` |
| `ρ̂(x,y) = μ̂(x)p̂(y∣x)` | distilled two-time joint law; `μ̂` = distilled one-time marginal |
| `P, P̂` | transfer (transition) operators on functions, `(Pf)(x)=∫p(y∣x)f(y)dy` |
| `{λ_i}, {ψ_i}` | eigenvalues / eigenfunctions of `P` (slow spectrum), `1=λ₀>λ₁≥…` |
| `t_i = −τ / ln λ_i` | `i`-th implied timescale (the primary kinetic observable) |
| `δ_i` | spectral gap isolating `λ_i`: `δ_i = dist(λ_i, spec(P)∖{λ_i})` |
| `D_χ` | χ²-type distance between joints (defined in §3) — the functional the bound is stated in |

---

## 1. What is being modeled

MDGen generates a **trajectory segment** `(x₀, x_τ, …)` **jointly** (not autoregressively). We only
need its induced **two-time law** at lag `τ`. Marginalizing the segment to consecutive pairs gives
a transition kernel `p̂(y∣x)`; the distilled (few-step) generator gives `p̂`. The scientific claim
to be made rigorous:

> **Audit correction:** a two-time law suffices for a one-lag pair-operator
> statement only. It does not determine actual multi-lag, MFPT, or committor
> behavior without path-law assumptions.

> Few-step distillation can leave every **per-frame** distribution (hence FID / structural
> plausibility) essentially unchanged while corrupting **kinetic** observables (implied timescales,
> metastable-set assignments, mean first-passage times). A bound must therefore be stated on the
> **joint two-time law**, not the marginal.

---

## 2. Assumptions (state each; flag realism for MD)

> **Historical registration:** the realism labels in this section were
> pre-audit judgments. They are not verified model facts; the companion
> solution's status table supersedes them.

- **(A1) Reversibility + discrete slow spectrum.** The reference dynamics are stationary and
  `μ`-reversible (detailed balance): `μ(x)p(y∣x) = μ(y)p(x∣y)`. Then `P` is self-adjoint on
  `L²(μ)` with real spectrum in `(−1,1]` **[historical interval; corrected to
  `[−1,1]`]** and an isolated slow part `1=λ₀>λ₁≥…≥λ_{m−1}` above the
  essential/continuous spectrum, gaps `δ_i>0`.
  *Historical realism assessment, unverified:* NVT/NVE equilibrium MD is reversible; discrete
  slow spectrum with a gap is the standing assumption that makes MSMs work. Ordinary
  configuration-space detailed balance, ergodicity, isolation, and multiplicity remain assumptions.
- **(A2) Bounded slow modes.** `‖ψ_i‖_∞ ≤ M_i < ∞` for the `m` slow eigenfunctions.
  *Historical realism assessment, unverified:* slow eigenfunctions are approximately smoothed
  indicators of metastable sets. Boundedness, and the stronger Lipschitz property needed by
  the Wasserstein route, remain model-specific assumptions.
- **(A3) Finite χ² / bounded density ratio.** `D_χ < ∞`. The registered equivalence with
  `ρ̂/(μ⊗μ) in L²` and the claimed student-only Doeblin sufficiency
  `p̂(y∣x)/μ(y) ≤ B` are **false without reference-side `L²` control**.
  *Realism:* **this is the demanding assumption.** It fails if the distilled model leaks mass into
  low-`μ` regions. It is the crux of §6 and the kill-gate of §8.
- **(A4) Marginal preservation (idealized) vs. drift (honest).** Idealized: `μ̂ = μ`, so `P` and
  `P̂` act on the *same* Hilbert space `L²(μ)`. Honest: `μ̂ ≠ μ`, `P̂` need not be `μ`-reversible or
  even normal — operators live on different spaces and a coupling term `δ_stat := ‖μ̂ − μ‖` enters.
  *Historical realism assertion, unverified:* distillation does not exactly preserve `μ`; the
  honest case (A4-honest) is what a real theorem must handle. See §5.2.

---

## 3. The distillation functional `D_χ` (Leg A object)

Define the **χ²-divergence between the two-time joints, referenced to the equilibrium product**:

```
D_χ²  :=  ∬  ( ρ̂(x,y) − ρ(x,y) )²  /  ( μ(x) μ(y) )   dx dy .
```

Under (A1)–(A4-idealized), `D_χ` equals the **Hilbert–Schmidt norm of the operator perturbation**:

```
‖P̂ − P‖_{op}  ≤  ‖P̂ − P‖_{HS}  =  D_χ .
```

**Why this functional and not the marginal.** `D_χ` is a distance on `ρ̂` (the *joint*).
The historical claim that MDGen training can control it is unproved; only marginal
insufficiency is proved — see §7.

---

## 4. Leg B — from operator perturbation to kinetic observables

> **Audit correction:** the scalar expansion below requires a simple isolated
> eigenvalue and explicit smallness. Repeated modes have cluster/subspace
> bounds. The eigenvector estimate alone does not prove set assignment, and
> the registered MFPT domain and full-gap amplification claim are false. See
> the companion solution.

Standard self-adjoint perturbation theory, once (A1)–(A3) give a self-adjoint `P` and `‖P̂−P‖_{op}≤D_χ`:

### 4.1 Eigenvalue / timescale error (the headline)
First-order self-adjoint perturbation with residual control (Weyl / Bauer–Fike):
```
λ̂_i − λ_i  =  ⟨ψ_i, (P̂−P) ψ_i⟩_μ  +  R_i ,        |R_i| ≤ 2 D_χ² / δ_i .
```
The leading term is `⟨ψ_i,(P̂−P)ψ_i⟩_μ = Ĉ_i(τ) − λ_i`, where
`Ĉ_i(τ) = 𝔼_{ρ̂}[ψ_i(x₀)ψ_i(x_τ)]` is the **distilled equilibrium autocorrelation of slow mode `i`**
and `C_i(τ)=λ_i` the true one. Hence
```
|λ̂_i − λ_i|  ≤  D_χ      (non-perturbative, Weyl)
```
and, propagating through `t_i = −τ/ln λ_i`,
```
| t̂_i − t_i |  ≤  ( t_i² / (τ λ_i) ) · D_χ  +  O(D_χ²) .        (★ MAIN BOUND)
```
**Interpretation:** kinetic error is *linear* in distillation error `D_χ`, *quadratically amplified
by the slow timescale* `t_i²`. Slow processes are the fragile ones — exactly the regime MD cares about.

### 4.2 Metastable assignment (eigenvectors)
Davis–Kahan sin-Θ:
```
‖ sin Θ(ψ̂_i, ψ_i) ‖  ≤  D_χ / δ_i .
```

### 4.3 Mean first-passage time / committors (resolvent)
For metastable sets `A,B`, MFPT solves `(I−P)m = 1` on `{A,B}ᶜ`; resolvent perturbation gives
```
| Δ MFPT |  ≤  C · ‖(I−P)⁻¹‖²_{0} · D_χ ,     ‖(I−P)⁻¹‖_{0} ~ 1/gap ~ t₁/τ ,
```
(`‖·‖_0` on the mean-zero subspace). MFPT error also scales with slow-timescale amplification.

**Status of Leg B:** textbook once §3 holds and `P̂` is self-adjoint. The intellectual work is (i)
establishing §3 from an achievable loss (§6), and (ii) removing the self-adjointness idealization (§5.2).

---

## 5. Two theorem versions

### 5.1 Clean theorem (idealized, `μ̂=μ`, `P̂` self-adjoint) — ~2-week lemma
> **Audit correction:** only the spectral portion follows after adding
> simplicity or a cluster formulation and explicit smallness. Assignment,
> hitting-time, and actual-path conclusions need their separate hypotheses.

Under (A1),(A2),(A3),(A4-idealized): bounds §4.1(★), §4.2, §4.3 hold with `‖P̂−P‖_op ≤ D_χ`.
Publishable only as a stepping stone; the assumptions are too strong to be the final claim.

### 5.2 Honest theorem (`μ̂≠μ`, `P̂` non-normal) — the real target
> **Superseded registration:** the additive-TV and `κ(V)` prescriptions below
> are disproved or unnecessary as stated; see the companion solution.

Two complications, each with a known remedy that must be assembled:
- **Different Hilbert spaces.** `P` on `L²(μ)`, `P̂` on `L²(μ̂)`. Introduce the transport/embedding
  and pay a `δ_stat = ‖μ̂−μ‖_TV` (or χ²) coupling term: bounds gain an additive `O(δ_stat)`.
- **Non-normality.** `P̂` not self-adjoint ⇒ Weyl fails. Replace with **Bauer–Fike** (eigenvalue
  perturbation scaled by the eigenvector-basis condition number `κ(V)`) or **pseudospectral**
  bounds. The clean `|λ̂_i−λ_i|≤D_χ` becomes `≤ κ(V)·D_χ + O(δ_stat)`.

**Target statement (honest):**
```
| t̂_i − t_i |  ≤  ( t_i² / (τ λ_i) ) · ( κ(V) · D_χ  +  c · δ_stat )  +  higher order .
```
*Historical difficulty assessment, superseded:* solid workshop / short-conference theorem,
not open. The audit instead shows that `κ(V)` is unnecessary once a common-space
operator bound exists, while obtaining that bound remains conditional.

---

## 6. THE LOAD-BEARING GAP — loss ⇒ `D_χ` (decides whether the clean bound exists)

The bounds are stated in `D_χ` (χ²/HS on the joint). The historical premise
that an intended consistency / shortcut / mean-flow objective controls an
`L²` velocity/score error that maps to segment `W₂` or `KL` is **unverified
because no exact objective or theorem is specified**.

**The obstruction is real:** `W₂` (and even `KL`) do **not** dominate `χ²`. One can have `W₂(ρ̂,ρ)→0`
while `D_χ = ∞` (distilled model places `o(1)` mass where `μ` is exponentially small). So the chain

```
distillation loss  →  L² score error  →  W₂/KL(segment law)  →  ??? →  D_χ
```

is broken at the last arrow **without an extra assumption**. Three registered ways to close it:

> **Audit result:** C1 is false as written, C2 controls Lipschitz pair
> observables but neither spectrum nor MFPT from pair `W₂` alone, and C3 is
> insufficient without anti-trapping control. The corrected conditional
> statements are in the companion solution.

- **(C1) Bounded importance weights (Doeblin), (A3).** Assume `p̂(y∣x)/μ(y) ≤ B`. Then
  `D_χ² ≤ B · KL(ρ̂‖ρ)`-type control becomes available and the clean spectral bound (★) holds with a
  constant in `B`. *Must argue `B` is realistic for MD* (e.g. via a confining-potential tail / the
  base-noise floor of the sampler). **This is the gate-#1 proof obligation.**
- **(C2) Weaker observable class via `W₂`.** Drop χ²; use `W₂`-contraction of Markov semigroups
  (Rudolf–Schweizer, arXiv:1503.04123) to bound **Lipschitz observables and MFPT** — but this route
  **cannot bound eigenvalues/timescales** (§4.1 is lost). Yields a strictly weaker paper.
- **(C3) Local χ² on a sublevel set.** Restrict to a high-probability set `𝒳_R={U≤R}`, prove a
  local `D_χ,R`, and bound the excluded-mass contribution to timescales separately. Middle ground.

**Pre-registered gate (see §8):** if (C1) closes with an MD-realistic `B`, the headline timescale
theorem (★) stands. If only (C2), L37 becomes a Lipschitz-observable/MFPT theorem (no timescale
bound). If none, there is no distillation→kinetics theorem and L37 reverts to a purely empirical paper.

---

## 7. The counterexample that IS the contribution (per-frame insufficiency)

**Claim (to be proven, easy):** there exist `P̂ ≠ P` with **identical one-time marginals**
(`μ̂=μ`) and identical per-frame statistics of every order, yet `λ̂₁ ≠ λ₁` by an `O(1)` amount.

**Construction sketch.** Any `μ`-preserving, `μ`-reversible re-coupling of the joint that keeps both
marginals `μ` but redistributes conditional mass changes `⟨ψ₁,P̂ψ₁⟩_μ` (a two-time functional) while
leaving all one-time functionals fixed. Concretely: on a 2-state metastable system, mix in a small
amount of "extra" inter-state transition probability symmetrically — one-time marginal untouched,
`λ₁` (hence `t₁`) shifted. Formalize on a finite-state `μ`-reversible chain, then lift.

**Consequence (the paper's thesis, now a theorem):** FID / per-frame plausibility / any marginal
distance provides **zero** control over implied timescales. One-lag spectral fidelity is a property
of the pair law, while MFPT and multi-lag fidelity require the path law. **Design prescription:**
distill against, and validate on, **two-time correlations**
(`Ĉ_i(τ)`), not per-frame quality. This both motivates `D_χ` as the right functional and refutes the
naive "few-step preserves FID so kinetics are fine" argument by construction.

> **Audit correction:** marginal insufficiency is proved. Sufficiency of a
> finite set of two-time correlations is not; the design prescription is a
> diagnostic heuristic unless an operator or density hypothesis is established.

---

## 8. Pre-registered proof kill-gates (do these before any compute)

> **Audit result:** GATE-0 passes. GATE-1's C1 premise is disproved, but none of
> its registered PASS/KILL branches exactly fits. GATE-2 is superseded
> conditional on common-space operator control. GATE-3 remains open.

Ordered cheapest-first. Each is pure math / small numerics on the released MDGen artifacts.

- **GATE-0 (counterexample, §7) — hours.** Prove the finite-state per-frame-insufficiency example.
  *PASS:* explicit `P̂` with equal marginals, `Δλ₁ = O(1)`. *If it fails:* the entire "measure the
  joint" thesis is wrong — stop. (Expected: passes trivially; it is a sanity anchor.)
- **GATE-1 (the loss→`D_χ` bridge, §6) — the decisive gate, days.**
  - *PASS-A:* prove (C1) with an MD-realistic bounded-weight constant `B` ⇒ headline timescale
    theorem (★) exists. Proceed to full write-up + empirical plan.
  - *PASS-B:* only (C2) `W₂` route closes ⇒ demote to MFPT/Lipschitz-observable theorem (no
    timescale bound); reassess whether that's worth a paper.
  - *KILL:* neither closes and `κ(V)`/pseudospectrum is uncontrollable ⇒ no theorem; L37 → empirical-only.
- **GATE-2 (non-normality constant, §5.2) — days.** Bound (or argue typical-case control of) `κ(V)`
  / the pseudospectral constant for the distilled operator. *KILL/soften:* if `κ(V)` can be arbitrarily
  large with no structural control, the honest theorem's constant is vacuous — retreat to §5.1 as a
  clearly-labeled idealization + empirical validation.
- **GATE-3 (constant calibration) — small numerics on cached data, no training.** On MDGen's released
  tetrapeptide data, estimate `t_i`, `δ_i`, and a proxy `D_χ` from subsampled/perturbed teacher output;
  check the predicted `(t_i²/τλ_i)` amplification is non-vacuous at realistic error levels. *KILL:* if
  the bound predicts error >100% of `t_i` at any achievable `D_χ`, the theorem is technically true but
  practically empty.

Only after GATE-1 = PASS-A (or an accepted PASS-B) does the empirical distillation run (Step 3) get built.

---

## 9. Prior-art boundary (historical and unverified)

> These are pre-audit literature notes, not conclusions of the mathematical
> proof. Novelty remains open pending a separate literature audit.

- **Static distribution-error bounds for distilled/score generators exist** (`W₂`/`KL` in score error).
  L37 is **not** that — it bounds *dynamical* observables.
- **MSM error theory exists** (Prinz, Sarich–Schütte, Djurdjevac; VAC/VAMP Noé–Nüske) — but expressed
  in terms of *discretization/estimation* error, **not** generative-model distillation error.
- **`W₂` Markov perturbation exists** (Rudolf–Schweizer 1503.04123) — static/marginal, not spectral,
  not distillation; it is exactly the (C2) fallback tool.
- **Historical claim:** no verified result bounds kinetic observables of a distilled trajectory generator. The pieces
  are standard-but-unassembled; the novelty is the assembly across the reversibility/normality/metric
  gaps plus the §7 insufficiency theorem.

Historical reference notes (not verified by this proof audit):
MDGen 2409.17808; Rudolf–Schweizer 1503.04123; ITO 2305.18046; BoPITO 2410.10605.
From-memory (must fetch-verify): Prinz 2011 JCP; Sarich–Noé–Schütte 2010; Djurdjevac–Sarich–Schütte
2012; Davis–Kahan 1970; VAC/VAMP; consistency-model convergence theory (Li/Chi).

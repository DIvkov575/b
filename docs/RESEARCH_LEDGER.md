# Research Ledger — Efficient / Minified Protein Structure Generation

Durable backlog of every candidate direction generated for this research thread.
Phase 1 appends all candidates as `pool` (no pruning). Phase 2 updates status in
place. Killed entries stay (they record dead ends). When an attempt fails, pull the
next-best `pool` entry here before regenerating.

**Status vocabulary:** `pool | picked | killed:<stage> | shipped`
Kill stages: `2a-triviality | 2a-scoop | 2a-thin | 2a-winnability | 2b-mvp | 2c-design`

> **2026-07-12 — L18 PICKED.** Ran a focused adversarial scoop-recheck on L18 alone
> (the one Phase-2a survivor). Verdict: **NOT SCOOPED**; both halves (per-sample
> designability-correlated diagnosis + distillation training-loss) remain open. Nearest
> neighbor = **Protein FID (arXiv:2505.08041)** — distributional, eval-only, validated vs
> CATH/Foldseek/OT (*not* designability), not a loss. L18's defensible slice is exactly
> what Protein FID leaves open. Pre-registered Phase-2b kill-gate locked in
> `docs/L18_PROTOCOL.md` (before any run). Next: run the H1 correlation gate on 1×A100
> once a teacher checkpoint loads.

**Compute budget:** 1× A100 (40/80GB), weeks. Feasible: short from-scratch training of
small models, LoRA/adapter fine-tuning of released teachers, few-step distillation,
consistency training of a small backbone model, multi-seed benchmark sweeps.
NOT feasible: from-scratch Proteina-scale pretraining. See [[project-biostat-compute]].

---

## ⚠️ PHASE 2A VERDICT (2026-07-08) — 1 survivor of 30, and the thread is largely SCOOPED

Ran literature filters (triviality → scoop w/ live web search → thin → winnability), one
verifier per candidate (workflow wf_3c66f2f0-be3, 31 agents). **Kill tally: scoop 16,
triviality 5, thin 5, winnability 3; survivors 1 (L18).**

**The landscape scan's "conspicuously absent" was OUT OF DATE.** A wave of Oct-2025 → mid-2026
papers already did protein few-step distillation. Four load-bearing scoops **verified by direct
arXiv fetch** (real papers, not hallucinated):

- **SiD-Protein** — *Distilled Protein Backbone Generation* (arXiv:2510.03095, Xie/Zhang/Wang/Tansey/Zhou; code github.com/LY-Xie/SiD_Protein). Score-identity-distills the **Proteina** teacher to few steps, **>20× speedup at matched designability/diversity/novelty**. Also empirically **refutes L02**: deterministic/low-noise sampling gives ~0 designability; inference-time noise modulation (γ≈0.45) is essential. → kills **L01, L02, L03, L05**.
- **Generalised Flow Maps** (arXiv:2510.21608, Davis/Albergo/Boffi/Bronstein/Bose). Lifts consistency/shortcut/meanflow to **arbitrary Riemannian manifolds** via self-distillation; names protein backbones. → kills **L01, L04**.
- **Riemannian MeanFlow** (arXiv:2602.07744, Woo/Skreta/Park/Neklyudov/Ahn). One-step manifold flow maps via semigroup identity, **demonstrated on protein backbone generation, ~10× fewer NFEs**. → kills **L04**.
- **Riemannian Consistency Model** (arXiv:2510.00983, Cheng et al., **NeurIPS 2025**). First few-step consistency on curved manifolds incl. **SO(3)**, closed-form, distillation (RCD) + training (RCT) variants. → kills **L07** ("teacher-free CT on frames" is done).

Other scoops (cited, **not all independently re-verified** — treat as strong signals): PI-Mamba
(linear-time long-protein FM, arXiv:2603.26705) kills L30; ProtDBench (end-to-end wall-clock
benchmark, arXiv:2605.04118) kills L23; Scaffold-Lab/PXDesign (cross-oracle) kills L22; Protein
FID/SHAPES (coverage metrics) kills L27; DeCAF (guidance in few-step, arXiv:2606.08375) kills L29;
FastDiSS (self-conditioning breakdown) kills L31.

**Per-candidate verdict:** L01 killed:2a-scoop · L02 killed:2a-scoop · L03 killed:2a-scoop ·
L04 killed:2a-scoop · L05 killed:2a-scoop · L06 killed:2a-winnability (only edge left = true 1–2
step, which hits the SO(3) geometry wall) · L07 killed:2a-scoop · L08 killed:2a-winnability
(BCH fixes O(d²) bias but the published failure is non-Gaussian latents, not commutators) ·
L09 killed:2a-thin · L10 killed:2a-triviality (textbook exp-map Heun port) · L11 killed:2a-triviality
(a KS test swap; can't overturn the headline) · L12 killed:2a-winnability (torsion FK error ceiling;
premise already falsified by ReQFlow) · L13 killed:2a-scoop · L14 killed:2a-triviality (one-line
double-cover metric swap) · L15 killed:2a-triviality (EDM-churn/TCD port) · L16 killed:2a-scoop ·
L17 killed:2a-scoop (FrameFlow already asymmetric per-channel) · **L18 SURVIVES** · L19 killed:2a-thin ·
L20 un-adjudicated (agent errored; ranker flags scoop-exposed, folds into L18 as eval layer) ·
L21 killed:2a-scoop · L22 killed:2a-scoop · L23 killed:2a-scoop · L24 killed:2a-thin ·
L25 killed:2a-scoop · L26 killed:2a-triviality · L27 killed:2a-scoop · L28 killed:2a-thin ·
L29 killed:2a-scoop · L30 killed:2a-scoop · L31 killed:2a-thin.

### Sole survivor — L18 (brought to user for pick)
**Structural-LPIPS for protein backbone distillation.** Few-step students suffer mean-structure
collapse because distillation losses use L2/Kabsch-RMSD/geodesic-L2; a designability-correlated
frozen-encoder feature distance (the protein LPIPS analog) both *demonstrates* and *fixes* it.
- **Why it survived:** changes thinking (LPIPS is load-bearing for image CMs; no protein analog exists);
  cheapest falsifier in the pool (zero-training correlation study); enabling infra that plugs into the
  now-open baselines (SiD-Protein, GFM) — the scoop wave is a *tailwind* (removes the from-scratch build).
- **MVP kill-gate (pre-registered, Phase 2b):** on one open teacher, generate 200–400 backbones across
  NFE∈{2,4,8,50}; compute designability (8 ProteinMPNN seqs @T=0.1 → ESMFold, scRMSD<2Å) and two
  distances to the many-step output (Kabsch-RMSD, frozen-encoder feature distance). **PASS** iff encoder
  distance reaches Spearman |ρ|>0.3 with designability AND beats RMSD by ≥0.15. **KILL** if RMSD already
  >0.6 (nothing to add) or encoder distance <0.3. ~1 day, no training.
- **Risk:** "first protein CD" novelty is dead — rests entirely on the distance + collapse story (a
  reviewer may see a loss-function swap); if the binding constraint is manifold geometry (per 2510.24732)
  not the loss, a perceptual distance gives valid-but-not-more-designable samples — which Phase A tests cheaply.
- **also_strong:** L20 (NFE×designability×diversity×novelty Pareto) — folded in as L18's measurement layer,
  not a standalone (scoop-exposed vs Scaffold-Lab / SiD-Protein diversity reporting).

**Assessment:** the efficient-protein-structure thread is a **red ocean as of mid-2026** — the
2510.03095 / 2510.21608 / 2602.07744 / 2510.00983 cluster occupies exactly the opening the scan
identified. L18 is the one narrow, defensible, cheap-to-falsify slot, and even it is method-adjacent
rather than a headline method. Honest SOP outcome: **one marginal survivor; strongly consider
returning to the pool / a different thread.** See "Next" below.

Generated 2026-07-08 via fan-out (8 shortfall lenses → 6 idea-generators, 14 agents).
Source scan: adversarially-verified deep-research (arXiv:2310.02391, 2405.20313,
2502.14637 ReQFlow, 2503.00710 Proteina, 2405.15489 Genie2, 2510.24732 "straight but not so fast").

---

## The shortfall map (Phase 1b) — where the frontier is brittle

The whole few-step story rests on **one technique (rectified flow) at ~50 steps**, benchmarked
inconsistently, and a **published negative result** (2510.24732) claims image-domain straightening
fails on proteins because 96% of backward SO(3) latents fail a **Gaussian** KS test. Four exploitable
seams emerged:

1. **The obstacle may be mis-diagnosed.** The correct base measure on compact SO(3) is Haar/IGSO3,
   whose angle marginal is (1−cos θ)/π — *not* Gaussian. Failing a Gaussian test is expected and may
   say nothing about well-posedness. (shortfalls 9, 14, 32)
2. **A whole class of methods structurally dodges the obstacle.** Consistency training/distillation,
   DMD, flow-map matching, and shortcut models never backward-integrate, so the non-Gaussian-latent
   failure *cannot arise*. All are verified-absent for proteins. (13, 15, 19, 22, 25, 31, 42, 44)
3. **The benchmarks are fragile and gameable.** Designability alone rewards mode collapse; speedups are
   single-length (N=300), single-oracle (ESMFold), single-seed, step-count (not wall-clock) framings,
   and never hold the teacher fixed. (1–7, 11, 12, 36, 39)
4. **Manifold-native math is unwritten.** Geodesic (log-map) losses, BCH-corrected group composition,
   per-channel (rot/trans) step budgets, higher-order manifold integrators, and a designability-correlated
   distance ("structural LPIPS") are all missing and cheaply testable. (18, 26, 28, 33, 34, 35, 46)

Full 47-shortfall dump archived in the workflow journal (run wf_aa399d0c-7ba).

---

## Candidate directions (Phase 1c) — deduped peak set

Near-duplicates from independent generators are merged (sharpest phrasing kept); merged
source IDs noted as `(Dxx)`. All entries `pool` until Phase 2.

### Cluster A — Manifold-native distillation of released teachers

- **[L01] Geodesic Consistency Distillation of a released SE(3) flow to 1–4 steps** — consistency loss in log-map geodesic distance d_SO3=‖log(fₐᵀf_b)‖_F (not Frobenius/quaternion-MSE); teacher PF-ODE integrated forward via exp-map, so the reflow backward-latent obstacle structurally cannot arise. · status: killed:2a-scoop · prior-art/delta: CM absent for backbones; closest = Consistency Policy (Prasad 2024, robotics SO(3)) + Euclidean CM (Song 2023); delta = geodesic-endpoint identity f(R_t,t)=R₀ on frames, never written/trained. · kill: median geodesic drift of teacher's own R₀-estimate over t∈[0,0.5] > 0.3 rad ⇒ target ill-posed. (D1/D19)

- **[L02] Consistency-distill RFdiffusion via the deterministic-sampler loophole** — refutes "RFdiffusion too stochastic to distill": its designability-optimal setting is noise_scale=0, a deterministic PF-ODE a CM can match. LoRA the IPA net as f(R_t,t)→R₀. · status: killed:2a-scoop · delta: teacher→student distill of RFdiffusion verified-absent; framed on the determinism loophole. · kill: run noise_scale=0 twice (confirm determinism), median frame-pred drift over t∈[0,0.5] > 30° ⇒ dead. (D9)

- **[L03] Riemannian Distribution-Matching Distillation (DMD) on SE(3)** — distill FoldFlow-2/ReQFlow to ≤4 steps by matching output *distribution* via a Lie-algebra score-difference (fake−real, IGSO3 score on the rotation channel); never inverts, and directly penalizes diversity collapse. · status: killed:2a-scoop · delta: DMD/DMD2 (Yin 2023/24) Euclidean/image-only; distributional distill of any structure model absent; forward-only ⇒ dodges obstacle. · kill: fit fake-score on frozen teacher, linear probe on score-diff gradient must separate on-/off-manifold (AUC>0.7) & be finite ⇒ else dead. (D4/D11/D23)

- **[L04] Riemannian Flow-Map Matching on SE(3)** — the landscape's explicitly-named "missing protein-native manifold distillation objective": learn two-time transport s_{t,s} via geodesic semigroup self-consistency s_{t,u}∘s_{u,s}=s_{t,s}; subsumes consistency + rectification in one forward-only objective, arbitrary step count from one run. · status: killed:2a-scoop · delta: Flow Map Matching (Boffi & Vanden-Eijnden 2024) Euclidean; SE(3) extension never written. · kill: teacher two-hop vs one-hop endpoint geodesic mismatch > 0.4 rad ⇒ semigroup too broken to compress. (D6/D15/D24/D55)

- **[L05] SO(3)-correct progressive distillation of Genie2 → ≤8 steps** — the robust step-halving baseline everyone skipped; produces the field's first *teacher-matched, sampler-only* NFE-vs-designability Pareto (vs cross-model 37×/62× apples-to-oranges). · status: killed:2a-scoop · delta: Progressive Distillation (Salimans-Ho 2022) canonical; never applied to Genie2; uses geodesic/frame loss not raw-coord MSE. · kill: along DDIM-ODE trajectory, 1-step-vs-2-step endpoint TM < 0.6 at coarsest level ⇒ PD floor > 8 steps (report actual floor). (D10)

- **[L06] Latent adversarial distillation with a frozen SE(3)-invariant discriminator** — ADD/LADD to 1–2 steps using a frozen structure encoder's invariant latents as the discriminator space, because raw-coordinate GANs are unstable & non-invariant. · status: killed:2a-winnability · delta: ADD/LADD/SDXL-Turbo (images) unported to structure precisely because coord-space discriminators fail; invariant-latent discriminator is SE(3)-invariant by construction. · kill: linear probe on frozen-encoder latents can't separate many-step vs few-step backbones (AUC<0.65–0.75) ⇒ no adversarial signal. (D12/D26)

### Cluster B — Teacher-free few-step generation

- **[L07] Teacher-free Consistency TRAINING on SO(3) via the closed-form IGSO3 score** — from-scratch 1–2 step generator, NO teacher, NO ODE sim; well-defined because the IGSO3 perturbation-kernel score (angle-derivative of the truncated character series) is analytic ⇒ unbiased single-sample CT gradient; enforces forward self-consistency ⇒ dodges the obstacle by construction. · status: killed:2a-scoop · delta: CT (Song 2023) Euclidean; never instantiated on frames. · kill: IGSO3 score autograd gradient NaN/variance-blowup for σ<0.05 that standard series-truncation can't tame ⇒ biased gradient, dead. (D3/D21/D41/D50)

- **[L08] Riemannian shortcut model on SE(3) with BCH correction** — single-run, teacher-free, step-size-conditioned one-step sampler; the true 2d-step target log(exp(d·ξ₁)exp(d·ξ₂)) = d(ξ₁+ξ₂)+(d²/2)[ξ₁,ξ₂]+… carries a commutator the Euclidean shortcut identity truncates away. · status: killed:2a-winnability · delta: Shortcut Models (Frans 2024) Euclidean; SE(3) instantiation absent; Lie-bracket correction is the mechanism. · kill: median ‖[ξ₁,ξ₂]‖/‖ξ₁+ξ₂‖ on released trajectories < 2–5% ⇒ correction negligible, collapses to known Euclidean shortcut, dead. (D2/D20/D45/D56)

- **[L09] Curvature-regularized OT-coupling training-time straightening** — achieve reflow's goal (straight paths → ≤5 steps) *without* backward-integrating: minibatch-OT coupling + explicit geodesic-acceleration (path-curvature) penalty on the SE(3) field during training. · status: killed:2a-thin · delta: only straightening tried on proteins (reflow) is dead because it inverts; FoldFlow-2 uses OT but no curvature penalty; forward-only lever. · kill: OT-coupled teacher trajectories already near-straight (normalized curvature <0.1) ⇒ no headroom, dead. (D48)

- **[L10] Manifold-native higher-order / exponential integrator for released checkpoints** — exp-map Heun / RKMK / DPM-Solver-on-SO(3); beats Euler-on-manifold below ~15 steps because constant positive SO(3) curvature makes 1st-order truncation dominant. Pure inference, zero retraining. · status: killed:2a-triviality · delta: all released SE(3) samplers use 1st-order Euler-on-manifold; RKMK (numerics) + DPM-Solver (Euclidean image) exist, never for biomolecular flows. · kill: exp-map-Heun vs Euler at NFE=8/10 fails to lift designability ≥5 pts ⇒ dead. Cheapest gate in the set (<40 lines). (D39/D51)

### Cluster C — Attack the obstacle / representation directly

- **[L11] Haar-KS reframe: the "SO(3) reflow wall" is a mis-specified test** — 2510.24732 KS-tested backward SO(3) latents against a *Gaussian* null (96% fail) and declared straightening impossible; the correct null on compact SO(3) is Haar, angle density (1−cos θ)/π. Retesting could overturn a headline negative result. · status: killed:2a-triviality · delta: nobody has run the correct diagnostic. · kill (IS the core experiment): if >50% of backward latents also fail KS against (1−cos θ)/π at p<0.005, the wall stands and this dies; if failure drops toward the 5% floor, the wall is an artifact. (~30 lines) (D36/D40/D49)

- **[L12] Move few-step off SO(3) onto the torsion torus Tⁿ (φ,ψ,ω)** — wrapped-normal base is analytically near-Gaussian ⇒ consistency/straightening transfers where SO(3) fails; trade rotation curvature for a friendly prior. · status: killed:2a-winnability · delta: accelerated samplers all live in SE(3)/quaternion space (where the obstacle is); Torsional Diffusion (Jing 2022)/FoldingDiff exist but no *few-step* backbone torsion flow. · kill: (1) backward-integrated torus latents must pass wrapped-normal KS materially more than SO(3) passes Haar KS; (2) NeRF/pnerf Cartesian reconstruction of 400-res chains must stay <~4Å global RMSD under small per-angle noise ⇒ else unusable. (D28/D44/D52)

- **[L13] Cartesian-Cα few-step breaks the 3.8Å bond constraint** — refutes the "go Cartesian (Proteina) and reflow works" reading: Proteina shows *training* straightness/scale, not few-step sampling; straight couplings on the thin Cα tube cross impossible bond geometries only many small steps repair. · status: killed:2a-scoop · delta: nobody has measured Cα-Cα bond violation vs NFE — the representation-specific low-NFE pathology the SO(3) analysis never touches. · kill: bond-violation fraction stays <5% even at NFE=2 ⇒ Cartesian route is clean, dead; if >20% at NFE≤5 decreasing to <5% at ≥50 ⇒ hard geometric wall (motivates bond-constrained objective). (D57)

- **[L14] Antipodal-aware quaternion consistency/shortcut loss for ReQFlow** — S³ double-covers SO(3), so naive quaternion-L2 consistency has a discontinuous fixed point at antipodal straddles; the double-cover distance min(‖qₐ−q_b‖,‖qₐ+q_b‖) or 2·arccos|⟨qₐ,q_b⟩| is required for a well-posed fixed point on the current SOTA frontier. · status: killed:2a-triviality · delta: ReQFlow uses unit quaternions; ill-posedness never stated. · kill: antipodal straddle fraction on released ReQFlow trajectories <1% ⇒ pathology measure-zero, not worth a paper, dead. (D8/D47)

- **[L15] TCD-style stochastic γ-sampling with IGSO3 re-noising** — inference-only self-correction on a frozen teacher: inject IGSO3-kernel re-noise between few-step Euler updates to correct accumulated curvature error — a *different lever* than reflow/straightening, zero training. · status: killed:2a-triviality · delta: TCD (images) unported; field treats rectification as the only speed knob. · kill: at fixed NFE=6–10, IGSO3 re-noise fails to beat deterministic Euler designability by > seed std ⇒ dead. Cheapest possible (hours). (D17/D22)

### Cluster D — Step-placement & scheduling (inference-only)

- **[L16] Manifold-aware step-schedule optimization ("Align Your Steps" on SO(3))** — measure WHERE few-step SE(3) error concentrates (hypothesis: near the IGSO3 prior t~1, max curvature) and reallocate steps; large inference-only speedup nobody has claimed. Includes phased-consistency variant with curvature-adaptive phase boundaries. · status: killed:2a-scoop · delta: all protein samplers use uniform Euclidean-ported discretization; Align Your Steps (NVIDIA 2024) is Euclidean/image. · kill: per-segment error profile flat (<2× variation) ⇒ non-uniform schedule yields <3 pts at fixed NFE, dead. Stacks on any distillation direction. (D18/D25/D37/D43)

- **[L17] Asymmetric per-channel NFE budget on R³×SO(3)** — rotations (curved) need strictly more steps than translations (flat); decoupled (N_rot, N_trans) at fixed FLOPs beats uniform. Inference-only diagnostic on released checkpoints. · status: killed:2a-scoop · delta: all SE(3) samplers integrate both channels with the same step count; per-channel sensitivity never measured. · kill: designability collapse symmetric between (N_rot=20,N_trans=2) and the swap (within ~3–5 pts) ⇒ channels equally step-hungry, dead; if one tolerates >5× fewer steps, real. (D5/D14/D42/D53)

### Cluster E — Enabling results (metrics & losses)

- **[L18] Designability-correlated distance function — the missing "structural LPIPS"** — mean-structure collapse in few-step students is caused by using L2/Kabsch-RMSD inside the consistency/shortcut loss; a designability-correlated (TM/lDDT-surrogate or learned invariant-latent) distance fixes it. Both distillation clusters silently depend on this. · status: picked (Phase 2a survivor) · delta: image CMs depend critically on LPIPS; no backbone analog; default RMSD/geodesic weighting unvalidated. · kill: no candidate distance achieves Spearman |ρ|>0.3 with designability-delta (or RMSD already >0.6, so nothing to add) ⇒ dead. (D7/D27/D46)

- **[L19] SE(3)-invariant (Procrustes-aligned) distillation target with an irreducible-variance bound** — for Cartesian models, a distillation target that is provably SE(3)-invariant, plus a lower bound on achievable student variance. · status: killed:2a-thin · delta: shortfall 17; theory-flavored, pairs with L03/L13. · kill: (to specify at 2b) — needs the invariance/variance statement made precise first. (shortfall 17)

### Cluster F — Benchmark / evaluation contributions (mostly inference-only)

- **[L20] Joint NFE-vs-(designability × diversity × novelty) Pareto surface** — first to show few-step/distilled samplers buy designability by silently collapsing diversity (pairwise-TM) and novelty (max-TM-to-PDB). Turns a benchmark gap into a method when paired with a diversity-repulsion distillation term. · status: pool (un-adjudicated; eval layer for L18) · delta: every efficiency claim is a scalar designability number at one NFE. · kill: as steps drop 50→5, diversity & novelty do NOT worsen (>0.02) while designability holds (within 0.03) ⇒ surface flat, thesis false. (D13/D29/D54)

- **[L21] Designability is gameable by collapse** — a trivial memorization baseline emitting canonical helix bundles beats SOTA few-step samplers on designability with near-zero diversity; proves designability alone cannot be a valid efficiency benchmark. · status: killed:2a-scoop · delta: nobody has shown the ranking metric is gameable by a degenerate generator. · kill: degenerate helix-bundle set scores <0.70 designability (below ReQFlow 0.912) ⇒ claim false. No training at all. (D30)

- **[L22] Cross-oracle designability: the 99%→35% swing is partly an ESMFold artifact** — the universal protocol uses one oracle (ESMFold), coupling evaluator to generator; cross-oracle (ESMFold/AF2/AF3) disagreement on few-step samples decouples the metric from its oracle. · status: killed:2a-scoop · delta: no efficiency paper reports cross-oracle designability. · kill: ESMFold & AF2 labels agree with Cohen's κ>0.9 ⇒ no artifact, dead. (D31)

- **[L23] Yield-weighted wall-clock inverts the NFE ranking** — deployment-honest cost = total wall-clock (incl. ProteinMPNN+ESMFold filtering) to obtain K designable backbones on fixed hardware; downstream filtering can dominate, inverting 37×/62× claims. · status: killed:2a-scoop · delta: all speedups are generator NFE ratios; no yield-weighted cost Pareto. · kill: lowest-NFE setting also gives lowest total wall-clock to K=100 across all lengths ⇒ ranking stable, dead. (D32/D54)

- **[L24] Few-step designability has an unmeasured length ceiling** — min-NFE for target designability grows with L, shrinking the headline 37× precisely on the long, multi-domain proteins people deploy. · status: killed:2a-thin · delta: 37×/62× are explicitly single-length (N~300); no length-resolved surface. · kill: designability at fixed 8 steps flat across L∈{100…800} (Δ<0.05) ⇒ length-invariant, dead. (D33)

- **[L25] Protocol-noise re-audit: published few-step rankings flip within CIs** — single-seed point estimates hide bootstrap CIs; the surrounding protocol (1/8/100 MPNN seqs, temperature, sample N, filters) is unstandardized. First variance-quantified re-audit. · status: killed:2a-scoop · delta: no efficiency paper reports error bars. · kill: 95% CI width <0.03 AND non-overlapping with nearest competitor ⇒ rankings real, dead. (D34)

- **[L26] Compute-normalized "step" axis (FLOPs / wall-clock)** — a step means different things across a quaternion SO(3) flow, a Cartesian rectified flow, and a 1000-step DDPM; a FLOP/wall-clock-normalized axis at fixed hardware & model size re-ranks the leaderboard. · status: killed:2a-triviality · delta: field compares "steps" as if hardware-neutral; never FLOP-counted. · kill: normalization preserves the existing ranking ⇒ marginal, dead. (shortfall 3; from D3-benchmark)

- **[L27] Gameability-resistant distributional-coverage metric (precision/recall)** — port improved-precision/recall (Kynkäänniemi et al.) to a frozen SE(3)-invariant structure encoder to expose the collapse designability structurally can't see. · status: killed:2a-scoop · delta: no distributional-coverage metric for backbones. · kill: coverage-recall rank-correlates >0.9 with plain designability ⇒ no orthogonal signal, dead. (D35)

- **[L28] Motif-scaffolding success-vs-NFE benchmark** — every few-step result is *unconditional* monomer generation; conditional generation (motifs/active-sites/symmetry) likely degrades faster under step reduction. First conditional-vs-NFE curve. · status: killed:2a-thin · delta: no motif-success-vs-NFE curve exists. · kill: motif success at low NFE tracks unconditional designability within Δ<0.05 ⇒ no faster degradation, dead. (D38)

### Cluster G — Deployment / method extensions

- **[L29] Guidance- & motif-preserving few-step distillation** — first accelerated sampler retaining RFdiffusion's auxiliary potentials + motif-scaffolding at single-digit steps; guidance (small per-step nudges over hundreds of steps) is fundamentally at odds with few-step, so re-inject guidance into the few remaining steps. · status: killed:2a-scoop · delta: all few-step results unconditional; guidance-vs-NFE conflict unaddressed. · kill: motif success at 20 steps already ≥90% of 200-step (no method needed) OR per-step guidance rescaling recovers <½ the gap when it collapses (conflict fundamental) ⇒ LoRA route dies. (D16)

- **[L30] Long-protein memory wall (space axis, not time axis)** — every method attacks steps (time); the deployment wall on long proteins is O(N²) attention memory per step, untouched by NFE reduction. · status: killed:2a-scoop · delta: shortfall 40; distinct from the whole few-step literature. · kill: (to specify) — profile peak memory vs L on released checkpoints; if memory isn't the binding constraint before wall-clock at deployable L, dead. (shortfall 40)

- **[L31] Self-conditioning breakdown under aggressive step reduction** — self-conditioning (feeding the previous x₀ estimate) silently breaks at low NFE because the estimate is poor early; characterize and fix. · status: killed:2a-thin · delta: shortfall 41; unexamined interaction. · kill: (to specify) — ablate self-conditioning on/off across NFE on a released checkpoint; if designability gap is NFE-independent, no interaction, dead. (shortfall 41)

---

## Notes for Phase 2

- **Highest-leverage cheap gates first** (effort ∝ 1/confidence): **L11** (Haar-KS reframe, ~30 lines, could overturn a headline result), **L10** (exp-map Heun, <40 lines, free speedup), **L15** (IGSO3 re-noise, hours), **L20/L21** (benchmark, inference-only) — all falsifiable in hours-to-days with no training.
- **Scoop risk is HIGH and rising** — consistency/distillation for proteins is an obvious next paper for several groups. Phase 2a scoop-check (deep-research-efficient, thorough) is mandatory before committing to L01/L03/L04/L05/L07.
- **Enabling dependencies:** L18 (structural-LPIPS distance) silently underpins L01/L07/L08 — consider gating it first.
- **Cross-cutting theme:** several methods (L01, L03, L04, L07, L08) share one thesis — *forward-only / non-inverting objectives dodge the 2510.24732 obstacle*. If L11 shows the obstacle is a test artifact, that weakens the novelty framing (but not the methods).
- **Metric discipline** (our own post-mortem lesson): pre-register the designability protocol EXACTLY (num MPNN seqs, temperature, oracle, seeds, CI) before any run — L25 exists because the field didn't.

---

## ADJACENT-THREAD SCAN (2026-07-12) — "cheapification of diffusion" beyond backbone gen

Zoomed out from de-novo backbone generation (this whole ledger's red-ocean thread) to the
broader question: *is there open work applying efficiency / few-step techniques to diffusion
in **folding** and **molecular simulation**?* Two verified scout maps (all arXiv IDs fetched):

### Thread F — Efficient diffusion sampling for FOLDING models (AF3 / Boltz / Chai) — 🟢 WIDE OPEN
Small (<15 papers), all mid-2025→now. Much less crowded than backbone-gen acceleration.
- **Direct prior art:** DeCAF (arXiv:2606.08375, flow-map cofolding distill, 5× fewer NFE) ·
  DCFold (arXiv:2605.17899, ICLR'26 oral, dual-consistency 1-step 15×) · Protenix-Mini /
  Mini+ (arXiv:2507.11839 / 2510.12842, 2-step ODE sampler, >90% compute cut, ~3% LDDT drop).
- **Architecture-side (complementary, not sampler):** Pairmixer (2510.18870), MegaFold
  (2506.20686, training), SeedFold (2512.24354), AF_Cache (2606.04566, MSA caching).
- **VERIFIED-OPEN GAPS:**
  1. **No ODE-solver swap** (DPM-Solver++/UniPC/LCM-solver) benchmarked on AF3-family
     diffusion — everyone *retrains/distills* instead of trying the training-free solver swap.
     ← **cheapest possible MVP: inference-only, hours on 1×A100.**
  2. No standard consistency-distillation (CT/CD) of **Boltz-2 or Chai-1** (DeCAF hit only
     Boltz-1x + Pearl; Protenix-Mini modifies arch, doesn't distill). Boltz-2 (affinity head) untouched.
  3. No step-caching / feature-reuse (DeepCache / FRDiff) across denoising steps.
  4. No structure-quality-adaptive step schedules (fewer steps rigid regions, more on loops/pocket).
  5. No unified sampler benchmark (every paper picks own NFE budget + eval set).

### Thread M — Efficient diffusion for MOLECULAR SIMULATION — 🟡 CROWDED, unevenly
Busy field, established groups (Noé/Klein lineage). ~20 verified papers.
- **Boltzmann generators:** EWFM (2509.03726), BoltzNCE (2507.00846), PITA (2506.16471,
  NeurIPS'25 spotlight), RegFlow (2506.01158), SBG (2502.18462), TBG (2406.14426).
- **Conformers (few-step):** FlashMol (2605.07020, DMD 4-step 250×), EQUIMF (2604.08189),
  EnFlow (2512.22597, 1-2 step), SO(3)-avg FM+Reflow (2507.09785), ET-Flow (2410.22388),
  GeoLDM-distill (2404.13491, 7.5×), EC-Conf (2308.00237, 1-step).
- **MD accel:** Hamiltonian Flow Maps (2601.22123), LiFlow (2410.01464), MDGen (2409.17808),
  CG-MD-NF (2406.01524), Timewarp (2302.01170).
- **Materials:** ADiT (2503.03965), FlowLLM (2410.23405), CrystalFlow (2412.11693), FlowMM (2406.04713).
- **Ligand/SBDD (thin):** FlowSBDD (2412.01174), Equiv-FM-HPT (2312.07168).
- **GAPS:** no consistency models on crystals; no CFG-in-few-steps for molecular guidance;
  no bespoke equivariant ODE solvers; no DPM-Solver on molecular tasks; ligand-diffusion
  distillation underdeveloped; no consistency-model Boltzmann generators; no survey.

**Assessment:** Thread F (folding-sampler acceleration) looked like the strongest slot —
under-attacked, real deployment pain. But the cheapest MVP (solver-swap) **DIED under
scoop+feasibility audit (2026-07-12)**, verified from source code:
- **Chai-1 already ships Heun 2nd-order by default** (`chai_lab/chai1.py`, `second_order=True`).
- **Multistep solvers (DPM-Solver++/UniPC) structurally can't run:** both Boltz-2 and Chai-1
  apply a fresh random SE(3) rotation+translation **every step** (`compute_random_augmentation`
  / `center_random_augmentation`); multistep methods reuse prior-step derivatives in a fixed
  frame, so cached history is in the wrong basis. This is why Chai chose single-step Heun.
- **Step-caching (DeepCache/FRDiff) barely applies:** the expensive trunk runs ONCE before the
  diffusion loop; only the small atom-transformer runs per-step — the big part is already amortized.
- **Circumstantial:** DeCAF/DCFold/Protenix-Mini all needed *training* to reach few-step; if a
  free solver swap worked they'd have done that. Score field too stiff in the low-NFE regime.
- **Sole survivor (thin):** port single-step Heun to Boltz-2 (Euler-only @ 200 steps), quantify
  NFE/accuracy. Unclaimed + feasible but upside modest (~200→60-100 NFE), workshop-tier.

**Meta-observation:** three consecutive efficient-diffusion probes (backbone-gen Phase 2a →
1 marginal survivor; L18 reframe → thin; folding solver-swap → dead) all came up dry. The
efficient-diffusion space is a genuine mid-2026 red ocean. Decision (2026-07-12): **re-audit
the 30 Phase-2a kills** — several scoop-kills rested on un-re-verified citations and may have
buried a candidate with more room than L18.

---

## RE-AUDIT OF PHASE-2A KILLS (2026-07-12) — 5 revivals, all citations re-fetched

Two adversarial re-auditors (one on un-verified scoop citations, one on judgment "thin/
winnability" kills). Every killer citation fetched from real arXiv. **Result: 5 REVIVE,
2 thin-revive, rest kill-stands.** Two kills rested on WRONG papers:

**REVIVED (kill was invalid or premise flawed):**
- **[L31] Self-conditioning breakdown at low NFE (protein)** — killed by "FastDiSS", which is
  **arXiv:2604.05551, a diffusion *LANGUAGE MODEL* (text seq2seq) paper** — wrong domain
  entirely. Invalid scoop. Protein-structure self-conditioning vs NFE is uncovered. → `pool`
- **[L27] Distributional precision/recall coverage on a frozen encoder** — killed by Protein
  FID (arXiv:2505.08041), which provides **only a single Fréchet scalar, no precision/recall**.
  A P/R pair sees mode-collapse the FID scalar can't. Fully open, reuses their encoder. → `pool`
- **[L24] Length × NFE designability surface** — killed on assumed "flat across L"; **Scaffold-Lab
  affirmatively documents designability degrades with length**, and all headline 20-37× speedups
  are single-length (N~300). Premise now evidence-backed the other way. Inference-only. → `pool`
- **[L22] Cross-oracle designability disagreement on FEW-STEP samples** — killers (Scaffold-Lab/
  PXDesign/ProtDBench) do general multi-oracle eval, none *conditioned on distilled samples*.
  Whether disagreement is amplified on few-step outputs is open. Inference-only. → `pool`
- **[L06] Adversarial distillation (ADD/LADD) to 2-4 step, frozen SE(3)-invariant discriminator**
  — killed by importing the 2510.24732 "SO(3) wall", which is a *backward-integration + Gaussian-
  null* artifact; ADD is forward-only so it can't arise. No adversarial distillation exists for
  proteins. Complements SiD's diversity weakness. Needs LoRA-scale training. → `pool`

**THIN-REVIVE / merge:** L09 (forward-only curvature straightening — headroom confirmed but
incremental, needs training); L19 (SE(3)-invariant variance bound — fold into L18 as its theory
layer); L23, L29 (largely occupied by ProtDBench / DeCAF).

**KILL-STANDS (now on correct grounds):** L30 (PI-Mamba arXiv:2603.26705 IS real linear-time
backbone gen to 2000+ res — memory wall genuinely dissolved); L08 (subsumed by RMF/GFM intrinsic
flow maps — no commutator to correct); L12 (torsion→Cartesian FK error ceiling is real, though the
ReQFlow rationale was bogus — ReQFlow is quaternion/SO(3), unrelated to the torus).

### THE PATTERN (key insight)
L18, L27, L22, L24, L31 are **all inference-only diagnostics of how few-step / distilled protein
generators silently degrade** — perceptual-distance collapse (L18), coverage collapse P/R (L27),
oracle disagreement (L22), length ceiling (L24), self-conditioning breakdown (L31). Individually
each is a workshop-tier note. **Combined, they are a single coherent paper:** *"What few-step
protein generators silently break — a diagnostic audit"* — no training, all on released
checkpoints, days on 1×A100, and it turns the red-ocean speedup wave into the thing being audited.
This is the strongest configuration to emerge from the whole thread. Candidate name: **L32 (audit
bundle)**. Next: scope L32 as the pick, or run one more scoop-check on the *bundle* framing.

---

## COMPUTE RE-OPENED → METHOD-DIRECTION FEASIBILITY CHECK (2026-07-13)

User confirmed **more than 1×A100 gettable** (ceiling TBD; scope at "few GPUs, weeks"). This
retires the inference-only constraint that forced audit-shaped directions, re-opening
training-based methods. Ran scoop+feasibility on the two strongest openings from the scout maps:

**[L33] Consistency-distill Boltz-2 / Chai-1 folding sampler → DEAD (scoop + infeasible).**
- Scooped by method: **DeCAF (arXiv:2606.08375)** owns endpoint-loss flow-map distillation of
  cofolding (distilled Boltz-1/1x + Pearl); **DCFold (arXiv:2605.17899)** owns dual-consistency
  1-step. Boltz-2/Chai-1 as literal targets are unclaimed, but only as a model-swap delta.
- **Infeasible at budget:** DeCAF disclosed **64 H200 GPUs, 100 epochs** — a cluster ~10-16×
  our tier. And **Boltz-2 + Chai-1 are inference-only (no training code released**; Boltz-2
  training code "coming soon"). Distillation needs a training loop that doesn't exist yet.
- Per-step SE(3) augmentation forces DeCAF's exact endpoint-loss formulation (velocity loss
  fails) → re-deriving their method. **Affinity-head angle = category error** (affinity is a
  separate PairFormer regressor, gradients detached from trunk, not in the diffusion loop).
- Sole surviving sliver (blocked on Boltz-2 training code): few-step effect on *affinity
  ranking* (a metric DeCAF/DCFold didn't report). Not actionable now.

**[L34] Few-step / DMD distillation of pocket-conditional SBDD → FEASIBLE but incremental.**
- Partially scooped: **TurboHopp (arXiv:2410.20660)** already did consistency-model few-step
  pocket-conditional SBDD, beats TargetDiff/DecompDiff on Vina — but scaffold-hopping-specific,
  consistency-*training* (not distillation), and only 50-150 steps (never 1-4 NFE). FlashMol
  (2605.07020) is DMD but **unconditional**. The cell {DMD/mean-flow} × {de-novo pocket-cond} ×
  {1-4 NFE} × {distilled from public teacher} is genuinely **empty**.
- **Feasible at budget:** DiffSBDD + TargetDiff have **public checkpoints AND training code**;
  CrossDocked2020 eval (100 pockets, Vina/PoseBusters) is cheap. This is the only training-based
  direction that is actually runnable at "few GPUs, weeks."
- **Two real risks:** (1) *Incrementalism* — reviewer sees "method-substitution delta on a
  solved problem" (TurboHopp + cheap flow-matching SBDD like MolCRAFT already exist). (2) *"Why
  it matters" is weak* — in a real SBDD pipeline docking/MD rescoring dominates cost, not the
  generator's denoising steps; the win is latency-per-sample (tight generate→dock RL loops), not
  pipeline throughput. (3) DMD is mode-seeking → validity/diversity collapse risk on conditioned
  targets (TurboHopp's own 25-step variant was "unstable").

**ANSWER to "are there any efficient-diffusion improvements left in drug discovery / molecule
modeling?":** Not exhausted, but the openings are narrow and each carries a specific wound —
folding-distillation is scooped+cluster-gated; SBDD-distillation is feasible but incremental with
a weak impact story. The genuinely OPEN method cells the scouts named and we have NOT yet checked:
consistency-models on **crystals/materials** (FlowMM lineage is all flow-matching), consistency-
model **Boltzmann generators**, and trajectory-level **MDGen** distillation. These are outside
"drug discovery" proper but inside molecular modeling.

---

## THREE OPEN CELLS — FEASIBILITY CHECK (2026-07-13) — 2 of 3 SURVIVE

First time in the whole thread that adversarial scoop+feasibility returned live candidates.

**[L35] Few-step consistency/mean-flow distillation of CRYSTAL generation → NOT SCOOPED + FEASIBLE ✅**
- **Scoop:** zero hits for consistency/mean-flow/DMD/shortcut applied to crystals across 4 search
  surfaces. Landscape (FlowMM 2406.04713, CrystalFlow 2412.11693, MatterGen 2312.03687, DiffCSP
  2309.04475, ADiT 2503.03965) is diffusion/flow-matching only, none few-step-distilled. The two
  Riemannian consistency papers (GFM 2510.21608, RMF 2602.07744) name proteins/DNA/RNA/geospatial —
  **crystals explicitly absent.**
- **Feasible:** DiffCSP + MatterGen have public training code AND checkpoints; MP-20 (~45k structs)
  trains in ~2-3 days on 2×A100 (real user report); eval uses **ML potentials (MatterSim/MACE/CHGNet),
  no DFT loop needed** for the research claim.
- **Narrowest safe version:** few-step (1-8 NFE) distillation of a DiffCSP/CrystalFlow teacher for the
  **CSP task (lattice + fractional coords, composition-conditioned so atom types fixed)** — sidesteps
  the discrete-atom-type channel (the analog of the SO(3) wall). Clean benchmark (match rate + RMSD).
- **Biggest risk:** significance, not scoop — flow-matching crystal models may already sample decently
  at ~20-50 NFE, so headroom for 1-8 NFE after distillation complexity may be thin. Scoop is safe.

**[L36] Consistency-model Boltzmann generators → FULLY SCOOPED ❌**
- **arXiv:2409.07323** (Sep 2024) already does CM + importance sampling → unbiased Boltzmann samples
  in 6-25 NFE. **arXiv:2606.29110 SCALLOP** (Jun 2026) does few-step flow-map BG with distilled
  likelihood. The exact "CMs lack likelihood" obstacle is already solved (post-CM Gaussian noise step
  → tractable path density for reweighting — the salvage IS the scooping paper's mechanism). Priority lost.

**[L37] Few-step distillation of trajectory-level MD (MDGen) → NOT SCOOPED + FEASIBLE ✅**
- **Scoop:** no few-step/distilled trajectory-level MD generator found. MDGen (2409.17808) is
  flow-matching, no distillation. Hamiltonian Flow Maps (2601.22123) is a per-step integrator (adjacent,
  not overlapping); Timewarp is an NF (already ~1-step). CM+MD intersection exists only for equilibrium
  samplers (2502.07337), not joint-trajectory generators.
- **Feasible:** MDGen repo (bjing2016/mdgen, MIT) has code + checkpoints + **downloadable cached
  tetrapeptide MD data (~300µs) — no MD regeneration needed**; eval scripts (torsion FES, MSM rates,
  dynamical content) reuse the released test set. Distilling the released checkpoint is a few-GPU/weeks job.
- **Narrowest version:** consistency/shortcut distillation of MDGen's forward-sim flow to 1-4 NFE,
  evaluated on preservation of *dynamical* observables (transition rates, FES, autocorrelation).
- **Biggest risk:** few-step distillation may wreck the *kinetic* observables that are the whole point
  (rare-transition mode collapse, mis-estimated MSM rates) — but "does step-reduction distillation
  preserve kinetics?" is itself the open scientific question, so a negative result is still a result.
  Secondary: MDGen already ~60 GPU-sec/inference, so speed payoff motivation is weaker than in images.

**STATE:** two live method candidates (L35 crystals, L37 MDGen), both not-scooped + feasible at
"few GPUs, weeks", both with public teacher code+checkpoints+data. L35's risk is significance
(baselines already semi-fast); L37's risk is that the hard part (kinetics) is exactly what breaks —
but L37 reframes that risk as the paper's scientific question. Next: pick one, or prototype both cheaply.

### THEORY LENS (user steer, 2026-07-13): aim for a novel math proof
User: "we could find some interesting results if we try to produce a novel math proof." A theorem
+ method beats an empirical distillation port (kills the incrementalism risk that wounded every
prior candidate). Where a real, unclaimed theorem plausibly lives:
- **L37 (MD kinetics):** a bound relating few-step distillation error (Wasserstein / score error) to
  error in *dynamical* observables — transfer-operator / Koopman spectrum, MSM implied timescales,
  spectral gap. "When does step-reduction preserve kinetics?" is a genuine open theoretical question
  with existing machinery (VAC/VAMP, transfer-operator perturbation theory) but no distillation result.
- **L35 (crystals):** well-posedness / symmetry-preservation of flow maps on the crystal product
  manifold (GL-quotient lattice × flat torus T³ × discrete simplex) — but GFM (2510.21608) already did
  general Riemannian flow-map theory, so a crystal-specific theorem risks being "apply GFM here" (weaker).
- **Cross-cutting:** the manifold-native math cluster from Phase 1b (log-map losses, BCH composition,
  higher-order manifold integrators, irreducible-variance bound L19) — theory-flavored, mostly unproven.
Investigating the theory surface before committing (whether a provable+unclaimed theorem exists).

### THEORY-SURFACE INVESTIGATION (2026-07-13) — SUPERSEDED BY THE 2026-07-14 PROOF AUDIT

> Historical scouting record only. Its novelty, MFPT, TV, non-normality, and
> go/no-go conclusions are superseded by the mathematical-resolution section
> below and must not be treated as current.

Two deep theory scouts (math reasoning + prior-art verification).

**L37 (MDGen kinetics) — NOVEL THEOREM EXISTS, and it doubles as the method's design principle.**
- **Machinery:** two legs — (A) distillation error → transfer-operator perturbation via χ²/HS
  distance **on the joint two-time law** ρ(x,y)=μ(x)p(y|x); (B) self-adjoint spectral perturbation
  (Weyl + Davis-Kahan sin-Θ, resolvent for MFPT). Standard-but-unassembled.
- **Theorem candidate (reversible + spectral-gap regime):** implied-timescale error
  `|t̂_i − t_i| ≤ (t_i²/(τ λ_i))·D_χ + O(D_χ²)`, i.e. **linear in distillation error, quadratically
  amplified by the slow timescale**; leading eigenvalue error = error in the slow mode's two-time
  autocorrelation. Plus Davis-Kahan for metastable-set assignment and a resolvent bound for MFPT.
- **Scoop:** static distribution-error bounds for distilled generators exist; a **dynamical-observable
  error bound for a distilled *trajectory* generator does not** (verified: Rudolf-Schweizer
  arXiv:1503.04123 is static-Wasserstein; ITO/BoPITO learn transfer operators but no kinetic error
  bound). Pieces are assemblable.
- **The killer insight (this IS the contribution):** the bound is **provably FALSE if stated on the
  per-frame marginal** — one can re-couple the joint law to hold every per-frame distribution / FID
  fixed while moving λ₁ (hence t₁) arbitrarily. So *marginal/per-frame distillation error gives ZERO
  control of kinetics.* Prescription: distill + validate on **two-time correlations, not per-frame
  plausibility.** This converts the project's central worry into a theorem-shaped, testable claim.
- **Difficulty:** clean reversible version ~2-week lemma; honest version (distilled P̂ is non-normal /
  not μ-reversible → Bauer-Fike + pseudospectra + a δ_stat coupling term; and bridging an achievable
  consistency/shortcut loss to D_χ) = solid workshop/short-conference theorem. Real, modest, not open.
- **Open technical gap:** the bridge from a specific distillation loss (L² velocity/score error, →
  W₂/KL of the segment law) to χ²/HS — W₂→χ² is NOT free (χ² can be ∞ while W₂ tiny). Needs a
  bounded-density-ratio (Doeblin/importance-weight) assumption, else only the W₂ route survives (bounds
  Lipschitz observables / MFPT but NOT eigenvalues). This is the deciding feasibility question for the
  clean theorem.

**L35 (crystals) — THEORY LARGELY SUBSUMED by GFM.**
- The "hard" parts (GL(3)/O(3) quotient, product structure) are **exactly what GFM (2510.21608)
  already covers**: the Gram-matrix rep turns GL(3)/O(3) into SPD(3), a Cartan-Hadamard manifold nicer
  than the tori/spheres GFM tests on; completeness/orientability/connectedness are closed under products.
  A continuous-manifold well-posedness theorem is a **one-line GFM corollary** — not publishable as new theory.
- **Non-subsumed sliver (obstruction, corollary-grade):** the *discrete atom-type channel* falls outside
  GFM's completeness+ODE assumptions — Fisher-Rao simplex is geodesically incomplete and crystal data
  lives on the vertices (log-map degenerates); a CTMC isn't a PF-ODE (no deterministic flow map). Plus a
  per-step group-augmentation-vs-semigroup incompatibility (crystal analog of the protein SO(3) wall).
  Real but assembled from known facts (Dirichlet-FM boundary pathology + CTMC discreteness) — a
  guardrail/synthesis note, not a hard theorem.

**SUPERSEDED VERDICT:** **L37 (MDGen kinetics distillation) was provisionally picked for a
theorem+method paper.** That pre-audit choice relied on three provisional
claims: public feasibility, an unclaimed theorem, and alignment between the
theorem and empirical method. The audit below leaves novelty and the
end-to-end theorem open, so this paragraph is not a current recommendation.

### MATHEMATICAL RESOLUTION (2026-07-14) — full-spectrum bridge fails as stated

The proof audit in `docs/L37_MATH_SOLUTION.md` changes the theory verdict:

- **GATE-0 passes:** equal one-frame marginals give no kinetic control.
- **GATE-1's C1 premise is disproved and the headline is blocked:** even with
  `p_hat/mu <= 1`, both KL directions can tend to zero while `D_chi` and the
  transfer-eigenvalue error remain fixed. No registered PASS/KILL branch exactly fits.
  The proved replacement assumes both normalized joints are bounded; it is sufficient,
  not necessary, and not established for MDGen. A stationary Ornstein-Uhlenbeck pair
  already shows that reference boundedness is not automatic on noncompact spaces.
- **The reference-uniform TV correction is false:** `TV(mu_hat,mu)` does not control the
  canonical Radon-Nikodym transport used to compare the state-aligned operators.
  Equivalence is required for that whole-space canonical comparison, and a lower
  density-ratio bound or uniform closeness is required for a useful norm bound.
- **GATE-2 is superseded conditionally:** once a common-space operator-norm bound exists,
  a non-normal student does not add a global `kappa(V)` because the reference operator is
  self-adjoint and normal.
- **Multiplicity matters:** the scalar first-order formula and individual eigenvector
  guarantee require a simple reference mode. Repeated modes admit spectral-cluster and
  invariant-subspace guarantees only.
- **Limited salvage:** an externally supplied segment-law KL bound controls bounded
  teacher-mode one-lag correlations and a derived relaxation proxy. That proxy is not a
  student implied timescale. A full pair-operator spectrum theorem remains conditional on
  strong, explicitly stated density regularity.
- **Path-law caveat:** a two-time law need not be Markov-consistent. This applies to
  MDGen and to configuration-only projections of phase-space MD. The surviving
  MFPT/committor result is a target-specific killed-kernel `L2` bound; it is not controlled
  by the full-chain gap and does not transfer to actual trajectories without
  infinite-horizon time-homogeneous Markov factorizations.
- **Still open:** the exact MDGen-loss-to-segment-KL theorem, GATE-3 constant calibration,
  pair stationarity, absolute continuity, MD-specific constants, finite-sample
  estimability, and novelty of the surviving result.

Current theory status: **the registered pure-math claims are proved, corrected, or disproved;
the end-to-end MDGen theorem and research go/no-go are not established.** Do not proceed on
the earlier unconditional-theorem verdict.

**INDEPENDENT SPOT-CHECK (2026-07-15):** hand-rederived from scratch (not just read) the two
highest-stakes claims: **GATE-0** (two-state per-frame-insufficiency: `D_χ=2|a−b|`, eigenvalue
gap `1−2a` vs `1−2b` — confirmed) and **No-go Theorem A** (the θ/e family that kills the
original C1 bridge: `g_e` mean-0 unit-norm ✓, `D_χ²=θ²` exactly independent of `e` ✓, `g_e` is
a genuine eigenvector with eigenvalue `θ` for the reference and `0` for the independence kernel
✓, `TV=2θe(1−e)→0` ✓, both KL directions →0 at the claimed rates ✓). **Both check out — the
audit is real, not a false-positive kill.** The friend's review is correct: my original
unconditional bridge is dead, replaced by the conditional Theorem 2 (needs reference-side
density boundedness, MD-realism unproven) and the weaker unconditional Pinsker fallback
(§8.2, bounds a fixed teacher-mode correlation, does not identify a student eigenvalue). Did
not independently re-verify every remaining algebraic step (Theorem 1's Schur complement,
No-go Theorem B's 3-state construction) — taken on the document's internal consistency.

### MD-REALISM RESOLUTION (2026-07-15) — local repair does not close L37

`docs/L37_MATH_SOLUTION_V2.md` resolves the follow-up attempt in
`docs/L37_MATH_SPEC_V2.md`:

- **Gate A proved:** MDGen uses temporal key-frame-relative residue poses, not
  backbone-threaded chain-relative or raw lab-frame poses. Tetrapeptide production is
  Langevin NVT (implicit solvent) or Langevin/barostatted NPT (explicit solvent);
  ATLAS protein production is Nose-Hoover/Parrinello-Rahman NPT. None is NVE.
- **Gate B proved conditionally:** ideal continuous underdamped Langevin has a jointly
  continuous phase-space transition density under explicit admissible-potential
  hypotheses, which gives compact local density-ratio bounds. This does not directly
  cover MDGen's projected, constrained, discretized, and sometimes deterministic/NPT
  reference pipeline.
- **Gate C blocked:** confinement at infinity does not imply a quantitative gap. A
  smooth one-dimensional potential with fixed quadratic tails can contain an
  equilibrium-rare well behind a growing barrier, with its spectral gap tending to
  zero.
- **Gate D blocked:** a two-state one-step student can have both KL directions, and both
  fixed-horizon segment KL directions, tend to zero while acquiring an eigenvalue
  tending to one. Few-step distillation has no automatic anti-trapping property.
- **Gate E blocked for current MDGen:** local agreement plus a common tail escape bound
  does not imply global operator closeness. The settled pair-spectrum theorem survives
  only after imposed global compactness/density/stationarity safeguards and an external
  loss-to-segment-KL theorem.

Current strongest assumption-light result remains `docs/L37_MATH_SOLUTION.md` Section
8.2: control of one fixed bounded teacher-mode autocorrelation, not identification of a
student eigenvalue. The end-to-end generic few-step-distillation theorem is formally
blocked, not open pending routine citation work.

**INDEPENDENT SPOT-CHECK (2026-07-15):** hand-rederived the two decisive v2 counterexamples
from scratch. **No-go Theorem D** (Gate D killer): teacher resamples i.i.d. from `μ_e=(e,1-e)`
(eigenvalue exactly 0); student `P̂_e` has eigenvalue `(1-2e)/(1-e)→1`. Recomputed both pair-law
tables and both KL divergences independently — match exactly, both →0 at `O(e·log(1/e))` while
the eigenvalue drifts to the worst possible value. Confirmed. **No-go Theorem C** (Gate C
killer): 1D potential family, fixed confining tail for every `n`, rare well of height `n` behind
a barrier of height `3n`. Redid the Rayleigh-quotient bound: `g_n ≤ C'·exp(-2n)→0` even as the
well's mass `→0` — standard Eyring-Kramers metastability, well-grounded. Confirmed. **Both real,
not false-positive kills — Gates C and D are genuinely blocked, no unconditional theorem is
recoverable by the local-boundedness repair.** Also spot-checked Gate B's core compactness lemma
(elementary extreme-value-theorem argument) — clean.

### FINAL DECISION (2026-07-15) — L37, method-only

Theory ambition dropped from "theorem+method" to "method alone, honestly scoped." Proceed with
empirical few-step distillation of MDGen (public code+checkpoints+cached tetrapeptide data,
feasible at "few GPUs, weeks") without claiming an end-to-end kinetic-error theorem. The
citable theoretical contribution, if wanted, is limited to the settled §8.2 bound (one fixed
teacher-mode correlation, no student-eigenvalue claim) — state it as exactly that, not more.
Status: **L37 → picked (method-only)**. Math side of this thread is closed; no further gate-work
planned. Next session: scope the empirical distillation pipeline (teacher checkpoint, distillation
objective, eval on dynamical observables per the original spec's §7 prescription — validate on
two-time correlations, not per-frame plausibility).

### PIPELINE SCOPED (2026-07-15) — `docs/L37_PIPELINE_SPEC.md`

Grounded against the live `bjing2016/mdgen` repo (not memory): teacher = HF `bjing-mit/mdgen`
`forward_sim.ckpt`; data = HF `bjing-mit/tetrapeptide-sims` (`4AA_sims_implicit`, 2,846 peptides,
use repo's own `splits/4AA_test.csv`); eval = `scripts/analyze_peptide_sim.py`, which **already
computes TICA + MSM + autocorrelation** (the two-time-correlation machinery the theory's §7
prescription calls for) alongside per-frame JSD — most of the needed eval infra already exists.

**Confirmed blocker (Gate 0, must patch first):** NFE is a real parameter of
`mdgen/transport/transport.py`'s `Sampler.sample_ode(num_steps=...)`, but the call site in
`mdgen/wrapper.py` has `num_steps=self.args.inference_steps` **commented out** — every release
run silently uses `num_steps=50` regardless of any flag. No `--inference_steps` CLI flag exists
anywhere. Small patch, but real; nothing runs at reduced NFE until it's fixed.

**Pipeline:** Gate 0 (patch) → Stage A (prep implicit-solvent tetrapeptides) → Stage B (teacher
reference @ NFE=50) → Stage C (consistency-style distillation, proposed not locked) → Stage D
(distilled student @ NFE∈{1,2,4,8} **+ same-teacher step-truncated control @ NFE∈{2,4,8}**, the
control needed to isolate distillation-specific collapse from generic low-NFE degradation) →
Stage E (eval both per-frame JSD and TICA/MSM/autocorrelation on every condition, same peptides).

**The actual test:** at matched per-frame JSD, do two-time metrics (TICA-FES, MSM transition
matrix, implied timescales) diverge between distilled-student and step-truncated-teacher? Divergence
= positive empirical result (distillation-specific kinetic collapse, the failure mode the blocked
theorem targeted). No divergence = citable negative result. **No theorem is validated either way**
— Gates C/D/E remain formally blocked; this is a clean measurement, not a proof check. The one
still-live theory object (§8.2's teacher-mode-autocorrelation bound) can optionally be logged as a
secondary sanity check against Stage C's achieved KL, clearly labeled as non-headline.

**Open for next session:** distillation objective not locked (consistency-style proposed, cheapest
given `Sampler.sample_ode` already exposes ODE-trajectory generation); compute path unresolved;
peptide-count/power for the pilot is a guess; unconfirmed whether `analyze_peptide_sim.py` already
calls pyEMMA's `.timescales()` (one-line add if not).

### IMPLEMENTATION (2026-07-15) — Gate 0 + Stage A/C built and validated end-to-end, real checkpoint

Per user direction (locked consistency distillation, used available local compute rather than
blocking on GPU access, defaulted small pilot batch, added `.timescales()`). Full detail in
`docs/L37_PIPELINE_SPEC.md` §5. Headline: **real teacher checkpoint + real trajectory data +
real training step, running on a laptop, not synthetic.**

- **Data-split bug caught before it mattered:** `forward_sim.ckpt` was trained on the
  **explicit**-solvent split (`4AA_sims`/`4AA_test.csv`); the spec's earlier suggestion to use
  implicit solvent was wrong — the two test splits are disjoint peptide sets. Corrected.
- **Three real upstream bugs found + patched** in vendored `third_party/mdgen`: (1) Gate 0's NFE
  wiring, confirmed and fixed exactly as scoped; (2) `prep_sims.py` referenced a nonexistent
  `args.atlas_dir` (should be `args.sim_dir`) — the README's own preprocessing command is broken
  as shipped; (3) `batched_gather`'s numpy list-indexing (`data[ranges]`) is a hard error on numpy
  2.x (legal-but-deprecated on MDGen's pinned 1.21.x) — blocked all real-data loading; fixed via
  `tuple(ranges)`. Bonus: this also pre-empts a documented upcoming PyTorch 2.9 break.
- **Distillation objective locked: consistency distillation** (not shortcut/mean-flow) — needs
  only the teacher's existing velocity field + one Euler step, no architecture change, lowest lift
  given `Sampler.sample_ode` already exposes the needed call. Built in `src/l37/`:
  `consistency_distill.py` (pure math core, boundary-condition parametrization reusing MDGen's own
  GVP path coefficients as `c_skip`/`c_out` — zero new hyperparameters; 13 tests, hardware-
  independent) → `training_step.py` (composes into one loss; 5 tests, verifies teacher gets zero
  gradient / student gets gradient on every param) → `train_distill.py` (runnable script: load
  teacher, warm-start student via `deepcopy`, load real data, Adam loop).
- **End-to-end validated on the real artifacts, not mocks:** downloaded real `forward_sim.ckpt`
  (34.15M params, confirmed `path_type=GVP, prediction=velocity, sim_condition=True`) and real
  trajectories for 3 test peptides (FLRH/IMRY/RTVD); ran the real (patched) `prep_sims.py`; ran
  3 real Adam steps through `train_distill.py` — finite losses (0.218/0.168/0.028), gradients on
  all 289 student tensors, zero on the frozen teacher. **70 tests pass repo-wide, zero regressions.**
- **Local environment limit (not a pipeline blocker):** `pyEMMA` (needed only for Stage E eval)
  fails to build from source on this machine — a genuine Clang-21 template-syntax
  incompatibility in pyEMMA's own C++ (reproduced independently, confirmed not a numpy/Python-
  version issue: `T::template compute(...)` needs an explicit template-argument list under Clang
  21's tightened conformance checking). Stages A–D (data, sampling, distillation) import and run
  without pyEMMA — confirmed by direct import test. Only eval needs a different toolchain.
- **Still open:** only mechanics validated (1 step × 3 peptides), not a real training run;
  sampling from the trained student not yet run; Stage E blocked locally on pyEMMA; full-pilot
  compute/throughput unassessed (this validated correctness, not scale).

**Concurrent independent audit:** a separate, concurrently-running Codex session (not dispatched
by this one — confirmed via process inspection) produced `docs/L37_LOSS_PREMISE_AUDIT.md`,
analyzing this session's actual `consistency_distillation_loss` code. Two concrete findings
verified true and fixed: `sim_inference.py` lacked an `--inference_steps` CLI override (added);
and **`dopri5`'s `num_steps` is output-point count, not NFE** — Stage D's NFE sweep needs the
fixed-step `euler` solver or real eval-count instrumentation to be valid; not yet resolved. Full
detail in `docs/L37_PIPELINE_SPEC.md` §5.6. Two agents editing this repo concurrently is a real
collision risk going forward — flagged to user.

### HONEST ASSESSMENT (2026-07-15) — novelty, expected result, and who would care

Asked directly after the implementation work: *do we have anything novel, what's the final
result, who cares.* Answering plainly, before more work goes in.

**Novelty — ingredients yes, finding no.** Three tiers:
1. The headline theorem is **dead**, not unproven — three independent passes (this session's
   audit, this session's v2 repair attempt, the concurrent session's loss-premise audit) all
   converged on the same block from different angles. That convergence is itself informative but
   is a negative result, not a discovery.
2. The obstruction proofs (GATE-0 per-frame-insufficiency; No-go Theorem D, a 2-state chain with
   KL→0 while the eigenvalue→1) are **real, independently hand-verified, correct math** — but
   guardrail-tier: "here is exactly why this doesn't work" is useful to the next person who tries,
   not a paper by itself.
3. The empirical setup (consistency-distill a joint-trajectory MD generator, check whether
   per-frame plausibility diverges from kinetic fidelity) is **verified unclaimed** in the
   literature — but "nobody's tried this" licenses the attempt, it doesn't predict the outcome.
   **Net: novel ingredients, no novel finding yet — zero eval has run.**

**Expected result — genuinely unknown, states both branches honestly:**
- **Branch A (divergence):** at matched per-frame JSD, the distilled student's two-time metrics
  (TICA free-energy surface, MSM transition matrix, implied timescales) are worse than a
  step-truncated teacher's — i.e. cutting sampling steps preserves per-frame structural accuracy
  while quietly degrading the speed/rate of motion. Real, citable, mechanistically motivated by
  the (blocked) theorem's own obstruction proofs. Workshop/eval-tier, not a headline.
- **Branch B (no divergence):** two-time metrics track the per-frame metric as steps drop — a
  valid but weaker negative result; without divergence, the theoretical motivation (which proves
  collapse is *possible*, not that it *happens* here) leaves the paper's hypothesis unconfirmed.
- Weak prior toward Branch A from SiD-Protein's independent finding that naive distillation
  already tanks designability for backbone generators (a related failure mode, different model
  class) — a prior, not evidence for this specific case.

**Who would care — narrow, and narrower than it first looks:**
- Direct audience: the small set of groups building generative trajectory/kinetics models (MDGen's
  own authors, the Noé/Klein Boltzmann-generator lineage, MD-surrogate researchers) — order tens of
  active researchers.
- **Complication that undercuts even that audience:** MDGen's teacher is already fast (~60 GPU-sec
  per rollout vs. ~3 GPU-hr for real MD, per the paper) — speed was never its bottleneck, so "make
  it faster via distillation" doesn't solve a pain this audience currently has.
- Broader relevance is possible only as a supporting case study for the wider "do fast neural
  surrogates for dynamical systems preserve dynamics or just statistics" question (echoes of the
  weather/climate-emulator extreme-event concern) — but that requires the paper to explicitly reach
  for that framing, and remains one model on tetrapeptides, not a general result.
- **Realistic ceiling: workshop/eval-track, cited by the next few papers in this specific
  subfield** — consistent with L18's and L32's ceiling earlier in this thread. Not a result that
  reaches practitioners or moves the broader field. If that ceiling isn't worth the remaining work
  (fix dopri5/NFE, get pyEMMA running, run a real pilot), that is a legitimate reason to stop here.

**Fallback if this fizzles:** L35 (crystals — not-scooped + feasible, never touched empirically)
and L32 (the diagnostic-bundle audit) both remain live, unexplored, recorded above — stopping on
L37 is not a dead end for the thread.

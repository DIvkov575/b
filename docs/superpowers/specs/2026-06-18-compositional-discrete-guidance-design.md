# Design: Composable Discrete Flows

## Title
**Composable Discrete Flows: Boolean Guidance Algebra for Discrete Flow Matching**

## One-line
Derive AND/NOT/OR composition operators for discrete flow matching, prove correctness, and demonstrate multi-property conditional generation.

## Gap (verified)
- Discrete Classifier-Free Guidance exists (D-CFG, ICLR 2025) — single condition only
- Composable Diffusion (Liu ECCV 2022) defines Boolean algebra for continuous scores — never extended to discrete
- ADFLIP, DPLM, EvoDiff all handle single-property guidance; multi-objective deferred to external BO wrappers
- 100-agent adversarial verification confirms: NO published work on compositional score arithmetic for discrete generative models as of mid-2026

## Contribution
A framework-agnostic Boolean guidance algebra for discrete flow matching:
- **AND**: Generate x satisfying condition A AND condition B simultaneously
- **NOT**: Generate x satisfying A but NOT B (selectivity/avoidance)
- **OR**: Generate x satisfying A OR B (diversity/coverage)
- Correctness proofs showing composed guidance converges to the correct conditional distribution
- Instantiation on 2 discrete flow formulations (CTMC + probability-path)
- Demonstration on controlled synthetic tasks + one real application

## Approach: Framework-Agnostic

Derive composition operators that work for ANY discrete flow formulation, then instantiate on specific ones. This is the strongest contribution because it's a general theory paper, not an application paper.

### Mathematical Framework

**Background:** In continuous diffusion, guidance works via scores:
- p(x|y) ∝ p(x)p(y|x) → ∇log p(x|y) = ∇log p(x) + ∇log p(y|x)
- AND: score_A+B = score_A + score_B (product of experts)
- NOT: score_A¬B = score_A - score_B
- OR: score_A∨B = log(exp(score_A) + exp(score_B)) (log-sum-exp)

**For discrete flows, the generative process uses transition RATES (CTMC) or probability vectors (path-based), not scores.** The key derivation:

#### CTMC Formulation (Campbell et al.)
- Generative process: dx ~ R_t(x'|x)dt where R_t is a rate matrix
- Unconditional: R_t^uncon
- Conditional on y: R_t^{y}(x'|x) = R_t^uncon(x'|x) · p(y|x', t) / p(y|x, t)
- **AND composition:** R_t^{A∧B}(x'|x) = R_t^uncon(x'|x) · [p(A|x',t)/p(A|x,t)] · [p(B|x',t)/p(B|x,t)]
- **NOT composition:** R_t^{A∧¬B}(x'|x) = R_t^uncon(x'|x) · [p(A|x',t)/p(A|x,t)] · [p(B|x,t)/p(B|x',t)]
- **OR composition:** R_t^{A∨B}(x'|x) = R_t^uncon(x'|x) · [p(A∨B|x',t)/p(A∨B|x,t)]

Key challenge: ensure composed rates remain valid (non-negative off-diagonal, rows sum to zero). Need to prove this holds or derive corrections.

#### Probability-Path Formulation (Gat et al.)
- Generative process: p_t(x) interpolates from source to target
- Learned posterior: p_θ(x_1|x_t) predicts clean data
- Guidance modifies the posterior: p_θ(x_1|x_t, y) ∝ p_θ(x_1|x_t) · p(y|x_1)^γ
- **AND:** p_θ(x_1|x_t, A∧B) ∝ p_θ(x_1|x_t) · p(A|x_1)^γ_A · p(B|x_1)^γ_B
- **NOT:** p_θ(x_1|x_t, A∧¬B) ∝ p_θ(x_1|x_t) · p(A|x_1)^γ_A · p(B|x_1)^{-γ_B}
- **OR:** p_θ(x_1|x_t, A∨B) ∝ p_θ(x_1|x_t) · [p(A|x_1)^γ + p(B|x_1)^γ - p(A|x_1)^γ·p(B|x_1)^γ]

Key challenge: negative exponents (NOT) can make probabilities > 1 or < 0 over discrete states. Need clamping/renormalization or alternative formulation.

### Base Model Choices

1. **CTMC instantiation:** Implement minimal CTMC discrete flow on synthetic categorical data (e.g., 8-state, sequence length 16-64). Controllable, fast, provable.
2. **Probability-path instantiation:** Use Gat et al.'s formulation. Train small model on text8 or synthetic sequences. Alternatively, use pretrained MDLM checkpoint if available.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│ Composable Discrete Flow Framework                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Base Model (unconditional discrete flow)            │
│  ├── CTMC variant (rate matrix R_t)                 │
│  └── Prob-path variant (posterior p_θ(x_1|x_t))    │
│                                                      │
│  Classifiers (per-property)                          │
│  ├── Classifier A: p(A|x_t, t) — time-conditional  │
│  ├── Classifier B: p(B|x_t, t)                     │
│  └── Classifier C: p(C|x_t, t)                     │
│                                                      │
│  Composition Engine                                  │
│  ├── AND(A, B) → composed rate/posterior            │
│  ├── NOT(A, B) → A ∧ ¬B composed rate/posterior    │
│  ├── OR(A, B) → composed rate/posterior             │
│  └── Nested: AND(A, NOT(B, C)) etc.                │
│                                                      │
│  Sampling                                            │
│  ├── Tau-leaping (CTMC)                             │
│  └── Euler step (prob-path)                         │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## Experiments

### Phase 1: Synthetic Validation (Week 1-3)
Prove the operators work on controlled tasks where ground truth is known.

**Task 1: Categorical sequence generation with known conditionals**
- State space: K=8 categories, sequence length L=32
- Train unconditional CTMC/prob-path discrete flow
- Train 3 binary classifiers on known properties (e.g., "contains pattern ABC", "starts with X", "has no repeated characters")
- Test: AND(prop1, prop2) should produce sequences satisfying both
- Test: NOT(prop1, prop2) should satisfy prop1, violate prop2
- Metric: % of generated samples satisfying the Boolean condition vs. ground-truth enumerable distribution
- Baseline: single-condition D-CFG applied independently (no composition)

**Task 2: Discrete graph generation with composed properties**
- DiGress-style categorical graph diffusion (small graphs, ~10-20 nodes)
- Properties: "graph is connected", "max degree ≤ 3", "has a cycle of length ≥ 5"
- Compose: AND(connected, max_deg≤3) — should generate 3-regular-ish connected graphs
- Compose: AND(connected, NOT(has_long_cycle)) — trees or short-cycle graphs

**Task 3: Scaling test**
- Increase K from 8 → 20 → 100
- Increase L from 32 → 128 → 512
- Measure: does composition quality degrade? At what scale?

### Phase 2: Real Application (Week 4-6)
One real domain demonstrating practical utility.

**Option A: Protein sequence design (preferred)**
- Base: DPLM or MDLM trained on UniRef50 (or use released checkpoint)
- Classifiers: stability predictor, secondary structure predictor, disorder predictor
- Compose: AND(stable, helical) AND NOT(disordered)
- Evaluate: ESMFold pLDDT, DSSP secondary structure assignment
- Compare: single-property guidance, Bayesian optimization wrapper, random filtering

**Option B: Small molecule SMILES generation**
- Base: discrete flow on SELFIES/SMILES (train on ZINC-250K)
- Classifiers: QED, SA score, lipophilicity, hERG liability
- Compose: AND(high_QED, low_SA) AND NOT(hERG_active)
- Evaluate: RDKit property calculators, novelty, diversity

### Phase 3: Analysis & Theory (Week 7-8)
- Prove convergence of composed sampling to correct conditional (or characterize approximation error)
- Ablation: guidance strength γ per operator
- Ablation: interaction between γ_A and γ_B (interference effects)
- Show failure modes: when does NOT over-suppress? When does AND mode-collapse?
- Compare CTMC vs prob-path instantiations

## Evaluation Metrics

| Metric | What it measures |
|--------|-----------------|
| Compositional accuracy | % samples satisfying full Boolean condition |
| Individual property satisfaction | Per-property success rate |
| Diversity (self-BLEU / Tanimoto) | Avoiding mode collapse under composition |
| Quality (perplexity / validity) | Not destroying base model quality |
| Interference | Does AND(A,B) hurt A-satisfaction vs guidance on A alone? |
| NOT effectiveness | Does NOT(B) actually suppress B without destroying A? |

## Compute Budget

| Component | Estimate |
|-----------|----------|
| Synthetic unconditional flow (K=8, L=32) | ~2 GPU-hours |
| Synthetic classifiers (3 properties) | ~1 GPU-hour each |
| Scaling tests (K=100, L=512) | ~8 GPU-hours |
| Real application base model (if training from scratch) | ~24-48 GPU-hours |
| Real application classifiers | ~4 GPU-hours each |
| Total | ~50-80 GPU-hours (< 1 week A100) |

If using pretrained checkpoints (DPLM, MDLM): reduce to ~20-30 GPU-hours total.

## Timeline (8 weeks)

| Week | Milestone |
|------|-----------|
| 1 | Math derivation + CTMC implementation + synthetic data |
| 2 | Prob-path implementation + synthetic classifiers |
| 3 | AND/NOT/OR operators implemented + synthetic validation |
| 4 | Kill gate: if synthetic accuracy < 80%, diagnose/pivot |
| 5-6 | Real application (protein or molecule) |
| 7 | Analysis, ablations, theory tightening |
| 8 | Paper writing |

## Kill Gates

1. **Week 3:** If composed guidance on synthetic task achieves < 80% Boolean accuracy (vs. > 95% for single-condition), the operators are broken. Diagnose whether it's the math or the implementation.
2. **Week 4:** If NOT operator consistently destroys sample quality (validity drops > 50%), the negative-exponent problem is fundamental. Consider alternative NOT formulations.
3. **Week 6:** If real application shows < 5% improvement over "generate + filter" baseline, the method works but isn't useful. Reframe as theory paper.

## Target Venues

- **NeurIPS 2026** (deadline ~May 2027) — main conference, generative models track
- **ICML 2026** (deadline ~Jan 2026) — if fast enough
- **ICLR 2027** (deadline ~Oct 2026) — most realistic timeline
- **LoG 2026** — backup, lower bar

## Key References

1. Liu et al. "Compositional Visual Generation with Composable Diffusion Models." ECCV 2022. (arXiv:2206.01714) — Boolean algebra for continuous diffusion
2. Gat et al. "Discrete Flow Matching." Jul 2024. (arXiv:2407.15595) — prob-path discrete FM
3. Campbell et al. "A Continuous Time Framework for Discrete Denoising Models." NeurIPS 2022. — CTMC formulation
4. Schiff et al. "Simple Guidance Mechanisms for Discrete Diffusion Models." ICLR 2025. (arXiv:2412.10193) — D-CFG single condition
5. Stark et al. "Dirichlet Flow Matching." ICML 2024. (arXiv:2402.05841) — simplex FM
6. Davis et al. "alpha-Flow." Apr 2025. (arXiv:2504.10283) — unified simplex geometry
7. Discrete Guidance Matching. ICLR 2026. (arXiv:2509.21912) — exact transition rates
8. ADFLIP. ICML 2025. (arXiv:2507.14156) — training-free classifier guidance for protein discrete FM
9. Zheng et al. "DPLM." 2024. (arXiv:2402.18567) — discrete protein language model with guidance

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Schiff et al. extend D-CFG to composition | High | Move fast; frame contribution as framework-agnostic (they're CTMC-only) |
| NOT operator numerically unstable | Medium | Alternative formulations: energy-based NOT, learned negation |
| Composed guidance mode-collapses | Medium | Temperature annealing, diversity regularization |
| "Just generate + filter" baseline is good enough | Medium | Show wall-clock efficiency advantage at low acceptance rates |
| Protein/molecule application doesn't work | Low | Paper stands on synthetic + theory alone; real app is bonus |

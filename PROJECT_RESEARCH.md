# Generative Biomolecule Project Research

## Overview

Research conducted June 2026 to identify a novel, compute-feasible generative biomolecule project.
Constraints: 1 GPU, no large-scale training, publishable at ML venue.

**Selected Project:** Dirichlet Flow Matching for Protein Inverse Folding

---

## Table of Contents

1. [Landscape Survey](#1-landscape-survey)
2. [Evaluated Projects](#2-evaluated-projects)
3. [Selected Project: Dirichlet FM for Inverse Folding](#3-selected-project-dirichlet-fm-for-inverse-folding)
4. [Rejected Projects with Rationale](#4-rejected-projects-with-rationale)
5. [Key References](#5-key-references)

---

## 1. Landscape Survey

### Key Players

#### Hannes Stark (MIT, PhD with Jaakkola & Barzilay)
- 58 publications, leading PhD student in ML for structural biology
- **EquiBind** (ICML 2022): Fast direct-shot docking via equivariant GNNs
- **DiffDock** (ICLR 2023): Molecular docking as diffusion on SE(3) × torsions
- **HarmonicFlow / FlowSite** (ICML 2024): Flow matching for docking + binding site design
- **Dirichlet Flow Matching** (ICML 2024): Flow matching on simplex for DNA sequence design
- **ProtComposer** (ICLR 2025 Oral): Compositional protein generation via 3D ellipsoids
- **BoltzGen** (2025): Universal binder design pipeline (proteins, peptides, antibodies, nanobodies)

#### Boltz (MIT spinoff: Gabriele Corso, Jeremy Wohlwend, Saro Passaro)
- **Boltz-1** (Nov 2024): First fully open-source AlphaFold3-level model (MIT license)
- **Boltz-2** (June 2025): Adds binding affinity prediction (log10 IC50 + binary)
- Architecture: PairFormer trunk + EDM diffusion + atom attention + confidence/affinity heads
- Handles: protein, DNA, RNA, ligands, modified residues, covalent bonds, glycans

### What's Solved (2024-2026)
- Protein backbone generation (RFdiffusion, Chroma, FoldFlow-2)
- Molecular docking with ~38% top-1 accuracy at <2Å (DiffDock)
- Structure prediction for biomolecular complexes (AF3, Boltz-1/2, Chai-1)
- De novo binder design with experimental validation
- Sequence generation from protein language models (ESM3, EvoDiff)

### What's NOT Solved
- Dynamics-aware design (no method designs for conformational ensembles)
- Function-first generation (methods design structure, then hope for function)
- Generalization to novel targets (ML docking fails on unseen protein families)
- Unified predict + design (separate models for each)
- Multi-component complex design (protein + ligand + nucleic acid holistically)
- Quantitative affinity control (can't design to a target Kd)
- Synthesizability-aware molecule generation

### Paradigm Comparison

| Approach | Strengths | Weaknesses | Examples |
|----------|-----------|------------|----------|
| Diffusion (DDPM/Score) | Well-understood; flexible conditioning | Slow sampling (many steps) | RFdiffusion, DiffDock, Chroma |
| Flow Matching | 5-10× fewer steps; simpler training | Less mature; fewer wet-lab validations | FrameFlow, FoldFlow-2, AlphaFlow |
| Autoregressive | Scales well; leverages LLM infra | Hard for 3D; order-dependent | ESM3, ProGen2 |
| Hybrid | Combines strengths | Complexity | ESM3, FoldFlow-2 |

---

## 2. Evaluated Projects

### Project A: Affinity-Guided BoltzGen (REJECTED)

**Idea:** Use Boltz-2's affinity head as gradient-based guidance during BoltzGen's diffusion sampling.

**Why rejected:**
- Boltz-2 affinity head has weak accuracy: Pearson r=0.24-0.45 on drug datasets, no significant correlation among top-100 compounds
- AIMS-Fold (May 2026) already guides Boltz-2's diffusion with experimental data (partial scoop)
- Boltz team could implement this trivially (GaussianSmearing class already in codebase)
- "Plugging existing head into existing framework" = systems contribution, not methodological
- Workshop-tier without wet-lab validation

**Key finding:** Boltz-2's affinity head uses hard distogram binning (.long() cast) that breaks gradients. An unused GaussianSmearing class exists as a differentiable alternative.

---

### Project B: Dirichlet FM for Protein Inverse Folding (SELECTED)

See [Section 3](#3-selected-project-dirichlet-fm-for-inverse-folding) for full details.

---

### Project C: Latent Dynamics — Generative Protein Motion

**Idea:** Equivariant autoencoder compresses MD conformations to latent space; flow/diffusion model generates ensembles in latent space. 50-500× faster than full-atom methods.

**Data:**
- mdCATH: 5,398 CATH domains, 3.6TB, HuggingFace, 62.6ms total simulation, CC-BY-4.0
- ATLAS: 1,938 proteins, 15+ TB, 3×100ns/protein, CC-BY-NC-4.0
- MISATO: 16,972 protein-ligand complexes, 193GB, 10ns each

**Novelty:** HIGH — Ophiuchus/LatentDiff do latent diffusion for static structures only. Nobody has done latent diffusion for dynamics/ensembles. AlphaFlow/MDGEN work in full-atom space (expensive).

**Compute:** A100 40GB, 3-4 weeks. Autoencoder (~5-15M params) + latent flow (~10-20M params).

**Risk:** (a) Reconstruction fidelity for dynamics-relevant detail, (b) correct Boltzmann weighting in latent space, (c) 3.6TB data loading.

**Status:** Strong #2 choice. Requires A100.

---

### Project D: Specificity-Aware Design — Binding A but NOT B

**Idea:** Apply Composable Diffusion's NOT operator (score subtraction) to molecular/protein generation for selectivity optimization.

**Data:**
- Davis et al.: 72 compounds × 442 kinases with Kd (31,800 measurements)
- Klaeger et al.: 243 compounds × 300+ kinases (~73,000 curves)
- ChEMBL: hundreds of thousands of extractable selectivity pairs
- BindingDB: tens of thousands of selectivity pairs

**Novelty:** HIGH — Composable Diffusion NOT operator (Liu et al., ECCV 2022) exists for images but has NEVER been applied to molecular/protein generation. REINVENT does RL-based selectivity but not diffusion guidance. BADGER does affinity guidance without negative/selectivity.

**Formulation:** `score_selective(x) = score(x|binds_A) - λ·score(x|binds_B)`

**Compute:** RTX 3090 24GB, 1-2 weeks. Train two classifiers + inference-time guidance on pretrained TargetDiff.

**Risk:** Simple docking-score subtraction might achieve similar results without ML.

**Status:** Excellent #2 choice if you want high practical impact. Cheapest compute.

---

### Project E: Coevolutionary Generation — Jointly Designing Protein Pairs

**Idea:** Extend MSA Transformer iterative generation to concatenated paired MSAs for co-generating interacting protein pairs.

**Data:**
- Bacterial two-component systems: 50,000-200,000+ HK-RR pairs across 10,756 bacteria
- PDB heteromeric complexes: ~60,000-80,000 structures
- STRING: 27.5B interactions, 59.3M proteins
- EVcouplings: precomputed for thousands of families

**Novelty:** HIGH — Malinverni & Babu (2023) did this with Potts models (MCMC). No deep learning method jointly generates both sides of an asymmetric PPI. Sgarbossa & Bitbol (2023) do MSA Transformer generation but single-chain only.

**Compute:** A100 40GB, 3-5 weeks. MSA Transformer fine-tuning (~100M params pretrained).

**Risk:** (a) Evaluation complexity (AF2-Multimer per pair), (b) must clearly beat Potts model.

---

### Project F: Multi-State Protein Design — Sequences That Fold Two Ways

**Idea:** Generative model conditioned on (backbone_A, backbone_B, trigger_type) → sequence that folds to both conformations.

**Data:**
- CoDNaS: tens of thousands of conformational pairs (~70% PDB coverage)
- Porter/Looger: ~100-192 confirmed fold-switching proteins
- ASD: 2,422 allosteric targets, 100K+ modulators

**Novelty:** HIGH — No generative model for multi-state design exists. Rosetta MSF does energy optimization. ProteinMPNN has multi-state mode but unvalidated.

**Compute:** RTX 3090 24GB, 2-3 weeks. Fine-tune ProteinMPNN with multi-state loss.

**Risk:** Only ~100-200 true fold-switches for training. Evaluation circular (AF2 can't predict both states; need AF-Cluster).

---

### Project G: Evolutionary Flow — Protein Evolution as Continuous Flow

**Idea:** Model evolution as flow matching in sequence space. Learn evolutionary velocity field conditioned on phylogenetic time.

**Data:**
- Pfam: 21,979 families with MSAs (22GB) + phylogenetic trees (29MB)
- ~10-50M transition pairs from tree cherries
- Viral longitudinal data (influenza, SARS-CoV-2, HIV)

**Novelty:** MODERATE — PEINT (Feb 2026, UC Berkeley) is very close: models p(descendant|ancestor, time) with transformers, experimentally validated. Gap is specifically the flow matching formulation with velocity field interpretation.

**Compute:** A100 40GB, 2-3 weeks. ESM-2 embeddings (frozen) + flow model (~20-50M params).

**Risk:** High risk of appearing incremental vs PEINT.

---

### Project H: Learning from Failures — Negative Data for Protein Generation

**Idea:** Use DPO/preference optimization with negative protein data (DMS failures, destabilizing mutations, aggregation) to improve generation.

**Data:**
- ProteinGym: ~3M variants across 217 DMS assays (majority loss-of-function)
- Tsuboyama: ~850K stability measurements (~75% destabilizing)
- GFP: 50K variants (~75% non-fluorescent)
- eSOL/PDBSol: 60K+ solubility annotations

**Novelty:** LOW-MODERATE — Physio-DPO (Jan 2026), g-DPO, EnerBridge-DPO, DDPP, CDM all cover this space. 6+ papers in 2025-2026.

**Compute:** RTX 3090 24GB, 1-2 weeks. DPO fine-tuning of ProtGPT2/EvoDiff.

**Risk:** HIGH scoop risk. Contribution would be incremental ("combined more types of negative data").

---

### Project I: Dirichlet FM for Antibody CDR Design

**Idea:** Same method as Project B but specialized for CDR loop design (6-20 residues).

**Data:** SAbDab (~7,000 antibody structures)

**Novelty:** Medium-High. Dirichlet FM's probabilistic output naturally captures CDR diversity.

**Compute:** RTX 4070 12GB, 1 week. Tiny model (~1-3M params).

**Risk:** Too niche for first paper. Better as follow-up to Project B.

---

## 3. Selected Project: Dirichlet FM for Inverse Folding

### Motivation

Stark et al. (ICML 2024) introduced Dirichlet Flow Matching — flow matching on the probability simplex using mixtures of Dirichlet distributions. They demonstrated it for DNA sequence design (4-dim simplex) but explicitly stated it "struggles to scale to higher simplex dimensions required for peptide and protein generation" (20 amino acids = 20-dim simplex).

Solving this scaling problem is an open research question with clear benchmarks and baselines.

### Problem Statement

Given a protein backbone structure, design a sequence of amino acids that will fold into that structure. This is "inverse folding" — the inverse of structure prediction.

**Current state-of-the-art:** ProteinMPNN (Dauparas et al., Science 2022) — autoregressive GNN.

**Limitation of ProteinMPNN:** Autoregressive decoding means O(L) sequential steps, no native diversity modeling, and no built-in mechanism for property-conditioned generation.

### Why Dirichlet FM

1. **Non-autoregressive:** All positions generated simultaneously (or in one step with distillation)
2. **Native uncertainty:** Dirichlet distribution directly represents per-position amino acid probabilities
3. **Guidance:** Classifier-free guidance for property conditioning at inference time
4. **Diversity:** Flow on simplex naturally captures multimodal sequence posteriors
5. **Theoretical elegance:** Dirichlet is the conjugate prior for categorical distributions — natural fit for amino acid modeling

### The 20-Dim Scaling Problem

Why Dirichlet FM works at 4 dims (DNA) but fails at 20 dims:
- **Concentration collapse:** Dirichlet distributions become highly concentrated on simplex vertices at high dimensions — interpolation paths degenerate
- **Path discontinuities:** Linear interpolation on high-dim simplices can pass through near-zero probability regions
- **Gradient vanishing:** Score functions on concentrated Dirichlet distributions have vanishing gradients away from modes
- **Normalization issues:** Higher-dim simplices have exponentially more volume near edges/faces than center

### Candidate Solutions

| Approach | Idea | Effort |
|----------|------|--------|
| Logit-normal reparameterization | Work in unconstrained logit space, transform to simplex at end | Low |
| Hierarchical factored simplex | Decompose 20-dim into nested sub-simplices (e.g., amino acid property groups: 4 groups × 5 members) | Medium |
| Annealed concentration schedule | Start with diffuse Dirichlet (low α), sharpen over flow time | Low |
| Score interpolation | Blend Dirichlet flow with simpler linear path on logit-simplex | Low |
| Learned noise schedule | Parameterize the Dirichlet concentration path with a neural network | Medium |
| Projected flow matching | Flow in R^19 (ambient space of simplex) with projection steps | Low |

### Architecture

```
Input: Backbone structure (N residues × backbone atom coords)
    │
    ▼
[GNN Encoder] — Message-passing on k-NN graph of Cα atoms
    │              Features: distances, angles, dihedrals, orientations
    │              Output: per-residue embeddings h_i ∈ R^d
    ▼
[Flow Network] — MLP per residue conditioned on h_i + time embedding
    │              Input: current point on simplex x_t ∈ Δ^19, time t
    │              Output: velocity vector v_t ∈ T(Δ^19) (tangent to simplex)
    ▼
[ODE Integration] — Euler/RK4 from t=0 (Dirichlet prior) to t=1
    │
    ▼
Output: Amino acid probabilities per residue ∈ Δ^19
        → argmax or sample for sequence
```

**Model size:** ~2-5M parameters (comparable to ProteinMPNN)

### Data

| Dataset | Description | Size | Usage |
|---------|-------------|------|-------|
| CATH 4.2 (ProteinMPNN splits) | Protein structures with sequences, train/val/test by topology | ~25,000 chains | Primary training |
| CATH 4.3 (updated) | Newer structures | ~30,000 chains | Extended training |
| PDB (redundancy-filtered) | All PDB at 40% seq identity | ~70,000 chains | Scale-up |
| TS50/TS500 | Standard test sets from ProteinMPNN paper | 50/500 structures | Evaluation |

**Data access:** CATH splits available from ProteinMPNN GitHub. Standard preprocessing pipeline exists.

### Evaluation Metrics

| Metric | What It Measures | Baseline (ProteinMPNN) |
|--------|-----------------|------------------------|
| Sequence recovery rate | % positions matching native | ~52% |
| Perplexity | Model confidence | ~3.5-4.0 |
| scTM (self-consistency TM-score) | Designed sequence refolds to target (via ESMFold/AF2) | >0.9 |
| Diversity | Avg pairwise sequence distance among N samples | Moderate |
| Sampling speed | Time per designed sequence | ~0.1s (autoregressive) |

**Success criteria:**
- Recovery ≥ 48% (within 4% of ProteinMPNN = publishable as competitive non-AR method)
- scTM ≥ 0.85 on TS50
- Diversity > ProteinMPNN (inherent advantage of flow-based)
- One-step distillation with <5% quality loss (unique capability)

### Implementation Plan

| Week | Milestone | Details |
|------|-----------|---------|
| **1** | Setup + DNA reproduction | Clone Dirichlet FM code. Reproduce DNA results. Set up CATH data pipeline. Identify failure mode at 20 dims with ablations. |
| **2** | Scaling fix implementation | Implement top 2-3 candidate fixes. Train small models on CATH subset (~5K chains). Quick eval to identify best approach. |
| **3** | Full training + evaluation | Train best approach on full CATH. Evaluate recovery, perplexity, scTM. Compare to ProteinMPNN, ESM-IF, LM-Design. |
| **4** | Extensions | One-step distillation. Classifier-free guidance (condition on stability/solubility). Diversity analysis. |
| **5-6** | Paper | Write up. Ablation figures. Submit to ICML/NeurIPS or ML4Molecules workshop. |

### Compute Requirements

| Resource | Requirement |
|----------|-------------|
| GPU | 1× RTX 3090 24GB (minimum: RTX 4080 16GB) |
| VRAM | 8-16GB |
| Storage | 5-10GB |
| RAM | 32GB |
| Training time | 2-5 days per experiment |
| Total wall-clock | 4-6 weeks |

### Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Can't beat ProteinMPNN | Medium | Paper framing: "first non-autoregressive flow model for inverse folding" + unique capabilities (guidance, diversity, speed) |
| 20-dim problem is fundamental | Low | Multiple candidate fixes; if all fail, fall back to CDR design (shorter, easier) |
| Scooped during execution | Low | Fast timeline (6 weeks); no known competing efforts |
| Dirichlet FM code not public | Low-Medium | Stark's code likely on GitHub; reimplementation from paper is straightforward (~500 lines core) |

### Unique Selling Points (vs ProteinMPNN)

1. **One-step generation:** After distillation, generate full sequence in a single forward pass (O(1) vs O(L))
2. **Native diversity:** Sample multiple valid sequences by varying the starting point on the simplex — no need for temperature hacking
3. **Property guidance:** Add classifier-free guidance for stability, solubility, or other properties without retraining
4. **Probabilistic output:** Full distribution over amino acids at each position (not just argmax) — useful for library design
5. **Interpolation:** Smooth paths between valid sequences by interpolating on the simplex — useful for exploration

### Related Code Resources

| Resource | URL/Location | Notes |
|----------|-------------|-------|
| Dirichlet FM (Stark) | github.com/HannesStark/dirichlet-flow-matching | Original DNA implementation |
| ProteinMPNN | github.com/dauparas/ProteinMPNN | Baseline + data splits |
| ESM-IF | github.com/facebookresearch/esm | Inverse folding baseline |
| CATH data | github.com/dauparas/ProteinMPNN/tree/main/training | Standard splits |
| Flow Matching (general) | github.com/atong01/conditional-flow-matching | Tong et al. framework |

---

## 4. Rejected Projects with Rationale

| Project | Primary Rejection Reason |
|---------|-------------------------|
| Affinity-Guided BoltzGen | Weak affinity signal (r=0.24-0.45), high scoop risk, systems-level contribution |
| Latent Dynamics | Requires A100; excellent project but higher compute bar |
| Specificity (NOT operator) | Excellent project; strong #2 if inverse folding doesn't work out |
| Coevolutionary Generation | Requires A100; longer timeline (3-5 weeks) |
| Multi-State Design | Limited training data (~100-200 fold-switches) |
| Evolutionary Flow | PEINT (Feb 2026) is too close; incremental contribution |
| Learning from Failures | Crowded field (6+ papers in 2025-2026); low novelty |
| Dirichlet FM for CDRs | Too niche for first paper; better as follow-up |

---

## 5. Key References

### Foundational Methods
- Lipman et al. (2022) "Flow Matching for Generative Modeling" — conditional flow matching framework
- Stark et al. (ICML 2024) "Dirichlet Flow Matching with Applications to DNA Sequence Design" — arXiv:2402.05841
- Ho & Salimans (2022) "Classifier-Free Diffusion Guidance" — arXiv:2207.12598

### Inverse Folding Baselines
- Dauparas et al. (Science 2022) "Robust deep learning–based protein sequence design using ProteinMPNN" — PMID:36108050
- Hsu et al. (2022) "Learning inverse folding from millions of predicted structures" (ESM-IF) — arXiv:2208.01112
- Zheng et al. (2023) "Structure-informed Language Models Are Protein Designers" (LM-Design) — arXiv:2302.01649

### Structure Prediction (for evaluation)
- Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure with a language model" (ESMFold)
- Jumper et al. (Nature 2021) "Highly accurate protein structure prediction with AlphaFold"

### Flow Matching for Biology
- Bose et al. (ICLR 2024) "SE(3)-Stochastic Flow Matching for Protein Backbone Generation" (FoldFlow)
- Jing et al. (ICML 2024) "AlphaFold Meets Flow Matching for Generating Protein Ensembles" (AlphaFlow)
- Campbell et al. (2024) "Generative Flows on Discrete State-Spaces" — CTMC-based discrete flows

### Boltz Ecosystem
- Wohlwend et al. (2024) "Boltz-1: Democratizing Biomolecular Interaction Modeling" — bioRxiv:2024.11.19.624167
- Passaro, Corso et al. (2025) "Boltz-2: Towards Accurate and Efficient Binding Affinity Prediction" — bioRxiv:2025.06.14.659707
- Stark et al. (2025) "BoltzGen: Toward Universal Binder Design" — bioRxiv:2025.11.20.689494

### Guidance Methods
- Dhariwal & Nichol (2021) "Diffusion Models Beat GANs on Image Synthesis" — classifier guidance
- Bansal et al. (2023) "Universal Guidance for Diffusion Models" — training-free guidance
- Liu et al. (ECCV 2022) "Compositional Visual Generation with Composable Diffusion Models" — NOT operator

### Competitive Landscape (Inference-Time Guidance)
- Shtrikman et al. (May 2026) "AIMS-Fold" — guides Boltz-2 diffusion with experimental data (arXiv:2605.26192)
- Jian et al. (2024) "BADGER" — affinity-guided small molecule diffusion (arXiv:2406.16821)
- Hartman et al. (Nov 2025) "Feynman-Kac Steering" — particle-based guidance for RFdiffusion (arXiv:2511.09216)
- Cremer et al. (Oct 2025) "FLOWR.root" — joint ligand generation + affinity (arXiv:2510.02578)

---

## Appendix: Compute Requirements Comparison

| Project | Min GPU | VRAM | Storage | Wall-Clock | Novelty | Scoop Risk |
|---------|---------|------|---------|------------|---------|------------|
| **Dirichlet FM (Inverse Fold)** | **RTX 3090** | **8-16GB** | **5GB** | **4-6 weeks** | **Med-High** | **Low** |
| Dirichlet FM (CDR) | RTX 4070 | 6-12GB | 2GB | 2-3 weeks | Med-High | Low |
| Specificity (NOT) | RTX 3090 | 15-24GB | 10GB | 2-4 weeks | High | Low |
| Multi-State | RTX 3090 | 12-20GB | 10GB | 3-4 weeks | High | Low |
| Latent Dynamics | A100 40GB | 30-40GB | 500GB | 4-6 weeks | High | Low |
| Coevolution | A100 40GB | 30-40GB | 100GB | 4-6 weeks | High | Low-Med |
| Evolutionary Flow | A100 40GB | 30-40GB | 100GB | 3-4 weeks | Moderate | Medium |
| Failures (DPO) | RTX 3090 | 16-24GB | 20GB | 2-3 weeks | Low-Med | High |
| Affinity-Guided Boltz | A100 40GB | 30-40GB | 10GB | 3-4 weeks | Low-Med | High |

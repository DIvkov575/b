# Literature Review: Compositional Guidance for Selectivity-Aware Molecular Generation

## Critical Finding: ActivityDiff (Zhou et al., 2025)

**arXiv:2508.06364** — Uses positive guidance toward desired target + negative guidance away from off-targets for selective molecular generation. Code NOT released. This is the closest prior work.

### Differentiation from ActivityDiff

| Aspect | ActivityDiff | Our Proposal |
|--------|-------------|--------------|
| Framing | Task-specific (activity guidance) | General compositional algebra (NOT/AND/OR) |
| Scope | Binary activity classifiers | Full selectivity ratio optimization |
| Composability | Two signals (on/off) | Arbitrary boolean logic over properties |
| Mathematical basis | Ad-hoc positive/negative | Principled from Composable Diffusion (Liu et al. 2022) |
| Code | Not released | Open-source |
| Evaluation | Unknown benchmarks | Kinase selectivity panels (Davis, Klaeger) with S-score, Gini, entropy |

---

## Theme 1: Compositional Score Algebra (Foundation)

| Paper | Year | Venue | Contribution | Relevance |
|-------|------|-------|-------------|-----------|
| Liu et al. — Composable Diffusion | 2022 | ECCV | AND (score addition), OR (mixture), NOT (subtraction) for images | Mathematical foundation |
| Ho & Salimans — CFG | 2022 | NeurIPS-W | Implicit composition; negative prompting | Theoretical basis |
| Du et al. — Reduce, Reuse, Recycle | 2023 | ICML | MCMC correction needed for composition | Critical engineering insight |
| Bansal et al. — Universal Guidance | 2023 | CVPR | Any differentiable network as guidance | Our paradigm |
| Gandikota et al. — Erasing Concepts | 2023 | ICCV | Score subtraction baked in via fine-tuning | Extension |

**Gap:** Full compositional algebra never applied to molecular generation.

---

## Theme 2: SBDD Diffusion Models (Base Generators)

| Model | Year | Venue | Checkpoints | License |
|-------|------|-------|-------------|---------|
| **TargetDiff** | 2023 | ICLR | Google Drive | MIT |
| **DiffSBDD** | 2024 | Nat Comp Sci | 8 models (Zenodo) | MIT |
| **DecompDiff** | 2023 | ICML | Google Drive | CC BY-NC 4.0 |
| **MolCRAFT** | 2024 | ICML | HuggingFace | CC BY-NC-SA |
| **OMTRA** | 2025 | — | Available | Apache 2.0 |

**Best for us:** TargetDiff (MIT, BADGER validated on it, checkpoints available).

---

## Theme 3: Guidance in Molecular Diffusion

| Paper | Year | Method | Relevance |
|-------|------|--------|-----------|
| **EEGSDE** (Bao et al.) | 2023 | Energy-guided SDE, linear combination | AND only; no NOT |
| **BADGER** (Jian et al.) | 2024 | Classifier + CFG on TargetDiff | Direct template; open-source |
| **ActivityDiff** (Zhou et al.) | 2025 | ± classifier guidance | Closest competitor; unreleased |
| **NOS** (Gruver et al.) | 2023 | Hidden-state guidance (proteins) | AND in discrete diffusion |
| **SVDD** (Li et al.) | 2024 | Value-based (derivative-free) | Alternative for non-differentiable rewards |
| **DiffSBDD negative design** | 2024 | Modified sampling | NOT for one pocket; not compositional |

**Key from BADGER:** Clean-data classifiers > noise-aware classifiers. Simplifies our pipeline.

---

## Theme 4: Kinase Selectivity Data

| Dataset | Compounds | Kinases | Measurement | Access |
|---------|-----------|---------|-------------|--------|
| Davis | 72 | 442 | Kd | TDC Python API |
| Klaeger | 243 | 300+ | Kd_app | ProteomicsDB |
| KIBA | 2,068 | 229 | Integrated | TDC Python API |
| PKIS2 | 645 | 403 | %Control 1µM | Public |
| KCGS v2.0 | 295 | 262 | Kd + %I | Zenodo |
| Wu et al. | 141,086 | 354 | Mixed | KIPP platform |

**Selectivity metrics:**
- S(10) < 0.01 = highly selective
- Gini > 0.7 = selective
- IC50 ratio > 100× = selective
- pIC50 window > 2 = 100× selective

---

## Theme 5: Selectivity-Aware Generation

| Method | Year | Approach | Limitation |
|--------|------|----------|------------|
| REINVENT 4 | 2024 | SMILES RL + multi-component scoring | No 3D in generator |
| DrugEx v2/v3 | 2021 | Pareto RL | SMILES-based |
| Tan et al. | 2022 | Generate-then-filter (S(10)=0.002) | Selectivity is filter, not objective |
| CMD-GEN | 2025 | Differential pharmacophores | Hierarchical, not end-to-end |
| ActivityDiff | 2025 | ± classifier guidance | Code unreleased; unknown benchmarks |
| FLOWR.root | 2025 | Flow + importance sampling (CK2α/CLK3) | Not released |

---

## Honest Novelty Assessment

Pure "NOT for molecules" is not fully novel (ActivityDiff 2025). Our contribution must be:

1. **General compositional algebra** — NOT/AND/OR/CONDITIONAL as a unified framework
2. **Rigorous kinase selectivity benchmarks** — first diffusion method evaluated on Davis/Klaeger with S-score/Gini/entropy
3. **Open-source implementation** — ActivityDiff code unreleased; we provide reproducible baselines
4. **MCMC-corrected composition** — Du et al. 2023 showed naive composition fails; never applied to molecules
5. **Methodology framing** — "Compositional Diffusion Guidance for Molecular Design" > "negative guidance for kinases"

---

## Key References

### Foundational
- Liu et al. (ECCV 2022) Composable Diffusion Models
- Ho & Salimans (2022) Classifier-Free Diffusion Guidance
- Du et al. (ICML 2023) Reduce, Reuse, Recycle
- Bansal et al. (CVPR 2023) Universal Guidance
- Dhariwal & Nichol (NeurIPS 2021) Diffusion Models Beat GANs

### SBDD Models
- Guan et al. (ICLR 2023) TargetDiff — arXiv:2303.03543
- Schneuing et al. (Nat Comp Sci 2024) DiffSBDD — arXiv:2210.13695
- Guan et al. (ICML 2023) DecompDiff — arXiv:2403.07902
- Hoogeboom et al. (ICML 2022) EDM — E(3) Equivariant Diffusion

### Guidance
- Jian et al. (JCIM 2026) BADGER — arXiv:2406.16821
- Bao et al. (ICLR 2023) EEGSDE — arXiv:2209.15408
- Zhou et al. (2025) ActivityDiff — arXiv:2508.06364
- Li et al. (2024) SVDD — arXiv:2408.08252

### Selectivity
- Davis et al. (Nat Biotech 2011) — kinase Kd panel
- Klaeger et al. (Science 2017) — kinase proteomics
- Tan et al. (J Med Chem 2022) — DDR1 selective design
- Zou et al. (Commun Biol 2025) CMD-GEN — differential pharmacophores
- Loeffler et al. (J Cheminform 2024) REINVENT 4

### Evaluation
- Karaman et al. (Nat Biotech 2008) — S-score
- Graczyk (J Med Chem 2007) — Gini coefficient
- Uitdehaag & Zaman (Br J Pharmacol 2012) — selectivity entropy

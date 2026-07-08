# Multi-Target Protein Binder Design via Diffusion Models

## Abstract

De novo protein binder design has achieved remarkable success for single targets, yet no computational method generates a single protein scaffold that simultaneously engages multiple independent protein targets. We present the first demonstration of multi-target binder design using structure-based diffusion models. By conditioning RFdiffusion on hotspot residues from two separate target proteins simultaneously, we generate novel protein backbones with binding interfaces for both targets on distinct surfaces. We evaluate designs using AF2-Multimer for each binder-target pair independently and verify geometric compatibility (no steric clash when both targets are simultaneously bound). Our approach requires no model retraining — it exploits the existing multi-chain conditioning capability of RFdiffusion in a configuration never before demonstrated. We validate on well-characterized target pairs and show that diffusion-generated multi-target binders achieve favorable predicted binding metrics (pae_interaction < 10) for both targets simultaneously, opening the door to computationally designed bispecific proteins, protein-based molecular glues, and synthetic hub proteins.

---

## Unique Value Proposition

First de novo backbone generation of proteins designed to bind multiple independent targets simultaneously — using existing diffusion infrastructure with no retraining.

---

## Highlights

1. **Genuinely unprecedented**: No published method generates a novel protein backbone that simultaneously binds 2+ independent protein targets. ProDualNet (2025) does dual-target *sequence* design on fixed backbones; all bispecific proteins are modular fusions of separately-discovered binders. We generate holistic, de novo backbones.

2. **Zero retraining**: Exploits RFdiffusion's existing multi-chain contig system and hotspot conditioning in a configuration never demonstrated. Pure inference-time innovation.

3. **Biologically relevant**: Natural hub proteins (14-3-3, calmodulin, p53) bind dozens of partners through multiple interfaces. We computationally design this capability from scratch.

4. **Clear evaluation pipeline**: AF2-Multimer predicts binding to each target independently; geometric compatibility verified by steric clash analysis of the ternary assembly.

5. **Extensible**: Once demonstrated, the approach generalizes to 3+ targets, conditional multi-target design (bind A only when B is present), and designed protein-based molecular glues.

6. **Low compute barrier**: Inference-only on single A100. No training data needed. ~$200-500 in cloud GPU for a full design campaign.

---

## Novelty Confirmation (Exhaustive Lit Review June 2026)

### This has NOT been done:
- No paper generates a de novo backbone binding 2+ independent targets
- RFdiffusion documentation and all Baker lab publications show single-target binder design only
- BindCraft, Chroma, Genie 3, EvoDiff — none demonstrate multi-target
- All bispecific proteins (DARPins, multibodies) use modular fusion of separate binders

### Closest prior work:

| Paper | Year | What It Does | Why We're Different |
|-------|------|-------------|---------------------|
| **ProDualNet** (Liu et al., Brief Bioinform 2025) | 2025 | Dual-target SEQUENCE design (fixed backbone) | We generate novel BACKBONES |
| **pMHC binders** (Liu et al., Science 2025) | 2025 | Binder contacts both peptide + MHC in a pre-formed complex | We target two INDEPENDENT proteins |
| **Sahtoe et al.** (Science 2022, Baker lab) | 2022 | Multi-interface building blocks for assemblies | Each interface designed independently, not holistically |
| **Zhang et al.** (2026) | 2026 | HDAC6/EZH2 dual peptides via RFdiffusion | Designed SEPARATE peptides then FUSED with linker |
| **Saragovi et al.** (bioRxiv 2025, Baker lab) | 2025 | Dual-interface proteins binding metal oxides | Interfaces for inorganic materials, not protein targets |

### Validates the problem matters:
- ProDualNet's existence (2025) proves the community recognizes dual-target design as important
- RFdiffusion GitHub Issue #149: user asked "Can a binder span two chains?" — community demand exists
- Baker lab 2026 Nature review: identifies "multistate protein systems" as near-term goal

---

## Technical Approach

### Method

```
Input: Target_A (PDB) + Target_B (PDB) + desired binder length (80-150 residues)

1. Place Target_A and Target_B in spatial arrangement
   (separated by ~40-80Å, non-overlapping, various relative orientations)

2. Specify dual hotspot residues:
   ppi.hotspot_res=[A30,A33,A34,B15,B20,B25]
   (3-6 residues per target, chosen from known binding interfaces or surface-exposed residues)

3. Run RFdiffusion with contig:
   A1-100/0 B1-100/0 80-150
   (fix both targets, generate binder of 80-150 residues)

4. Generate 10,000-50,000 backbone designs

5. Design sequences with ProteinMPNN (multi-chain mode)

6. Evaluate:
   a. AF2-Multimer(binder + Target_A) → pae_interaction_A
   b. AF2-Multimer(binder + Target_B) → pae_interaction_B
   c. Steric check: superimpose binder, verify Target_A and Target_B don't clash
   d. Pass criteria: pae_interaction < 10 for BOTH, no steric clash
```

### Key Design Decisions

**Target placement**: The relative geometry of targets matters. We sample multiple orientations (same-face, opposite-face, 90°, 120°) and filter post-generation.

**Binder size**: Natural bispecific interactions require ~1500Å² per interface. A 100-150 residue protein has ~6000-8000Å² surface area — enough for two non-overlapping interfaces using 40-50% of available surface.

**Hit rate expectation**: Standard single-target RFdiffusion yields ~1-10% success. Multi-target adds a multiplicative constraint. Expected: ~0.01-0.1% (1 in 1000-10000). Hence generating 50,000 designs.

### Fallback if hit rate too low

1. **Add auxiliary potentials**: Use `olig_contacts` potential on both target chains simultaneously
2. **Partial diffusion from hub proteins**: Start from natural multi-interface scaffolds (14-3-3, calmodulin) and diversify
3. **Two-stage approach**: Design binder for target_A first, then partial-diffuse to add interface for target_B
4. **Fine-tuning** (last resort): Create synthetic training data from pairs of binary complexes sharing a mediator

---

## Evaluation Metrics

| Metric | Threshold | What It Measures |
|--------|-----------|-----------------|
| pae_interaction (target A) | < 10 | Predicted binding to target A |
| pae_interaction (target B) | < 10 | Predicted binding to target B |
| Binder pLDDT | > 70 | Structural confidence of designed binder |
| Interface SASA (A) | > 800 Å² | Sufficient binding interface area |
| Interface SASA (B) | > 800 Å² | Sufficient binding interface area |
| Steric clash score | < 5 clashes | Geometric compatibility of ternary complex |
| Rosetta ddG (each interface) | < -10 REU | Favorable binding energy |
| ProteinMPNN recovery | > 30% | Designability of the backbone |

---

## Datasets / Resources

| Resource | What | Access |
|----------|------|--------|
| RFdiffusion | Backbone generator | Open source (BSD), GitHub |
| ProteinMPNN | Sequence design | Open source, GitHub |
| AF2-Multimer | Binding prediction | Open source (ColabFold) |
| dl_binder_design | Evaluation pipeline | Open source, GitHub (nrbennet) |
| PDB hub proteins | Natural multi-interface examples for validation | RCSB PDB |
| 14-3-3 complexes | Multi-partner scaffold reference | PDB (~50 structures) |
| Target pairs for testing | Well-characterized proteins with known surfaces | PDB (see below) |

### Proposed Target Pairs for Validation

| Pair | Why | Surface geometry |
|------|-----|-----------------|
| IL-2 + IL-15 | Similar cytokines, well-characterized | Both globular, ~15kDa |
| PD-1 + CTLA-4 | Both immune checkpoints, clinical relevance | Ig-domains, flat surfaces |
| EGFR + HER2 | Both receptor tyrosine kinases | Extracellular domains |
| GFP + mCherry | Easy to express and test binding | Both barrels, orthogonal |
| Ubiquitin + SUMO | Small, well-characterized, distinct surfaces | Both ~8-12 kDa |

---

## Compute Budget

| Task | GPU-hours | Cost @ $1.50/hr |
|------|-----------|-----------------|
| RFdiffusion generation (50K designs × 3 target pairs) | 50-100 | $75-150 |
| ProteinMPNN sequence design | 5-10 | $8-15 |
| AF2-Multimer evaluation (2 × top 1000 per pair) | 50-150 | $75-225 |
| Steric clash analysis | <1 | <$2 |
| **Total** | **~100-260** | **$150-400** |

---

## Timeline

| Week | Milestone |
|------|-----------|
| 1 | Set up RFdiffusion + dl_binder_design pipeline. Test dual-hotspot conditioning on one target pair. Confirm designs contact both targets. |
| 2-3 | Scale generation to 50K designs per target pair. Run ProteinMPNN + AF2 evaluation pipeline. |
| 4-5 | Filter results. Analyze hit rates. If <0.01%: implement fallback (potentials, partial diffusion). |
| 6-8 | Full evaluation across 3 target pairs. Ablations (binder length, hotspot density, target geometry). |
| 9-10 | Analysis, figures, writing. |
| 11-12 | Paper submission. |

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Hit rate too low (<0.001%) | HIGH | Auxiliary potentials; partial diffusion from hub scaffolds; two-stage design |
| Model only contacts one target | HIGH | Filter for dual-interface contact; increase hotspot density on weaker target |
| AF2-Multimer unreliable for designed ternary complexes | MEDIUM | Use Boltz-2 as orthogonal validator; report uncertainty |
| Baker lab publishes this first | MEDIUM | Move fast; paper-ready in 12 weeks; preprint immediately |
| Designed interfaces are weak (Kd > μM) | MEDIUM | Optimize with Rosetta FastRelax; iterate with ProteinMPNN |
| Steric clash between bound targets | LOW | Sample diverse target geometries; filter post-hoc |

---

## Publication Strategy

**Target venues (in order of preference):**
1. Nature Methods / Nature Biotechnology — if hit rates are good + experimental validation
2. ICML / NeurIPS GenBio workshop — fast, establishes priority
3. bioRxiv preprint immediately upon first positive results — priority claim
4. ICLR 2027 main conference — if comprehensive computational + any experimental validation

**Paper framing:**
"Multi-Target Protein Binder Design with Structure Diffusion Models" — demonstrate the capability, characterize when it works/fails, provide the community with a reproducible protocol.

---

## Key References

- Watson et al. (Nature 2023) RFdiffusion — backbone diffusion for binder design
- Liu et al. (Brief Bioinform 2025) ProDualNet — dual-target sequence design (our baseline/comparison)
- Liu et al. (Science 2025) pMHC binders — multi-chain target complex binding with RFdiffusion
- Sahtoe et al. (Science 2022) — multi-interface protein building blocks
- Yang, Wang, Baker (Nature 2026) — review identifying multistate systems as frontier
- Dauparas et al. (Science 2022) ProteinMPNN — sequence design
- Bennett et al. (Nature 2026) — antibody binder design with RFdiffusion

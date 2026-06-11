# Multi-Target Protein Binder Design with Structure Diffusion Models

---

## 1. Introduction

### 1.1 Problem

Designing proteins that bind a single target is now routine — RFdiffusion (Watson et al., Nature 2023) generates de novo binders with ~1-10% computational success rates validated by AlphaFold2-Multimer. However, no existing method generates a single protein scaffold that simultaneously engages two or more independent protein targets.

This capability would enable:
- **Bispecific therapeutics** without modular fusion (smaller, more stable, cooperative binding)
- **Protein-based molecular glues** that template interactions between two proteins
- **Synthetic hub proteins** for programmable signaling circuits
- **Multi-component assembly connectors** designed holistically rather than interface-by-interface

### 1.2 Current State of the Art

| Method | Capability | Limitation |
|--------|-----------|------------|
| RFdiffusion | De novo binder for ONE target | Single-target hotspot conditioning only |
| ProteinMPNN | Sequence design for multi-chain complexes | Fixed backbone required; no structure generation |
| ProDualNet (Liu et al., 2025) | Dual-target sequence design | Fixed backbone; no de novo topology |
| Bispecific antibodies | Dual binding | Modular fusion of separate domains, not holistic design |
| Chroma | Composable protein conditioning | Multi-target binding not demonstrated |

### 1.3 Our Contribution

We demonstrate that RFdiffusion's existing multi-chain conditioning architecture — designed for binding pre-formed complexes — can be repurposed for simultaneous multi-target binder design by specifying hotspot residues on two independent targets. This requires:
- No model retraining
- No architectural modification
- Only inference-time configuration changes

We characterize when this approach succeeds, when it fails, and what factors (target geometry, binder length, hotspot density) control the dual-binding hit rate.

---

## 2. Background

### 2.1 RFdiffusion

RFdiffusion (Watson et al., 2023) is a denoising diffusion model over SE(3) residue frames, fine-tuned from RoseTTAFold. It generates protein backbones by iteratively denoising from random coordinates over T timesteps.

**Binder design mode:** Given a target protein structure and hotspot residues (desired contact points), RFdiffusion generates binder backbones that make contacts at specified positions. The target is fixed context; the binder is generated.

**Multi-chain support:** The contig map syntax supports multiple fixed chains:
```
A1-76/0 B1-97/0 100-150
```
This specifies: fix chain A (76 residues), fix chain B (97 residues), generate a new chain of 100-150 residues.

**Hotspot conditioning:** `ppi.hotspot_res=[A8,A44,B35,B37]` — specifies residues on target chains that the generated binder should contact. During training, 0-20% of true interface residues were provided as hotspots.

### 2.2 Why Multi-Target Should Work (In Principle)

1. **The contig system already handles multiple fixed chains.** Specifying two independent target proteins as two fixed chains is syntactically valid.
2. **Hotspots from both chains can be specified simultaneously.** The model sees conditioning from both targets.
3. **Natural hub proteins exist.** 14-3-3 (~250 residues) binds 200+ partners via multiple distinct interfaces. Calmodulin (~148 residues) binds different targets on two lobes. The protein physics allows multi-interface scaffolds.
4. **Geometric feasibility:** A 120-residue binder has ~7000Å² solvent-accessible surface. Two interfaces of ~1200Å² each consume ~35% — leaving sufficient surface for structural integrity.

### 2.3 Why It Might Fail

1. **Training distribution bias:** RFdiffusion was trained on single-target binder complexes. It may have never seen simultaneous dual-interface backbones during training.
2. **Surface allocation competition:** Building one interface may consume residues/surface that the other interface needs.
3. **Geometric constraint:** Both targets must fit around the binder without sterically clashing with each other.
4. **Hit rate collapse:** If P(binds_A) = 5% and P(binds_B | binds_A) < P(binds_B), the joint probability could be vanishingly small.

---

## 3. Method

### 3.1 Overview

```
Input: Two target PDB structures + hotspot residues for each
  ↓
Stage 1: Place targets in shared coordinate frame (separation ~60Å)
  ↓
Stage 2: Run RFdiffusion with dual-hotspot contig (generate N=50,000 backbones)
  ↓
Stage 3: Filter by dual contact (keep designs with ≥5 Cα contacts to EACH target at 8Å)
  ↓
Stage 4: Sequence design with ProteinMPNN (multi-chain mode, 4 seqs/backbone)
  ↓
Stage 5: Validate with AF2-Multimer (score binder+A and binder+B independently)
  ↓
Stage 6: Steric compatibility check (verify targets don't clash when both bound)
  ↓
Output: Ranked designs with dual binding evidence
```

### 3.2 Target Placement

Two target proteins are placed in a shared coordinate frame:
- Target A centered at origin
- Target B placed along the x-axis at distance `d` (default 60Å)
- Multiple relative orientations sampled (identity + 4 random rotations of target B)

The separation distance `d` is chosen such that:
- `d > radius_A + radius_B` (no initial clash)
- `d < radius_A + radius_B + 2 * binder_radius` (binder can bridge the gap)

### 3.3 Hotspot Selection

Hotspot residues are selected per target:
- **Manual:** Known binding epitopes from literature (e.g., ubiquitin Ile44 patch: residues 8, 44, 48, 63, 68)
- **Automatic:** Surface-exposed residues (low neighbor count at 10Å radius) sampled with spatial diversity

3-6 hotspots per target, specified as `ppi.hotspot_res=[A8,A44,A48,B35,B37,B39]`.

### 3.4 Generation

RFdiffusion command:
```bash
python run_inference.py \
  inference.input_pdb=combined_target.pdb \
  'contigmap.contigs=[A1-76/0 B1-231/0 100-150]' \
  'ppi.hotspot_res=[A8,A44,A48,A63,A68,B35,B37,B39,B55,B57]' \
  inference.output_prefix=outputs/design \
  inference.num_designs=50000 \
  diffuser.T=50
```

**Key parameters:**
- Binder length: 100-150 residues (large enough for two interfaces)
- Diffusion steps: T=50 (standard for binder design)
- Number of designs: 50,000 (compensating for expected low hit rate)

### 3.5 Contact Filtering

For each generated backbone, compute Cα-Cα contacts with each target:
- Contact = any binder Cα within 8Å of a target Cα
- Pass criterion: ≥5 contacts with target A AND ≥5 contacts with target B

This is a fast geometric filter (seconds per design) that eliminates the majority of single-target-only binders.

### 3.6 Sequence Design

ProteinMPNN (Dauparas et al., 2022) designs sequences for passing backbones:
- Binder chain (C) is designable
- Target chains (A, B) are fixed context
- 4 sequences per backbone at temperature 0.1

### 3.7 Validation

**AF2-Multimer scoring (per design, two predictions):**
1. AF2-Multimer(binder_sequence + target_A_sequence) → pae_interaction_A
2. AF2-Multimer(binder_sequence + target_B_sequence) → pae_interaction_B

Pass criterion: pae_interaction < 10 for BOTH predictions.

**Steric compatibility:**
Superimpose binder backbone from both AF2 predictions. Place target A and target B at their predicted binding positions. Count atom pairs with distance < 2.0Å between the two targets. Pass: ≤5 clashes.

### 3.8 Ranking

Designs passing all filters are ranked by composite score:
```
score = (pae_A + pae_B) - 0.1 × (contacts_A + contacts_B)
```
Lower score = better (low PAE + high contacts).

---

## 4. Experimental Design

### 4.1 Target Pairs

| Pair | Targets | Rationale |
|------|---------|-----------|
| Ubiquitin + SUMO | 1UBQ (76 aa) + 2BF8 (231 aa) | Small, well-characterized, distinct binding surfaces |
| PD-1 + CTLA-4 | 3RRQ + 3OSK | Clinical relevance (immune checkpoints), Ig-fold geometry |
| GFP + mCherry | 1EMA + 2H5Q | Easy expression, fluorescence-based binding assay possible |

### 4.2 Controls

- **Positive control:** Single-target binder design (standard RFdiffusion) — establishes baseline hit rate per target
- **Negative control:** Random backbone generation without hotspots — establishes false positive rate of contact filter
- **Comparison:** ProDualNet (fixed-backbone dual-target sequence design) — establishes the bar for the sequence-only approach

### 4.3 Ablations

| Variable | Values | Question |
|----------|--------|----------|
| Binder length | 80, 100, 120, 150 residues | Minimum size for dual interface |
| Target separation | 40, 50, 60, 80 Å | Optimal bridging distance |
| Hotspots per target | 3, 5, 8 | Stronger conditioning vs over-constraint |
| Diffusion steps | 25, 50, 100 | Quality vs speed tradeoff |
| Orientation | 5 random rotations | Geometry dependence |

### 4.4 Metrics

| Metric | Definition | Success Threshold |
|--------|-----------|-------------------|
| Dual-contact hit rate | % of designs with ≥5 contacts to each target | >0.1% (>50 hits from 50K designs) |
| AF2 dual-binding rate | % passing pae_interaction < 10 for both targets | >0.01% (>5 hits from 50K) |
| Steric compatibility | % of AF2-passing designs with <5 target-target clashes | >50% of AF2-passing |
| Interface area (per target) | Buried SASA at each interface | >800 Å² |
| Designability | ProteinMPNN sequence recovery after AF2 refolding | scTM > 0.7 |

### 4.5 Failure Modes and Fallbacks

| Failure | Detection | Fallback |
|---------|-----------|----------|
| 0 designs contact both targets | Stage 3 returns empty set after 50K | Add `olig_contacts` potential; increase hotspot count |
| Hit rate < 0.01% | Insufficient passing designs for statistics | Partial diffusion from hub protein scaffolds |
| AF2 doesn't validate any design | Stage 5 returns no passes | Relax threshold to pae < 15; try Boltz-2 as alternative validator |
| Targets always clash | Stage 6 fails most designs | Sample more orientations; increase separation |

---

## 5. Expected Outcomes

### 5.1 Optimistic Scenario
- Dual-contact hit rate: ~1-5% (500-2500 designs from 50K)
- AF2 validation: ~10% of contact-passing designs (50-250 final candidates)
- Clear demonstration that diffusion models can generate multi-target binders without retraining

### 5.2 Realistic Scenario
- Dual-contact hit rate: ~0.1-1% (50-500 designs from 50K)
- AF2 validation: ~5% of contact-passing (3-25 final candidates)
- Paper demonstrates feasibility with caveats (specific geometries work, others don't)

### 5.3 Pessimistic Scenario
- Dual-contact hit rate: <0.01% (<5 designs from 50K)
- Requires fallback strategies (potentials, partial diffusion)
- Paper becomes "characterizing the limits of multi-target design" + successful fallback

### 5.4 What Makes This Publishable in Each Scenario

All three scenarios produce a paper because:
1. **First systematic attempt** at multi-target binder generation with diffusion models
2. **Characterization of the design space** (what target geometries work, what sizes are needed, how hit rate scales with parameters)
3. **Open protocol** others can use immediately (no retraining, just contig configuration)

---

## 6. Compute Budget

| Stage | GPU-hours | Cost @ $1.50/hr |
|-------|-----------|-----------------|
| RFdiffusion (50K × 3 pairs × 5 orientations) | 150-300 | $225-450 |
| ProteinMPNN (on filtered subset, ~1000 designs) | 2-5 | $3-8 |
| AF2-Multimer (top 200 × 2 targets) | 30-60 | $45-90 |
| **Total** | **~180-370** | **$270-550** |

---

## 7. Timeline

| Week | Milestone |
|------|-----------|
| 1 | Pilot run: 5K designs, one pair, one orientation. Measure dual-contact rate. Go/no-go gate. |
| 2-3 | Full generation: 50K designs × 3 pairs. Filter + ProteinMPNN. |
| 4-5 | AF2 evaluation of top candidates. Steric clash analysis. |
| 6-8 | Ablations (binder length, separation, hotspot density). |
| 9-10 | Analysis, figures, writing. |
| 11-12 | Paper submission + preprint. |

---

## 8. Related Work

### De Novo Protein Binder Design
- Watson et al. (Nature 2023) — RFdiffusion: structure diffusion for binder design
- Bennett et al. (Nature 2026) — Antibody binder design with RFdiffusion
- Dauparas et al. (Science 2022) — ProteinMPNN: inverse folding

### Dual-Target / Bispecific Design
- Liu et al. (Brief Bioinform 2025) — ProDualNet: dual-target sequence design (fixed backbone, GNN + ESM2)
- Zhang et al. (Comb Chem HTS 2026) — RFdiffusion peptides fused with linker for HDAC6/EZH2 (NOT holistic)
- Sahtoe et al. (Science 2022) — Multi-interface building blocks (each interface designed independently)

### Multi-Interface Natural Proteins
- 14-3-3 scaffold proteins: bind 200+ phosphopeptide partners via conserved groove
- Calmodulin: two-lobed structure binding different targets on each lobe
- p53: 100+ interaction partners through multiple distinct surfaces

### Protein Design Tools
- ProteinMPNN (Dauparas et al., 2022) — sequence design given backbone
- AF2-Multimer (Evans et al., 2022) — complex structure prediction for validation
- dl_binder_design (Bennett lab) — evaluation pipeline for designed binders

---

## 9. Limitations

1. **No experimental validation** in this work — purely computational proof-of-concept
2. **AF2-Multimer reliability** for designed ternary complexes is untested
3. **Binding affinity** likely weak (μM range) due to split surface area — affinity maturation needed
4. **Target pair geometry** constrains which pairs are feasible — not all combinations will work
5. **Single-model generalization** — RFdiffusion was not trained for this task; we exploit emergent capability

---

## 10. Broader Impact

If successful, this work demonstrates that structure diffusion models possess latent multi-target design capability accessible through inference-time configuration alone. This implies:
- The protein physics learned by RFdiffusion is general enough to handle multi-interface design without explicit training
- A single model can be repurposed for qualitatively different design tasks through creative conditioning
- The gap between "what the model can do" and "what has been demonstrated" may be larger than assumed for other design tasks

This opens exploration of other undocumented RFdiffusion capabilities: conditional binding (bind A only when B is present), allosteric switches, self-assembling multi-component systems.

# High-Level Design: Multi-Target Protein Binder Generation

## 1. Objective

Generate de novo protein scaffolds that simultaneously bind two independent protein targets using RFdiffusion's existing multi-chain conditioning — no model retraining required. Validate computationally with AF2-Multimer.

## 2. Hypothesis

RFdiffusion's denoising diffusion over SE(3) frames, when conditioned on hotspot residues from two co-placed target chains, will produce binder backbones that form interfaces with both targets. The model was never explicitly trained for this, but the physics of multi-interface proteins (14-3-3, calmodulin) is represented in its training distribution.

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Pipeline (src/pipeline.py)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────┐   ┌─────────────┐   ┌──────────┐   ┌──────────────┐  │
│  │ Stage 0  │──▶│   Stage 1   │──▶│ Stage 2  │──▶│   Stage 3    │  │
│  │ Validate │   │Place Targets│   │RFdiffusion│   │Contact Filter│  │
│  │ Hotspots │   │  (CPU)      │   │  (GPU)    │   │   (CPU)      │  │
│  └──────────┘   └─────────────┘   └──────────┘   └──────────────┘  │
│                                          │                 │          │
│                                    5K backbones     dual-contact      │
│                                                     survivors         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐    │
│  │   Stage 4    │──▶│   Stage 5    │──▶│       Stage 6        │    │
│  │ProteinMPNN   │   │AF2-Multimer  │   │ Steric Clash + Rank  │    │
│  │  (GPU)       │   │   (GPU)      │   │      (CPU)           │    │
│  └──────────────┘   └──────────────┘   └──────────────────────┘    │
│                                                      │               │
│                                              ranked designs           │
│                                         (CSV + JSON in outputs/)      │
└─────────────────────────────────────────────────────────────────────┘
```

## 4. Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target placement | Fixed separation (60Å) along x-axis + multi-orientation sampling | Simple geometry; orientations capture rotational degrees of freedom |
| Binder length | 100–150 residues | Surface budget analysis: two interfaces of ~1200Å² each requires ≥100 residues |
| Hotspot count | 5 per target | Matches RFdiffusion training regime (0–20% of interface residues as conditioning) |
| Contact threshold | 8Å Cα–Cα | Standard PPI contact cutoff; generous enough to avoid false negatives |
| Pilot size | 5K designs | Go/no-go gate; 50× cheaper than full run; >0.1% hit rate ⟹ 5+ hits expected |
| Validation | AF2-Multimer (binder+A, binder+B separately) | Gold-standard computational binding prediction; avoids ternary-complex modeling limitation |

## 5. Inputs & Outputs

### Inputs
- **Target PDB structures** — two protein chains (any size; tested with 76–231 residues)
- **Hotspot residues** — 3–8 residue IDs per target specifying desired contact points
- **Configuration** — separation distance, binder length range, design count, filter thresholds

### Outputs
- **Backbone PDBs** — generated binder structures (chain C) in complex with targets (chains A, B)
- **Sequences** — ProteinMPNN-designed amino acid sequences for passing backbones
- **Rankings** — composite score (PAE + contacts) in `results.csv` and `summary.json`
- **Go/no-go metric** — dual-contact hit rate (% of designs contacting both targets)

## 6. Computational Stages

### Stage 0: Hotspot Validation (CPU, <1s)
Verify that specified hotspot residue numbers exist in the input PDBs. Reject pairs with <3 valid hotspots per target.

### Stage 1: Target Placement (CPU, <1s)
- Center target A at origin
- Place target B at `(separation, 0, 0)` with optional rotation
- Write combined PDB with chain A + chain B
- Generate RFdiffusion contig string: `A1-76/0 B1-231/0 100-150`
- Format hotspot string: `[A8,A44,A48,A63,A68,B35,B37,B39,B55,B57]`

### Stage 2: Backbone Generation (GPU, ~2-4hr for 5K)
- RFdiffusion inference with `Complex_base_ckpt.pt`
- 50 denoising steps per design
- Output: N backbone PDB files (binder chain only, in target frame)

### Stage 3: Dual-Contact Filter (CPU, ~minutes)
- For each design, compute Cα–Cα distance matrix between binder and each target
- Count contacts at 8Å threshold per target
- Pass: ≥N contacts to each target (N=3 for pilot, N=5 for full run)
- This eliminates single-target-only binders (~95–99% expected)

### Stage 4: Sequence Design (GPU, ~minutes)
- ProteinMPNN multi-chain mode
- Fix target sequences (chains A, B); design binder (chain C)
- 4 sequences per backbone at temperature 0.1
- Output: FASTA files for AF2 input

### Stage 5: AF2-Multimer Validation (GPU, ~30min/design)
- Two independent predictions per design: binder+A, binder+B
- Extract predicted aligned error (PAE) at the interface
- Pass: PAE_interaction < 10 for both complexes

### Stage 6: Steric Check & Ranking (CPU, <1s/design)
- Superpose binder from both AF2 predictions
- Check target A vs target B atom distances (clash = <2Å)
- Pass: ≤5 clashing atom pairs
- Rank by: `score = (PAE_A + PAE_B) - 0.1 × (contacts_A + contacts_B)`

## 7. Infrastructure

### Pilot Run
| Resource | Spec | Cost |
|----------|------|------|
| Instance | g5.2xlarge (1× A10G, 24GB VRAM) | ~$0.50–0.80/hr spot |
| Storage | 100GB gp3 EBS | ~$0.08/GB-month |
| Duration | 2–4 hours (5K designs) | **$2–4 total** |
| Software | RFdiffusion + SE3-Transformer + model weights (~5GB) | — |

### Full Run (if pilot succeeds)
| Resource | Spec | Cost |
|----------|------|------|
| Instance | p4d.24xlarge (8× A100) or 8× g5.xlarge | ~$12/hr spot |
| RFdiffusion | 50K designs × 3 pairs × 5 orientations | ~150–300 GPU-hours |
| ProteinMPNN | ~1000 designs × 4 seqs | ~2–5 GPU-hours |
| AF2-Multimer | ~200 designs × 2 predictions | ~30–60 GPU-hours |
| **Total** | | **$270–550** |

### Deployment Flow
```
Local machine                         AWS (g5.2xlarge)
─────────────                         ────────────────
deploy/launch.sh ──spot request──▶  Ubuntu 22.04 DL AMI
                                     │
                                     ▼
                                    deploy/setup_instance.sh
                                     ├── clone RFdiffusion
                                     ├── install SE3-Transformer
                                     ├── download model weights (5GB)
                                     └── pip install project deps
                                     │
                                     ▼
                                    deploy/fetch_pdbs.sh
                                     └── download 1UBQ, 1A5R from RCSB
                                     │
                                     ▼
                                    deploy/run_pilot.sh
                                     └── python3 -m src.pipeline --config configs/pilot.yaml
                                     │
                                     ▼
                                    outputs/ubiquitin_sumo/
                                     ├── designs/*.pdb
                                     ├── results.csv
                                     └── summary.json
```

## 8. Success Criteria

### Pilot (5K designs, go/no-go)
| Metric | Threshold | Action if failed |
|--------|-----------|------------------|
| Dual-contact hit rate | >0.1% (>5 designs) | Proceed to full run |
| Dual-contact hit rate | 0.02–0.1% (1–5 designs) | Add RFdiffusion potentials, re-run |
| Dual-contact hit rate | 0% | Fallback: partial diffusion from hub scaffolds |

### Full Run (50K designs)
| Metric | Threshold |
|--------|-----------|
| Dual-contact hit rate | >0.1% |
| AF2 dual-binding rate | >0.01% (>5 validated designs) |
| Steric compatibility | >50% of AF2-passing |
| Interface area per target | >800Å² buried SASA |

## 9. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RFdiffusion ignores second target's hotspots | Medium | High — 0 dual-contact hits | Add `olig_contacts` potential to force contacts on both chains |
| Binder too small for two interfaces | Medium | High — thin interfaces, no AF2 validation | Increase binder range to 150–200 residues |
| Targets clash sterically in all geometries | Low | Medium — limits applicable pairs | Sample 20+ orientations; increase separation to 80Å |
| AF2-Multimer unreliable for designed ternary complexes | Medium | Medium — false negatives kill good designs | Cross-validate with Boltz-2; relax PAE threshold to 15 |
| Spot instance preempted mid-run | Medium | Low — lost partial progress | Checkpoint every 500 designs; use `--start-from` flag |

## 10. Modules

| File | Responsibility | Lines |
|------|---------------|-------|
| `src/pipeline.py` | Orchestration, CLI entry point | 210 |
| `src/prepare.py` | Target placement, contig/hotspot formatting, PDB cleaning | 182 |
| `src/generate.py` | RFdiffusion CLI wrapper, mock mode | 78 |
| `src/filter.py` | Contact analysis, dual-contact filter | ~50 |
| `src/design.py` | ProteinMPNN wrapper, FASTA I/O | ~100 |
| `src/evaluate.py` | AF2 command builder, steric clash check | ~50 |
| `src/results.py` | CSV/JSON export, ranking | ~80 |
| `src/utils.py` | PDB I/O, coordinate extraction | ~60 |

## 11. Configuration

**`configs/pilot.yaml`** — target pair definition:
```yaml
pairs:
  - name: "ubiquitin_sumo"
    target_a:
      pdb: "data/pdbs/1ubq.pdb"
      chain: "A"
      hotspots: [8, 44, 48, 63, 68]
    target_b:
      pdb: "data/pdbs/1a5r_sumo.pdb"
      chain: "A"
      hotspots: [35, 37, 39, 55, 57]
    separation: 60.0
    binder_length: [100, 150]
```

**`configs/pilot_defaults.yaml`** — pipeline parameters:
```yaml
generation:
  num_designs: 5000
  diffusion_steps: 50
filtering:
  min_contacts_per_target: 3
  contact_distance_threshold: 8.0
evaluation:
  pae_interaction_threshold: 10.0
  max_steric_clashes: 5
```

## 12. Experiment Phases

```
Phase 1: Pilot (this deployment)
├── 5K designs, 1 pair (ubiquitin + SUMO), 1 orientation
├── Measure dual-contact rate → go/no-go
└── Cost: $2–4, Duration: 2–4 hours

Phase 2: Full Generation (if hit rate > 0.1%)
├── 50K designs × 3 pairs × 5 orientations = 750K total
├── Contact filter → ProteinMPNN → AF2
└── Cost: $270–550, Duration: 2–3 weeks

Phase 3: Ablations
├── Vary: binder length, separation, hotspot count, diffusion steps
├── Characterize design space
└── Cost: ~$200 additional

Phase 4: Analysis & Paper
├── Figures, statistical analysis
├── Failure mode characterization
└── Submission target: 12 weeks from pilot
```

## 13. What's Novel

1. **First demonstration** of simultaneous multi-target binder generation with a diffusion model
2. **Zero retraining** — exploits emergent multi-chain capability through inference-time conditioning
3. **Systematic characterization** of when dual-hotspot conditioning works vs. fails
4. **Open protocol** — any lab can replicate with `pip install` + RFdiffusion weights

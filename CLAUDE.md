# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research repo exploring novel ML contributions. Current codebase contains a **concluded** DPP margin-aware subgraph selection experiment (failed — DPP MAE 0.539 worse than baselines). The **active direction** is **Composable Discrete Flows: Boolean Guidance Algebra for Discrete Flow Matching** (spec at `docs/superpowers/specs/2026-06-18-compositional-discrete-guidance-design.md`).

## Commands

```bash
# Run tests (DPP subgraph GNN codebase)
pytest                              # all tests
pytest tests/test_dpp_selector.py   # single file
pytest -k "test_greedy"             # single test by name

# Run ZINC ablation (requires torch-geometric + ZINC dataset)
python -m src.experiments.run_zinc --config configs/zinc.yaml

# Run full multi-seed ablation (4 models × 5 seeds × 100 epochs)
python -m src.experiments.run_full_ablation

# Install
pip install -e .
```

## Architecture (DPP Subgraph Selection — concluded)

ESAN-style bag-of-subgraphs GNN with learned DPP-based selection:

- `src/data/subgraph_policies.py` — subgraph generation (node-del, edge-del, ego)
- `src/data/datasets.py` — ZINC/TU/OGB loaders with subgraph bags
- `src/models/base_gnn.py` — 4-layer GINConv encoder
- `src/models/dpp_selector.py` — DPP kernel (quality + similarity) with greedy MAP and differentiable relaxation
- `src/models/margin_scorer.py` — per-subgraph margin contribution estimator
- `src/models/bag_aggregator.py` — DeepSets/weighted/attention aggregation
- `src/models/dpp_subgraph_gnn.py` — end-to-end model
- `src/training/batched_trainer.py` — mega-batch encoding (12× speedup over naive)
- `src/training/losses.py` — classification + DPP margin loss
- `src/training/margin_utils.py` — margin computation
- `src/baselines/` — ESAN uniform, ESAN full-bag, centrality selection (HyMN-style)
- `src/experiments/` — ablation scripts, selection analysis

## Key Results (DPP — project failed)

ZINC-12K MAE (lower is better): full-bag 0.409 > centrality 0.453 > uniform 0.496 > DPP 0.539.
V2 (LOO supervision) also failed (0.550). Conclusion: diversity-based selection anti-correlated with utility on molecular graphs. See `docs/results_analysis.md`.

## Next Direction: Composable Discrete Flows

Derive AND/NOT/OR composition operators for discrete flow matching. No implementation yet — spec only. Key details:
- Framework-agnostic Boolean algebra over CTMC and probability-path discrete flows
- Phase 1: synthetic validation (K=8→100 categorical sequences)
- Phase 2: real app (protein sequences or small molecules)
- Kill gate: <80% Boolean accuracy at week 3
- Target: ICLR 2027

## Research Context

Full project history and evaluated alternatives in `PROJECT_RESEARCH.md`. Previous directions (all abandoned with rationale): multi-target protein binder, Dirichlet FM inverse folding, specificity-aware design, latent dynamics, coevolutionary generation, multi-state design, evolutionary flow, learning from failures.

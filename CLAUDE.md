# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Composable Discrete Flows: Boolean Guidance Algebra for Discrete Flow Matching.** Derive AND/NOT/OR composition operators for discrete flow matching. Spec at `docs/superpowers/specs/2026-06-18-compositional-discrete-guidance-design.md`. Kill gate: >80% Boolean accuracy on synthetic tasks.

## Commands

```bash
# Run tests
pytest                              # all 25 tests
pytest tests/test_composition.py    # single file
pytest -k "test_compose_and"        # single test by name

# Run full synthetic experiment (K=8, L=32, 100 epochs — ~30min CPU)
python -m src.experiments.run_synthetic --config configs/synthetic.yaml

# Smoke test (K=4, L=8, 10 epochs — ~30sec)
python -m src.experiments.run_synthetic --config configs/smoke.yaml

# Install
pip install -e .
```

## Architecture

Two discrete flow base models + classifiers + composition operators:

- `src/data/synthetic.py` — K-category L-length sequences with 3 Boolean properties
- `src/models/ctmc_flow.py` — CTMC discrete flow (rate matrix denoiser, tau-leaping)
- `src/models/prob_path_flow.py` — probability-path flow (posterior predictor, Euler sampling)
- `src/models/classifier.py` — time-conditional binary classifier p(y|x_t, t)
- `src/guidance/compose.py` — AND/NOT/OR operators (framework-agnostic interface)
- `src/guidance/ctmc_guidance.py` — CTMC-specific guided sampling with per-transition rates
- `src/guidance/prob_path_guidance.py` — prob-path guided posterior modification
- `src/training/train_flow.py` — training loops for both flow models
- `src/training/train_classifier.py` — classifier training with noised inputs
- `src/experiments/run_synthetic.py` — end-to-end experiment with kill-gate evaluation

## Research Context

Full project history in `PROJECT_RESEARCH.md`. Previous directions (abandoned): DPP subgraph selection (failed experimentally), multi-target protein binder, Dirichlet FM inverse folding. Target venue: ICLR 2027.

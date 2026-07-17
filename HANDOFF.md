# Handoff — 2026-06-12

## In-Flight
- Branch: `main` (clean except untracked HANDOFF.md)
- Uncommitted: `HANDOFF.md` (this file)
- Unpushed commits: 0 (all synced with origin/main)
- Open tasks: none

## Next Steps
1. **Deploy on AWS for RFdiffusion pilot run.** User wants AWS deployment — need to set up EC2 instance (p4d.24xlarge or g5.xlarge for A100/A10G), install RFdiffusion + model weights, transfer `configs/` and `src/`, run `python3 -m src.pipeline` for 5K designs as go/no-go gate. User requested confirmation before any actual deployment.
2. **If pilot succeeds (>0.1% dual-contact rate):** Scale to 50K designs, install ProteinMPNN + ColabFold, run full stages 4-5 (sequence design + AF2 validation).
3. **If pilot fails (<0.01% hit rate):** Implement fallback in `src/generate.py` — add RFdiffusion potentials (`olig_contacts` on both chains) or partial diffusion from hub protein scaffolds.

## Decisions This Session
- **Fixed git upstream tracking:** `git branch --set-upstream-to=origin/main main` (was missing, caused `git pull` failures)
- **User wants AWS deployment:** Explicitly asked to "setup project for this" and required confirmation before actual deployment. No deployment has been initiated yet.

## Tried and Abandoned
Nothing abandoned this session — session was brief (upstream fix + deployment request).

## Mental Model
The pipeline (`src/pipeline.py`) orchestrates 7 stages for generating de novo protein binders to two targets simultaneously. Currently runs locally in mock mode (42 tests pass). Real execution requires an NVIDIA GPU (A100 preferred) for RFdiffusion (stage 2), ProteinMPNN (stage 4), and AF2-Multimer (stage 5). Stages 0, 3, 5-steric, 6-results run on CPU. The AWS deployment needs: (1) GPU instance, (2) RFdiffusion repo + SE3-transformer + model weights (~5GB), (3) our `src/` + `configs/` uploaded, (4) run the pipeline without `--mock`. The critical unknown remains whether dual-hotspot conditioning produces any designs that contact both targets — this has never been tested by anyone.

## Key Files
- `src/pipeline.py` — Full 7-stage orchestration (entry point: `python3 -m src.pipeline`)
- `src/generate.py` — RFdiffusion CLI wrapper + mock mode
- `configs/targets.yaml` — Target pair config (ubiquitin + SUMO, hotspots, separation)
- `configs/defaults.yaml` — Pipeline parameters (50K designs, 8Å contact threshold, etc.)
- `PAPER_DRAFT.md` — Full paper draft with method + experimental design
- `HANDOFF.md` — This file
- `docs/superpowers/plans/2026-06-11-multi-target-binder.md` — Original implementation plan
- `scripts/run_cloud.sh` — Shell script for GPU execution (needs RFdiffusion path configured)

## Blockers / Caveats
- **User must confirm before any AWS deployment actions** (explicit request)
- **RFdiffusion model weights are ~5GB** — need to download on the instance, not transfer
- **RFdiffusion install is non-trivial:** requires SE3-Transformer (separate repo), specific PyTorch/CUDA versions, Hydra configs
- **I don't know actual RFdiffusion inference speed** — cannot give reliable time estimates for the pilot run
- **Git upstream was broken** — fixed this session, `git pull` now works

## Test Status
42 passed, all green
Run: `python3 -m pytest tests/ -v`

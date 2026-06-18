# Experiment Results: DPP Margin-Aware Subgraph Selection

## Setup

- **Dataset:** ZINC-12K (molecular graph regression)
- **Budget k:** 5 subgraphs per graph
- **Policy:** Node-deletion (bag size = n_nodes per graph, avg ~23)
- **Epochs:** 100
- **Seeds:** 5 (0-4)
- **Architecture:** 4-layer GIN, hidden_dim=64, out_dim=1
- **Metric:** MAE (lower is better)

## Results

| Method | MAE (mean ± std) | Time/seed |
|--------|-----------------|-----------|
| Full (no selection, all ~23 subgraphs) | **0.409 ± 0.005** | 6970s |
| Centrality (delete highest-degree nodes) | 0.453 ± 0.006 | 657s |
| Uniform (random k=5) | 0.496 ± 0.008 | 641s |
| DPP margin-aware (ours) | 0.539 ± 0.020 | 2647s |

### Per-seed breakdown

| Seed | DPP | Uniform | Centrality | Full |
|------|-----|---------|------------|------|
| 0 | 0.529 | 0.499 | 0.450 | 0.405 |
| 1 | 0.527 | 0.491 | 0.450 | 0.408 |
| 2 | 0.574 | 0.508 | 0.459 | 0.417 |
| 3 | 0.540 | 0.495 | 0.460 | 0.409 |
| 4 | 0.523 | 0.486 | 0.445 | 0.409 |

## Analysis: Why DPP Failed

**1. The supervision signal is vacuous.**
The margin loss broadcasts the same per-graph margin to ALL subgraphs in the bag. Every subgraph gets the same "reward" signal regardless of its individual contribution. The quality scorer learns to predict the graph-level margin (a constant per graph), not per-subgraph utility. This is equivalent to no signal.

**2. More subgraphs = strictly better on ZINC.**
Full (all ~23) >> Centrality (k=5) >> Uniform (k=5) >> DPP (k=5). The ranking is monotone with information quantity. This suggests ZINC benefits from maximal expressivity — the Franks-Morris "expressivity hurts margin" regime does NOT manifest on this dataset. ZINC is a clean, low-noise molecular benchmark where more information always helps.

**3. DPP overhead hurts optimization.**
DPP has extra learnable parameters (margin scorer MLP) competing for gradient signal with the main task. With only 100 epochs, these extra parameters dilute learning without contributing useful selection.

**4. Diversity is not the bottleneck on ZINC.**
Node-deletion subgraphs on small molecular graphs are already fairly diverse (different atoms removed produce meaningfully different structures). DPP's diversity pressure doesn't add value when the candidate pool is already diverse enough.

## Implications for the Research Direction

The hypothesis "margin-aware diverse selection improves generalization for subgraph GNNs" is **not validated** on ZINC. This could mean:

- **Wrong dataset:** ZINC is too clean / not in the "expressivity hurts" regime. Need a dataset where full-bag ESAN actually underperforms simpler GNNs (the Franks-Morris condition).
- **Wrong supervision:** Per-subgraph attribution (leave-one-out, influence functions) rather than broadcast margin.
- **Wrong problem:** The budget-constrained regime (k << n) is dominated by "just use all subgraphs" — selection only matters when full-bag is prohibitively expensive (larger graphs).

## V2 Experiment: LOO Supervision + Hard Selection Fix

After diagnosing V1's failures, implemented V2 with:
1. **Hard DPP selection at train AND eval** (eliminates train/eval distribution mismatch)
2. **Leave-one-out supervision** (per-subgraph signal: how much does removing each subgraph change the prediction)
3. **Exploration** (20% random swap of one selected subgraph to avoid self-reinforcing collapse)
4. **LOO loss only on selected subgraphs** (prevents suppressing unselected subgraphs to zero)

### V2 Results (dev desktop, 100 epochs, 5 seeds)

| Method | MAE (mean ± std) | Time/seed |
|--------|-----------------|-----------|
| Centrality | **0.453 ± 0.012** | 523s |
| Uniform | 0.493 ± 0.009 | 455s |
| DPP v2 (LOO + exploration) | 0.550 ± 0.022 | 2592s |

### V2 Per-seed

| Seed | DPP v2 | Uniform | Centrality |
|------|--------|---------|------------|
| 0 | 0.547 | 0.509 | 0.442 |
| 1 | 0.533 | 0.489 | 0.449 |
| 2 | 0.560 | 0.494 | 0.448 |
| 3 | 0.583 | 0.482 | 0.474 |
| 4 | 0.528 | 0.489 | 0.461 |

### V2 Conclusion

**The fix did not help.** DPP v2 (0.550) is marginally worse than DPP v1 (0.539). Fixing the supervision signal and train/eval consistency did not change the fundamental result: learned diversity-based selection is anti-correlated with what helps on ZINC.

## Final Conclusions

1. **The Franks-Morris regime does not manifest on ZINC.** On this clean molecular dataset, more expressivity (more subgraphs) monotonically improves performance. There is no "expressivity hurts generalization" effect to exploit.

2. **Diversity is the wrong inductive bias for molecular graphs.** DPP selects structurally diverse subgraphs (peripheral, atypical). Centrality selects structurally central subgraphs (hubs, core). On molecules, centrality is the better heuristic — core atoms carry more chemical information.

3. **Learned selection adds overhead without benefit when the task is clean.** The DPP scorer adds parameters that compete for gradient signal without contributing useful selection on a low-noise dataset.

4. **Budget selection only matters at scale.** At k=5 of 23 on small molecular graphs, "just use all subgraphs" dominates everything. Selection becomes relevant only when the full bag is prohibitively expensive (large graphs with 100+ nodes).

## Project Status: Abandoned

The thesis "margin-aware diverse selection improves generalization for subgraph GNNs" is empirically refuted on the primary benchmark (ZINC-12K). Two complete implementation attempts (V1: broadcast margin, V2: LOO supervision) both produce worse results than a zero-parameter centrality heuristic.

### Evaluated pivots (all thin)

- **Scaling laws / distillation paper:** "Bag size beats learned selection" framing. Evaluated via 5-agent council (theorist, contrarian, expansionist, executor, outsider). Verdict: the audience is ~30-50 ESAN/subgraph-GNN researchers. Practical impact near-zero (ZINC graphs have 23 nodes; nobody is compute-bottlenecked). "More compute = better" is not a surprising finding. The test-time-compute framing is a stretch. Abandoned.
- **Phase diagram paper:** Characterize when selection helps vs hurts across datasets × budgets. Pure empirical study, no method contribution. Niche audience.
- **Negative result workshop paper:** Document that centrality dominates learned selection. Curiosity value only.

**Conclusion:** The subgraph GNN selection subfield is too niche to continue in. Pivoting to a different Graph ML problem entirely.

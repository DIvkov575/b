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

## What Could Still Be Publishable

1. **Negative result paper (LoG workshop):** "Does Diversity Help? A Controlled Study of Subgraph Selection Strategies" — show that on ZINC, simple centrality beats learned methods, and more subgraphs always helps. Frame as empirical guidance for practitioners.

2. **Fix the supervision, re-run:** Replace broadcast margin with leave-one-out attribution (remove each subgraph, measure margin change). This gives per-subgraph signal. If that works, it's the original contribution.

3. **Find the right dataset:** Need a dataset where the full bag hurts (or at minimum, doesn't help proportionally). Possible candidates: noisy social graphs, heterogeneous graphs, or synthetic graphs designed to exhibit the Franks-Morris condition.

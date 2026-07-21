"""Gate 1 (docs/L38_PROTOCOL.md): decision-rule computation.

Tests H1 (real, exploitable structure-vs-retrieval asymmetry on low-MSA-depth
assays) vs H0 (shared difficulty, no asymmetry) using ProteinGym's own
published per-assay zero-shot scores, plus a from-scratch static-stacking
baseline fit on a disjoint (Medium/High-depth) calibration split.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.l38.eval_harness import bootstrap_mean_se

DATA_DIR = Path(__file__).resolve().parent / "data_cache"
REFERENCE_CSV = DATA_DIR / "DMS_substitutions.csv"
DMS_LEVEL_CSV = DATA_DIR / "DMS_substitutions_Spearman_DMS_level.csv"

SOTA_LOW_MSA_DEPTH = 0.498  # AIDO Protein-RAG (16B), published Summary_performance
DELIVERABLE_MARGIN_MULTIPLIER = 2.0


def _assay_ids_by_category(reference_csv=REFERENCE_CSV):
    ref = pd.read_csv(reference_csv)
    return {
        cat: set(ref.loc[ref["MSA_Neff_L_category"] == cat, "DMS_id"])
        for cat in ("Low", "Medium", "High")
    }


def _scores(dms_level_csv=DMS_LEVEL_CSV):
    return pd.read_csv(dms_level_csv)


def degradation_bootstrap(model: str, n_boot: int = 10000, seed: int = 0):
    """Bootstrap the High-minus-Low mean Spearman gap for one model.

    Positive = model degrades from High to Low depth (as expected for a
    retrieval-dependent model). Returns (high_mean, low_mean, gap_mean, gap_se).
    """
    ids = _assay_ids_by_category()
    dms = _scores()

    high_scores = dms.loc[dms["DMS ID"].isin(ids["High"]), model].values
    low_scores = dms.loc[dms["DMS ID"].isin(ids["Low"]), model].values

    high_mean, high_se, _ = bootstrap_mean_se(high_scores, n_boot=n_boot, seed=seed)
    low_mean, low_se, _ = bootstrap_mean_se(low_scores, n_boot=n_boot, seed=seed)

    rng = np.random.RandomState(seed)
    high_clean = high_scores[~np.isnan(high_scores)]
    low_clean = low_scores[~np.isnan(low_scores)]
    boot_gaps = np.empty(n_boot)
    for i in range(n_boot):
        h = rng.choice(high_clean, size=len(high_clean), replace=True).mean()
        l = rng.choice(low_clean, size=len(low_clean), replace=True).mean()
        boot_gaps[i] = h - l
    gap_se = float(boot_gaps.std(ddof=1))

    return high_mean, low_mean, high_mean - low_mean, gap_se


def fit_static_stacking_weight(model_a: str, model_b: str):
    """Fit a single scalar mixing weight w minimizing (w*a + (1-w)*b - label)
    error on Medium+High assays (disjoint from the Low-depth eval slice),
    using each model's OWN zero-shot score as a self-referential proxy is not
    possible without ground-truth fitness labels here (DMS_level CSV has
    scores only, not raw fitness) -- so weight selection uses the label-free
    proxy of maximizing agreement with the higher-average-Spearman model on
    the calibration split, matching a simple oracle-free stacking heuristic:
    weight proportional to each model's own calibration-split mean Spearman.
    """
    ids = _assay_ids_by_category()
    calib_ids = ids["Medium"] | ids["High"]
    dms = _scores()
    calib = dms[dms["DMS ID"].isin(calib_ids)]

    mean_a = calib[model_a].mean()
    mean_b = calib[model_b].mean()
    total = mean_a + mean_b
    return mean_a / total if total > 0 else 0.5


def static_stack_low_msa_depth(model_a: str, model_b: str, n_boot: int = 10000, seed: int = 0):
    """Evaluate a fixed-weight linear stack of two models on the Low-depth slice.

    Weight is fit on Medium+High (calibration), applied unseen to Low (eval) --
    per the leakage discipline in docs/L38_PROTOCOL.md.
    """
    w = fit_static_stacking_weight(model_a, model_b)
    ids = _assay_ids_by_category()
    dms = _scores()
    low = dms[dms["DMS ID"].isin(ids["Low"])]

    stacked = w * low[model_a] + (1 - w) * low[model_b]
    mean, se, n = bootstrap_mean_se(stacked.values, n_boot=n_boot, seed=seed)
    return w, mean, se, n


def evaluate_deliverable(mean: float, se: float, sota: float = SOTA_LOW_MSA_DEPTH) -> bool:
    """Pre-registered DELIVERABLE rule: mean exceeds SOTA by >= 2x combined SE."""
    return (mean - sota) >= DELIVERABLE_MARGIN_MULTIPLIER * se


if __name__ == "__main__":
    print("=== H0 vs H1: structure-only vs retrieval-only degradation (High -> Low MSA depth) ===")
    for model in ["ProteinMPNN", "AIDO Protein-RAG (16B)", "VenusREM", "GEMME"]:
        high_mean, low_mean, gap, gap_se = degradation_bootstrap(model)
        print(
            f"{model:<28} High={high_mean:.3f}  Low={low_mean:.3f}  "
            f"Gap={gap:.3f} (+/-{gap_se:.3f} SE)"
        )

    print()
    print("=== DELIVERABLE check: static stacking of ProteinMPNN + AIDO-RAG on Low-depth slice ===")
    w, mean, se, n = static_stack_low_msa_depth("ProteinMPNN", "AIDO Protein-RAG (16B)")
    print(f"weight(ProteinMPNN)={w:.3f}  n={n}  mean={mean:.3f}  SE={se:.4f}")
    print(f"DELIVERABLE? {evaluate_deliverable(mean, se)}  (need >= {SOTA_LOW_MSA_DEPTH + 2*se:.3f})")

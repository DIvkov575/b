"""Gate 0/1 eval harness: low-MSA-depth slice selection + bootstrap statistics.

Matches ProteinGym's own aggregation convention: one Spearman score per
DMS assay already exists in the official per-assay score files, so the unit
of bootstrap resampling here is the *assay*, not the individual mutant --
mirroring ProteinGym's compute_bootstrap_standard_error. See
docs/L38_PROTOCOL.md for the protocol this harness serves.
"""
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def bootstrap_mean_se(values, n_boot: int = 10000, seed: int = 0):
    """Bootstrap mean and standard error of the mean over assay-level scores.

    NaNs are dropped before resampling (an assay a model didn't score).
    Returns (mean, bootstrap_se, n_used).
    """
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    n = len(values)
    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    return float(values.mean()), float(boot_means.std(ddof=1)), n


def load_low_msa_depth_assay_ids(reference_csv: Path) -> set:
    """Return the DMS_id set where MSA_Neff_L_category == 'Low'.

    reference_csv is ProteinGym's reference_files/DMS_substitutions.csv.
    """
    ref = pd.read_csv(reference_csv)
    return set(ref.loc[ref["MSA_Neff_L_category"] == "Low", "DMS_id"])


def per_assay_scores_for_models(
    dms_level_csv: Path, assay_ids: Iterable[str], model_names: Iterable[str]
) -> pd.DataFrame:
    """Return the per-assay Spearman score table restricted to assay_ids/model_names.

    dms_level_csv is ProteinGym's DMS_substitutions_Spearman_DMS_level.csv
    (one row per assay, one column per model). Raises KeyError if a requested
    model column doesn't exist, rather than silently dropping it.
    """
    dms = pd.read_csv(dms_level_csv)
    model_names = list(model_names)
    missing = [m for m in model_names if m not in dms.columns]
    if missing:
        raise KeyError(f"Model column(s) not found in {dms_level_csv}: {missing}")

    filtered = dms[dms["DMS ID"].isin(set(assay_ids))]
    return filtered[["DMS ID", *model_names]].reset_index(drop=True)

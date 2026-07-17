"""L18 H1 correlation gate: distances -> Spearman -> bootstrap CI -> PASS/KILL verdict.

Pre-registered rule (docs/L18_PROTOCOL.md), operating on absolute Spearman rho
between a distance and per-backbone binary designability:

  KILL-A  if |rho(d_rmsd, designable)| > 0.6          (coord distance suffices)
  KILL-B  if |rho(d_feat, designable)| < 0.3          (feature distance has no signal)
  PASS    if |rho(d_feat)| > 0.3 AND |rho(d_feat)| - |rho(d_rmsd)| >= 0.15
  WEAK    if |rho(d_feat)| > 0.3 AND 0 <= margin < 0.15

KILL-A takes precedence (if coord distance already predicts designability, there is
nothing for a perceptual distance to add, regardless of how good d_feat looks).
"""
from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr

# Pre-registered thresholds — do not move after seeing data (the L25 lesson).
RMSD_SUFFICIENT = 0.6   # KILL-A: coord distance already predicts designability
FEAT_MIN_SIGNAL = 0.3   # KILL-B floor for the feature distance
FEAT_MARGIN = 0.15      # how much d_feat must beat d_rmsd for a full PASS


@dataclass
class Verdict:
    status: str          # PASS | WEAK | KILL-A | KILL-B
    rho_feat: float
    rho_rmsd: float
    margin: float
    reason: str


def spearman_abs(distance, designable):
    """Absolute Spearman correlation between a distance and binary designability."""
    distance = np.asarray(distance, dtype=float)
    designable = np.asarray(designable, dtype=float)
    rho, _ = spearmanr(distance, designable)
    if np.isnan(rho):
        return 0.0
    return float(abs(rho))


def bootstrap_ci(distance, designable, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for |Spearman rho|, resampling over backbones."""
    distance = np.asarray(distance, dtype=float)
    designable = np.asarray(designable, dtype=float)
    n = len(distance)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = spearman_abs(distance[idx], designable[idx])
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return lo, hi


def adjudicate(rho_feat, rho_rmsd):
    """Apply the pre-registered PASS/KILL rule to two absolute Spearman rho's."""
    margin = rho_feat - rho_rmsd
    if rho_rmsd > RMSD_SUFFICIENT:
        return Verdict("KILL-A", rho_feat, rho_rmsd, margin,
                       f"coord distance already predicts designability "
                       f"(|rho_rmsd|={rho_rmsd:.2f} > {RMSD_SUFFICIENT}); nothing to add")
    if rho_feat < FEAT_MIN_SIGNAL:
        return Verdict("KILL-B", rho_feat, rho_rmsd, margin,
                       f"feature distance has no designability signal "
                       f"(|rho_feat|={rho_feat:.2f} < {FEAT_MIN_SIGNAL})")
    if margin >= FEAT_MARGIN:
        return Verdict("PASS", rho_feat, rho_rmsd, margin,
                       f"feature distance beats coord by {margin:.2f} (>= {FEAT_MARGIN})")
    return Verdict("WEAK", rho_feat, rho_rmsd, margin,
                   f"feature distance has signal but margin {margin:.2f} < {FEAT_MARGIN}")

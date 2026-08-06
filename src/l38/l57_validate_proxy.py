"""L57 criterion-4 check, runnable: re-derive every candidate proxy's
correlation against eSol's real experimental labels, and re-derive the
evidence that eSol is a distinct measurement from L43's cached solubility
dataset.

This exists so the r=+0.305 claim in l57_expression_yield_steering.py's
docstring is verifiable rather than asserted -- L50 criterion 4 requires the
proxy to be validated against real labels BEFORE a steering run, and L43's
whole effort was wasted by discovering r=-0.03 afterward.

    python3 -m src.l38.l57_validate_proxy

Fetches and caches eSol from HuggingFace (AI4Protein/eSOL) on first run.
Exits non-zero if the chosen proxy's correlation falls below the 0.2 bar or
if eSol stops being distinguishable from L43's dataset.
"""
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.l38.l57_expression_yield_steering import expression_yield_proxy

CACHE_DIR = Path(__file__).resolve().parent / "data_cache" / "expression"
CLEAN_PATH = CACHE_DIR / "esol_clean.csv"
L43_DIR = Path(__file__).resolve().parent / "data_cache" / "solubility"
HF_BASE = "https://huggingface.co/datasets/AI4Protein/eSOL/resolve/main/"

STANDARD_RESIDUES = set("ACDEFGHIKLMNPQRSTVWY")
MAX_SEQ_LEN = 400
MIN_ABS_R = 0.2  # the bar this proxy must clear; L43's GRAVY scored 0.03

KYTE_DOOLITTLE = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
    "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
    "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}


def _fraction(seq, residues):
    return sum(1 for c in seq if c in residues) / len(seq)


def gravy(seq):
    return sum(KYTE_DOOLITTLE[c] for c in seq) / len(seq)


def net_charge(seq):
    """L51's signed net charge -- included to show it is NOT the same proxy."""
    return (_fraction(seq, "DE") - _fraction(seq, "KR")) * 1.0


def turn_forming_fraction(seq):
    return _fraction(seq, "NGPS")


def wilkinson_harrison_neg_cv(seq):
    """Wilkinson & Harrison 1991's two-term recombinant-solubility
    discriminant, negated so higher = more soluble-like.
    """
    return -(15.43 * turn_forming_fraction(seq) - 29.56 * expression_yield_proxy(seq))


CANDIDATE_PROXIES = {
    "absolute_charge_average (CHOSEN)": expression_yield_proxy,
    "wilkinson_harrison_negCV (composite)": wilkinson_harrison_neg_cv,
    "aromatic_fraction_negated": lambda s: -_fraction(s, "FWY"),
    "beta_fraction_negated": lambda s: -_fraction(s, "VILFYWCT"),
    "gravy_negated (L43's dead proxy)": lambda s: -gravy(s),
    "net_charge (L51's proxy)": net_charge,
    "turn_forming_fraction (NGPS)": turn_forming_fraction,
    "cys_fraction_negated": lambda s: -_fraction(s, "C"),
    "seq_length_negated (confound probe)": lambda s: -float(len(s)),
}


def load_esol():
    if CLEAN_PATH.exists():
        return pd.read_csv(CLEAN_PATH)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for split in ("train", "valid", "test"):
        req = urllib.request.Request(HF_BASE + f"{split}.csv", headers={"User-Agent": "python-urllib"})
        raw = urllib.request.urlopen(req, timeout=120).read()
        part = pd.read_csv(io.BytesIO(raw))
        part["split"] = split
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df["aa_seq"] = df["aa_seq"].astype(str).str.strip().str.upper()
    df = df[df["aa_seq"].str.len() > 1]
    df = df[df["aa_seq"].apply(lambda s: set(s) <= STANDARD_RESIDUES)]
    df = df.dropna(subset=["label"])
    df = df[df["aa_seq"].str.len() <= MAX_SEQ_LEN]
    clean = df[["name", "gene", "aa_seq", "label", "split"]].rename(columns={"aa_seq": "sequence"})
    clean.to_csv(CLEAN_PATH, index=False)
    return clean


def main():
    df = load_esol()
    print(f"eSol, length-filtered <={MAX_SEQ_LEN}: n={len(df)}, splits={df['split'].value_counts().to_dict()}")

    failures = []
    for label, subset in [("FULL SET", df), ("HELD-OUT TEST SPLIT ONLY", df[df["split"] == "test"])]:
        y = subset["label"].astype(float).values
        print(f"\n=== {label} (n={len(subset)}) ===")
        print(f"{'proxy':40s} {'pearson':>9s} {'p':>10s} {'spearman':>9s}")
        rows = []
        for name, fn in CANDIDATE_PROXIES.items():
            x = np.array([fn(s) for s in subset["sequence"]])
            pr, pp = stats.pearsonr(x, y)
            sr, _ = stats.spearmanr(x, y)
            rows.append((name, pr, pp, sr))
        for name, pr, pp, sr in sorted(rows, key=lambda r: -abs(r[1])):
            print(f"{name:40s} {pr:+9.4f} {pp:10.2e} {sr:+9.4f}")
        chosen_r = next(pr for name, pr, _, _ in rows if name.endswith("(CHOSEN)"))
        if abs(chosen_r) < MIN_ABS_R:
            failures.append(f"{label}: chosen proxy |r|={abs(chosen_r):.4f} < {MIN_ABS_R}")

    seqs = df["sequence"]
    chosen = np.array([expression_yield_proxy(s) for s in seqs])
    g = np.array([gravy(s) for s in seqs])
    nc = np.array([net_charge(s) for s in seqs])
    lens = seqs.str.len().values.astype(float)
    y = df["label"].astype(float).values
    print("\n=== the chosen proxy is not a relabeling of a known-dead proxy ===")
    print(f"vs GRAVY (L43's dead proxy):  r={stats.pearsonr(chosen, g)[0]:+.4f}")
    print(f"vs net_charge (L51's proxy):  r={stats.pearsonr(chosen, nc)[0]:+.4f}")
    resid_x = chosen - np.polyval(np.polyfit(lens, chosen, 1), lens)
    resid_y = y - np.polyval(np.polyfit(lens, y, 1), lens)
    print(f"partial r vs label | length:  r={stats.pearsonr(resid_x, resid_y)[0]:+.4f} "
          f"(length itself: r={stats.pearsonr(-lens, y)[0]:+.4f}; masked-fill "
          f"generation never changes length, so length cannot confound the run)")

    l43_train, l43_test = L43_DIR / "train.csv", L43_DIR / "test.csv"
    if l43_train.exists() and l43_test.exists():
        l43 = pd.concat([pd.read_csv(l43_train), pd.read_csv(l43_test)])
        l43["s"] = l43["sequences"].astype(str).str.strip().str.upper()
        overlap = df.merge(l43[["s", "labels"]].drop_duplicates("s"), left_on="sequence", right_on="s")
        print(f"\n=== eSol vs L43's cached solubility dataset ({len(overlap)} shared sequences) ===")
        if len(overlap) >= 30:
            r, p = stats.pointbiserialr(overlap["labels"].astype(int), overlap["label"].astype(float))
            print(f"eSol continuous label vs L43 binary label: r={r:+.4f} (p={p:.3f})")
            print("  -> uncorrelated => genuinely distinct measurements, not a rerun of L43"
                  if p > 0.05 else "  -> CORRELATED: eSol may not be distinct from L43 after all")
            if p <= 0.05:
                failures.append(f"eSol label correlates with L43's (r={r:+.4f}, p={p:.3f}) -- not a distinct target")
        print(f"His-tagged (MAHHHHHH) sequences -- L43: "
              f"{l43['s'].str.startswith('MAHHHHHH').sum()}, eSol: "
              f"{df['sequence'].str.startswith('MAHHHHHH').sum()}")
    else:
        print("\n(L43 solubility cache absent -- skipped the distinctness check)")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"\nOK: chosen proxy clears |r|>={MIN_ABS_R} on both the full set and the held-out split.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

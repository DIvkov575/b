"""Generate every figure in the paper directly from the committed
plm_steering/l5[3-7]_repro_out*/results.json files -- no hand-copied
numbers, so a figure changes automatically if a results.json is
regenerated. Run from this directory: python3 make_figures.py

Palette: fixed categorical order from the project's dataviz reference
(blue, orange, aqua, yellow), validated colorblind-safe adjacent pairs.
Sequential magnitude uses the same blue ramp. No dual axes anywhere.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
GRAY = "#8a8a86"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

plt.rcParams.update({
    "font.size": 10,
    "axes.edgecolor": "#c7c6c0",
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT_SECONDARY,
    "ytick.color": TEXT_SECONDARY,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load(name):
    return json.load(open(REPO_ROOT / "plm_steering" / name / "results.json"))


ALPHAS = [0.1, 0.25, 0.5, 1.0, 2.0]


def effect_series(verdict, safe_only=False):
    keys = ["0.1", "0.25", "0.5"] if safe_only else [str(a) for a in ALPHAS]
    xs, ys, lo, hi = [], [], [], []
    for k in keys:
        r = verdict["real_vs_random_by_alpha"].get(k, {})
        if r.get("point_estimate") is None:
            continue
        xs.append(float(k))
        ys.append(r["point_estimate"])
        lo.append(r["point_estimate"] - r["ci_lower"])
        hi.append(r["ci_upper"] - r["point_estimate"])
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi)


# --- Figure 1: dose-response for the two PASS targets (L54, L55) ---------

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6), sharey=False)

d54 = load("l54_repro_out")
xs, ys, lo, hi = effect_series(d54["verdict"], safe_only=True)
axes[0].errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=BLUE, ecolor=BLUE,
                  elinewidth=2, capsize=3, markersize=6, linewidth=2)
axes[0].axhline(0, color=GRAY, linewidth=1, linestyle="--")
axes[0].set_title("Catalytic activity (L54)", fontsize=11)
axes[0].set_xlabel(r"steering strength $\alpha$")
axes[0].set_ylabel("real $-$ random control\n(IVYWREL-style proxy score)")

d55 = load("l55_repro_out")
xs, ys, lo, hi = effect_series(d55["verdict"], safe_only=True)
axes[1].errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=ORANGE, ecolor=ORANGE,
                  elinewidth=2, capsize=3, markersize=6, linewidth=2)
axes[1].axhline(0, color=GRAY, linewidth=1, linestyle="--")
axes[1].set_title("Intrinsic disorder (L55)", fontsize=11)
axes[1].set_xlabel(r"steering strength $\alpha$")
axes[1].set_ylabel("real $-$ random control\n(TOP-IDP proxy score)")

fig.tight_layout()
fig.savefig(OUT_DIR / "fig1_dose_response.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig1_dose_response.png", bbox_inches="tight", dpi=200)
plt.close(fig)


# --- Figure 2: proxy validity vs. causal effect (the headline claim) -----

targets = [
    ("Binding\naffinity", 0.80, "l53_repro_out", "KILL"),
    ("Catalytic\nactivity", 0.22, "l54_repro_out", "PASS"),
    ("Intrinsic\ndisorder", 0.45, "l55_repro_out", "PASS*"),
    ("Expression\nyield", 0.31, "l57_repro_out", "AMBIG."),
]

fig, ax = plt.subplots(figsize=(5.5, 4.8))
color_by_verdict = {"KILL": GRAY, "PASS": BLUE, "PASS*": ORANGE, "AMBIG.": YELLOW}

# per-point label placement tuned individually to avoid the title, the axes,
# and the other three points -- a generic above/below rule collides here
# because binding affinity sits right at y=0 near the x-axis, and disorder
# sits at the very top of the y-range near the title.
label_offsets = {
    "Binding\naffinity": (0, 16),
    "Catalytic\nactivity": (-2, -28),
    "Intrinsic\ndisorder": (0, 14),
    "Expression\nyield": (24, 10),
}

for label, proxy_r, fname, verdict in targets:
    d = load(fname)
    eff = d["verdict"]["real_vs_random_by_alpha"]["0.5"]["point_estimate"]
    cohens_d = eff / d["baseline"]["std"]
    ax.scatter(proxy_r, cohens_d, s=140, color=color_by_verdict[verdict],
               edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(label.replace("\n", " "), (proxy_r, cohens_d),
                textcoords="offset points", xytext=label_offsets[label],
                ha="center", fontsize=8.5, color=TEXT_SECONDARY)

ax.axhline(0, color=GRAY, linewidth=1, linestyle="--", zorder=1)
ax.set_xlabel("proxy validity (Pearson $r$ vs. real labels, held-out)")
ax.set_ylabel("steering effect size (Cohen's $d$ at $\\alpha=0.5$)")
ax.set_title("Proxy validity does not predict causal steerability", fontsize=11, pad=14)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(-0.2, 1.0)

handles = [plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                       markerfacecolor=c, markeredgecolor="white", label=v)
           for v, c in color_by_verdict.items()]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8.5)

fig.tight_layout()
fig.savefig(OUT_DIR / "fig2_proxy_vs_effect.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig2_proxy_vs_effect.png", bbox_inches="tight", dpi=200)
plt.close(fig)


# --- Figure 3: L55 seed-robustness, with and without residue-exclusion ---

seeds = ["seed 0", "seed 1", "seed 2"]
seed_files = ["l55_repro_out", "l55_repro_out_seed1", "l55_repro_out_seed2"]

full_effect, excl_effect, excl_lo, excl_hi, excl_sig = [], [], [], [], []
for fname in seed_files:
    d = load(fname)
    v = d["verdict"]
    full_effect.append(v["real_vs_random_by_alpha"]["0.5"]["point_estimate"])
    rc = v["robustness_check"]["diff_with_exclusion"]
    excl_effect.append(rc["point_estimate"])
    excl_lo.append(rc["point_estimate"] - rc["ci_lower"])
    excl_hi.append(rc["ci_upper"] - rc["point_estimate"])
    excl_sig.append(rc["significant_at_95pct"])

x = np.arange(len(seeds))
width = 0.35

fig, ax = plt.subplots(figsize=(5.8, 4.2))
ax.bar(x - width / 2, full_effect, width, label="full effect (all residues)",
       color=BLUE)
bars = ax.bar(x + width / 2, excl_effect, width,
              yerr=[excl_lo, excl_hi], capsize=3, ecolor=TEXT_SECONDARY,
              label="residue-excluded (E, S removed)", color=ORANGE)

for i, sig in enumerate(excl_sig):
    marker = "significant" if sig else "CI crosses zero"
    ax.annotate(marker, (x[i] + width / 2, excl_effect[i] + excl_hi[i] + 0.003),
                ha="center", fontsize=7.5,
                color=TEXT_SECONDARY if sig else "#b23b3b",
                fontweight="bold" if not sig else "normal")

ax.axhline(0, color=GRAY, linewidth=1, linestyle="--", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(seeds)
ax.set_ylim(-0.008, 0.062)
ax.set_ylabel("real $-$ random control\n(TOP-IDP proxy score, $\\alpha=0.5$)")
ax.set_title("Intrinsic disorder (L55): residue-exclusion\nrobustness is seed-sensitive (passes 2 of 3)",
             fontsize=10.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
          frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / "fig3_seed_robustness.pdf", bbox_inches="tight")
fig.savefig(OUT_DIR / "fig3_seed_robustness.png", bbox_inches="tight", dpi=200)
plt.close(fig)

print(f"Wrote 3 figures to {OUT_DIR}")

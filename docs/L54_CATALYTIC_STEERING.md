# L54 — Catalytic activity (kcat) steering: PASS, replicated across 3 seeds

Status: **PASS.** The first genuinely new-property (not-a-reproduction)
capability gain in the whole L41-L57 arc, and the only target of the 5
tried in this batch (L53 binding, L54 catalytic, L55 disorder, L56
immunogenicity, L57 expression yield) to clear the full protocol on every
one of 3 independent seeds. Proxy validated against real labels first, per
`docs/L50_CAPABILITY_GAIN_PROTOCOL.md` criterion 4.

## Dataset choice: DLKcat, not the single-backbone DMS assays

Three catalytic datasets were on hand in `src/l38/data_cache/catalytic/`:

| dataset | n | shape | verdict |
|---|---|---|---|
| `AMIE_PSEAE_Wrenbeck_2017` | 6227 | 6227 single point mutants of ONE 346-aa amidase | rejected |
| `OXDA_RHOTO_Vanella_2023_activity` | 6396 | 6396 single point mutants of ONE 364-aa oxidase | rejected |
| `dlkcat_wt_mut` | 17010 | cross-protein kcat, many enzymes/organisms/substrates | **used** |

The two DMS assays are literally single-backbone: `mutated_seq` length has
std 0.0, `num_mutations` is 1 for every row. A compositional proxy therefore
varies over one of ~350 positions. BLOSUM62-similarity-to-wildtype does reach
**r=+0.229** (p=4e-75) on AMIE and +0.171 on OXDA — numerically the best
correlation found anywhere in this task — but its entire validated range is a
one-substitution window, while `mask_fill_generate` at `MASK_FRACTION=0.3`
rewrites ~100 positions. Validating on 1-mutation variation and then scoring
100-mutation generations is L43's GRAVY error wearing a different hat, so it
was rejected despite the better headline r. DLKcat's cross-protein spread is
the regime the steering eval actually operates in.

DLKcat filtering: sequences <=400 aa, canonical residues only, kcat>0,
deduplicated by sequence with **median** log10 kcat across a sequence's
records (a sequence recurs once per substrate; taking one arbitrary record
would let substrate choice set the label). Yields **4,370 unique enzymes**,
70/30 split at `SEED=0` → 3,059 train / 1,311 test. Available after
splitting: 612 low-kcat and 617 high-kcat vector candidates (need 150 each),
656 eval candidates (need 150). Criterion 6 satisfied with large margin.

## Proxy: glycine fraction − arginine fraction

`catalytic_activity_proxy` in `src/l38/l54_catalytic_activity_steering.py`.
Validated against real log10(kcat) BEFORE the harness was written:

| check | Pearson r | note |
|---|---|---|
| full set (n=4370) | **+0.220** (p=8e-49) | spearman +0.22 |
| held-out test (n=1311) | **+0.214** (p=5e-15) | spearman +0.21 |
| length-residualized | +0.223 | not a length confound (label vs length r=-0.02) |
| wildtype-only subset (n=1738) | +0.146 (p=1e-09) | not driven by engineered mutants |
| within EC class 1 / 2 / 3 / 4 / 5 | +0.13 / +0.23 / +0.13 / +0.35 / +0.01 | significant in 4 of 5; not a fold/EC confound |

Mechanism is the published **activity–stability tradeoff**: high-turnover
(notably cold-adapted) enzymes buy active-site conformational flexibility
with Gly-rich, Arg- and salt-bridge-poor sequences, while rigid thermostable
enzymes trade turnover for Arg-mediated ion pairs (Fields 2001; Siddiqui et
al. 2006; Berezovsky & Shakhnovich 2005). Confirming the tradeoff is present
in this data with the predicted sign, **L42's IVYWREL thermostability proxy
correlates negatively with kcat here (r=-0.118)**. This is a documented
biophysical correlate, not a formula fit to this project's numbers.

### Alternatives measured and rejected (same held-out split)

| proxy | test r |
|---|---|
| glycine fraction alone | +0.193 |
| catalytic-residue fraction (HCDESKRYNTW) | -0.149 |
| aromatic fraction (FWY) | -0.133 |
| charged fraction (DEKR) | -0.109 |
| GRAVY | +0.092 |
| Arg/(Arg+Lys) ratio | +0.075 |
| net charge (L51's proxy) | -0.009 (null) |
| sequence length | -0.022 (null) |
| composites adding A, IVYWREL, or Arg/(Arg+Lys) normalization | all below the two-term form |

## Criterion 4 is enforced in code, not just documented

`l54_run_repro.py` recomputes the proxy-vs-real-label correlation on both
splits and **asserts `|r_test| >= 0.15` (and correct sign) before the model is
loaded**. L43 ran its entire experiment and only afterwards found its proxy
correlated r=-0.03 with real labels; that failure mode is now unreachable
without the run stopping first. Measured in-script at the gate: train
r=+0.2218, test r=+0.2139 → PASS.

## Deviations from the L51/L52 pattern

1. **JSON loader instead of `pd.read_csv`** — DLKcat ships as JSON records;
   `load_dlkcat()` does the filter + median-aggregate + dedup.
2. **Random 70/30 split** — DLKcat has no built-in `stage` column like L51's
   aggregation CSV, so the split is a seeded permutation.
3. **Residue-exclusion fallback.** The proxy is a G-vs-R contrast, so if the
   top-2 substituted residues are exactly {G, R}, excluding both makes the
   proxy identically 0 and the bootstrap vacuous. In that case the check
   falls back to excluding only the top-1 residue and records
   `exclusion_note` in the results JSON. Still a real exclusion check — and
   note this is a *stricter* criterion-3 test than L42's, since excluding G
   or R removes one of the proxy's own two terms.
4. **Criterion 5 recorded as `null`, not `true`** — no prior technique exists
   for this property (same as L51). The verdict logic only requires the
   operative criteria.

## Smoke test (run; full sweep deliberately not run)

Both files parse. Proxy unit checks pass (`GGGG`→+1.0, `RRRR`→-1.0,
`GRAA`→0.0, empty and fully-excluded inputs raise). On ESM2-650M via MPS,
one real DLKcat sequence:

- `mean_pooled_activation_all_layers` → 33 layers, (1, 1280) each
- `mask_fill_generate` baseline → 107 chars, proxy +0.084
- alpha=0 hooked vs baseline: **0 chars differ** (true no-op)
- alpha=5 hooked vs baseline: **31 chars differ** (hook really modifies generation)
- degeneracy filter and paired-bootstrap helpers wired correctly

## Steering results (SEED=0)

All 33 layers hooked (fresh-property test, not the L52 layer-subset
question). Same harness as L42/L51/L52: `MultiLayerSteeringHook`,
`mask_fill_generate` (mask_frac=0.3, max_len=400), matched-norm random
control, paired bootstrap (10,000 resamples), degeneracy filter, n=150 eval
sequences (low-kcat, held out from vector construction).

| alpha | real mean | random mean | diff | significant | n | degenerate (real) |
|---|---|---|---|---|---|---|
| 0.1 | 0.0218 | 0.0205 | +0.0013 | **yes** | 150 | 0/150 |
| 0.25 | 0.0255 | 0.0203 | +0.0052 | **yes** | 150 | 0/150 |
| 0.5 | 0.0358 | 0.0206 | +0.0152 | **yes** | 150 | 0/150 |
| 1.0 | 0.0966 | 0.0214 | +0.0761 | yes | 140 | 10/150 |
| 2.0 | 0.0334 | 0.0251 | +0.0100 | yes | 116 | 34/150 |
(baseline mean 0.0203, n=150, 0 degenerate)

Clean monotonic dose-response through the whole safe range (+0.0013 ->
+0.0052 -> +0.0152), degeneracy staying at 0/150 through alpha=0.5 and only
appearing at alpha>=1.0 — the expected shape.

**Residue-exclusion robustness (criterion 3):** dominant substituted
residues at alpha=0.5 are G and A — note G is literally one of the proxy's
own two terms, the strictest possible version of this check. Excluding
both: effect drops from +0.0152 to +0.0052 (a real 3x shrinkage) but
**stays significant** (CI [0.0037, 0.0067]) — the effect survives with
~34% of its magnitude intact even after removing its own defining residue.

## Seed-robustness check: PASS replicates on 3/3 independent seeds

Every number above uses `SEED=0` for the train/vector-building split AND
the random-control direction draw. Reran twice more with `SEED=1` and
`SEED=2` (same script, only the seed changed) to check this isn't a
lucky split:

| seed | real-vs-random diff @ alpha=0.1/0.25/0.5 | decision | residue-robust? |
|---|---|---|---|
| 0 | +0.0013 / +0.0052 / +0.0152 | PASS | yes (34% retained, still sig.) |
| 1 | +0.0009 / +0.0035 / +0.0118 | PASS | yes |
| 2 | +0.0017 / +0.0035 / +0.0124 | PASS | yes |

Same monotonic shape, same order of magnitude, all 3 seeds independently
clear every criterion. **This is the one target in the 5-target batch with
fully unanimous seed-robustness** (contrast with L55's disorder result,
2 of 3 seeds — see `docs/L55_DISORDER_STEERING.md`). Proxy-vs-real-label
correlation also held across seeds' different train/test splits: test
r=+0.214 (seed 0), +0.164 (seed 1), +0.287 (seed 2) — all comfortably above
the 0.15 gate.

## Verdict: PASS (5 of 6 criteria; criterion 5 not operative)

| criterion | result |
|---|---|
| 1. beats random control, real CI | **PASS**, 3/3 seeds |
| 2. dose-response | **PASS**, monotonic across all 3 safe alphas, 3/3 seeds |
| 3. residue-exclusion robust | **PASS**, ~34% magnitude retained, 3/3 seeds |
| 4. proxy pre-validated | **PASS**, re-validated in-script by assert before the model loads |
| 5. beats prior technique | N/A — no existing technique for this property |
| 6. adequately powered | **PASS**, n=150 |

## What this means

First successful new-property capability gain in the L41-L57 arc that
isn't a reproduction of someone else's published finding (L42's
thermostability result reproduces Huang et al. 2025). Difference-of-means
steering on ESM2-650M pushes masked-fill generation toward a compositional
signature (glycine-enrichment/arginine-depletion) that tracks a real
biophysical property (enzyme turnover) never directly trained for.

**Notable alongside L53 (binding affinity, same batch):** L53's proxy is
far more strongly validated (r=0.80 vs. this target's r=0.22) yet steers
nothing (flat null at every alpha, see `docs/L53_BINDING_STEERING.md`).
Proxy-validation strength and steerability are not the same thing — the
same "correlational strength doesn't predict causal effect" lesson from
L48/L49's head-ablation work, recurring one level up: at the level of
which target *properties* are steerable at all, not just which model
*components* are causally load-bearing.

## What this is NOT

Not evidence the effect is mechanistically specific to catalysis — the
proxy is compositional (glycine/arginine balance), and the steering vector
could equally be described as "push toward a flexibility-associated
amino-acid profile," which happens to correlate with real kcat. No
structural or functional check (e.g. whether generated sequences preserve
plausible active-site geometry) was run. Not a claim this generalizes to
other catalytic properties (Km, substrate specificity) — only kcat/turnover
was tested.

## To run the full sweep

```bash
source .venv-l38/bin/activate
python3 -m plm_steering.l54_run_repro     # writes plm_steering/l54_repro_out/results.json
```

11 arms x 150 sequences on a shared MPS GPU, ~5 minutes wall clock.

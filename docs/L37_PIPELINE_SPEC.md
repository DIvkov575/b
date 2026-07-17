# L37 — Empirical Pipeline Spec: Few-Step Distillation of MDGen (Method-Only)

**Date:** 2026-07-15 · **Status:** Gate 0 + Stage A + Stage C mechanics implemented and
validated end-to-end against the real teacher checkpoint and real trajectory data (see §5).
Stages B/D/E (full sampling + eval sweep) not yet run. · **Scope:** method-only per
`docs/RESEARCH_LEDGER.md` "FINAL DECISION (2026-07-15)" — no end-to-end kinetic-error theorem is
claimed or required. This is an empirical measurement pipeline, not a theorem-validation pipeline.

Eval discipline follows the original spec's own prescription (`docs/L37_MATH_SPEC.md` §7,
reaffirmed by the audit's GATE-0): **per-frame plausibility does not control kinetics — distill
and validate on two-time correlations.** Every design choice below exists to make that comparison
clean: per-frame metrics and two-time metrics are computed side by side, on the same samples, so
if they diverge it's visible rather than assumed.

---

## 0. Grounded facts (verified against the live repo this session — not from memory)

**Repo:** `github.com/bjing2016/mdgen` (paper arXiv:2409.17808), MIT license.

| Object | Location | Fact |
|---|---|---|
| Teacher checkpoint | HF `bjing-mit/mdgen`, file `forward_sim.ckpt` | The tetrapeptide joint trajectory-segment generator — the correct teacher (distinct from `atlas.ckpt`, `interpolation.ckpt`, `upsampling.ckpt`, `inpainting.ckpt`, which are other tasks) |
| Reference MD data | HF dataset `bjing-mit/tetrapeptide-sims` | `4AA_sims/<NAME>/<NAME>.{pdb,xtc,out}` (explicit solvent, 3,309 peptide codes) and `4AA_sims_implicit/<NAME>/...` (implicit, 2,846 codes) |
| Preprocessing | `scripts/prep_sims.py` | Converts `.xtc`+`.pdb` → `.npy`, shape `(num_frames, num_residues, 14, 3)`, dtype `float16` (atom14 coords, Å); superposes trajectories (removes rigid drift) |
| Sampling script | `sim_inference.py` | `python sim_inference.py --sim_ckpt forward_sim.ckpt --data_dir <dir> --split splits/4AA_test.csv --num_rollouts N --num_frames K --xtc --out_dir <dir>` |
| Sampler internals | `mdgen/transport/transport.py`, class `Sampler` | `sample_ode(sampling_method="dopri5", num_steps=50, ...)` and `sample_sde(..., num_steps=250)` — **NFE is a real, exposed parameter of the class** |
| **NFE is NOT wired to CLI** | `mdgen/wrapper.py`, `NewMDGenWrapper.inference` | The call site reads `sample_fn = self.transport_sampler.sample_ode(sampling_method=self.args.sampling_method)` with `# num_steps=self.args.inference_steps)` **commented out** — `sample_ode` silently runs at its hardcoded default `num_steps=50` regardless of any flag. No `--inference_steps`/`--num_steps` flag exists in `mdgen/parsing.py` or any inference script. |
| Eval script (forward sim) | `scripts/analyze_peptide_sim.py` | Already computes, per test peptide: JSD on torsion marginals and on TICA coords (`out['JSD'][feat]`, `out['JSD']['TICA-0']`); autocorrelation/decorrelation on sin/cos of torsions and on TICA modes (`statsmodels.tsa.stattools.acovf`/`acf`); TICA via `pyemma.coordinates.tica(..., kinetic_map=True)`; full MSM construction via `pyemma.msm.estimate_markov_model` with k-means clustering (`k=100`), PCCA coarse-graining (`nstates=10`), transition matrices, and stationary distributions (`out['msm_pi']`, `out['traj_pi']`, `out['msm_transition_matrix']`, `out['traj_transition_matrix']`) |

**Consequence for scoping:** the kinetics-eval infrastructure this project needs largely **already
exists** in `analyze_peptide_sim.py` — TICA and MSM are exactly two-time-correlation analyses
(TICA is a lagged-correlation-matrix eigendecomposition; it is the empirical cousin of the
transfer-operator eigenanalysis in `docs/L37_MATH_SOLUTION.md`). The one confirmed gap is the
**NFE wiring**, which must be patched before anything else — see Gate 0.

**Not yet confirmed:** whether a pyEMMA MSM object's standard `.timescales()` call is invoked
anywhere in `analyze_peptide_sim.py` (the fetched summary quotes `msm_transition_matrix`/`msm_pi`
but not an explicit timescales call). If absent, adding it is a one-line addition — pyEMMA MSM
objects expose it natively — not new infrastructure. Verify before Gate 3.

---

## 1. Pipeline stages

```
[Gate 0]  patch wrapper.py to expose --inference_steps          (must happen first, blocks everything)
   │
[Stage A] data prep: prep_sims.py on a chosen split (implicit solvent, smaller/cheaper)
   │
[Stage B] teacher reference: sample forward_sim.ckpt at NFE=50 (current default) — the reference
   │       trajectories every comparison is measured against
   │
[Stage C] distillation: train a few-step student against the teacher's flow-matching ODE
   │
[Stage D] student sampling: sample the distilled student at NFE ∈ {1,2,4,8}, plus the UNDISTILLED
   │       teacher at NFE ∈ {2,4,8,50} as a same-model step-truncation control (see §4)
   │
[Stage E] eval: analyze_peptide_sim.py (as-is + timescales check) on every condition, same test
             peptides, same seeds — per-frame AND two-time metrics side by side
```

### Gate 0 — NFE wiring (small, required, do first)

Restore the commented-out line in `mdgen/wrapper.py` (`num_steps=self.args.inference_steps`) and
add the corresponding CLI flag in `mdgen/parsing.py` / `sim_inference.py`, or bypass the CLI
entirely and call `model.transport_sampler.sample_ode(num_steps=N, sampling_method=...)` directly
in a small wrapper script. Either is a few lines; this is confirmed necessary, not speculative —
without it every run silently uses `num_steps=50` no matter what's requested.

### Stage A — Data

Use **implicit-solvent** tetrapeptides (`4AA_sims_implicit`, 2,846 codes) over explicit — smaller,
cheaper, and the theory audit's Gate A already established implicit-solvent production is plain
Langevin NVT (simpler reversibility story than explicit-solvent's barostatted NPT, though this
project makes no theorem claim that depends on it). Use the repo's own `splits/4AA_test.csv` held-
out split for eval peptides — reusing MDGen's own train/test split avoids introducing a new,
unvalidated split.

**Peptide count for the pipeline:** start with a small subset (order 10–20 test peptides) for the
first full pass through Stages B–E before committing to the full test split — this is a new,
unbuilt pipeline; validate the mechanics cheaply before scaling the sweep.

### Stage B — Teacher reference

Run `sim_inference.py` with the patched NFE control at `num_steps=50` (the field's current
standard, and the value the repo silently runs at today) to produce the reference trajectories.
Same `--num_rollouts` / `--num_frames` for every condition in Stage D, so all trajectories are
directly comparable at the eval stage.

### Stage C — Distillation (the new code; not in the repo)

**Design choice, open for revision:** distill the flow-matching probability-flow ODE (`sample_ode`,
`dopri5`) using a **consistency-distillation-style objective** — train a student network to map
any point on the teacher's ODE trajectory directly to the trajectory's endpoint, self-consistently
across timesteps, so a single (or few) forward pass replaces the 50-step `dopri5` integration.
This is the same family the theory audit's counterexamples (No-go Theorems A, D) analyzed, so the
eval below is testing on exactly the failure mode the math investigation characterized — not a
disconnected empirical add-on.

Rationale for consistency-style over an alternative (e.g. progressive distillation, shortcut
models): MDGen's `Sampler` already exposes the ODE trajectory machinery (`sample_ode`) needed to
generate teacher (input, trajectory, endpoint) triples for a consistency loss without modifying
the teacher; this is the lowest-engineering-lift route to a first working student. **This is a
starting point, not a locked decision** — revisit once Stage B/D mechanics are validated.

### Stage D — Student (and control) sampling

Two families of samples, both needed for the comparison in §4 to mean anything:

1. **Distilled student**, sampled at NFE ∈ {1, 2, 4, 8}.
2. **Undistilled teacher, step-truncated** — run the *same* teacher checkpoint's `sample_ode` at
   NFE ∈ {2, 4, 8} (in addition to the NFE=50 reference from Stage B). This is the control the
   original math spec's own framing implicitly needs: it isolates *distillation-specific*
   degradation from *generic low-NFE-integration* degradation. Without it, any kinetic collapse
   seen in the student is ambiguous — it could be a distillation artifact or simply what happens
   to this ODE at few steps regardless of method.

### Stage E — Eval

Run `scripts/analyze_peptide_sim.py` (as released, plus the `.timescales()` addition if confirmed
absent) on every condition from Stage D against the Stage B reference, same peptides/seeds. Read
the **per-frame** metrics (JSD on raw torsion marginals) and the **two-time** metrics (decorrelation/
autocorrelation curves, TICA-projected free-energy surfaces, MSM transition matrices, stationary
distributions, implied timescales) side by side per condition.

---

## 2. What "validate on two-time correlations, not per-frame plausibility" means operationally

Two concrete comparisons, both required — this is the pipeline's actual scientific content:

- **Per-frame track:** `out['JSD'][feat]` for raw torsion angles — the metric that, per GATE-0's
  formal counterexample, *cannot* by itself detect a kinetic problem.
- **Two-time track:** `out['JSD']['TICA-0']`, the decorrelation curves, and the MSM-derived
  quantities (transition matrix deviation, stationary-distribution deviation, implied timescales
  if added) — the metrics that *can*.
- **The actual test:** at matched per-frame JSD (i.e., comparing conditions that look equally good
  by the per-frame metric), do the two-time metrics diverge? If distilled students at low NFE show
  materially worse two-time metrics than step-truncated teachers at the *same* per-frame JSD, that
  is the paper's empirical finding — a clean, positive demonstration of exactly the failure mode
  the (now-blocked) theorem was trying to bound. If they track together, that's the honest negative
  result, and still worth reporting (per the original spec's own framing: "even a KILL is a citable
  negative result").

---

## 3. What this pipeline does NOT claim

No bound, formula, or theorem is being validated or calibrated — Gates C/D/E of the math
investigation are formally blocked (`docs/L37_MATH_SOLUTION_V2.md`), not merely unproven. This is
a clean empirical measurement, motivated by and using the vocabulary of that investigation (per-
frame vs. two-time), but the result is a reported effect size with whatever statistical caveats
apply to a ~10-20-peptide pilot, not a confirmed or falsified inequality.

The one theoretical object that *is* still live is `docs/L37_MATH_SOLUTION.md` §8.2 — a bound on
one fixed, bounded teacher-mode autocorrelation from an assumed segment-KL bound. If a specific
distillation loss and its achieved KL are logged during Stage C, that number could in principle be
checked against §8.2's inequality on the TICA slow mode as a secondary, clearly-labeled sanity
check — not a headline result, since §8.2 explicitly does not identify a student eigenvalue.

---

## 5. Implementation log (2026-07-15) — what was actually built and validated

Per user direction: distillation objective locked to **consistency distillation** (not
shortcut/mean-flow — see rationale below); compute unblocked using whatever's available locally
rather than waiting on a GPU allocation; pilot peptide count defaulted small (3, scaled to 100 —
the repo's own `4AA_test.csv` size); `.timescales()` gap closed (confirmed absent, added).

### 5.1 Environment (this machine: Apple M3 Pro, no CUDA GPU)

- `third_party/mdgen/` — real clone of `github.com/bjing2016/mdgen` (MIT), commit at clone time.
- `.venv-mdgen311` — Python 3.11 venv (not 3.14 — `pyEMMA`'s C++ extension `deeptime` fails to
  compile on Clang 21 under 3.14; template-syntax error `Metric::template compute(...)` is a
  genuine Clang-21 standards-conformance regression against pyEMMA's own vendored C++ source, not
  a Python-version issue — confirmed by reproducing the exact error in a 3-line test file).
  torch 2.13.0 (MPS available, `torch.backends.mps.is_available()==True`); numpy 2.4.6.
- **pyEMMA still does not build** even on 3.11, same Clang-21 template error, now in pyEMMA's own
  `clustering_module.cpp` (which includes deeptime's headers) rather than deeptime itself (a
  prebuilt deeptime *wheel* installs fine — only pyEMMA's from-source extension is blocked). This
  confirms Stage E (eval) is blocked on THIS machine specifically — an older/different toolchain
  (e.g. a standard Linux GPU box) will very likely not hit this. Frozen env: `pip freeze` saved to
  `third_party/mdgen_requirements_freeze.txt`.
- Confirmed via direct import: `mdgen.transport.transport`, `mdgen.wrapper` (the model, sampler,
  training-step machinery) import and run **without** pyEMMA — only `mdgen/analysis.py` and the
  `scripts/analyze_*.py` eval scripts need it (pyEMMA is imported at module level there). So
  Stages A–D (data, sampling, distillation) are fully unblocked here; only Stage E is deferred.

### 5.2 Three real upstream bugs found and patched in the vendored `third_party/mdgen` copy

1. **Gate 0, as predicted:** `mdgen/wrapper.py`'s `inference()` had
   `num_steps=self.args.inference_steps` commented out — confirmed by reading the file, exactly as
   the earlier scoping pass reported. Patched: uncommented, and added
   `--inference_steps` (default 50, preserving current behavior) to `mdgen/parsing.py`.
2. **`scripts/prep_sims.py` references `args.atlas_dir`, which does not exist** — the actual flag
   is `--sim_dir`. Both the ATLAS and non-ATLAS branches of `do_job` hit this; the README's
   documented preprocessing command is broken as shipped. Patched: `args.atlas_dir` → `args.sim_dir`.
3. **`mdgen/tensor_utils.py`'s `batched_gather`** indexed a numpy array with a plain Python list of
   mixed index objects (`data[ranges]`) — legal (if deprecated) on numpy 1.21.x (MDGen's pin), a
   hard `ValueError` on numpy 2.x (installed here). This blocked `atom14_to_atom37`, hence the
   entire real-data loading path. Patched: `data[ranges]` → `data[tuple(ranges)]`. Bonus finding:
   PyTorch's own indexing (used elsewhere via this same function on tensors) already warns this
   exact pattern becomes an error in PyTorch 2.9, so the fix pre-empts a second future break too.

All three are environment/version-compatibility patches to the vendored dependency, not changes to
project logic — consistent with the class of patch already applied for Gate 0. Regression tests
for (1) and (3) are in `tests/l37/` (see §5.4); (2) has no dedicated test (a path-string typo with
no interesting failure mode beyond "wrong attribute name").

### 5.3 Distillation objective: locked to consistency distillation

**Chosen over shortcut models / mean-flow:** consistency distillation needs only the teacher's
existing velocity field and one Euler step to build a training target — no architecture change,
no new conditioning input (shortcut models need a step-size input baked into the network; mean-
flow needs an average-velocity reformulation). Given MDGen's `Sampler.sample_ode` already exposes
exactly the velocity-field call the distillation loss needs, this is the lowest-engineering-lift
route to a *first working* student, which is what "continue the work" needed most concretely.
Revisiting shortcut/mean-flow after this student's quality is characterized remains open.

**Design, implemented in `src/l37/`:**
- `consistency_distill.py` — pure math core: `gvp_alpha_sigma` (transcribed from and verified
  against `mdgen/transport/path.py::GVPCPlan`, not re-derived), `consistency_output` (boundary-
  condition parametrization `f_θ(x,t)=α_t·x+σ_t·rawoutput`, giving `f_θ(x,1)=x` exactly for any
  raw output, using MDGen's own path coefficients as `(c_skip,c_out)` — zero new hyperparameters),
  `euler_step`, `sample_timestep_pair` (grid sampling of adjacent `(t_n,t_next)` steps),
  `masked_mse` (mirrors MDGen's own `mean_flat` normalization convention). **13 tests, hardware-
  independent, TDD'd against the real GVP formula.**
- `training_step.py` — `consistency_distillation_loss(teacher, student, x_n, num_steps, mask,
  generator)`: samples `(t_n,t_next)`, takes one frozen-teacher Euler step to build `x_next`,
  forms both sides' consistency-function outputs (target under `torch.no_grad()`, prediction
  with grad), returns masked-MSE loss. Self-distillation (same student network on both sides),
  matching consistency distillation rather than progressive distillation. **5 tests**, including
  an explicit "teacher gets zero gradient, student gets gradient on every parameter" check.
- `train_distill.py` — the runnable script: loads the teacher checkpoint (`load_teacher`,
  correctly re-applying the `inference_steps` patch since old checkpoints predate the flag),
  builds a warm-started student (`build_student` = `deepcopy` of the teacher + explicit
  `requires_grad_(True)` re-enable), loads one real preprocessed trajectory frame per peptide
  (`load_one_frame_batch`, following `sim_inference.py`'s own `get_batch` convention), and runs
  a plain Adam training loop over a peptide split.

### 5.4 End-to-end validation against REAL checkpoint + REAL data (not synthetic)

- Downloaded the real `forward_sim.ckpt` (136MB, HF `bjing-mit/mdgen`) — loads via
  `NewMDGenWrapper.load_from_checkpoint` after allowlisting `argparse.Namespace` for
  `torch.load`'s `weights_only=True` default (checkpoint is from the paper authors' official HF
  release — verified in the original grounding pass — safe to allowlist). Confirmed saved hparams:
  `path_type='GVP', prediction='velocity', sim_condition=True` — exactly the forward-simulation
  teacher, 34,152,521 parameters.
- **Data-split correction (caught by checking real files, not assumed):** the pilot MUST use the
  **explicit-solvent** split (`4AA_sims` + `splits/4AA_test.csv`), not implicit-solvent as this
  spec originally suggested in §0/§1 above — `forward_sim.ckpt`'s training command (per the
  README) uses the explicit split; the implicit and explicit test splits are disjoint peptide sets
  (verified by diffing the two CSVs — only ~1 of the first 10 rows overlap).
- Downloaded real trajectories for 3 test-split peptides (FLRH, IMRY, RTVD; ~17s each for the
  100ns `.xtc`), ran the real `prep_sims.py` (patched) to get real `(10000,4,14,3)` float16 arrays.
- Ran `src/l37/train_distill.py` for real: **3 real Adam optimizer steps, one per peptide, against
  the real teacher, on real trajectory data** — finite losses (0.218, 0.168, 0.028), gradients on
  all 289 student parameter tensors, zero gradient on the frozen teacher, different `(t_n,t_next)`
  pairs sampled per step as expected. This is genuine, not a mocked or synthetic smoke test.
- Full local suite after all patches: **70 tests pass** (20 in `tests/l37/`, 8 in `tests/l18/`, 42
  pre-existing binder-pipeline tests) — zero regressions.

### 5.5 What remains (updated open items)

- **Stage B/D at scale** — only 1 peptide has run a training step so far (×3 different peptides,
  1 step each); a real training run (many steps, real loss curve, convergence check) has not.
- **Sampling from the distilled student** (`sample_ode` at low NFE using the trained student's
  weights) — not yet run; needed before any eval (Stage E) can compare distilled-student vs.
  step-truncated-teacher trajectories.
- **Stage E (eval)** — blocked on pyEMMA locally (§5.1); needs a different machine/toolchain, or a
  from-source pyEMMA build with an older Clang, or a lighter reimplementation of just the
  TICA+MSM+timescales calls this project needs (not attempted — reusing MDGen's own eval script
  is still preferred once a working toolchain is found).
- **Compute for a real (not 3-peptide, 1-step) run** — mechanics are proven on CPU/MPS; a full
  pilot (100 test peptides × many steps × the NFE/control conditions in §1 Stage D) will be slow
  on this laptop. Whether to run a longer pilot here anyway, or move to different hardware, is
  still open — this session validated correctness, not throughput.
- **Peptide count / statistical power** for the eventual full pilot beyond the 3-peptide mechanics
  check remains a judgment call, not a power calculation.

### 5.6 Independent concurrent audit (2026-07-15) — `docs/L37_LOSS_PREMISE_AUDIT.md`

A separate, concurrently-running Codex session in this same repo (confirmed via process/telemetry
inspection, not dispatched by this session) produced `docs/L37_LOSS_PREMISE_AUDIT.md`, a rigorous
follow-on to `L37_MATH_SOLUTION_V2.md`'s still-open "exact loss-to-segment-KL premise" item — it
analyzes the actual `consistency_distillation_loss` built in §5.3 above. Two claims spot-checked
against the real code/repo this session (both confirmed true):

1. **`sim_inference.py` had no `--inference_steps` override** — Gate 0's patch added the flag to
   `mdgen/parsing.py` (used by `train.py`) and wired it in `wrapper.py`, but the standalone
   sampling script has its own separate, smaller argparse block that never got the flag. **Fixed
   this session**: added `--inference_steps` to `sim_inference.py`, applied as an explicit
   override on the loaded model's args after `load_from_checkpoint` (same pattern already used
   ad hoc in the §5.4 smoke tests). Verified against the real checkpoint.
2. **For the default `dopri5` adaptive solver, `num_steps` is the number of requested/interpolated
   output points, not the number of model evaluations** — confirmed directly from
   `Sampler.sample_ode`'s own docstring. **This means Stage D's planned "NFE sweep" is not
   actually an NFE sweep under `dopri5`** — varying `num_steps` there changes output granularity,
   not adaptive-solver function-evaluation count. Genuine correction to §1 Stage D, not yet
   resolved: either switch to the fixed-step `euler` sampling_method for true NFE control, or
   instrument dopri5's actual eval count directly. **Open decision for the next work session.**

The audit's broader thesis — that no loss currently proves the end-to-end kinetic-error theorem
(distinguishing student-vs-teacher error from student-vs-physical-MD error, and requiring premises
on stationarity/density/isolated-spectrum this pipeline doesn't establish) — is consistent with,
and sharpens, this ledger's existing "method-only, no theorem claimed" scope
(`docs/RESEARCH_LEDGER.md` "FINAL DECISION (2026-07-15)"). It does not change this pipeline's
plan: Stage C's consistency loss remains the empirical training objective; §3 above already states
no bound is being validated. Its Route A (a fixed, pre-registered MSM-based `L_chi` certificate
comparing the student directly against held-out physical MD, not just the teacher) is a
substantive design idea for a stronger empirical/theoretical hybrid result, not yet incorporated.

**Process note:** two agents (this session + the concurrent Codex session) are editing the same
repo without coordination. No conflicting edits occurred this time (the concurrent session only
added a new file), but this is a real collision risk going forward — worth surfacing to the user.

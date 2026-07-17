# L35 Pipeline Spec — Few-Step Consistency Distillation of DiffCSP

Status: **mechanics validated end-to-end on real artifacts, real MPS hardware.
No real training run yet.** Grounded against the live `jiaor17/DiffCSP` repo
(commit `7121d15`, 2024-04-27), not memory. Written for the LoG 2026 Extended
Abstract Track (abstract due 2026-07-29, full paper 2026-08-01).

## Why L35, not L37

L37 (MDGen kinetics distillation) was the prior candidate. Checked against
three past LoG proceedings (2022 v198, 2023 v231, 2024 v269, 141 papers total):
**1/141 papers touches MD/trajectory/kinetics; 8/141 touch molecules/proteins
at all, and every one of those 8 is built on a real graph/message-passing
architecture.** MDGen's own architecture (`third_party/mdgen/mdgen/model/ipa.py`,
`mha.py`) is IPA + attention — zero graph components (`edge_index`, message
passing, `torch_geometric`). L37 would read as a paper at the wrong venue.
DiffCSP's denoiser (`CSPNet`, below) is a genuine periodic-boundary
message-passing GNN — a natively on-topic LoG submission.

## The teacher: DiffCSP, CSP task, MP-20

- Repo: `github.com/jiaor17/DiffCSP`, MIT license, vendored at
  `third_party/diffcsp` (shallow clone, commit `7121d15`).
- Architecture: `diffcsp/pl_modules/cspnet.py` — `CSPNet`/`CSPLayer`, real
  message passing (`edge_model`/`node_model`, `scatter(..., reduce='mean')`
  aggregation over an explicit `edge_index`). Default config
  (`conf/model/decoder/cspnet.yaml`): `hidden_dim=512, num_layers=6,
  edge_style=fc` (fully-connected graph — no PBC neighbor search needed for
  this dataset scale).
- Diffusion wrapper: `diffcsp/pl_modules/diffusion.py::CSPDiffusion`. Two
  independent noise tracks, confirmed by reading `forward()`/`sample()`
  directly:
  - **Lattice** (3×3 matrix): VP/DDPM, epsilon-prediction. `forward()`:
    `loss_lattice = mse(pred_l, rand_l)` — `pred_l` IS the noise estimate.
  - **Fractional coordinates**: VE-SDE / annealed Langevin on the wrapped
    torus (mod 1). `sample()`'s predictor step cites "Score-Based Generative
    Modeling through SDEs" (Song & Ermon) directly in a code comment.
    `pred_x` is a **normalized** score; `sample()` rescales it via
    `pred_x = pred_x * torch.sqrt(sigma_norm)` before use — this
    un-normalization is load-bearing and was lifted verbatim into the
    training step, not re-derived.
- Teacher checkpoint: `mp_csp` (CSP task, MP-20), from the paper's own
  Google Drive release (README-linked), downloaded via `gdown`.
  `third_party/diffcsp_checkpoints/mp_csp/last.ckpt`, **sha256
  `c06719217c940718823f7c67b58c74a0fcfba4c3e4312b23cdd22b7a15ea6b5f`**
  (47M, verified in `tests/l35/test_real_checkpoint_load.py`). Config
  (`hparams.yaml`, same directory): `hidden_dim=512, num_layers=6,
  edge_style=fc, timesteps=1000, sigma_begin=0.005, sigma_end=0.5` — matches
  the README's documented default exactly.
- Data: MP-20 (45,231 structures) is bundled directly in the repo as CSV
  (`third_party/diffcsp/data/mp_20/{train,val,test}.csv`) — real CIF strings,
  no external download needed.

## What's been validated (all real artifacts, all tests pass)

29 tests across `tests/l35/` (run via `.venv-diffcsp`, a dedicated venv —
frozen at `third_party/diffcsp_requirements_freeze.txt`):

1. **`torch_scatter` compat shim** (`src/l35/torch_scatter_compat.py`,
   `torch_scatter_compat_shim.py`). `torch_scatter` is a C++ extension that
   fails to build on this machine — pybind11 header conflict vs. torch 2.13
   on Apple Silicon (`redefinition of 'remove_class'` etc.), same class of
   toolchain wall as L37's pyEMMA/Clang-21 problem. DiffCSP's default config
   (`edge_style='fc'`) exercises exactly one `torch_scatter` call —
   `scatter(..., reduce='mean')` — which `torch.Tensor.scatter_reduce_` covers
   natively (`include_self=False` matches torch_scatter's untouched-group-is-
   zero, not NaN, semantics). Verified numerically identical to a manual
   groupby-mean reference on both CPU and real MPS. The shim installs fake
   `torch_scatter`/`torch_scatter.composite` modules into `sys.modules`
   *before* import, so the vendored DiffCSP source runs completely
   unmodified; `scatter_softmax`/`segment_coo`/`segment_csr` (unused on the
   default `fc` path) are wired to raise loudly if ever actually called,
   rather than silently returning a wrong answer.
   Consequence: **no need to fight DiffCSP's pinned old deps (torch==1.9.0,
   torch-geometric==1.7.2)** — the modern stack (torch 2.13, PyG 2.8) works
   with this one small, verified shim.
2. **Real DiffCSP model imports** (`test_diffcsp_imports.py`). `CSPNet`
   forward pass and full `CSPDiffusion` module instantiation, from the
   unmodified vendored source, on modern torch+PyG. Also surfaced and fixed
   two real environment quirks: `diffcsp.common.utils` requires a
   `PROJECT_ROOT` env var (per the repo's own `.env.template`) and does
   `os.chdir(PROJECT_ROOT)` as an *import side effect* — every test fixture
   here sets the env var and restores cwd on teardown.
3. **Real data slice** (`test_real_data_slice.py`). A handful of real MP-20
   test-split rows through DiffCSP's own `process_one()` — real CIF parsing
   via pymatgen, real graph construction — without preprocessing the full
   45k-structure dataset.
4. **Real checkpoint load** (`test_real_checkpoint_load.py`). `mp_csp`
   loads into a from-source `CSPDiffusion` with `strict=True` — exact
   parameter-name and shape match, zero missing/unexpected keys. Confirms
   finite losses on a real 4-structure batch.
5. **Consistency-distillation math core** (`src/l35/consistency_distill.py`,
   12 tests). Derivation:
   - **Lattice** (`lattice_x0_estimate`, `lattice_ddim_step`): DDIM/Tweedie
     x0-estimate. `alphas_cumprod[0]==1` exactly (zero-padded in
     `BetaScheduler.__init__`), so the boundary condition
     `f(l_0, ac_t{=}1) == l_0` holds **exactly for any pred_eps** — no
     c_skip/c_out blend needed, this formula already IS a valid consistency
     function by construction. DDIM as the deterministic distillation
     trajectory for a discrete-time epsilon-prediction VP model is directly
     precedented by Latent Consistency Models (Luo et al. 2023, verified via
     the paper's Eq. 9 and its stated `c_skip(0)=1, c_out(0)=0` boundary
     condition) — same setup as DiffCSP's lattice track.
   - **Coordinates** (`coord_x0_estimate`, `coord_pfode_step`): Tweedie
     x0-estimate, wrapped mod 1. `sigmas[0]==0` exactly (zero-padded in
     `SigmaScheduler.__init__`), so the score term's coefficient
     (`sigma_t**2`) is exactly zero at the boundary — `f(x_0, sigma_t{=}0) ==
     x_0` holds exactly and unconditionally, *regardless* of a real, flagged
     approximation error away from the boundary (Tweedie's formula is exact
     for a Gaussian kernel; DiffCSP's forward process wraps mod 1, so the
     true kernel is a wrapped normal, not Gaussian — the same
     wrapped-vs-Gaussian caveat noted for L18/L11 elsewhere in this
     project's research ledger). The deterministic step
     (`coord_pfode_step`) is the VE probability-flow ODE (Song et al. 2021,
     Eq. 13, verified via direct fetch: `dx = -0.5*g(t)^2*score*dt` for
     zero-drift VE) — exactly **half** of DiffCSP's own predictor
     `step_size = sigma_t**2 - sigma_next**2` (`diffusion.py::sample()`),
     with the stochastic term dropped.
   - Both boundary formulas are algebraic identities, tested as such (exact
     equality regardless of an adversarially large/garbage network output at
     the boundary) — not approximate/tolerance-based assertions.
6. **Training step** (`src/l35/training_step.py`, 5 unit tests + 2
   integration tests). Composes the math core with `CSPNet`'s real calling
   convention. One real bug caught and fixed before it shipped: an initial
   draft approximated `sigma_norm ≈ sigma**2`; the real `sigma_norm`
   (`diff_utils.sigma_norm`) is a **10,000-sample Monte Carlo estimate of the
   wrapped-normal score's expected squared norm with no closed form** —
   checked directly (`sigma_norm[500]≈401` vs. `sigma[500]**2≈0.0025`, ~5
   orders of magnitude apart) before the bug reached a passing test. Fixed by
   requiring callers to pass real `BetaScheduler`/`SigmaScheduler` instances;
   `test_loss_uses_real_sigma_norm_not_a_sigma_squared_approximation` is a
   regression test comparing real-vs-naive losses directly. A second bug
   (per-graph sigma broadcast against per-atom coordinate tensors) was caught
   the same way, from a real shape-mismatch `RuntimeError`, not a silent
   wrong answer.
7. **End-to-end integration** (`test_train_distill_integration.py`). Real
   `mp_csp` checkpoint → real teacher decoder (frozen) → student warm-started
   via `copy.deepcopy` → real MP-20 structures → real consistency-
   distillation loss → real `.backward()` → real `Adam.step()`, on real MPS
   hardware where available. Confirmed: teacher receives zero gradient,
   student receives finite gradients on every parameter, and an Adam step
   measurably changes student weights.

## What's still open

- **No real training run** — only mechanics (a handful of structures, a
  few steps) validated. No student checkpoint, no sample-quality numbers.
- **Distillation-target NFE not chosen.** DiffCSP's own paper only ablates
  down to 100 steps (default) vs. 1000/5000 — the 1000→{1,2,4,8} regime is
  fully unclaimed territory per the 2026-07-15 scoop-check, but the actual
  target step count for this submission isn't locked.
- **Eval protocol not written.** DiffCSP's own metrics (match rate, RMSD
  against the ground-truth structure, via `scripts/eval_utils.py` /
  `compute_metrics.py`) exist and are the obvious choice, but haven't been
  wired to run against a distilled student yet.
- **Real A100 training run deferred until compute access is confirmed**, per
  explicit user instruction (2026-07-16). Until then: continue validating
  pipeline mechanics locally on MPS with small real batches; do NOT patch
  `diffusion.py`'s hardcoded `sample()` step loop (needed only for
  generating actual sample structures at reduced NFE, not for the
  consistency-distillation training step itself, which never calls
  `sample()`).
- **Boundary-condition math is newly derived for DiffCSP specifically** — no
  paper does this exact derivation. Built entirely from cited, verified
  sources (DDIM/Song et al. 2020, LCM/Luo et al. 2023, VE-PFODE/Song et al.
  2021) plus DiffCSP's own scheduler code, but flagged here as new work, not
  lifted from an existing paper the way L37 reused MDGen's own GVP
  coefficients.
- **CrystalFlow's code provenance is unverified** (a scoop-check agent found
  no official repo — the codebase floating around, `ixsluo/crystalflow`, is
  an unofficial fork of DiffCSP) — shelved as a teacher candidate; DiffCSP
  remains the sole teacher for this submission.

## Environment

- `.venv-diffcsp` (Python 3.11.15, torch 2.13.0, PyG 2.8.0), separate from
  the main repo venv and from `.venv-mdgen311` (L37's env) — same pattern as
  L37's dedicated venv.
- Full frozen requirements: `third_party/diffcsp_requirements_freeze.txt`.
- Real dependency chain installed clean (no other C++ build failures):
  pandas, pytorch_lightning, networkx, scikit-learn, pymatgen, python-dotenv,
  p_tqdm/pathos, pyxtal, pyyaml.

# L37 Loss and Premise Audit: Exact Contract for an End-to-End MDGen Theorem

**Date:** 2026-07-15

**Status:** Audit complete. The missing contract is now explicit. No current
MDGen or L37 objective satisfies it end to end.

**Depends on:** `L37_MATH_SOLUTION.md`, especially Theorems 1-3 and Section
8; `L37_MATH_SOLUTION_V2.md`, especially No-go Theorem D and Section 6.

## 1. Executive verdict

The phrase "unspecified loss and model-specific premises" concealed four
separate gaps:

1. The L37 prototype does not define a complete population loss. In
   particular, the law of the input state at each flow time is unspecified.
2. Standard deterministic flow matching or consistency MSE does not imply
   either direction of segment KL at finite nonzero loss.
3. Distillation compares the student with an MDGen teacher, while the kinetic
   theorem compares the student with physical MD. KL has no triangle
   inequality, so teacher error cannot be silently omitted.
4. Even a valid segment-KL bound does not imply spectral fidelity without the
   global stationarity, density, and isolated-spectrum premises already exposed
   by the two math resolutions.

The existing papers do not contain a theorem that closes all four gaps.

There are three mathematically valid ways forward:

| Route | Population quantity that must be controlled | What it can prove | Current status |
|---|---|---|---|
| Direct operator certificate | `||U P_S U^-1-P_R||_op`, or the sufficient `D_chi` loss below | Directly supplies the perturbation needed by the spectral theorem | Not implemented; most direct route |
| Exact pair/segment KL | Either exact KL direction for the **deployed** student law | Supplies the perturbation only after global `a,B` density premises | No tractable physical-MD density or certified student density exists |
| Common-diffusion score/drift loss | Likelihood-weighted score MSE or Girsanov drift energy | Supplies target-to-student segment KL for an exact stochastic sampler | Requires a stochastic-model redesign and a finite-step sampler theorem |

The proposed deterministic consistency loss remains usable as an empirical
training objective. At best it supplies a teacher-relative map or Wasserstein
statement under additional assumptions. It cannot be inserted as
`L_distill` in the existing KL-to-spectrum theorem.

## 2. Objects that must be fixed before choosing a loss

Let `c` denote a fixed peptide and all conditioning information. Let

```text
Q_R,c = physical-MD segment law,
Q_T,c = MDGen-teacher segment law,
Q_S,c = law of the actually deployed few-step student.
```

The subscript `R` means physical reference, not MDGen teacher. Let `rho_R,c`,
`rho_T,c`, and `rho_S,c` be the selected lag-pair marginals of those segment
laws. Let `mu_R,c` and `mu_S,c` be the corresponding one-frame marginals when
the pairs are stationary.

The physical lag `tau` is distinct from the generative flow time
`t in [0,1]`. MDGen's flow evolves an entire vectorized physical trajectory
segment from Gaussian noise. A loss integrated over flow time is not itself a
physical-time kinetic loss.

The theorem target must state all of the following:

* whether `c` fixes only peptide identity or also the initial conformation;
* whether the target is physical MD or the MDGen teacher;
* the modeled state space and its sigma-algebra or intrinsic base measure;
* the physical lag and pair projection extracted from a generated segment;
* whether the claim concerns a one-lag pair operator, a finite segment, or an
  actual recursively generated path.

An average over peptide identities does not imply a guarantee for every
peptide. A per-peptide theorem needs per-peptide bounds. For a finite sampling
law with weights `w_c>0`, the weak fallback
`L_c <= L_average/w_c` is valid but can be vacuous for rare peptides.

## 3. The exact operator contract

For one fixed context `c`, the strongest surviving theorem in the existing
solution assumes:

1. `rho_R` is stationary and reversible with transfer operator `P_R` on
   `L2(mu_R)`;
2. `rho_S` is stationary with equal endpoint marginal `mu_S`;
3. `h=dmu_S/dmu_R >= a>0`;
4. relative to `mu_R tensor mu_R`,

   ```text
   q_R = d rho_R/d(mu_R tensor mu_R),
   q_S = d rho_S/d(mu_R tensor mu_R),
   0 <= q_R,q_S <= B < infinity;
   ```

5. the target reference eigenvalue `lambda in (0,1)` is simple and isolated,
   with

   ```text
   delta = dist(lambda,spec(P_R)\{lambda});
   ```

6. the common-space perturbation

   ```text
   epsilon = ||U P_S U^-1-P_R||_op
   ```

   is small enough, where `Uf=sqrt(h)f`.

The existing proof gives the sufficient KL bound

```text
K_op(a,B)
  = 1/a + B(1+a^(-1/2))/(2a^(3/2)),

epsilon
  <= K_op(a,B) sqrt(2 B K),
```

when either pair-KL direction is at most `K`. The required smallness condition
is

```text
epsilon < min(delta/2,lambda,1-lambda).
```

It then gives a unique local real student eigenvalue and the stated
one-lag implied-timescale error bound. Repeated reference eigenvalues support
only cluster and invariant-subspace conclusions.

This is a pair-operator theorem. To identify the eigenvalue with actual
multi-lag relaxation, both physical and generated path laws must additionally
factor through their corresponding time-homogeneous kernels at every horizon.
MFPT and committor claims need the further killed-resolvent hypotheses in
`L37_MATH_SOLUTION.md`.

### 3.1 If the loss targets the MDGen teacher

The physical-MD conclusion additionally needs an operator-level teacher error.
On one common `L2(mu_R)` space, define

```text
eta_teacher = ||B_T-P_R||_op,
eta_student_teacher = ||B_S-B_T||_op.
```

Then, and only then, the triangle inequality gives

```text
||B_S-P_R||_op <= eta_student_teacher + eta_teacher.
```

A student-to-teacher KL bound is not an acceptable replacement for this line
without a separate density/transport lemma. KL itself has no triangle
inequality. The smallness test must use the total physical-reference
perturbation, including teacher and sampler error.

## 4. Route A: direct operator or `D_chi` loss

The near-minimal theorem-facing loss is

```text
L_op,c = ||U_c P_S,c U_c^-1-P_R,c||_op^2.
```

Then `epsilon_c=sqrt(L_op,c)` by definition. This is the cleanest mathematical
contract, although it is not generally a convenient neural-network training
loss.

A stronger but more estimable sufficient loss is

```text
L_chi,c
  = integral |q_S,c(x,y)-q_R,c(x,y)|^2
      dmu_R,c(x) dmu_R,c(y).
```

For a common marginal, `epsilon_c <= sqrt(L_chi,c)`. For different marginals,
the existing proof gives

```text
epsilon_c <= K_op(a_c,B_c) sqrt(L_chi,c).
```

This route needs no loss-to-KL theorem. It directly measures the quantity that
the proof uses.

### 4.1 Finite-state MSM version

For a fixed, preregistered partition with states `i,j`, the loss becomes

```text
L_chi,c
  = sum_(i,j) [rho_S,c(i,j)-rho_R,c(i,j)]^2
      /[mu_R,c(i) mu_R,c(j)].
```

The other constants are directly defined by

```text
a_c = min_i mu_S,c(i)/mu_R,c(i),
B_c = max_(i,j) max(q_R,c(i,j),q_S,c(i,j)).
```

This is the most practical route to a complete theorem in the current
pipeline because `analyze_peptide_sim.py` already constructs MSM transition
matrices and stationary distributions. It should be used as a held-out
certificate comparing the student directly with physical MD; the consistency
loss may remain the optimizer.

The resulting theorem is about that fixed finite-state MSM, not the underlying
continuous molecular dynamics. Zero cells, adaptive cluster selection, and
finite correlated trajectories require simultaneous confidence bounds.
Pseudocounts are an estimator choice, not a proof that the true `a_c` is
positive.

## 5. Route B: exact density-ratio or likelihood loss

For the actual deployed segment laws, either of the following is an exact
loss:

```text
L_KL,c^(S||R)
  = E_(Y~Q_S,c) log[dQ_S,c/dQ_R,c](Y)
  = KL(Q_S,c||Q_R,c),

L_KL,c^(R||S)
  = E_(Y~Q_R,c) log[dQ_R,c/dQ_S,c](Y)
  = KL(Q_R,c||Q_S,c).
```

Data processing gives the same-direction pair-KL bound. Thus `C_loss=1` in
the conditional spectral theorem, provided the global `a_c,B_c` premises
hold.

Ordinary student negative log likelihood is not itself KL:

```text
E_QR[-log q_S] = KL(Q_R||Q_S) + H(Q_R).
```

It supplies an absolute numerical KL bound only if the reference entropy or
reference log density is known or separately bounded.

This route requires:

* normalized densities for both laws on the same state space;
* exact or certified log-density evaluation for the deployed few-step sampler;
* support/absolute-continuity compatibility;
* numerical integration and stochastic trace-estimation error bounds if a CNF
  likelihood is used;
* a full-dimensional density, an intrinsic manifold density, or explicit
  endpoint dequantization.

MDGen contains an ODE-likelihood routine, but it numerically integrates the
flow and uses a Hutchinson divergence estimate. The current checkout has no
`forward_sim.ckpt` with which to validate it. More fundamentally, MDGen's
quaternion and torsion representations are normalized onto constraints during
decoding, so an ambient Euclidean endpoint density needs a precise
dequantization or manifold interpretation. The current one-to-four-step
consistency map has no defined likelihood.

## 6. Route C: common-diffusion score or drift loss

There is a real loss-to-KL theorem in the score-SDE literature, but it applies
to a stochastic model rather than MDGen's deployed deterministic ODE.

Let `p_c` be the target distribution of an entire vectorized physical segment.
Choose a forward noising SDE

```text
dX_t = f(X_t,t) dt + g(t) dW_t,   X_0~p_c,
```

with marginal `p_c,t` and terminal law `p_c,T`. Define the student by the
reverse SDE with score `s_theta`. The likelihood-weighted population loss is

```text
L_SDE,c
  = (1/2) integral_0^T
      E_(X_t~p_c,t)[g(t)^2
        ||s_theta(X_t,t,c)-grad log p_c,t(X_t)||^2] dt.
```

Under the regularity and Novikov assumptions in Song et al.
`arXiv:2101.09258`, exact continuous-SDE sampling gives

```text
KL(p_c||p_theta^SDE)
  <= L_SDE,c + KL(p_c,T||pi).
```

Equivalently, for two reverse SDEs with common diffusion matrix `sigma`,
compatible initial laws, and drift difference in the range of `sigma`,
Girsanov gives

```text
KL(Q_target^path||Q_student^path)
  = KL(nu_target||nu_student)
    + (1/2) E_(Q_target)
        integral ||sigma^dagger(b_target-b_student)||^2 dt.
```

Endpoint or segment KL follows by data processing.

The load-bearing premises are:

* the same nondegenerate diffusion for target and student;
* the displayed target-path expectation, not an arbitrary interpolation law;
* absolute continuity, existence/uniqueness, growth and Lipschitz regularity,
  and the relevant Novikov condition;
* a known terminal-prior mismatch term;
* exact continuous-SDE sampling, or a separate theorem for the law of the
  actual finite-step sampler.

MDGen's `Sampler.sample_sde` and velocity-to-score conversion do not by
themselves establish these premises. The released forward-simulation wrapper
uses `sample_ode`. The native unweighted velocity MSE is not the displayed
likelihood-weighted score or drift loss. For an approximate velocity
checkpoint, the converted quantity is not automatically the score of the
checkpoint's own ODE occupancy.

### 6.1 A finite-step stochastic alternative

For an actual `K`-step stochastic sampler with target transitions `r_k` and
student transitions `s_theta,k`, the exact chain-rule loss is

```text
L_chain,c
  = KL(nu_R||nu_S)
    + sum_k E_(Q_R)
        KL(r_k(.|X_k)||s_theta,k(.|X_k))
  = KL(Q_R^(0:K)||Q_S^(0:K)).
```

For equal-covariance Gaussian transitions, each conditional KL is a weighted
mean-squared drift error. This route controls the deployed finite-step law,
not an ideal continuous SDE. It requires a tractable target chain whose final
state has the desired physical-MD or teacher segment law. No such one-to-four
step chain is currently specified for L37.

## 7. What deterministic MDGen losses actually support

### 7.1 Native MDGen teacher loss

For a data segment `Y`, Gaussian `Z`, and `t~Uniform[0,1]`, MDGen uses

```text
alpha_t = sin(pi t/2),
sigma_t = cos(pi t/2),

X_t = alpha_t Y + sigma_t Z,
U_t = alpha_dot_t Y + sigma_dot_t Z,

L_MDGen
  = E masked_mean ||v_phi(X_t,t;C(Y))-U_t||^2.
```

The mask covers the seven rigid coordinates and valid torsion coordinates,
and the implementation averages per example. The conditioned first physical
frame remains in the loss. Conditioning is embedded; it is not hard clamping
of the generated first frame.

This is conditional flow-matching MSE. Lipman et al.
`arXiv:2210.02747` prove that conditional and marginal flow-matching
objectives have the same gradients and differ by a parameter-independent
constant. They do not prove a finite-loss KL bound. The raw loss contains an
irreducible conditional-variance term; a field-error theorem needs the
population excess above the Bayes optimum, not the unadjusted logged loss.

At inference, MDGen samples Gaussian noise for the entire segment and solves

```text
dX_t/dt = v_phi(X_t,t;C).
```

This is a deterministic probability-flow ODE conditional on its noise and
context. Recursive forward simulation generates a whole segment jointly,
retains its final frame as the next context, and samples fresh noise. The
generated first frame of the next segment is not clamped to the previous
endpoint.

### 7.2 A valid velocity-MSE-to-Wasserstein statement

Let `X_t^T` follow the exact teacher ODE from the shared Gaussian latent, and
define the teacher-occupancy loss

```text
L_v,c
  = integral_0^1 E
      ||v_S(X_t^T,t,c)-v_T(X_t^T,t,c)||^2 dt.
```

If `v_S(t,.,c)` is `L_t`-Lipschitz, a common-noise coupling and Gronwall give

```text
W2^2(Q_T,c,Q_S,c)
  <= [integral_0^1 exp(2 integral_t^1 L_r dr) dt] L_v,c.
```

For constant `L`, the coefficient is `(exp(2L)-1)/(2L)`. This result requires
teacher ODE occupancy in the expectation. MDGen's analytic interpolation
samples have that occupancy only for the exact population flow field; an
approximate checkpoint needs an occupancy-density-ratio argument.

This controls the whole segment as one vector. Passing it through the MDGen
decoder requires a global decoder Lipschitz constant or norm floors for every
quaternion and torsion normalization. It still does not imply KL, `D_chi`, or
spectral fidelity.

### 7.3 A valid idealized consistency-to-Wasserstein statement

On one compatible teacher grid `X_0,...,X_N`, let the same consistency map
`F_theta` satisfy `F_theta(X_N,1)=X_N`, and define

```text
ell_n
  = E ||F_theta(X_n,t_n)-F_theta(X_(n+1),t_(n+1))||^2,

L_cons = (1/N) sum_n ell_n.
```

Telescoping along that same trajectory gives

```text
W2^2(Law(F_theta(X_0,t_0)),Law(X_N))
  <= N^2 L_cons.
```

The current prototype does not yet instantiate this population loss. It
receives `x_n` from its caller rather than sampling the corresponding teacher
occupancy; it uses one Euler teacher step; it has no EMA definition, MDGen
conditioning adapter, or student sampler. Solver, target-network, and
multistep-sampling errors must be included if those objects differ from the
idealized statement.

Song et al. `arXiv:2303.01469` prove a different zero-population-loss,
uniform-Lipschitz map result with numerical-solver error
`O(Delta t^p)`. Lyu, Chen, and Feng `arXiv:2308.11449` give finite-residual
Wasserstein results under an OU-specific setup and per-timestep assumptions.
Neither paper supplies raw-output KL for the current MDGen loss.

## 8. MDGen-specific premises: found versus missing

| ID | Required fact | Evidence found | Exact missing human input |
|---|---|---|---|
| TARGET-1 | Reference target | Specs intend physical MD; code distills MDGen | Choose physical-MD-relative or teacher-relative theorem in writing |
| TARGET-2 | Context and scope | MDGen conditions on sequence, first-frame poses, and first-frame latent tokens | Define `c`, initial-frame law, physical lag, pair projection, and per-peptide versus average claim |
| STATE-1 | Common state space | Decoder uses rigid offsets and normalized torsions; trajectories are aligned | Define intrinsic manifold/quotient, finite MSM partition, or dequantized Euclidean state and base measure |
| CKPT-1 | Exact teacher | No checkpoint is present in this checkout | Supply checkpoint file/hash and serialized hyperparameters |
| DATA-1 | Teacher/data compatibility | Released `forward_sim.ckpt` is documented with explicit-solvent `4AA`; the pipeline selected implicit-solvent data | Use matching explicit data, or supply and identify an implicit forward-simulation checkpoint |
| LOSS-1 | Complete population loss | Only an algebraic adjacent-step kernel exists | Select Route A, B, or C and specify every expectation law and normalization |
| STUDENT-1 | Deployed law | Student architecture and inference sampler are absent | Define one/few-step update, all injected noise, conditioning, decoder, and recursive rollout |
| FLOW-1 | Current CD state law | `x_n` is a caller argument | Specify teacher ODE/grid occupancy and how it is sampled |
| EMA-1 | Target network | Prototype uses the current student under stop-gradient | Supply EMA update, decay schedule, convergence premise, or use a fixed target |
| NUM-1 | Numerical semantics | `dopri5` is adaptive; requested output times are not NFE | Define solver, tolerances, actual NFE measurement, and a bound for the deployed sampler law |
| REF-1 | Reference stationarity/reversibility | Ideal equilibrium pair facts are discussed; literal constrained/projected MD is not certified | Supply ensemble, equilibration, time-reversal argument, and observed-state pair symmetry |
| STAT-1 | Student stationarity | Not enforced; rollout starts from a data frame | Supply invariant law or stationarity construction and equal endpoint marginals |
| DENS-1 | Global `a,B` or direct operator control | No global constants; local density facts were insufficient | Supply per-context certified constants or `L_op/L_chi` bounds |
| SPEC-1 | Isolated mode | Not measured per target/context | Supply `lambda`, simplicity, isolation gap `delta`, and ordering gap if called the leading mode |
| TEACHER-1 | MDGen-to-MD error | Absent | Supply direct operator error on the common state space, or certify the student directly against MD |
| DECODER-1 | Latent-to-state stability | Quaternion/torsion normalization can be singular at zero norm | Supply norm floors/Lipschitz bounds, or formulate the certificate after decoding |
| PATH-1 | Actual kinetic path | Joint segment model is not first-order Markov; endpoint recursion is a separate skeleton | Choose pair-proxy scope or prove the relevant all-horizon homogeneous Markov factorization |
| EST-1 | Population-to-estimate validity | Current metrics are point estimates | Supply dependence-aware confidence bounds for `L_chi`, `a`, `B`, `lambda`, and `delta` |

## 9. Checkpoint and sampler corrections to the pipeline

Three implementation facts affect the mathematical target:

1. The documented `forward_sim.ckpt` uses explicit-solvent `4AA` data sampled
   at 10 ps intervals. The implicit-solvent command in the MDGen README is the
   separate upsampling configuration with conditioning every 100 frames.
2. The local worktree now passes `self.args.inference_steps` into
   `sample_ode` and defines that argument in `mdgen/parsing.py`, but
   `sim_inference.py` has no CLI override. A checkpoint-loaded value therefore
   still controls the run unless the script or loaded args are changed.
3. For adaptive `dopri5`, `num_steps` is the number of requested output times,
   not the number of model evaluations. A theorem or speed claim must record
   the solver's actual NFE.

These are model-definition issues, not minor experiment bookkeeping.

## 10. Paper audit

| Source | What it actually supplies | What it does not supply |
|---|---|---|
| MDGen, `arXiv:2409.17808` and released code | Conditional GVP flow-matching objective and deterministic ODE segment sampler | Finite-loss KL, student stationarity, teacher-to-MD certificate |
| Consistency Models, `arXiv:2303.01469`, Theorem 1 | Zero-CD-loss plus Lipschitz and solver consistency implies uniform map error | Finite-positive-loss KL or TV |
| Lyu, Chen, Feng, `arXiv:2308.11449` | Finite-residual endpoint Wasserstein under OU-specific assumptions; TV only after smoothing/correction | Raw-output KL for MDGen CD |
| Flow Matching, `arXiv:2210.02747`, Theorems 1-2 | Conditional vector fields generate the marginal path; CFM and FM have identical gradients | Nonzero velocity-loss-to-KL theorem |
| Maximum Likelihood Training of Score-Based Diffusion Models, `arXiv:2101.09258`, Theorem 1 | Likelihood-weighted score MSE upper-bounds target-to-model KL for an exact reverse SDE | Deterministic ODE or few-step consistency-map guarantee |
| Score-Based Generative Modeling through SDEs, `arXiv:2011.13456` | Reverse SDE and probability-flow ODE framework | A finite MDGen velocity-loss-to-KL bridge |
| Progressive Distillation, `arXiv:2202.00512` | Practical weighted regression for matching teacher steps | Distributional or kinetic theorem |

The vague prior reference to a "Li/Chi" consistency-convergence result could
not be matched to a relevant theorem. The identifiable paper is Lyu, Chen, and
Feng, `arXiv:2308.11449`, and its conclusion is Wasserstein, not KL.

## 11. Recommended next decision

If the goal is a defensible theorem without redesigning MDGen, use Route A on
a fixed MSM:

1. Freeze peptide set, lag, features, clustering, and state partition before
   looking at student results.
2. Compare the deployed student directly with physical MD, not only with the
   MDGen teacher.
3. Generate or reweight both pair tables to stationary endpoint marginals.
4. Produce dependence-aware upper confidence bounds for `L_chi` and lower/
   upper bounds for `a,B`.
5. Verify the reference mode is simple and isolated and evaluate the explicit
   smallness condition.
6. State the result as an MSM pair-spectrum theorem. Add path-level language
   only if the generated process used by the claim is demonstrably the
   corresponding homogeneous Markov chain.

If the goal is a theorem whose right side is the neural training loss, Route C
is the clean literature-supported option, but it changes the method to a
stochastic score/drift model and still needs a theorem for the actual
one-to-four-step sampler.

If the deterministic consistency method is retained unchanged, the honest
theoretical endpoint is a teacher-relative Wasserstein or bounded-observable
statement. It is not an end-to-end physical-MD spectral theorem.

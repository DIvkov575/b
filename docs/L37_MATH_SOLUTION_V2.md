# L37 Math Resolution v2: The MD-Realism Gap

**Date:** 2026-07-15

**Resolves:** `docs/L37_MATH_SPEC_V2.md`

**Depends on without re-proving:** `docs/L37_MATH_SOLUTION.md`, especially
Theorems 1-3, No-go Theorems A-B, and Section 8.2

## 1. Verdict

The proposed local-boundedness plus anti-trapping repair does not produce an
MDGen-realistic end-to-end timescale theorem. Local boundedness is valid for an
ideal continuous Langevin diffusion under explicit regularity assumptions, but
mere confinement of a molecular potential does not provide a quantitative
anti-trapping bound. More decisively, neither a generic few-step sampler nor a
small average distillation or segment-KL loss has any automatic anti-trapping
property.

| Gate | Verdict | Reason |
|---|---|---|
| A | **PROVED** | MDGen uses per-residue offsets relative to the same residue in temporal key frames. It is neither backbone-threaded chain-relative nor a raw lab-frame representation. Tetrapeptide data use a Langevin thermostat, with implicit-solvent NVT and explicit-solvent NPT production. The protein ATLAS data use Nose-Hoover/Parrinello-Rahman NPT production. None is NVE. |
| B | **PROVED CONDITIONALLY** | The compactness lemma is immediate and does not require positivity of `p`. Continuous underdamped Langevin has a continuous phase-space density under standard Hormander/admissible-potential hypotheses. Those theorems do not directly cover MDGen's literal constrained splitting integrator, Monte Carlo barostat, hidden solvent/momenta, trajectory alignment, or position-only projection. |
| C | **BLOCKED** | Cheeger converts a *global* conductance lower bound into a gap; it does not derive that bound from local geometry. Foster-Lyapunov drift also needs a quantitative small-set minorization. Confinement at infinity alone permits an arbitrarily rare well behind an arbitrarily high barrier and hence an arbitrarily small gap. |
| D | **BLOCKED** | An explicit one-step student has both pair-KL directions, and both fixed-horizon full-segment KL directions, tending to zero while its nontrivial eigenvalue tends to one. Average consistency/flow losses likewise cannot control behavior on a training-measure-small set without a uniform structural constraint. |
| E | **BLOCKED** | Local `D_chi` plus tail escape does not yield the global operator perturbation required by Theorems 1-3. A full theorem survives only after imposing stronger global compactness/density/stationarity conditions and an external loss-to-KL premise. Those are not properties of generic few-step distillation. |

The pre-registered fallback in `L37_MATH_SOLUTION.md` Section 8.2 therefore
remains the strongest assumption-light conclusion for an MDGen-style model.

## 2. GATE A: model facts

### Verdict: PROVED

### 2.1 Coordinate representation

The dichotomy in the v2 spec is not exhaustive.

MDGen first constructs a backbone rigid frame `g_t^j in SE(3)` for each residue
`j` at each physical time `t`. Given temporal key frames `t_1,...,t_K`, its
generated token is

```text
([g_{t_1}^j]^{-1} g_t^j, ..., [g_{t_K}^j]^{-1} g_t^j, torsions_t^j).
```

This is equation (4) in Section 3.1 of the MDGen paper. The reference and target
poses have the **same residue index `j`**. Thus:

1. It is not a chain-relative representation of residue `j` against residue
   `j-1`.
2. It is not a raw fixed lab-frame representation either.
3. It is a temporal key-frame-relative representation, invariant to a common
   left action of `SE(3)`.

The released code implements exactly this operation:

```text
get_offsets(ref_frame, rigids)
    = ref_frame.invert().compose(rigids)
```

and normally takes `ref_frame=rigids[:,0:1]`; interpolation also makes offsets
from the last temporal frame. Decoding left-composes the generated offset with
the first key frame. See `mdgen/utils.py` lines 7-14 and
`mdgen/wrapper.py` lines 292-317 and 456-470 at repository commit
`81482a403b91c5a8437046da1d8d321ba97089cc`.

The released simulation script superposes each trajectory before saving the XTC
used downstream, and the preparation script superposes again
(`scripts/run_peptide_sim.py`, lines 124-127; `scripts/prep_sims.py`, lines
69 and 75). Thus the prepared training trajectories have overall rigid drift
removed. The original HDF5 reporter output is written before that alignment, so
it should not itself be called an aligned raw trajectory. Alignment still does
not compactify the model's translation channel: equation (4) explicitly places
every key-frame-relative translation in `R^3`. A generated residue can move
arbitrarily far from its own key-frame pose, and no neighboring-residue
coordinate identity prevents a chain break. Consequently the proposed
"chain-relative coordinates make translation intrinsically compact" shortcut is
unavailable.

### 2.2 Ensemble and thermostat

For the tetrapeptide data, MDGen Appendix B.2 states:

* OpenMM with Amber14 and either GBN2 implicit solvent or TIP3P-FB explicit
  solvent;
* a Langevin thermostat at 350 K;
* 2 fs integration, hydrogen-bond constraints, and friction `0.1 ps^-1`
  (implicit) or `0.3 ps^-1` (explicit);
* 20 ps NVT equilibration;
* 100 ns production in NVT for implicit solvent and NPT with a Monte Carlo
  barostat at 1 bar for explicit solvent.

The released code agrees: it constructs `LangevinMiddleIntegrator`, runs 10,000
2-fs NVT steps, then adds a Monte Carlo barostat for explicit solvent before
production (`scripts/run_peptide_sim.py`, lines 67-121).

Thus the answer is not simply "NVT":

* implicit-solvent tetrapeptide production is thermostatted **NVT**;
* explicit-solvent tetrapeptide production is thermostatted/barostatted
  **NPT**;
* neither is NVE.

MDGen also trains its protein model on ATLAS trajectories. MDGen Appendix B.3
delegates their protocol to the ATLAS dataset. The ATLAS paper's "Molecular
dynamics simulation protocol" states that production is NPT using a
Nose-Hoover thermostat and Parrinello-Rahman barostat after NVT and NPT
equilibration. This is also not NVE, but it is not stochastic Langevin.

### 2.3 Consequences for the standing mathematical assumptions

For continuous NVT underdamped Langevin on full phase space, the invariant law
is canonical,

```text
pi(dq dp) proportional to exp[-beta(U(q)+p^T M^{-1}p/2)] dq dp.
```

The process obeys generalized detailed balance under momentum reversal, not
ordinary self-adjointness on phase space. If `Theta(q,p)=(q,-p)`, the
generalized detailed-balance identity reverses a stationary phase-space pair
and applies `Theta` to both endpoints. Integrating that identity over both
momenta removes the two `Theta` operations, so the equilibrium
**position-pair law** is symmetric in its two positions. Its one-lag pair
operator can therefore be self-adjoint, although the projected position process
is generally not Markov.

For NPT data, the invariant law includes box/volume and pressure variables; it
is not the fixed-volume density `Z^-1 exp(-beta U(q)) dq`. After discarding
solvent, hydrogens, momenta, thermostat/barostat variables, and box variables,
the peptide-coordinate marginal is a potential-of-mean-force distribution, not
the bare Amber or CHARMM force-field density appearing in the original spec.

Finally, a finite-step numerical splitting scheme need not preserve the exact
continuous-time Gibbs law or generalized detailed balance. Those properties
must be checked for the actual integrator if the theorem is intended literally,
rather than assumed for an idealized continuous Langevin reference.

## 3. GATE B: local boundedness

### Verdict: PROVED CONDITIONALLY

The elementary lemma is proved exactly. Its application to ideal continuous
Langevin is standard under explicit hypotheses. Its application to the literal
MDGen reference pipeline is not established by the cited diffusion theorems.

### 3.1 Compactness lemma

Let `v` be the base measure with respect to which `mu` has density `m(y)` and
the kernel has density `p(y|x)`. Assume:

1. `Omega_R` is compact;
2. `m` is continuous and strictly positive on a neighborhood of `Omega_R`;
3. `p` is jointly continuous on `Omega_R x Omega_R`.

Then

```text
m_R := min_{y in Omega_R} m(y) > 0,
M_R := max_{(x,y) in Omega_R^2} p(y|x) < infinity.
```

Both statements are the extreme-value theorem. Therefore

```text
0 <= q(x,y) = p(y|x)/m(y) <= M_R/m_R =: B_R
```

on `Omega_R x Omega_R`.

Everywhere positivity of `p` is **not needed** for this upper bound. It would be
relevant to lower bounds and minorization. The assumptions also must say which
base measure is used. With holonomic constraints, it is manifold volume rather
than ambient Lebesgue measure.

### 3.2 Ideal continuous underdamped Langevin

In unconstrained coordinates, write

```text
dQ_t = M^{-1} P_t dt,
dP_t = -grad U(Q_t) dt - Gamma M^{-1}P_t dt
       + sqrt(2 beta^{-1} Gamma) dW_t,
```

with positive-definite `M` and full-rank friction/noise on every momentum
direction. The diffusion vector fields span the momentum directions. If `X_0`
denotes the drift and `X_i` a momentum-noise direction, then

```text
[X_i, X_0]
```

has a nonzero position component proportional to `M^{-1}e_i`. Hence the noise
directions and their first brackets with the drift span all position and
momentum directions. This is the parabolic Hormander bracket condition.

Hairer, *On Malliavin's proof of Hormander's theorem*, Theorem 1.3
(bounded derivatives of all orders) and the more flexible Theorem 4.5
(Assumption 4.2: smooth coefficients plus the stated moment and derivative-flow
bounds) then give a smooth transition density at every `t>0`. This is a
smoothness theorem, not an everywhere-positivity theorem and not a quantitative
Aronson bound.

A source closer to molecular potentials is Herzog and Mattingly,
*Ergodicity and Lyapunov functions for Langevin dynamics with singular
potentials*. Their Definition 2.3 assumes, in particular:

* `U` is smooth on the open finite-energy domain `O`;
* `O` is path connected;
* energy sublevel sets are precompact;
* the Gibbs density is integrable; and
* along every sequence with `U -> infinity`,
  `|grad U| -> infinity` and
  `|Hess U|/|grad U|^2 -> 0`.

Their Proposition 2.11 proves for the continuous underdamped process that, for
every `t>0`, the phase-space transition law has full support, is absolutely
continuous, and has a density `r_t(z,z')` continuous jointly in
`(t,z,z')`. The paper verifies its admissibility assumptions for a class with
polynomial external confinement and Lennard-Jones-type repulsion.

Full support plus continuity does not by itself imply
`r_t(z,z')>0` at every individual pair. Proposition 2.11 therefore does not
prove the everywhere-positive clause proposed in the v2 spec. That clause is
unnecessary for the upper bound: on any compact **phase-space** set
`K subset X`,

```text
sup_{K x K} r_tau(z,z') / pi(z') < infinity.
```

This is a fully rigorous local boundedness result for that idealized reference.
Norris's lemma is one ingredient used to control inverse Malliavin covariance in
proofs of Hormander smoothing; it is not itself a two-sided heat-kernel estimate.
No Aronson-type theorem is needed merely to establish finiteness on a compact
set.

### 3.3 Why this does not literally settle MDGen's configuration kernel

There are five nontrivial changes between the preceding theorem and the MDGen
object:

1. **Projection.** MDGen retains peptide heavy-atom positions but discards
   momenta, solvent, hydrogens, thermostat variables, and barostat variables.
   A marginal density exists by Fubini when the full joint has a density, but
   joint continuity after integrating unbounded hidden variables requires a
   locally uniform integrable heat-kernel bound. Proposition 2.11 alone does not
   state that bound.
2. **Rigid symmetries.** A bare molecular force field is invariant under global
   translation and ordinarily global rotation. The noncompact translation
   symmetry alone makes its lab-frame energy sublevels non-precompact and its
   Gibbs density non-normalizable; rotation adds a compact redundancy. One must
   work in a periodic cell, fix a center/orientation, or construct a quotient
   process. Superposition in a saved trajectory is preprocessing; it does not by
   itself prove that the aligned coordinates follow such a Markov diffusion.
3. **Noncompact hidden momentum.** Even after quotienting rigid motion, a set
   `{U<=R}` is compact only in position. The corresponding full phase-space set
   still has unbounded momentum. The phase-space compactness result applies to
   a Hamiltonian sublevel such as `{U+K<=R}`, not directly to `{U<=R}`.
4. **Constraints and discretization.** OpenMM uses hydrogen-bond constraints
   and a finite-step `LangevinMiddleIntegrator`. The continuous unconstrained
   bracket calculation does not prove a density for that constrained splitting
   chain. A separate rank/bracket theorem on the constraint manifold, or a
   theorem for this exact integrator, is needed.
5. **NPT and deterministic thermostat data.** The explicit-solvent chain also
   has Monte Carlo volume moves. The ATLAS protein data use deterministic
   Nose-Hoover/Parrinello-Rahman dynamics, to which velocity-noise
   hypoellipticity does not apply at all.

Thus Gate B is proved for an ideal continuous Langevin reference on full phase
space, and for a position-only kernel if one additionally assumes the required
projection continuity. It is not yet a theorem about the literal MDGen data
pipeline.

### 3.4 Overdamped alternative

For

```text
dQ_t = -grad U(Q_t) dt + sqrt(2 beta^{-1}) dW_t,
```

the diffusion matrix is uniformly elliptic, so the noise fields already span
the tangent space without taking brackets. Under the smoothness,
nonexplosion, and derivative-flow moment hypotheses in Hairer Assumption 4.2,
Theorem 4.5 gives a smooth density at every positive time. Joint heat-kernel
continuity, if assumed or supplied by the applicable elliptic theorem, then
lets Section 3.1 apply directly in configuration space. Pointwise positivity is
again a separate conclusion and is not needed for this upper bound.

MDGen's tetrapeptide source is explicitly underdamped Langevin, so overdamped
dynamics is a possible coarse model, not the actual stated reference.

### 3.5 Quantitative realism

Even where `B_R<infinity` is proved, the argument gives

```text
B_R = (local heat-kernel maximum)/(local equilibrium-density minimum).
```

In many dimensions the denominator can be exponentially small in `R` and the
heat-kernel constant can have severe short-time, dimension, and force-derivative
dependence. Hypoellipticity establishes existence; it does not show that the
constant entering Theorem 2 is numerically useful at MDGen scale.

## 4. GATE C: reference anti-trapping

### Verdict: BLOCKED

Both proposed tools are valid, but neither derives a non-vacuous reference gap
from the generic statement "the force field is confining." In fact that
implication is false.

### 4.1 Exact Cheeger statement

For a finite reversible chain with stationary law `pi`, define

```text
Q(S,S^c) = sum_{x in S, y in S^c} pi(x)P(x,y),
Phi(S)   = Q(S,S^c)/pi(S),
Phi_*    = min_{0<pi(S)<=1/2} Phi(S).
```

Levin, Peres, and Wilmer, *Markov Chains and Mixing Times*, second edition,
Theorem 13.10, states, for `gamma=1-lambda_2`,

```text
Phi_*^2/2 <= gamma <= 2 Phi_*.
```

They attribute the result to Sinclair-Jerrum and Lawler-Sokal. Lawler and Sokal,
*Bounds on the L2 spectrum for Markov chains and Markov processes: a
generalization of Cheeger's inequality*, Trans. AMS 309 (1988), treats general
reversible Markov kernels, with constants depending on convention.

This is the right theorem **after a global conductance lower bound has been
proved**. The infimum defining `Phi_*` ranges over every measurable set,
including bottlenecks inside the core and rare sets outside it. A local statement
that one selected region has no bottleneck does not lower-bound this global
infimum.

For a reversible kernel, a genuinely uniform tail escape condition does have a
useful local consequence. If `D=Omega_R^c` and

```text
P(x,D) <= 1-kappa for every x in D,
```

then the killed tail operator `K_D=1_D P 1_D` satisfies

```text
||K_D||_{L2(mu)->L2(mu)} <= 1-kappa.
```

Indeed, Jensen gives

```text
|K_D f(x)|^2 <= P(x,D) P(1_D f^2)(x).
```

Integrating over `D`, using reversibility, and applying the same escape bound
again gives `||K_D f||_2^2 <= (1-kappa)^2 ||f||_2^2`.
This rules out a mode supported entirely in the excluded region above
`1-kappa`. It does not compare the full reference and student operators or
control core-tail coupling.

### 4.2 What Foster-Lyapunov actually supplies

The drift condition alone is not the complete theorem. The user's displayed
drift inequality implies the global bound below with `K=b`, but it does not
imply the second, minorization bound. Hairer and Mattingly,
*Yet another look at Harris' ergodic theorem for Markov chains*, assume

```text
PV <= gamma V + K,                 gamma<1,
P(x,.) >= alpha nu(.) on {V<=R},  R>2K/(1-gamma).
```

Their Theorem 1.3 gives an explicit one-step contraction in a weighted total
variation metric. For `K>0` and any

```text
alpha_0 in (0,alpha),
gamma_0 in (gamma+2K/R,1),
beta = alpha_0/K,
```

one may take

```text
bar_alpha =
  max(1-alpha+alpha_0, (2+R beta gamma_0)/(2+R beta)) < 1.
```

Thus a quantitative result needs both the drift constants and a quantitative
minorization `alpha` on the core. It gives a spectral gap in a weighted
supremum/total-variation setting. Converting it to the exact `L2(mu)` gap used
by the reversible transfer-operator theorem requires additional comparison
arguments.

The suggested choices `V=U` or `V=H` do not give the proposed underdamped drift:

```text
L U = p dot grad U,
L H = -gamma |p|^2 + gamma d/beta.
```

The first has no sign. The second has no coercivity when potential energy is
large and momentum is small. Valid Langevin proofs add position-momentum cross
terms and usually exponentiate a modified Hamiltonian.

Herzog-Mattingly Theorem 2.6 constructs such a function
`W=exp(bH(1+o(1)))` for their admissible singular potentials and proves
weighted-TV convergence `C exp(-eta t)`, but `C` and `eta` are existential.
Their paper explicitly notes why `H` alone fails.

More quantitative results expose the missing constants rather than eliminating
them:

* Eberle, Guillin, and Zimmer, Ann. Probab. 47 (2019), Assumption 2.1 and
  Theorem 2.3, require globally Lipschitz `grad U` and the explicit
  dissipativity inequality

  ```text
  x dot grad U(x)/2
    >= lambda [U(x)+u^(-1) gamma^2 |x|^2/4] - A.
  ```

  Their Wasserstein contraction rate is

  ```text
  c = (gamma/384) min(
        lambda L u gamma^(-2),
        Lambda^(1/2) exp(-Lambda) L u gamma^(-2),
        Lambda^(-1/2) exp(-Lambda)),
  ```

  where their equations (2.19)-(2.20) define `Lambda`, which is at least
  `6(d+A)/5`. This displays the exponential dimension/nonconvexity
  deterioration. Global Lipschitzness excludes Lennard-Jones singularities,
  and this is a Wasserstein rate rather than the required `L2` gap.
* Cao, Lu, and Wang, Arch. Rational Mech. Anal. 247:90 (2023), Theorem 1,
  obtain an `L2` rate

  ```text
  nu = m gamma / [c (sqrt(m)+R+gamma)^2]
  ```

  under a position-space Poincare inequality with constant `m`, derivative
  bounds, and compact embedding. The input `m` already contains the global
  bottleneck information at issue here.
* Baudoin, Gordina, and Herzog, *Gamma calculus beyond Villani...*,
  Theorem 2.23 and Corollary 2.34, give an explicit weighted-`H1` rate for a
  class including Lennard-Jones repulsion plus polynomial confinement. The
  displayed rate in their equation (2.32) is

  ```text
  sigma = min(
            alpha/[2(1+lambda)],
            gamma/[1+beta rho_K']).
  ```

  where `rho_K'` contains the local Poincare constant of the compact core and
  `alpha,beta,lambda` contain the force-growth, temperature, and dimension
  constants defined in their equations (2.28)-(2.33). Their estimates therefore
  still require the core bottleneck input and deteriorate with dimension.

These are real conditional results. They do not imply a useful numerical gap
from bonded-force growth alone.

### 4.3 No-go Theorem C: confinement does not prevent a rare slow well

Consider one-dimensional overdamped Langevin at inverse temperature one,

```text
dX_t = -U_n'(X_t) dt + sqrt(2) dW_t,
mu_n(dx) = Z_n^{-1} exp(-U_n(x)) dx.
```

There is a sequence of smooth potentials `U_n` with all of the following
properties:

1. `U_n(x)=x^2` outside one fixed compact interval, so the tail dissipativity
   is identical for every `n`;
2. a fixed-width local well `A` has potential height `n`;
3. fixed-width collars separating `A` from the rest of the line have potential
   height at least `3n-O(1)`; and
4. a fixed core interval has potential zero.

Such functions are obtained by smooth interpolation between constant plateaus
inside the fixed compact interval and `x^2` outside it. The normalization
constants `Z_n` are bounded above and below independently of `n`, because the
zero-potential core has fixed positive length and the quadratic tails are fixed.
Consequently

```text
mu_n(A) is asymptotic to exp(-n).
```

Choose a smooth `f_n` equal to one on the well, zero outside the two barrier
collars, and changing only where `U_n>=3n-O(1)`, with `|f_n'|` bounded
independently of `n`. The interpolation can be chosen so that
`supp(f_n) subset {U_n>=n-O(1)}`. Hence `int f_n dmu_n=O(exp(-n))`,
while `f_n=1` on a fixed-width set of mass comparable to `exp(-n)`. The
reversible generator's Dirichlet form and variance therefore obey

```text
E_n(f_n,f_n)
  = integral |f_n'|^2 dmu_n
  <= C exp(-3n),

Var_mu_n(f_n) >= c exp(-n).
```

The variational characterization of the generator spectral gap `g_n` gives

```text
g_n <= E_n(f_n,f_n)/Var_mu_n(f_n)
    <= C' exp(-2n) -> 0.
```

For the lag-`tau` semigroup, spectral calculus gives

```text
sup spec(P_tau,n restricted to 1-perp)
  = exp(-tau g_n) -> 1.
```

For any fixed sublevel threshold `R`, the well belongs to `{U_n>R}` once
`n>R`, while its equilibrium mass tends to zero. It is therefore exactly a
rare excluded region hiding an arbitrarily slow mode.

This example is already smooth, reversible, nonexplosive, and quadratically
confining. An underdamped analogue has the same barrier metastability; the
overdamped construction alone suffices to disprove the proposed implication
from generic confinement to a uniform anti-trapping constant. It does not say
that one fixed, fully specified peptide has zero gap. It says any quantitative
lower bound for that peptide must contain internal barrier, conductance, or
Poincare information not supplied by tail dissipativity.

### 4.4 Force-field realism

Real peptide force fields are nonconvex by design and support metastability.
Bond terms may control dissociation after quotienting global translation, while
torsion, nonbonded, and solvent effects create wells and barriers inside the
confined region. Lennard-Jones forces are singular, electrostatics are
long-ranged, and production simulations use constraints, cutoffs/PME, and
sometimes a variable periodic box. These facts are compatible with qualitative
ergodicity under a carefully specified model. They do not imply a useful
uniform conductance or Poincare constant.

Gate C can be made true only by assuming or independently estimating a
numerical global conductance/Poincare/minorization constant, or by imposing
unrealistically strong structure such as global strong convexity. That constant
is not supplied by "the force field is confining."

## 5. GATE D: student anti-trapping

### Verdict: BLOCKED

There is no automatic anti-trapping theorem for generic few-step
distillation. The obstruction persists even when the student is a one-step
stochastic generator and every fixed-length stationary segment law converges to
the corresponding teacher law in both KL directions.

### 5.1 No-go Theorem D: pair KL can hide a one-step student trap

Let the state space be `{A,B}` and, for `0<e<1/4`, let

```text
mu_e = (e, 1-e).
```

Take the teacher to resample independently from `mu_e`:

```text
P_e =
  [[e, 1-e],
   [e, 1-e]].
```

It is stationary and reversible, with nontrivial eigenvalue zero. Define the
student

```text
P_hat_e =
  [[1-e,               e],
   [e^2/(1-e), 1-e^2/(1-e)]].
```

Detailed balance holds because

```text
mu(A) P_hat(A,B) = e^2
                 = mu(B) P_hat(B,A).
```

Thus the student has the same stationary marginal and is reversible. Its
nontrivial eigenvalue is

```text
lambda_hat_e
  = 1 - e - e^2/(1-e)
  = (1-2e)/(1-e) -> 1.
```

The conductance of the rare state is

```text
Phi_hat({A}) = P_hat(A,B) = e -> 0.
```

The teacher and student stationary pair tables are

```text
rho_e:
  AA=e^2,       AB=e(1-e),
  BA=e(1-e),    BB=(1-e)^2;

rho_hat_e:
  AA=e(1-e),    AB=e^2,
  BA=e^2,       BB=1-e-e^2.
```

Direct substitution gives

```text
KL(rho_hat_e || rho_e)
  = e(1-e) log((1-e)/e)
    + 2e^2 log(e/(1-e))
    + (1-e-e^2) log((1-e-e^2)/(1-e)^2)
  = O(e log(1/e)) -> 0,
```

and

```text
KL(rho_e || rho_hat_e)
  = e^2 log(e/(1-e))
    + 2e(1-e) log((1-e)/e)
    + (1-e)^2 log((1-e)^2/(1-e-e^2))
  = O(e log(1/e)) -> 0.
```

This obstruction persists for a full fixed-length segment, not only a pair.
Let `Q_e^(m)` and `Q_hat_e^(m)` be the stationary Markov path laws on
`(X_0,...,X_m)` generated by `P_e` and `P_hat_e`. The chain rule for relative
entropy and stationarity give, for every fixed integer `m>=1`,

```text
KL(Q_hat_e^(m) || Q_e^(m))
  = m KL(rho_hat_e || rho_e),

KL(Q_e^(m) || Q_hat_e^(m))
  = m KL(rho_e || rho_hat_e).
```

Both full-segment KL divergences therefore tend to zero for every fixed
horizon, while the student eigenvalue still tends to one. A joint one-shot
generator can sample either finite path table directly; autoregressive
implementation is not essential to the counterexample.

The normalized student density on `AA` is `(1-e)/e`, so the missing uniform
density constant diverges exactly where the trap forms.

This student is constructible with one random draw: conditional on the current
state, compare a uniform random variable with the corresponding row
probability. There is also a lift with a **fixed reference process** rather than
a changing two-state teacher. Take `X=[0,1]`, let `mu` be Lebesgue measure, and
let the teacher independently resample from `mu` for every `e`. Partition
`X=A_e union B_e` with `A_e=[0,e]`; apply `P_hat_e` to the region label and
then sample from `mu` conditioned on the selected region. The piecewise-constant
region-label eigenfunction has eigenvalue `(1-2e)/(1-e)`, and disintegration
over the two regions gives exactly the same pair and fixed-horizon path KL
formulas above.

Thus even a full fixed-horizon segment-KL guarantee cannot provide student
anti-trapping. A fortiori, a weaker unspecified distillation objective cannot.

### 5.2 Why an average flow or consistency loss does not repair this

The general obstruction is elementary. Let a training objective have the form

```text
L(F_hat) = integral ell(F_hat(z),F(z)) nu(dz).
```

If the model class can equal `F` off a set `A_e` with `nu(A_e)=e` and make a
bounded modification on `A_e`, then

```text
L(F_hat) <= e sup_{A_e} ell(F_hat,F).
```

The modification can be smoothed over a shell of arbitrarily small additional
`nu`-mass. Hence an average squared velocity, flow-map, or consistency error
does not imply a pointwise escape probability or Lyapunov inequality on rare
conditioning states. To obtain such an implication one would need extra
uniform regularity plus a quantitative covering/lower-density argument, or the
condition must be built into the architecture.

Exact equality to the teacher map everywhere would of course transfer every
property. Practical small expected loss is not such a statement. The L37 specs
also do not identify an exact distillation objective or a theorem giving a
uniform conditional error, so there is no narrower structural premise to prove.

### 5.3 What could be engineered

A safeguard must be quantitative. The following observations rule out several
informal versions:

* A Gaussian noise floor on unbounded `R^d` is not a global Doeblin
  minorization: translated Gaussian densities have zero infimum as the center
  ranges over `R^d`.
* Rejecting an energetically invalid proposal by keeping the current state can
  **increase** trapping and introduces a singular self-loop. Rejection is useful
  only if rejection triggers a quantified return/reset mechanism.
* A plausibility score or designability gate is diagnostic unless failure
  forces return to a safe set with a stated minimum probability.

One sufficient, explicitly engineered package is:

1. both reference and student are defined on a common compact connected state
   space `K` (possibly a deliberately truncated model);
2. every student transition ends with a fixed positive-time heat-kernel or
   wrapped-noise step on `K`, whose density `k_sigma` satisfies
   `0<m_sigma<=k_sigma(x,y)<=M_sigma<infinity`;
3. endpoint stationarity is enforced;
4. the reference equilibrium density is bounded above and below on `K`; and
5. the reference transition density is bounded above on `K x K`, as supplied
   by Gate B only for the idealized continuous reference (or imposed as a
   separate hypothesis).

The noise bound yields a uniform minorization and upper density bound. If
`v` is normalized volume and the student kernel density lies in `[m,M]`,
stationarity implies its invariant density also lies in `[m,M]`. This supplies
both anti-trapping and the density-ratio constants needed by Theorem 3.

An alternative is an explicit state-independent reset of weight `alpha` to the
intended invariant law, combined with a residual kernel that preserves that
same law. On invariant-mean-zero functions the reset term vanishes, so this
construction caps nontrivial eigenvalues by `1-alpha`. It also exposes a
practical conflict: representing a physical timescale `t>>tau` requires roughly
`alpha <= tau/t`, so a noise/reset floor strong enough to make constants useful
can erase the slow kinetics the model is meant to reproduce.

These are imposed method-design constraints. MDGen and a generic few-step
distillation objective do not contain them.

## 6. GATE E: assembly

### Verdict: BLOCKED

There are two separate blockers:

1. Gates C and D do not hold for the target model without extra assumptions.
2. Even if both chains have a tail escape bound, local `D_chi` does not become
   the global operator perturbation required by Theorems 1-3.

### 6.1 Local agreement plus escape is not global operator agreement

The second point has a finite-state witness. Let

```text
mu(C)=1-e,  mu(A_1)=mu(A_2)=e/2,
```

fix `kappa in (0,1)`, and restrict

```text
0 < e < 1/(1+kappa).
```

For both reference and student set

```text
P(A_i,C)=kappa,
P(C,A_i)=e kappa/[2(1-e)],
P(C,C)=1-e kappa/(1-e).
```

On the two tail states let the reference kernel be

```text
P(A_i,A_1)=P(A_i,A_2)=(1-kappa)/2,
```

while the student uses

```text
P_hat(A_i,A_i)=1-kappa,
P_hat(A_i,A_j)=0 for i!=j.
```

Both kernels are reversible with stationary law `mu`. They have:

* exactly the same core pair law on `{C}x{C}`;
* excluded mass `e`;
* the same pointwise escape probability `kappa` from every tail state.

For the antisymmetric tail function

```text
f(C)=0, f(A_1)=1, f(A_2)=-1,
```

the reference has `Pf=0`, while the student has
`P_hat f=(1-kappa)f`. Hence

```text
||P_hat-P||_op >= 1-kappa
```

for every `e`, even as the excluded mass tends to zero and the local discrepancy
is exactly zero.

A fixed `kappa` does prevent this tail eigenvalue from approaching one. It does
not make the global operators close. To recover a localization theorem one
would need, at minimum, a spectral cutoff separating all tail modes from the
target reference cluster and quantitative control of both core-tail blocks.
Those hypotheses and that theorem are absent from the v2 proposal.

### 6.2 Strongest conditional student-spectrum theorem

The existing Theorem 3 can still be applied under stronger **global**
conditions. The exact surviving statement is:

> **Conditional engineered pair-spectrum theorem.**
>
> Let the reference pair be stationary and reversible with transfer operator
> `P` on `L2(mu)`. Let `lambda_i in (0,1)` be a simple isolated eigenvalue and
> define
>
> ```text
> delta_i = dist(lambda_i, spec(P) \ {lambda_i}).
> ```
>
> Let the generated pair be stationary with marginal `mu_hat` and Markov pair
> operator `P_hat` on `L2(mu_hat)`. Relative to the common reference product
> measure `mu tensor mu`, define
>
> ```text
> h     = dmu_hat/dmu,
> q     = drho/d(mu tensor mu),
> q_hat = drho_hat/d(mu tensor mu).
> ```
>
> Assume, almost everywhere,
>
> ```text
> h >= a>0,
> 0<=q,q_hat<=B<infinity,
> KL(rho_hat||rho) <= C_loss L_distill.
> ```
>
> Define
>
> ```text
> K_op(a,B)
>   = 1/a + B(1+a^(-1/2))/(2a^(3/2)),
>
> epsilon_*
>   = K_op(a,B) sqrt(2 B C_loss L_distill).
> ```
>
> If
>
> ```text
> epsilon_* < min(delta_i/2, lambda_i, 1-lambda_i),
> ```
>
> then `P_hat` has one algebraic eigenvalue `lambda_hat_i` in the
> `delta_i/2` disk around `lambda_i`, it is real, and
>
> ```text
> |lambda_hat_i-lambda_i| <= epsilon_*,
>
> |t_hat_i-t_i|
>   <= L_tau(lambda_i,epsilon_*) epsilon_*.
> ```
>
> Here `t_i=-tau/log(lambda_i)` and
> `t_hat_i=-tau/log(lambda_hat_i)`.
>
> Equivalently, the Taylor version in `L37_MATH_SOLUTION.md` gives
>
> ```text
> |t_hat_i-t_i|
>   <= [t_i^2/(tau lambda_i)] epsilon_*
>      + (1/2) M_2(lambda_i,epsilon_*) epsilon_*^2.
> ```

This is exactly Theorem 3 plus Corollary 2 of the settled solution, with the
external loss-to-pair-KL premise substituted. If a theorem controls segment KL,
data processing supplies pair KL.

The compact/noise design in Gate D is one sufficient route to finite `a,B`.
For example, on compact `K` with normalized volume `v`, suppose

```text
m_mu <= dmu/dv <= M_mu,
p_ref(y|x) <= M_ref,
m <= p_hat(y|x) <= M.
```

Writing `m_hat=dmu_hat/dv`, stationarity gives

```text
m_hat(y) = integral m_hat(x) p_hat(y|x) dv(x),
```

so `m<=m_hat<=M`. Moreover,

```text
q(x,y)     = p_ref(y|x)/(dmu/dv)(y),
q_hat(x,y) = [m_hat(x)/(dmu/dv)(x)]
             [p_hat(y|x)/(dmu/dv)(y)].
```

Therefore one may take

```text
a = m/M_mu,
B = max(M_ref/m_mu, M^2/m_mu^2).
```

This makes every constant explicit in principle. It is not the local-tail
theorem proposed in v2. It assumes that the whole modeled state space has been
compactified/truncated and that the student has a global density floor and
ceiling.

### 6.3 Why this does not apply to a current MDGen-style run

No checked MDGen fact establishes:

* a compact generated state space;
* a positive global student noise floor or density ceiling;
* equal generated endpoint marginals/stationarity;
* a lower bound `dmu_hat/dmu>=a`;
* an exact loss-to-segment-KL theorem and constant `C_loss`; or
* time-homogeneous Markov consistency of the jointly generated path.

The last point means that, even if the pair theorem applied, its eigenvalue would
be a one-lag pair-operator implied-timescale proxy. It would describe actual
multi-lag generated relaxation only under the path-consistency conditions
already stated in `L37_MATH_SOLUTION.md`.

The engineered theorem is therefore materially weaker than an unconditional
claim about generic few-step distillation.

## 7. Verified references

1. Jing, Stark, Jaakkola, and Berger, *Generative Modeling of Molecular
   Dynamics Trajectories*, arXiv:2409.17808v1, NeurIPS 2024. Coordinate
   tokenization: Section 3.1, equations (3)-(4), pages 3-4. Simulation protocol:
   Appendix B.2, page 16.
2. MDGen code, `github.com/bjing2016/mdgen`, commit
   `81482a403b91c5a8437046da1d8d321ba97089cc`: `mdgen/utils.py` lines 7-14;
   `mdgen/wrapper.py` lines 292-317 and 456-470;
   `scripts/run_peptide_sim.py` lines 67-127; `scripts/prep_sims.py` lines
   69 and 75.
3. Vander Meersche et al., *ATLAS: protein flexibility description from
   atomistic molecular dynamics simulations*, Nucleic Acids Research
   52(D1):D384-D392 (2024), DOI `10.1093/nar/gkad1084`, "Molecular dynamics
   simulation protocol."
4. Hairer, *On Malliavin's proof of Hormander's theorem*,
   arXiv:1103.1998, Theorems 1.3 and 4.5 and Assumption 4.2.
5. Herzog and Mattingly, *Ergodicity and Lyapunov functions for Langevin
   dynamics with singular potentials*, arXiv:1711.02250, Definition 2.3,
   Proposition 2.11, Theorem 2.6, and Corollaries 5.10-5.12.
6. Levin, Peres, and Wilmer, *Markov Chains and Mixing Times*, second edition,
   Theorem 13.10. General-state source: Lawler and Sokal, Trans. AMS 309(2):
   557-580 (1988).
7. Hairer and Mattingly, *Yet another look at Harris' ergodic theorem for
   Markov chains*, arXiv:0810.2777, Assumptions 1-2 and Theorem 1.3.
8. Eberle, Guillin, and Zimmer, *Couplings and quantitative contraction rates
   for Langevin dynamics*, Ann. Probab. 47(4):1982-2010 (2019),
   Assumptions 2.1/2.7 and Theorem 2.3.
9. Cao, Lu, and Wang, *On explicit L2-convergence rate estimate for
   underdamped Langevin dynamics*, Arch. Rational Mech. Anal. 247:90 (2023),
   DOI `10.1007/s00205-023-01922-4`, Assumptions 1-3 and Theorem 1.
10. Baudoin, Gordina, and Herzog, *Gamma calculus beyond Villani and explicit
    convergence estimates for Langevin dynamics with singular potentials*,
    arXiv:1907.03092v3, Assumptions 2.6-2.7, Theorem 2.23, and Corollary 2.34.

## 8. Strongest honest conclusion

For an ideal continuous underdamped Langevin diffusion satisfying explicit
Hormander and admissible-potential hypotheses, the reference transition density
is locally bounded relative to equilibrium on compact phase-space sets. That
fact does not close the MD-realism gap. Generic confinement does not exclude a
rare slow well, and a generic one-step or few-step student can have vanishing
fixed-horizon segment KL while acquiring an eigenvalue tending to one. Local
density control plus tail escape also does not supply the global operator
perturbation used by Theorems 1-3. A student implied-timescale theorem survives
only under imposed global compactness/density/stationarity safeguards and an
external loss-to-segment-KL theorem; this is an idealized engineered model, not
a theorem about current MDGen-style distillation. Without those additions, the
strongest honest result remains the settled Section 8.2 bound on a fixed bounded
teacher mode's one-lag autocorrelation, which does not identify a student
eigenvalue.

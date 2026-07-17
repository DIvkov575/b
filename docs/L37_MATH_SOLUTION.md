# L37 Math Resolution: What Is Provable, What Fails, and the Corrected Theorem

**Date:** 2026-07-14  
**Resolves:** `docs/L37_MATH_SPEC.md`

## 1. Verdict

The proposed same-marginal isolated-spectrum theorem is valid for a simple
isolated eigenvalue after adding explicit small-perturbation conditions. For a
repeated isolated eigenvalue, only its spectral cluster and invariant subspace
are intrinsically stable. MFPT and assignment claims need separate hypotheses
given below. The proposed bridge from an achievable KL loss to the spectral
theorem is not valid under condition (C1) as written.

| Gate | Result | Reason |
|---|---|---|
| GATE-0 | **PASS** | A two-state reversible chain gives identical one-frame laws and an order-one eigenvalue and implied-timescale change. |
| GATE-1 | **REGISTERED C1 DISPROVED; HEADLINE BLOCKED** | Bounding only `p_hat(y|x)/mu(y)` does not let KL control `D_chi`. An explicit reversible counterexample has both KL directions tending to zero while `D_chi` and the eigenvalue error stay fixed. None of the registered PASS/KILL branches exactly describes this outcome. |
| GATE-1, corrected | **CONDITIONAL THEOREM ONLY** | If both two-time densities relative to `mu x mu` are uniformly bounded, then `D_chi^2 <= 2 B KL`. This boundedness is not automatic and already fails for the reference joint of a stationary Ornstein-Uhlenbeck process. No MD-specific PASS-A is established. |
| GATE-2 | **SUPERSEDED, CONDITIONAL ON COMMON-SPACE OPERATOR CONTROL** | The reference operator is self-adjoint, hence normal. Once a common-space operator-norm perturbation is available, an arbitrary non-normal student has local spectral control without a global `kappa(V)` factor. |
| Honest marginal drift | **TV CLAIM DISPROVED** | `TV(mu_hat,mu)` alone cannot couple the two Hilbert spaces in operator norm. The valid replacements proved below impose stronger density-ratio hypotheses. |
| GATE-3 | **OPEN** | It is an empirical constant-calibration gate on released MDGen data, not a pure proof obligation. |

Thus there is no unconditional theorem of the form

```text
distillation loss -> KL/W2 -> student transfer spectrum -> implied timescale
```

under the assumptions in the draft. There are two rigorous replacements:

1. a conditional transfer-spectrum theorem under stronger density assumptions; and
2. an assumption-light bound on the generated autocorrelation of each bounded
   teacher slow mode. The second result does not identify a student eigenvalue.

## 2. Mathematical scope and notation

### 2.1 Three objects that must not be conflated

Let `X` be a standard Borel space (in particular, a Polish configuration space
with its Borel sigma-algebra). This assumption guarantees existence of regular
conditional probabilities.

The reference lag-`tau` pair law is

```text
rho(dx,dy) = mu(dx) p(dy|x).
```

It has both marginals `mu`, and reversibility means `rho(dx,dy)=rho(dy,dx)`.

A jointly generated segment has some path law `Q_hat`. Marginalizing two
positions gives a pair law `rho_hat`, with first and second marginals
`mu_hat_0` and `mu_hat_1`. Disintegration with respect to its first marginal
gives a regular conditional kernel:

```text
rho_hat(dx,dy) = mu_hat_0(dx) p_hat(dy|x).
```

This fact alone supports a **pair operator**

```text
(P_hat f)(x) = integral f(y) p_hat(dy|x).
```

Jensen's inequality shows that this operator is a contraction from
`L2(mu_hat_1)` to `L2(mu_hat_0)`. It is defined only
`mu_hat_0`-almost everywhere and is not yet an iterable endomorphism.

```text
||P_hat f||_{L2(mu_hat_0)}^2
  <= integral P_hat(|f|^2) dmu_hat_0
  = ||f||_{L2(mu_hat_1)}^2.
```

There are three distinct levels:

1. **Pair law:** `rho_hat` and one application of `P_hat`.
2. **Iterated Markov surrogate:** the artificial chain obtained by repeatedly
   applying `p_hat`.
3. **Generated path law:** the actual multi-frame distribution `Q_hat`.

If `mu_hat_1=mu_hat_0`, write the common marginal as `mu_hat`; then `mu_hat`
is stationary for the pair kernel and `P_hat` is a contraction endomorphism
on `L2(mu_hat)`. This still does not imply that `Q_hat` is Markov or
time-homogeneous.

The contraction follows from Jensen and stationarity:

```text
||P_hat f||_2^2
  <= integral P_hat(|f|^2) dmu_hat
  = ||f||_2^2.
```

MFPTs, committors, and multi-lag observables can be defined for non-Markov
path laws. Identifying them with iterations or resolvents of the pair kernel
requires Markov consistency of the relevant path law. Through a finite
horizon `n`, this is

```text
Q_hat(dx_0,...,dx_n)
  = mu_hat(dx_0) product_{k=0}^{n-1} p_hat(dx_{k+1}|x_k)       (MC)
```

with the same kernel at each step. An infinite-horizon MFPT or committor
requires this factorization for every finite `n`, a consistent infinite-path
law, and the integrability assumptions stated with the killed resolvent below.
A finite generated segment supports only finite-horizon or truncated hitting
observables. The analogous requirements apply to the reference path whenever
its hitting quantities are represented by the resolvent of `P`.

This matters for the spec's MD interpretation. Hamiltonian or underdamped
dynamics is naturally Markov on full phase space and is commonly reversible
only after momentum reversal. Its projection onto configurations can fail to
be Markov even when each equilibrium pair law is symmetric. In that case the
one-lag pair operators remain well-defined and can be self-adjoint, but their
iterations need not reproduce either actual path law. The formula
`-tau/log(lambda_i)` is then a one-lag implied timescale, not a semigroup decay
constant, and the killed-resolvent MFPT is a surrogate.

The MD assumptions are therefore model premises, not consequences of
equilibrium notation. Canonical NVT and microcanonical NVE equilibrium laws
are different; a density `Z^(-1) exp(-beta U)` describes the former, not the
latter. A free translational or rotational coordinate also requires a quotient,
confinement, or finite periodic box for normalization. On phase space,
underdamped dynamics commonly satisfies generalized detailed balance
`P_tau^*=R P_tau R` under momentum reversal `R`, rather than ordinary
self-adjointness. A position-only pair may still be symmetric because position
is invariant under `R`, but the projected position process need not be Markov.

### 2.2 Density notation

Except for the explicitly singular example in Section 5, all density-based
statements below assume

```text
rho << nu and rho_hat << nu, where nu=mu x mu.
```

If either pair law has an unmatched singular component relative to `nu`, the
draft's density formula for `D_chi` is not finite and the Hilbert-Schmidt route
does not apply. This is a substantive assumption for deterministic dynamics.

Let

```text
nu       = mu x mu,
q        = d rho / d nu,
q_hat    = d rho_hat / d nu.
```

When both one-time marginals equal `mu`,

```text
q(x,y)     = p(y|x) / mu(y),
q_hat(x,y) = p_hat(y|x) / mu(y).
```

Write

```text
D_chi = ||q_hat - q||_{L2(nu)}.
```

This is an `L2(mu x mu)` distance between normalized joint densities. It is not
the usual `chi^2(rho_hat || rho)`, whose denominator would be `rho`.

## 3. Same-marginal theorem

### Lemma 0: detailed balance gives a self-adjoint contraction

Let `p` be a Markov kernel with stationary law `mu`, and let
`rho(dx,dy)=mu(dx)p(dy|x)`. Then `P` is a contraction on `L2(mu)`. If
`rho(dx,dy)=rho(dy,dx)`, then `P` is self-adjoint.

#### Proof

Jensen's inequality and stationarity give

```text
||P f||_2^2
  <= integral P(|f|^2) dmu
  = ||f||_2^2.
```

For `f,g in L2(mu)`, detailed balance gives

```text
<f,Pg>_mu
  = integral f(x)g(y) rho(dx,dy)
  = integral f(x)g(y) rho(dy,dx)
  = <Pf,g>_mu.
```

Thus `P=P*` and `||P||_op<=1`; consequently
`spec(P) subset [-1,1]`.

The endpoint `-1` is possible, for example for a reversible period-two chain.
Likewise, simplicity of the eigenvalue `1` requires an ergodicity assumption;
neither fact follows from reversibility alone.

### Lemma 1: Hilbert-Schmidt identity

Assume `rho` and `rho_hat` both have first and second marginal `mu`, and
`q_hat-q` belongs to `L2(mu x mu)`. On `H = L2(mu)`,

```text
((P_hat - P)f)(x)
    = integral (q_hat(x,y) - q(x,y)) f(y) dmu(y).
```

Therefore, for `E = P_hat - P`,

```text
||E||_op <= ||E||_HS = D_chi.                       (1)
```

#### Proof

Relative to `mu`, the integral kernel of `E` is `q_hat-q`. The defining
identity for Hilbert-Schmidt integral operators therefore gives

```text
||E||_HS^2
  = integral integral |q_hat(x,y)-q(x,y)|^2 dmu(x)dmu(y)
  = D_chi^2.
```

Every Hilbert-Schmidt operator is bounded and its operator norm is at most its
Hilbert-Schmidt norm. This proves (1).

This identity needs only `q_hat - q` in `L2(nu)`. The statement in the draft
that finite `D_chi` is equivalent to `q_hat` being in `L2(nu)` additionally
requires `q` itself to be in `L2(nu)`.

The draft's student-only Doeblin condition is not even sufficient for
finiteness without reference-side control. Let `X` be the unit circle with
Haar probability `mu`, and choose an even probability density
`h in L1(mu)\L2(mu)`; for example, normalize
`h(t)=|t|^(-3/4)` on `[-1/2,1/2)`. Define

```text
q(x,y) = h(y-x),
q_hat(x,y) = 1.
```

Translation invariance gives both marginals `mu`, and evenness gives
reversibility. The student satisfies `q_hat<=1`, but

```text
D_chi = ||1-q||_2 = infinity,
```

because `q` is not square-integrable. Thus A3 must be stated directly as
`q_hat-q in L2(mu x mu)` or supplemented by reference-side assumptions.

### Theorem 1: isolated eigenvalue under an arbitrary student perturbation

Work on the complexification of `H` if `H` was initially real. Let `P` be
bounded and self-adjoint. Let `lambda` be a simple isolated eigenvalue with
normalized eigenfunction `psi`, and define

```text
delta = dist(lambda, spec(P) \ {lambda}).
```

For a self-adjoint Markov contraction, `spec(P)` is contained in `[-1,1]`.
The timescale corollary below additionally assumes `lambda in (0,1)`.

Let `P_hat = P + E`, where `E` need not be self-adjoint, and set
`epsilon = ||E||_op`. If `epsilon < delta/2`, then:

1. the disk `|z-lambda| < delta/2` contains exactly one eigenvalue
   `lambda_hat`, counted with algebraic multiplicity;
2. `|lambda_hat-lambda| <= epsilon`; and
3. the first-order expansion obeys

```text
lambda_hat - lambda
    = <psi, E psi>_mu + R,

|R| <= epsilon^2 / (delta - 2 epsilon).              (2)
```

   If `epsilon<=delta/4`, this implies
   `|R|<=2 epsilon^2/delta`.
4. if `P` and `E` preserve the underlying real Hilbert space, then
   `lambda_hat` is real.

Under Lemma 1, `epsilon` can be replaced by `D_chi`.

#### Proof

For `z` outside `spec(P)`,

```text
||(z-P)^(-1)|| = 1 / dist(z,spec(P))
```

because `P` is normal. On the circle `|z-lambda|=delta/2`, the Neumann
series for

```text
z-P_hat = (I - E(z-P)^(-1))(z-P)
```

converges when `epsilon < delta/2`. The Riesz projection inside that circle
therefore has the same rank as for `P`, namely one: along the homotopy
`P+tE`, `0<=t<=1`, the contour stays in the resolvent set, so its Riesz
projection varies continuously, and a continuous finite-rank projection has
constant rank. The same resolvent argument also gives

```text
spec(P_hat) subset {z : dist(z,spec(P)) <= epsilon},
```

For the eigenvalue inside the isolating disk, every point of
`spec(P)\{lambda}` is more than `epsilon` away because
`epsilon<delta/2`; hence its nearby reference spectral point must be
`lambda`. This proves `|lambda_hat-lambda|<=epsilon`.

For completeness, let `Pi f=<psi,f>psi`, `Q=I-Pi`, and write `P_hat` in block
form on `span{psi} + psi_perp`:

```text
P_hat =
  [[lambda+e_00, e_01],
   [e_10,         B_Q]],

e_00 = <psi,E psi>,
B_Q  = Q(P+E)Q.
```

The local eigenvalue satisfies `|lambda_hat-lambda|<=epsilon`. Moreover,
`lambda_hat-B_Q` is invertible because the resolvent identity gives

```text
||(lambda_hat-B_Q)^(-1)||
  <= 1/(delta-|lambda_hat-lambda|-epsilon)
  <= 1/(delta-2epsilon).
```

The first component of a corresponding eigenvector is nonzero; otherwise
`lambda_hat` would belong to `spec(B_Q)`, contradicting this invertibility.
Eliminating the complementary component gives the scalar Schur-complement
identity

```text
lambda_hat-lambda
  = e_00 + e_01 (lambda_hat-B_Q)^(-1) e_10.
```

Both off-diagonal blocks have norm at most `epsilon`, so the second term has
absolute value at most `epsilon^2/(delta-2epsilon)`. This is (2).

The proof uses normality of the reference `P`, not of `P_hat`. Consequently,
the global eigenvector condition number of `P_hat` does not appear.

For the last assertion, complex conjugation preserves the spectrum and
algebraic multiplicity of a real operator. If `lambda_hat` were nonreal, its
distinct conjugate would also lie in the disk centered at the real number
`lambda`, contradicting the fact that the disk has total algebraic
multiplicity one.

### Corollary 1: the autocorrelation term

For a real slow mode,

```text
<psi, E psi>_mu
    = E_rho_hat[psi(x0) psi(x_tau)]
      - E_rho[psi(x0) psi(x_tau)]
    = C_hat_psi(tau) - lambda.                       (3)
```

The last equality uses
`E_rho[psi(x0)psi(x_tau)]=<psi,Ppsi>_mu=lambda`
for normalized `psi`.

Thus the draft's first-order interpretation is correct in the same-marginal
setting.

### Corollary 2: implied-timescale error

Suppose `lambda in (0,1)`, the compared eigenvalue is real, and

```text
epsilon < min(delta/2, lambda, 1-lambda).
```

For `t(u) = -tau/log(u)`, the mean-value theorem gives the non-asymptotic bound

```text
|t(lambda_hat)-t(lambda)|
    <= L_tau(lambda,epsilon) epsilon,                (4)

L_tau(lambda,epsilon)
    = max over u in [lambda-epsilon,lambda+epsilon]
      tau / (u (log u)^2).
```

For an explicit second-order form, define

```text
M_2(lambda,epsilon)
  = sup over u in [lambda-epsilon,lambda+epsilon] of |t''(u)|.
```

Taylor's theorem yields

```text
|t_hat-t|
    <= (t^2 / (tau lambda)) epsilon
       + (1/2) M_2(lambda,epsilon) epsilon^2.         (5)
```

Indeed,

```text
t'(u) = tau/(u(log u)^2),
```

and `t''` is bounded on the displayed compact interval in `(0,1)`.
Taylor's theorem and `|lambda_hat-lambda|<=epsilon` prove (5). In particular,
the usual `O(epsilon^2)` notation is valid for fixed `lambda` with `epsilon`
restricted to a fixed compact neighborhood inside `(0,1)`; it is not uniform
as `lambda` approaches either endpoint.

The coefficient in the draft is therefore correct. The omitted conditions are
important: for a slow mode, a useful bound requires roughly
`epsilon << 1-lambda`; as `lambda` tends to one,
`1-lambda` is asymptotic to `-log(lambda)=tau/t`.

### Corollary 3: handling a non-reversible student

If `mu` is stationary for `P_hat`, its adjoint is the time-reversed Markov
kernel and the additive reversibilization is

```text
S_hat = (P_hat + P_hat*) / 2.
```

Then `S_hat` is a self-adjoint Markov operator,

```text
||S_hat-P||_op <= ||P_hat-P||_op,
<psi,(S_hat-P)psi> = <psi,(P_hat-P)psi>,
```

where the second identity is for real kernels and a real choice of `psi`.
If `||S_hat-P||<min(delta/2,lambda,1-lambda)`, equations (4)-(5) apply to
the unique eigenvalue of `S_hat` in the isolating disk. This eigenvalue is a
**reversibilized surrogate**, not an eigenvalue or decay rate of `P_hat`.

For the real Markov operator `P_hat` itself, Theorem 1 already shows that its
unique local eigenvalue is real. Thus no additive reversibilization or
`kappa(V)` factor is needed for that local eigenvalue; reversibilization is a
separate choice of kinetic summary.

### Corollary 4: eigenvectors

Let `psi_hat` be a normalized right eigenvector associated with the unique
local eigenvalue of `P_hat`. For `epsilon < delta/2`,

```text
sin angle(psi_hat,psi)
    <= epsilon / (delta-epsilon)
    <= 2 epsilon/delta.                              (6)
```

The draft's `epsilon/delta` expression suppresses the perturbed-gap
denominator.

Indeed, for a normalized perturbed eigenvector `psi_hat`, project
`(P+E)psi_hat=lambda_hat psi_hat` onto `psi_perp`. Since
`dist(lambda_hat,spec(P|psi_perp))>=delta-epsilon`,

```text
||Q psi_hat||
  <= ||(Q P Q-lambda_hat)^(-1)|| ||Q E psi_hat||
  <= epsilon/(delta-epsilon).
```

The left side is the sine of the angle from `psi_hat` to `span{psi}`, proving
(6). This argument does not require `P_hat` to be self-adjoint.

An eigenvector-angle estimate alone does not prove a metastable-set assignment
bound. For example, suppose `psi` and `psi_hat` are real and sign-aligned, and
threshold both at `c`:

```text
A     = {x : psi(x) >= c},
A_hat = {x : psi_hat(x) >= c}.
```

For every `a>0`,

```text
mu(A symmetric_difference A_hat)
  <= mu({|psi-c| <= a}) + ||psi_hat-psi||_2^2/a^2
  <= mu({|psi-c| <= a})
     + (2/a^2) [epsilon/(delta-epsilon)]^2.           (6a)
```

The first inequality follows because disagreement away from the threshold
margin forces `|psi_hat-psi|>a`, followed by Chebyshev's inequality. For
sign-aligned unit vectors,
`||psi_hat-psi||_2 <= sqrt(2) sin angle(psi_hat,psi)`, which proves the second
inequality. A useful assignment guarantee therefore needs an explicit margin
condition on `mu({|psi-c|<=a})`.

### Repeated isolated eigenvalues

The scalar expansion (2) and individual eigenvector bound (6) require
simplicity. Let instead `lambda` be an isolated eigenvalue of finite
multiplicity `r`, let `Pi` be its orthogonal spectral projection, and retain
the gap `delta`. If `epsilon<delta/2`, the disk
`|z-lambda|<delta/2` contains a spectral cluster of `P_hat` with total
algebraic multiplicity `r`, and every point of that cluster obeys

```text
|z-lambda| <= epsilon.                               (6b)
```

If `Pi_hat` is its Riesz projection, the resolvent identity on the circle of
radius `delta/2` gives

```text
||Pi_hat-Pi||_op <= 2 epsilon/(delta-2 epsilon).      (6c)
```

Indeed, the reference and perturbed resolvents on that circle have norms at
most `2/delta` and `2/(delta-2epsilon)`, respectively. Integrating

```text
(z-P_hat)^(-1)-(z-P)^(-1)
  = (z-P_hat)^(-1) E (z-P)^(-1)
```

around a contour of length `pi delta` proves (6c); homotopy invariance of the
Riesz rank proves the multiplicity statement, and spectral inclusion proves
(6b).

No individual basis vector in `Ran(Pi)` is stable without extra structure.
This failure occurs even for reversible Markov chains. On three states with
uniform `mu`, let

```text
P = lambda I + (1-lambda) Pi_1,
```

where `Pi_1` projects onto constants and `0<lambda<1`. Choose an orthonormal
basis `psi_1,psi_2` of the mean-zero eigenspace and set

```text
E = epsilon(psi_1 tensor psi_2 + psi_2 tensor psi_1).
```

For sufficiently small `epsilon`, `P+E` remains a symmetric stochastic
matrix because `E 1=0` and `P` has strictly positive entries. Its two
nontrivial eigenvalues are `lambda+epsilon` and `lambda-epsilon`, whereas
`<psi_1,E psi_1>=0`; its eigenvectors are
`(psi_1+psi_2)/sqrt(2)` and `(psi_1-psi_2)/sqrt(2)`. Thus the registered
scalar first-order formula and individual assignment claim are false for a
repeated mode. The cluster bound (6b), the subspace bound (6c), and the
scalar timescale bound (4) for each real cluster eigenvalue are the valid
replacements.

### Corollary 5: killed-resolvent bounds

Work first in the common space `H=L2(mu)`. For a target set `B`, let
`D=X\B`, let `M_D` denote multiplication by `1_D`, and define the killed
lag-skeleton operators on `H_D=M_D H` by

```text
K     = M_D P M_D,
K_hat = M_D P_hat M_D.
```

These are the killed operators of the reference kernel and the **iterated
Markov surrogate** from Section 2.1. Assume `I-K` has a bounded inverse and
set

```text
R0 = ||(I-K)^(-1)||,
epsilon_K = ||K_hat-K||.
```

If `R0 epsilon_K < 1`, then `I-K_hat` is also invertible. The physical-time
MFPT solution functions for the lag-`tau` skeleton are

```text
m_B     = tau (I-K)^(-1) 1_D,
m_hat_B = tau (I-K_hat)^(-1) 1_D.
```

They obey the `L2(mu)` bound

```text
||m_hat_B-m_B||_2
  <= tau R0^2 epsilon_K
     / (1-R0 epsilon_K) * ||1_D||_2.                 (7)
```

To prove it, factor

```text
I-K_hat
  = (I-K)[I-(I-K)^(-1)(K_hat-K)].
```

The bracket is invertible by the Neumann series, and hence

```text
||(I-K_hat)^(-1)|| <= R0/(1-R0 epsilon_K).
```

Now use

```text
(I-K_hat)^(-1)-(I-K)^(-1)
  = (I-K_hat)^(-1)(K_hat-K)(I-K)^(-1).
```

This is a function-norm bound, not a pointwise MFPT bound. If an initial law
`alpha` on `D` has density `g=dalpha/dmu in L2(mu)`, then

```text
|E_alpha[m_hat_B]-E_alpha[m_B]|
  <= ||g||_2 ||m_hat_B-m_B||_2.                      (7b)
```

Point evaluation is not a continuous functional on `L2(mu)` in general.

The controlling resolvent is target-specific and is not determined by the
full-chain slow gap. If `K` is self-adjoint and
`sup spec(K)<1`, then

```text
R0 = 1/(1-sup spec(K)).
```

For example, the independent-resampling kernel has mean-zero spectral gap one,
but for a target of mass `b`, its killed operator has eigenvalue `1-b` and
`R0=1/b`. This disproves the draft's claimed generic identification of the
MFPT amplification with the full-chain timescale.

For the `A`-to-`B` committor, let `A` and `B` be disjoint measurable sets,
set `C=X\(A union B)`, and redefine `R0` and `epsilon_K` for the following
compression:

```text
K     = M_C P M_C,
K_hat = M_C P_hat M_C,
b     = M_C P 1_B,
b_hat = M_C P_hat 1_B,

u     = (I-K)^(-1) b,
u_hat = (I-K_hat)^(-1) b_hat,
beta  = ||b_hat-b||_2.
```

Here `u` and `u_hat` are the interior restrictions. The full committors equal
zero on `A`, one on `B`, and these functions on `C`.

The same resolvent identity gives

```text
||u_hat-u||
  <= R0/(1-R0 epsilon_K)
     * (beta + epsilon_K R0 ||b||).                  (7a)
```

When the Hilbert spaces and boundary sets agree, compression cannot increase
operator norm, so

```text
epsilon_K <= ||P_hat-P||_op <= D_chi,
beta <= ||P_hat-P||_op ||1_B||_2 <= D_chi ||1_B||_2.
```

Thus (7a) is a genuine `D_chi` bound under the displayed killed-resolvent
hypotheses; neither a boundary source term nor its norm may be omitted.

Equations (7), (7a), and (7b) describe actual infinite-horizon reference and
generated-path quantities only when both paths have the time-homogeneous
Markov factorization for every finite horizon and the displayed solutions are
integrable. Without those facts, they compare iterated pair-kernel surrogates
and must not be reported as MD or MDGen trajectory MFPT/committor guarantees.

## 4. GATE-0: per-frame statistics cannot control kinetics

Let `mu=(1/2,1/2)` and, for `a in (0,1/2)`, define

```text
P_a = [[1-a, a],
       [a, 1-a]].
```

Every `P_a` is `mu`-stationary and `mu`-reversible. At stationarity every
one-time law is exactly `mu`, so all per-frame statistics agree for all choices
of `a`. The nontrivial eigenvalue is

```text
lambda_1(P_a) = 1-2a.
```

Take `a=1/8` and `b=3/8`. Then

```text
lambda_1(P_a) = 3/4,
lambda_1(P_b) = 1/4,
|Delta lambda_1| = 1/2.
```

Their implied-timescale difference is the fixed positive quantity

```text
tau [1/log(4/3) - 1/log(4)].
```

Direct calculation also gives

```text
D_chi(P_a,P_b) = 2|a-b| = 1/2,
```

because for uniform `mu`,

```text
D_chi^2
  = sum_{i,j} (P_b(i,j)-P_a(i,j))^2
  = 4(a-b)^2.
```

so the spectral bound is tight in this example. This proves GATE-0.

A continuous-state lift follows by partitioning the state space into two sets
of `mu`-mass `1/2`, using `P_a` for transitions between the sets, and drawing
conditionally from `mu` inside the selected set.

GATE-0 proves that one-frame information is insufficient. It does **not**
prove that matching any finite list of two-time correlations is sufficient for
spectral, MFPT, or path-law fidelity. Such correlations are useful diagnostics;
they become guarantees only under an operator or density hypothesis such as
those proved below.

## 5. GATE-1: condition (C1) in the draft is insufficient

The missing issue is the density of the true joint relative to `mu x mu`.
Bounding only the student's normalized kernel cannot convert KL into
`L2(mu x mu)`.

The draft's stronger claim that Wasserstein convergence can coexist with
infinite `D_chi` is also exact. Let `mu` be uniform on `[0,1]`,
`rho=mu x mu`, and let `gamma` be the law of `(X,X)` for `X~mu`. For
`alpha_n` decreasing to zero, set

```text
rho_hat_n = (1-alpha_n) rho + alpha_n gamma.
```

Both marginals are `mu`, and `rho_hat_n` is the reversible pair law of

```text
p_hat_n(dy|x) = (1-alpha_n) mu(dy) + alpha_n delta_x(dy).
```

The diagonal component is singular relative to `mu x mu`, so `D_chi=infinity`
under the extended convention. On any fixed bounded product metric, mixture
coupling gives

```text
W_2(rho_hat_n,rho)^2
  <= alpha_n W_2(gamma,rho)^2 -> 0.
```

Thus Wasserstein convergence alone cannot enter the Hilbert-Schmidt argument.
The next theorem is stronger in a different direction: both pair laws remain
absolutely continuous and the registered student-side bound holds.

### No-go Theorem A: registered (C1) plus KL/Wasserstein cannot control spectrum

Fix any `theta in (0,1)` and equip `[0,1]^2` with any fixed bounded product
metric. There is a family of pairs of stationary reversible Markov kernels
with the same marginal `mu` such that

```text
q_hat <= 1,
KL(rho_hat||rho) -> 0,
KL(rho||rho_hat) -> 0,
W_p(rho_hat,rho) -> 0 for every finite p,
```

while `D_chi=theta` and the nontrivial eigenvalue error equals `theta`.
Consequently, no bound whose right side tends to zero with either KL direction
or finite-order Wasserstein distance, while depending only on the student-side
constant in (C1), can control `D_chi` or the transfer spectrum.

#### Construction and proof

Take `X=[0,1]` with uniform `mu`. For `0<e<1/2`, choose a set `A_e` of measure
`e` and define

```text
g_e(x) = sqrt((1-e)/e)       for x in A_e,
         -sqrt(e/(1-e))      otherwise.
```

Then `integral g_e dmu=0` and `||g_e||_2=1`. Fix `theta in (0,1)` and define

```text
q_e(x,y) = 1 + theta g_e(x)g_e(y),
q_hat(x,y) = 1.
```

Both are nonnegative because the smallest value of
`g_e(x)g_e(y)` is `-1`. Both integrate to one in each variable, so they define
`mu`-stationary, `mu`-reversible kernels. The student satisfies the draft's
condition with the best possible constant:

```text
p_hat(y|x)/mu(y) = q_hat(x,y) = 1.
```

Nevertheless,

```text
D_chi^2
    = ||q_hat-q_e||_2^2
    = theta^2.                                       (8)
```

The last equality uses
`||g_e(x)g_e(y)||_{L2(mu x mu)}=||g_e||_2^2=1`.

The reference operator is

```text
P_e f = integral f dmu + theta g_e <g_e,f>,
```

so `g_e` has eigenvalue `theta`; the student independent kernel has eigenvalue
zero on `g_e`. The eigenvalue error is the same fixed value `theta`.

In contrast,

```text
q_AA = 1 + theta(1-e)/e,
q_AB = q_BA = 1-theta,
q_BB = 1 + theta e/(1-e),

KL(rho_hat || rho_e)
  = -e^2 log(q_AA)
    -2e(1-e) log(1-theta)
    -(1-e)^2 log(q_BB)
  = O(e),

KL(rho_e || rho_hat)
  = e^2 q_AA log(q_AA)
    +2e(1-e)(1-theta) log(1-theta)
    +(1-e)^2 q_BB log(q_BB)
  = O(e log(1/e)),

TV(rho_hat,rho_e) = 2 theta e(1-e).
```

For the TV identity,

```text
||g_e||_1 = 2 sqrt(e(1-e)),

TV
  = (theta/2) integral |g_e(x)g_e(y)| dmu(x)dmu(y)
  = 2 theta e(1-e).
```

The asymptotics follow directly from
`e^2 log(1/e)=o(e)`, `log(1+O(e))=O(e)`, and
`e log(1/e)->0`. All three distances tend to zero. If the product space has
diameter `L`, a maximal coupling gives

```text
W_p(rho_hat,rho_e)^p <= L^p TV(rho_hat,rho_e),
```

so every finite-order Wasserstein distance tends to zero as well. This proves
No-go Theorem A and realizes the obstruction described informally in Section 6
of the draft.

## 6. A correct KL-to-`D_chi` bridge

### Theorem 2: bounded normalized joints

Assume `0<B<infinity` and

```text
0 <= q(x,y) <= B,
0 <= q_hat(x,y) <= B
```

almost everywhere. Then

```text
D_chi^2 <= 2 B KL(rho_hat || rho),
D_chi^2 <= 2 B KL(rho || rho_hat).                   (9)
```

#### Proof

For `0 <= a,b <= B`, use the extended-value convention and let

```text
H(a|b) = a log(a/b) - a + b.
```

If `a,b>0`, Taylor's theorem around the minimum `a=b` and
`partial_a^2 H=1/a>=1/B` on the interval between `a` and `b` gives

```text
H(a|b) >= (a-b)^2/(2B).
```

If `b=0<a`, the left side is infinite. If `a=0<b`, the inequality reduces to
`b>=b^2/(2B)`. The remaining zero cases follow by continuity. Thus the
pointwise inequality holds everywhere.

Integrating with `a=q_hat`, `b=q`, and using that both densities integrate to
one proves the first inequality in (9), because

```text
integral H(q_hat|q) dnu
  = KL(rho_hat||rho)
    - integral q_hat dnu + integral q dnu
  = KL(rho_hat||rho).
```

Swapping `a` and `b` proves the reverse-KL inequality.

If a distillation theorem gives

```text
KL(rho_hat || rho) <= C_loss L_distill,
```

then, for a simple reference eigenvalue `lambda_i in (0,1)`, set

```text
epsilon_loss = sqrt(2 B C_loss L_distill).
```

If `epsilon_loss<min(delta_i/2,lambda_i,1-lambda_i)`, the exact
same-marginal consequence is

```text
|t_hat_i-t_i|
  <= L_tau(lambda_i,epsilon_loss) epsilon_loss.       (10)
```

For fixed `lambda_i`, equation (5) further gives

```text
|t_hat_i-t_i|
  <= (t_i^2/(tau lambda_i)) epsilon_loss
     + (1/2) M_2(lambda_i,epsilon_loss)
       epsilon_loss^2.
```

For a real Markov student, the unique local eigenvalue is real by Theorem 1.
The corresponding statement for the additive reversibilization concerns a
different, explicitly reversibilized surrogate.

If the available theorem controls full segment laws instead, data processing
first gives

```text
KL(rho_hat || rho)
  <= KL(Q_hat_segment || Q_segment).
```

The draft does not specify `L_distill` precisely enough to prove this KL
premise, its direction, or `C_loss`; (10) is conditional on a separate,
model-specific score/flow theorem.

Some additional control of the reference joint is necessary for a universal
implication based only on the registered student bound: the counterexample in
Section 5 violates the reference-side boundedness used in Theorem 2. It does
not show that uniform upper bounds are the only possible extra hypothesis.

### MD realism check

The uniform-density hypothesis of Theorem 2 is not automatic on a noncompact
state space. Even the reference pair of a stationary one-dimensional
Ornstein-Uhlenbeck process violates it. If
`(X_0,X_tau)` is standard bivariate Gaussian with correlation `r>0`, then its
density ratio relative to the product of its marginals is

```text
q(x,y)
  = (1-r^2)^(-1/2)
    exp((2rxy-r^2(x^2+y^2))/(2(1-r^2))).
```

Along `x=y=s`, this grows as `exp(r s^2/(1+r))`. Since `q` is continuous,
every sufficiently large value persists on an open set of positive Gaussian
product measure, so its essential supremum is infinite. On a compact energy
sublevel set, continuity may bound this reference ratio, but Theorem 2 still
requires absolute continuity and a bound for the student ratio. The constants'
dimension and lag-time dependence must be quantified before (10) can be
called non-vacuous.

The corrected hypothesis therefore closes the conditional mathematics, but
the OU example alone says nothing definitive about MDGen. The draft's
**PASS-A** remains unearned without a target-specific proof and non-vacuous
constants.

## 7. Different marginals: why TV is not enough

Assume first that `mu_hat << mu` and let

```text
h = d mu_hat / d mu
```

and assume the generated pair law is stationary with marginal `mu_hat`. Define
`q_hat=d rho_hat/d(mu x mu)` and assume
`q_hat-q in L2(mu x mu)`. Let `T` be the Hilbert-Schmidt operator on
`L2(mu)` with kernel `q_hat-q`, and set `J_hat=P+T`. Then `J_hat` has kernel
`q_hat` on bounded test functions and

```text
||J_hat-P||_HS = D_chi.
```

To compare the operators using the canonical pointwise Radon-Nikodym transport
that preserves the underlying state coordinate, require equivalence:

```text
mu_hat ~ mu, equivalently 0 < h < infinity mu-almost everywhere.       (E)
```

Under (E), `U:L2(mu_hat)->L2(mu)` defined by `Uf=sqrt(h)f` is unitary. The
transported student operator `B_hat=U P_hat U^(-1)` has kernel
`q_hat(x,y)/(sqrt(h(x))sqrt(h(y)))` relative to `mu`:

```text
(B_hat f)(x)
  = integral q_hat(x,y)
      /[sqrt(h(x))sqrt(h(y))] f(y) dmu(y).           (11)
```

If `h` vanishes on a set of positive `mu`-measure, `U` is unitary only onto
the support subspace `L2({h>0},mu)` and (11) does not provide a whole-space
spectral comparison. This is why equivalence or a stronger lower bound appears
in every valid replacement below. Under equivalence alone,
`M_(1/sqrt(h))` may be unbounded on `L2(mu)`, so a same-space multiplier
factorization of (11) is only formal. The two replacements below impose a
positive lower bound, making that factorization bounded.

Small TV does not control the multiplier `h^(-1/2)`.

### No-go Theorem B: `D_chi` plus marginal TV cannot control honest spectrum

Fix any `lambda in (0,1)`. There is a family of reversible reference/student
pairs with equivalent but different stationary marginals such that the
reference has spectrum `{1,lambda,0}` and a uniformly bounded normalized
`lambda`-eigenfunction, while

```text
D_chi -> 0,
TV(mu_hat,mu) -> 0,
lambda_1(P_hat) -> 1.
```

Thus no reference-uniform universal function `F(u,v)` satisfying
`F(u,v)->0` as `(u,v)->(0,0)` can bound the ordered nontrivial spectral error
by `F(D_chi,TV(mu_hat,mu))`. This does not exclude an estimate whose constants
depend on additional quantitative properties of one fixed reference.

#### Construction and proof

Let the states be `{A,B,C}` and fix `lambda in (0,1)`. Choose `e>0` small
enough that

```text
e + e^3/(1-e^2) < 1-lambda.
```

This inequality makes every entry of the kernels below nonnegative and makes
the student's additional eigenvalue larger than `lambda`. Set

```text
mu_e = (e, (1-e)/2, (1-e)/2),
psi_e = (0, 1/sqrt(1-e), -1/sqrt(1-e)),
q_e(i,j) = 1 + lambda psi_e(i)psi_e(j).
```

The reference operator has eigenvalues `{1,lambda,0}`, and
`||psi_e||_infinity` stays bounded as `e` tends to zero. Indeed,
`psi_e` has zero `mu_e`-mean and unit `L2(mu_e)` norm, and
`||psi_e||_infinity<=sqrt(2)` whenever `e<=1/2`. Moreover,

```text
P_e f = E_mu_e[f] 1 + lambda <psi_e,f>_mu_e psi_e.
```

Set

```text
mu_hat_e = (e^2, (1-e^2)/2, (1-e^2)/2),
r = e,
d = e^3/(1-e^2),

P_hat_e =
  [[1-r, r/2, r/2],
   [d, (1-d+lambda)/2, (1-d-lambda)/2],
   [d, (1-d-lambda)/2, (1-d+lambda)/2]].
```

This student is ergodic and reversible with respect to `mu_hat_e`. Its
eigenvalues are

```text
1, lambda, 1-e-e^3/(1-e^2).
```

Reversibility follows from
`mu_hat_e(A) r/2 = mu_hat_e(B) d = e^3/2` and the symmetry between `B` and
`C`. The antisymmetric vector `(0,1,-1)` has eigenvalue `lambda`; aggregating
`B,C` into one state gives the remaining eigenvalue `1-r-d`.

Hence its slowest nontrivial eigenvalue tends to one while the reference's is
the fixed value `lambda`. Here

```text
h = (e,1+e,1+e),
```

so the measures are equivalent for every fixed `e`, but the lower density
ratio degenerates as `e->0`.

For an exact calculation, set

```text
A_+(e) = (2+lambda)(1-e)-e^2,
A_-(e) = (2-lambda)(1-e)-e^2.
```

Entrywise substitution in the definition of `D_chi` gives

```text
cell(s)          rho_hat-rho                         mu_i mu_j
A,A              -e^3                                e^2
A<->B, A<->C     -e(1-e-e^2)/2                      e(1-e)/2
B,B and C,C      e A_+(e)/4                          (1-e)^2/4
B,C and C,B      e A_-(e)/4                          (1-e)^2/4
```

There are four directed `A`-to-`B/C` cells and two cells in each of the last
two rows. Therefore

```text
TV(mu_hat_e,mu_e) = e(1-e),

D_chi^2
  = e^4
    + 2e(1-e-e^2)^2/(1-e)
    + e^2(A_+(e)^2+A_-(e)^2)/(2(1-e)^2)
  = 2e + O(e^2).                                     (12)
```

The right sides of (12) tend to zero, whereas the leading nontrivial spectral
error tends to `1-lambda>0`. This proves No-go Theorem B.

### Consequence for the draft's local-sublevel route (C3)

In No-go Theorem B, exclude the rare state `A`. Its reference and student
masses are `e` and `e^2`, both tending to zero. The last two rows of the
entrywise table are exactly the contribution on `{B,C} x {B,C}`; their squared
normalized contribution is `O(e^2)`, so the corresponding local `D_chi` is
`O(e)`. Nevertheless the student has an eigenvalue tending to one.

Therefore, excluded mass plus local `D_chi` alone cannot control global
timescales. A valid C3 theorem must add some hypothesis that rules out the
exhibited trapping mechanism, such as a uniform conductance,
escape-probability, or return-time bound for the excluded region.

### Two valid replacements

Assume (E). If

```text
||h-1||_infinity <= eta < 1,
```

then (11), `||P||<=1`, and the multiplication-operator norm give

```text
||B_hat-P||_op
    <= (D_chi + eta)/(1-eta).                        (13)
```

Indeed, with `s=h^(-1/2)`,

```text
B_hat-P = s(J_hat-P)s + (s-I)Ps + P(s-I).
```

Here all multiplier operators are bounded because `h>=1-eta`.

Here `||s||_infinity <= (1-eta)^(-1/2)` and

```text
||s-I||_infinity (||s||_infinity+1)
    <= eta/(1-eta),
```

which proves (13).

This is a valid, if strong, replacement for the draft's additive TV term.

A less pointwise alternative is available if

```text
h >= a > 0
and
||q||_infinity <= C.
```

Retain the standing assumption `q_hat-q in L2(mu x mu)`, so `D_chi<infinity`.

Marginalizing `q_hat-q` and applying Cauchy-Schwarz gives

```text
||h-1||_2 <= D_chi.
```

Explicitly,

```text
h(x)-1 = integral (q_hat(x,y)-q(x,y)) dmu(y),
```

and Cauchy-Schwarz in `y`, followed by integration in `x`, proves the bound.

Using (11) and the fact that `u -> u^(-1/2)` is Lipschitz on `[a,infinity)`
then gives

```text
||B_hat-P||_HS <= K(a,C) D_chi,                      (14)

K(a,C)
  = 1/a + C(1+a^(-1/2))/(2a^(3/2)).
```

To see (14), the integral kernel of `B_hat` relative to `mu` is

```text
s(x) q_hat(x,y) s(y).
```

Subtracting `q` gives

```text
s(x)s(y)(q_hat-q)(x,y)
  + q(x,y)(s(x)s(y)-1).
```

The first term has `L2(nu)` norm at most `D_chi/a`. Also,

```text
||s-1||_2 <= D_chi/(2a^(3/2)),
||s(x)s(y)-1||_2
    <= (1+a^(-1/2)) ||s-1||_2.
```

Multiplying the second line by `||q||_infinity <= C` and adding the two
contributions gives (14).

Equations (13) or (14), followed by Theorem 1, are honest different-marginal
theorems. Neither uses a `kappa(V)` factor.

Combining (9) and (14) yields a conditional pair-KL-to-spectrum theorem if
`q`, `q_hat` are bounded and `h` has a positive lower bound. Connecting that
result to a distillation loss is a separate premise.

### Theorem 3: conditional different-marginal pair-spectrum bound

Assume:

1. the reference pair is stationary and reversible with transfer operator `P`;
2. the generated pair is stationary with marginal `mu_hat`;
3. `h=dmu_hat/dmu >= a>0`;
4. `0<=q,q_hat<=B<infinity`;
5. `lambda` is a simple isolated eigenvalue of `P` with gap `delta`; and
6. `KL(rho_hat||rho)<=K`.

Then the unitary transport `B_hat=U P_hat U^(-1)` satisfies

```text
||B_hat-P||_op
  <= ||B_hat-P||_HS
  <= K_op(a,B) sqrt(2 B K),                          (14a)

K_op(a,B)
  = 1/a + B(1+a^(-1/2))/(2a^(3/2)).
```

#### Proof

Theorem 2 gives `D_chi<=sqrt(2 B K)`. Equation (14), with `C=B`, gives

```text
||B_hat-P||_op
  <= ||B_hat-P||_HS
  <= K_op(a,B) D_chi.
```

Combining the two inequalities proves (14a). The spectral and timescale
conclusions are then exactly Theorem 1 and Corollary 2 applied with
`epsilon=epsilon_*`. Unitary conjugation preserves the student spectrum.

If the right side, denoted `epsilon_*`, is less than `delta/2`, Theorem 1
gives one algebraic eigenvalue of `B_hat` in the isolating disk and

```text
|lambda_hat-lambda| <= epsilon_*.
```

Because `B_hat` is a real operator and the local algebraic multiplicity is
one, this eigenvalue is real. If
`epsilon_*<min(delta/2,lambda,1-lambda)`, its one-lag implied-timescale error
obeys the exact bound (4) with `epsilon=epsilon_*`.

If a self-adjoint surrogate is desired, define

```text
S_B = (B_hat+B_hat*)/2
    = U (P_hat+P_hat*)/2 U^(-1).
```

Then `||S_B-P||_op<=epsilon_*`, so Theorem 1 and (4) give the same bound for
the local eigenvalue of the additive reversibilization. That eigenvalue is a
reversibilized surrogate, not an eigenvalue of `P_hat`. No `kappa(V)` factor
is required for either local perturbation statement.

This theorem concerns induced pair operators. Their positive real eigenvalues
in `(0,1)` define the stated one-lag implied-timescale proxies. They describe
actual multi-lag reference and generated path relaxation through operator
powers only when the corresponding time-homogeneous Markov factorizations
hold for both paths.

## 8. Weak observable control without density-ratio assumptions

### 8.1 Wasserstein controls Lipschitz pair observables, not MFPT

Equip `X x X` with the sum metric

```text
d_2((x,y),(x',y')) = d(x,x') + d(y,y').
```

Assume the two pair laws have finite first moments in `d_2`. For every
`L_F`-Lipschitz pair observable `F`, Kantorovich-Rubinstein duality gives the
first inequality below. If they have finite second moments, then

```text
|E_rho_hat[F]-E_rho[F]|
  <= L_F W_1(rho_hat,rho)
  <= L_F W_2(rho_hat,rho).                           (14b)
```

If `psi_i` is bounded by `M_i` and is `L_i`-Lipschitz, then
`F(x,y)=psi_i(x)psi_i(y)` is `M_i L_i`-Lipschitz under `d_2`, because

```text
|psi_i(x)psi_i(y)-psi_i(x')psi_i(y')|
  <= M_i L_i [d(x,x')+d(y,y')].
```

Thus Wasserstein distance controls this fixed teacher-mode correlation. It
does not control the operator norm or student spectrum, as No-go Theorem A
shows. A first-passage time is a path functional, generally unbounded, rather
than a function of one pair, so (14b) gives no MFPT theorem.

The same counterexample makes this failure explicit. Take `A_e` as the target
and start outside it. Under the independent student kernel the hitting time in
lag steps is geometric with mean `1/e`. Under the reference kernel `q_e`, the
one-step probability of entering `A_e` from any point outside it is
`e(1-theta)`, so the mean is `1/[e(1-theta)]`. Their difference diverges while
every finite-order pair Wasserstein distance tends to zero. A valid MFPT route
therefore needs direct killed-operator control as in Corollary 5, or separate
uniform kernel, contraction, Lyapunov, and value-function regularity
hypotheses. Pair-law `W_2` alone supplies none of these.

### 8.2 Segment KL controls bounded pair observables

This section begins from a segment-law KL bound; it does not derive one from an
unspecified distillation objective. Suppose a separate theorem proves

```text
KL(Q_hat_segment || Q_segment) <= K_seg.             (K)
```

Data processing gives the same upper bound for any two-time marginal. For a
real-valued, normalized nonconstant teacher eigenfunction `psi_i`, orthogonal
to constants and satisfying `||psi_i||_infinity <= M_i`, Pinsker's inequality
therefore gives

```text
|E_rho_hat[psi_i(x0)psi_i(x_tau)] - lambda_i|
    <= M_i^2 sqrt(2 K_seg).                          (15)
```

The orthogonality is automatic for `lambda_i!=1`, since self-adjointness and
`P1=1` imply
`lambda_i<psi_i,1>=<P psi_i,1>=<psi_i,P1>=<psi_i,1>`.

Here total variation is `TV(P,Q)=sup_A |P(A)-Q(A)|`, so
`TV<=sqrt(KL/2)` and
`|E_P f-E_Q f|<=2||f||_infinity TV`. These two inequalities prove (15).

In the same-marginal case, `psi_i` has zero mean and unit variance under both
laws, so (15) controls its normalized one-lag correlation. If the generated
pair is stationary with a different marginal, let

```text
m_hat = E_mu_hat[psi_i],
v_hat = E_mu_hat[(psi_i-m_hat)^2],
c_hat = E_rho_hat[(psi_i(x0)-m_hat)(psi_i(x_tau)-m_hat)] / v_hat.
```

Set

```text
r_i = M_i^2 sqrt(2 K_seg) + 2 M_i^2 K_seg.
```

Because `E_mu[psi_i]=0` and `E_mu[psi_i^2]=1`, data processing and Pinsker give

```text
|m_hat|^2 <= 2 M_i^2 K_seg,
|v_hat-1| <= r_i.
```

More explicitly, if

```text
a_hat     = E_mu_hat[psi_i^2],
gamma_hat = E_rho_hat[psi_i(x0)psi_i(x_tau)],
```

then Pinsker gives

```text
|a_hat-1| <= M_i^2 sqrt(2 K_seg),
|gamma_hat-lambda_i| <= M_i^2 sqrt(2 K_seg).
```

Since `v_hat=a_hat-m_hat^2` and the centered lag covariance is
`gamma_hat-m_hat^2`, each differs from its reference value by at most `r_i`.
Thus `v_hat>=1-r_i`. Consequently, if `r_i<1`,

```text
|c_hat-lambda_i|
    <= (1+|lambda_i|) r_i / (1-r_i).                 (16)
```

Indeed, writing `gamma_c_hat=gamma_hat-m_hat^2`,

```text
|c_hat-lambda_i|
  = |gamma_c_hat-lambda_i v_hat|/v_hat
  <= (|gamma_c_hat-lambda_i|
      + |lambda_i| |v_hat-1|)/(1-r_i),
```

which is (16).

Define the **one-lag teacher-mode relaxation proxy**

```text
t_proj_i = -tau / log(c_hat).
```

Set

```text
epsilon_proj_i = (1+|lambda_i|) r_i / (1-r_i).
```

If `lambda_i in (0,1)` and
`epsilon_proj_i<min(lambda_i,1-lambda_i)`, then (16) keeps `c_hat` in
`(0,1)` and the scalar mean-value theorem gives

```text
|t_proj_i-t_i|
  <= L_tau(lambda_i,epsilon_proj_i) epsilon_proj_i.  (17)
```

No spectral-gap condition is needed for this scalar observable bound.
This is a segment-KL-to-one-lag-observable theorem with no density-ratio
assumption. It is not yet a distillation-loss theorem because premise (K) is
external.

For the reference pair, the analogous centered correlation equals `lambda_i`,
so its proxy equals `t_i`. The generated `t_proj_i` generally does not equal
`t_i`; equation (17) only bounds their difference. It is **not** a student
implied timescale unless the centered teacher observable is also a student
pair-operator eigenfunction. It is an actual multi-lag relaxation time only
with the further Markov/semigroup consistency that makes powers of the
one-lag operator describe the path. It gives no MFPT guarantee. The
counterexamples above show why it cannot be promoted to a student-spectrum
statement without stronger assumptions.

## 9. Corrected final statement

A consequence of the segment-law KL premise that needs no density-ratio
assumption is:

> Segment-law KL controls bounded teacher-mode autocorrelations and their
> one-lag relaxation proxies through (15)-(17), subject to the displayed
> small-error condition. This does not identify a student eigenvalue. Theorem 3
> controls a student pair-operator eigenvalue only under all its assumptions,
> including isolation and pair KL, and only when
> `epsilon_*<delta/2`. Once common-space operator control exists, local
> non-normal perturbation of a simple isolated reference mode needs no global
> student eigenvector condition number.

For L37, the theory decision is therefore:

- Use (15)-(17) only as a segment-KL-to-teacher-mode one-lag result.
- Present Theorem 3 only as a conditional pair-operator spectrum theorem.
- Do not claim an MD-realistic full-spectrum result until the required density
  constants are proved finite and shown to be non-vacuous.
- Do not claim that one-time marginals, KL/Wasserstein alone, or TV marginal
  drift control student implied timescales.
- Do not report pair-kernel MFPTs or committors as properties of generated
  trajectories unless infinite-horizon Markov consistency and the
  killed-resolvent hypotheses are established.
- Do not describe additive-reversibilization eigenvalues as student
  eigenvalues, or an eigenvector angle as a set-assignment guarantee without a
  margin condition.
- Do not call this an end-to-end distillation theorem until a specific loss,
  KL direction, and loss-to-segment-KL constant are supplied and proved.

## 10. Claim-by-claim resolution of the original spec

| Spec claim | Resolution | Evidence |
|---|---|---|
| Section 1, a pair law alone supports generated kinetics | **FALSE beyond one lag**; it defines a pair operator, not the actual path law | Section 2.1 |
| Generated pair has equal endpoint marginals | **MODEL-SPECIFIC OPEN**; required for a stationary student pair operator | Section 2.1 |
| A1, reversible stationary pair gives self-adjoint contraction | **PROVED**; the spectrum is in `[-1,1]`, not necessarily `(-1,1]`; ergodicity, isolated modes, and their multiplicities remain assumptions | Lemma 0 |
| A1, ordinary configuration-space reversibility for MD | **MODEL-SPECIFIC OPEN**; phase-space momentum reversal and non-Markov projections must be handled | Section 2.1 |
| A2, bounded slow modes | **ASSUMPTION; MODEL-SPECIFIC OPEN**; not needed for operator perturbation, but used for the KL observable bound | Section 8 |
| Lipschitz slow modes for the Wasserstein route | **ADDITIONAL ASSUMPTION; MODEL-SPECIFIC OPEN** | Section 8.1 |
| A3, finite `D_chi` equivalent to `q_hat in L2` | **FALSE without also assuming `q in L2`**; the correct condition is `q_hat-q in L2` | Lemma 1 discussion |
| A3, student-only Doeblin bound implies finite `D_chi` | **FALSE unless the reference density has additional `L2` control** | Torus counterexample after Lemma 1 |
| Absolute continuity with respect to `mu x mu` | **MODEL-SPECIFIC OPEN**; it can fail for deterministic dynamics | Section 2.2 and the singular Wasserstein example |
| A4, marginal drift contributes additive TV error | **NO REFERENCE-UNIFORM MODULUS**; the registered universal TV claim is disproved | No-go Theorem B |
| Section 3, `D_chi` equals the Hilbert-Schmidt perturbation | **PROVED with same marginals** | Lemma 1 |
| Section 4.1, simple isolated eigenvalue location | **PROVED if `D_chi<delta/2`** | Theorem 1 |
| Section 4.1, scalar first-order remainder | **PROVED for a simple mode**; the simplified `2 D_chi^2/delta` bound requires `D_chi<=delta/4` | Theorem 1 and (2) |
| Section 4.1, leading term equals teacher-mode correlation error | **PROVED for a real normalized simple mode with common marginal** | Corollary 1 |
| Section 4.1, repeated eigenvalue scalar formula | **DISPROVED**; spectral-cluster and Riesz-subspace replacements are proved | (6b)-(6c) |
| Section 4.1, real implied-timescale bound | **PROVED for a local real eigenvalue remaining in `(0,1)`**; without path consistency it is a one-lag proxy | (4)-(5) |
| Section 4.2, eigenvector bound | **PROVED for a simple mode with corrected denominator** | (6) |
| Section 4.2, eigenvector angle implies metastable assignment | **FALSE without a threshold/clustering margin**; a conditional threshold bound is proved | (6a) |
| Section 4.3, MFPT and committor | **CONDITIONAL LAG-SKELETON `L2` BOUND**; requires target-specific killed-resolvent and boundary-source control | (7), (7a), (7b) |
| Section 4.3, MFPT equation on `X\(A union B)` | **FALSE**; MFPT to `B` is solved on `X\B`, while `X\(A union B)` is the committor interior | Corollary 5 |
| Section 4.3, MFPT amplification equals full-chain slow-timescale amplification | **DISPROVED** | Independent-resampling rare-target example after (7b) |
| Section 5.1, clean theorem | **SPECTRAL PART PROVED** with simplicity or the cluster replacement and explicit smallness; hitting and actual-path parts need their additional hypotheses | Sections 3-4 |
| Section 5.2, additive `TV` plus `kappa(V)` theorem | **TV PART DISPROVED; `kappa(V)` PART SUPERSEDED** once common-space operator control is obtained | Section 7 and Theorems 1, 3 |
| Section 6, `W2 -> D_chi` obstruction with `D_chi=infinity` | **PROVED** | Singular diagonal-mixture example in Section 5 |
| Section 6, condition (C1) | **DISPROVED as written** | Section 5 |
| Section 6, corrected bounded-density bridge | **PROVED CONDITIONALLY**; target-specific MD applicability is open | Theorem 2 and OU check |
| Section 6, `W2` fallback | **PROVED for Lipschitz pair observables; DISPROVED for spectrum and standalone MFPT control** | No-go Theorem A, (14b), and its hitting-time example |
| Section 6, local sublevel-set route | **INSUFFICIENT as stated** | No-go Theorem B; requires tail and anti-trapping estimates |
| Section 7, per-frame insufficiency | **PROVED** | Two-state example |
| Section 7, matching selected two-time correlations suffices | **HEURISTIC ONLY**; no sufficiency theorem follows from GATE-0 | Sections 4 and 8 |
| GATE-0 | **PASS** | Section 4 |
| GATE-1 | **REGISTERED C1 DISPROVED; NO REGISTERED BRANCH FITS; HEADLINE MD THEOREM BLOCKED** | Sections 5-6 |
| GATE-2 | **SUPERSEDED CONDITIONAL ON COMMON-SPACE OPERATOR CONTROL** | Theorem 1 |
| GATE-3 | **OPEN empirical gate** | Requires released MDGen data |
| Loss-to-segment-KL premise | **OPEN / unspecified by the spec** | Must be proved for the exact distillation objective |
| MDGen stationarity, Markov consistency, and density constants | **MODEL-SPECIFIC OPEN** | Sections 2, 6, and 7 |
| Prior-art novelty and fetched-reference claims | **LITERATURE PREMISES UNVERIFIED BY THIS PROOF** | Require a separate literature audit |

The original spec explicitly allowed either proving a step or showing that it
is blocked. The table resolves the pure operator, metric, and counterexample
claims by proof, correction, or disproof. It deliberately does **not** claim an
end-to-end MDGen theorem: the exact loss, stationarity, absolute continuity,
path consistency, target-specific constants, GATE-3 calibration, and novelty
remain external obligations and must not be represented as theorem
consequences.

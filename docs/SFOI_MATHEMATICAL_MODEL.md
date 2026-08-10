# SFOI mathematical model and implementation map

This document defines the statistical model used by MOI. It separates the
mathematics from the iterative numerical method so that a small numerical
`delta` is not confused with scientific accuracy or goodness of fit.

## 1. State vector

For `n` reaches and `K` runoff regions, the physical state is

\[
x = [Q_1,\ldots,Q_n,R_1,\ldots,R_K]^T,
\]

where `Q_i` is reach discharge and `R_k` is regional runoff. In a gaged
constrained Mean-flow run with genuine FLPE rows, the augmented state is

\[
z = [x^T,b,\eta_1,\ldots,\eta_G]^T.
\]

`b` is one multiplicative FLPE bias per algorithm and `eta_g` is a latent
correlated-error effect for each runoff region that contains at least one
genuine FLPE row. Prior/Fill reaches do not create or receive latent effects.

Implementation:

- `build_sparse_sfoi_problem` in `moi/sfoi_math_core.py` builds `x` and the
  sparse physical system.
- `adjust_lsq_bias_correlated_sparse` builds `z` only when bias and/or
  correlation is identifiable.
- `Integrate._augmented_solver_features` requires both a calibration gage and
  at least one genuine FLPE row.

## 2. Observation equations

### 2.1 Genuine FLPE discharge

For a reach selected by `flpe_eligible_mask`,

\[
y_i^{FLPE} = (1+b)Q_i
  + \sigma_i\sqrt{\rho}\,\eta_{g(i)}
  + \sigma_i\sqrt{1-\rho}\,\epsilon_i,
\qquad \epsilon_i\sim N(0,1).
\]

The multiplicative marginal scale is

\[
\sigma_i =
\frac{c_i\max(|(1+b)Q_i|,\theta_{floor})}{\sqrt{f_i}},
\]

where `c_i` is the coefficient of variation and `f_i` is the robust factor in
`(0,1]`. The independent residual is divided by
`sigma_i * sqrt(1-rho)`. Therefore `rho` is the shared fraction of variance,
not the shared fraction of error amplitude.

Only genuine FLPE rows receive `(1+b)` and `eta`. The implementation uses
`flpe_eligible_mask` in `marginal_sigma`, `prediction`, and the augmented
Jacobian. Rows marked Prior or Fill remain ordinary discharge priors.

### 2.2 Bias and regional-effect priors

\[
b\sim N(0,s_b^2), \qquad \eta_g\sim N(0,1).
\]

The defaults are `s_b=0.50` and `rho=0.20`. In the WLS objective these add

\[
(b/s_b)^2 + \sum_g \eta_g^2.
\]

The bias prior is one row regardless of basin size, so many informative FLPE
rows can dominate it. Each regional effect has its own unit-normal penalty.

### 2.3 Prior and Fill discharge

If an algorithm does not supply a usable FLPE estimate, MOI first uses the SoS
discharge prior:

\[
y_i^{prior}=Q_i+e_i, \qquad
\sigma_i=c_{prior}\max(|Q_i|,\theta_{floor}).
\]

The default is `c_prior=0.60`. If the SoS prior is also unavailable, a
runoff-derived Fill value is used with default `c_fill=1.0`. Neither row is
bias/correlation augmented.

Provenance is preserved by `Integrate.get_pre_mean_q` and
`Integrate.initialize_integration_vars` as `FLPE`, `Prior`, or `Fill`.

### 2.4 Regional runoff prior

Each region contributes

\[
y_k^R=R_k+e_k^R,
\]

with default coefficient of variation `0.50`. These rows regularize regional
runoff and couple it to reach discharge through mass balance.

### 2.5 Gage observations

For a calibration gage on reach `j`,

\[
y_j^{gage}=Q_j+e_j^{gage}.
\]

The default coefficient of variation is `0.10`. Gage sigma is frozen at its
observation scale and gages are excluded from robust downweighting. Gages do
not receive FLPE bias or regional correlation.

Implementation: `Integrate.build_gage_observation_rows` appends selector rows
after the physical system has been built.

### 2.6 Mass balance

For every dependent reach,

\[
Q_i - \sum_{u\in upstream(i)} c_{ui}Q_u
    - \Delta A_i\,u_c\,R_{g(i)} = 0.
\]

Source/headwater reaches have no mass row and retain independent discharge
degrees of freedom. In hard mode this is an equality constraint. In default
soft mode it is a protected pseudo-observation with explicit absolute
uncertainty

\[
\sigma_{mass}=5\;\mathrm{m^3\,s^{-1}}.
\]

Because the target is zero, a relative CV is not meaningful. The builder
encodes `SFOI_Soft_Mass_Sigma / SFOI_Theta_Floor` into the legacy multiplicative
weight interface so the resulting fixed sigma is exactly the configured
absolute value.

## 3. Objective and Gauss-Newton step

At outer iteration `k`, multiplicative scales and robust factors are frozen.
The nonlinear FLPE prediction is linearized around `z_k`. The important
Jacobian terms are

\[
\frac{\partial \hat y_i}{\partial Q_i}=1+b,
\qquad
\frac{\partial \hat y_i}{\partial b}=Q_i,
\qquad
\frac{\partial \hat y_i}{\partial \eta_g}=\sigma_i\sqrt{\rho},
\]

for genuine FLPE rows. The sparse constrained WLS subproblem is

\[
z_k^*=\arg\min_z
\|W_k(J_kz-r_k)\|_2^2
+(b/s_b)^2+\sum_g\eta_g^2,
\]

subject to physical bounds and optional hard mass equalities.

Implementation:

- `prediction` evaluates the nonlinear observation model.
- `J_x`, `J_bias`, and `J_effect` assemble the sparse Jacobian.
- `_solve_wls_sparse` solves the bounded/equality-constrained quadratic
  subproblem.

## 4. Robust IRLS update

After applying a state step, MOI computes standardized residuals and the global
chi-square statistic. If it exceeds the upper two-sided chi-square bound,
eligible rows with `|s_i| > s_limit` receive

\[
f_i^{target}=\left(\frac{s_{limit}}{|s_i|}\right)^2.
\]

The default limit is `2.5`. Robust factors are damped:

\[
f_i^{new}=(1-d_r)f_i^{old}+d_rf_i^{target},
\]

with default `d_r=0.5`. If chi-square is below the lower bound, factors recover
toward one. Gage and soft-mass rows are fixed at `f_i=1`.

`So` in the augmented result is reduced chi-square. It diagnoses fit but is not
a convergence delta.

## 5. Oscillation control

Let `d_k=z_k^*-z_k`. Components are scaled so large discharges do not hide a
reversal in dimensionless bias/effects. A period-two signature is detected when

\[
\cos(d_k,d_{k-1})\le -0.5.
\]

The applied step is

\[
z_{k+1}=z_k+\lambda_kd_k.
\]

On an oscillation, `lambda` is multiplied by `0.5`, down to `0.05`. After three
non-oscillating iterations it is multiplied by `1.25` toward one. A solution
using this mechanism is labeled `converged_after_relaxation`, not
`forced_converged`; no Mike-style convergence-by-construction is claimed.

## 6. Component-wise convergence

For physical state components,

\[
r_j=\frac{|x_{j,new}-x_{j,old}|}
{\max(|x_{j,new}|,\theta_{floor})}.
\]

MOI records

\[
\delta_{phys,RMS}=\sqrt{mean(r_j^2)},
\qquad
\delta_{phys,p95}=percentile_{95}(r_j),
\qquad
\delta_{phys,max}=\max(r_j).
\]

The other component changes are

\[
\delta_b=|b_{new}-b_{old}|,
\qquad
\delta_\eta=RMS(\eta_{new}-\eta_{old}),
\qquad
\delta_f=\max|f^{new}-f^{old}|.
\]

Both the raw candidate step and the applied relaxed step must satisfy their
state tolerances. The default gates are:

| Gate | Default |
| --- | ---: |
| physical RMS | `1e-2` (1%) |
| physical p95 | `2e-2` (2%) |
| absolute bias fraction | `1e-3` (0.1 percentage point) |
| regional-effect RMS | `1e-2` SD |
| maximum robust-factor change | `1e-3` |

These values are initial engineering defaults and must be validated through
threshold ablations. Convergence means iteration stability; it does not imply
small truth error or an acceptable reduced chi-square.

Implementation: the metrics and gates are in
`adjust_lsq_bias_correlated_sparse`. Final component deltas, reduced chi-square,
thresholds, and relaxation diagnostics are copied by
`Integrate.integrator_optimization_calcs` and written by
`Output._write_bias_correlation_diagnostics`.

## 7. Prior-only solver reuse

If an algorithm has zero genuine FLPE rows, augmentation is disabled. Its
ordinary sparse problem is fingerprinted using the sparse design, observation
vector, covariance, constraints, bounds, and fixed/robust masks. An identical
prior-only problem already solved for another algorithm reuses a deep copy of
that result. This prevents HiVDI/SADS or other unavailable algorithms from
re-solving the same SoS-prior system while preserving separate downstream
algorithm outputs.

Implementation: `Integrate._prior_only_solver_signature` and the local
`prior_only_solver_cache` in `Integrate.integrator_optimization_calcs`.

## 8. Ordinary MOI and Mike-style fallback

The ordinary multiplicative solver remains separate. Its default convergence
threshold is 5%, and it retains Mike's iteration-7 permanent-outlier and
post-iteration-14 weight-mixing fallback. Its status distinguishes natural
from constructed convergence.

The augmented solver deliberately does not freeze iteration-7 outliers or
force weight equality. That stronger fallback should be evaluated only after
the component-threshold and relaxation approach has been tested on real
basins.

"""Deterministic flow-law parameter fitting for the MOI integrator.

Why this module exists
----------------------
``compute_FLPs()`` used to hand every algorithm's parameter vector to a single
``scipy.optimize.minimize`` call started from the FLPE prior.  On a basin that
works most of the time.  On a global run it does not: L-BFGS-B from one start
point has no way to tell "converged" from "stalled in a local basin", four of
the six algorithms never even checked ``res.success``, and the failures are
silent -- a stalled fit exports finite parameters that look exactly like a good
fit.

For every algorithm except MOMMA the roughness parameter is a pure *scale*: the
flow law is linear in it.  That means the fit is a separable (variable
projection) least-squares problem.  Fix the non-scale parameters and the
optimal scale has a closed form; the search then collapses to one or two
dimensions, which a bounded grid can cover exhaustively.

The result is deterministic -- same inputs, same parameters, no dependence on
where the optimiser happened to start -- and it separates two failure modes
that the single-start optimiser conflated:

  * the optimiser stalled          -> the grid finds the better optimum
  * the flow law cannot fit at all -> the grid confirms the residual is real

Closed forms
------------
Write the flow law as ``q_t = s * C_t(theta)``, where ``s`` is the scale and
``theta`` the remaining parameters.  Minimising over ``s`` for fixed ``theta``:

  linear space   sum_t w_t (s C_t - y_t)^2      ->  s* = S(w C y) / S(w C^2)
  log space      sum_t w_t (log s + log C_t - log y_t)^2
                                                ->  log s* = S(w (log y - log C)) / S(w)

``n`` enters as ``q = C / n`` so ``s = 1/n``; HiVDI's ``alpha`` enters directly
so ``s = alpha``.  MOMMA has no scale parameter -- ``n_b = 0.11 * Save^0.18`` is
pinned by slope -- so it gets a plain two-dimensional grid over ``(B, H)``.
That asymmetry is itself informative: it is part of why MOMMA's residual
behaves differently from the dA-driven laws.

Nothing here reaches the SFOI integrator solve.  This module only decides which
flow-law parameters reproduce a hydrograph that has already been integrated.
"""

import numpy as np

# ---------------------------------------------------------------------------
# fit_status values, written to the output netCDF as an integer code.
# ---------------------------------------------------------------------------
FIT_GOOD = 'good'
FIT_UNIDENTIFIABLE = 'unidentifiable'
FIT_FALLBACK_PRIOR = 'fallback_prior'
FIT_NO_SERIES = 'no_series'
FIT_FAILED = 'failed'

FIT_STATUS_CODE = {
    FIT_GOOD: 0,
    FIT_UNIDENTIFIABLE: 1,
    FIT_FALLBACK_PRIOR: 2,
    FIT_NO_SERIES: 3,
    FIT_FAILED: 4,
}

# param_at_bound is a bitmask: bit i set means parameter i sits on a bound.
BOUND_TOL = 1.0e-6

# A fit is refused outright below this many usable timesteps.  Two free
# parameters against four points is not a fit, it is interpolation.
MIN_VALID_OBS = 5


class FitOutcome(object):
    """Parameters plus everything needed to judge whether to trust them."""

    __slots__ = (
        'params', 'status', 'nrmse_lin', 'nrmse_log', 'param_at_bound',
        'n_valid_obs', 'n_grid_evals', 'objective', 'method',
    )

    def __init__(self, params, status, nrmse_lin=np.nan, nrmse_log=np.nan,
                 param_at_bound=0, n_valid_obs=0, n_grid_evals=0,
                 objective=np.nan, method='varpro'):
        self.params = tuple(float(p) for p in params)
        self.status = status
        self.nrmse_lin = float(nrmse_lin)
        self.nrmse_log = float(nrmse_log)
        self.param_at_bound = int(param_at_bound)
        self.n_valid_obs = int(n_valid_obs)
        self.n_grid_evals = int(n_grid_evals)
        self.objective = float(objective)
        self.method = method


# ---------------------------------------------------------------------------
# residual helpers
# ---------------------------------------------------------------------------

def _as_1d(values):
    return np.asarray(values, dtype=float).ravel()


def residual_weights(loss, resid, scale=None):
    """IRLS weights that turn a least-squares solve into a robust one.

    ``soft_l1`` and ``huber`` match ``scipy.optimize.least_squares``' loss
    functions.  The point is dA: a single bad ``d_x_area`` timestep -- a
    geometry dropout, a jump after a missing pass -- otherwise drags the whole
    fit, because squared error rewards chasing it.  Down-weighting it is the
    right response; deleting the reach is not.
    """
    if loss in (None, 'l2', 'linear'):
        return np.ones_like(resid)
    r = np.abs(resid)
    if scale is None or not np.isfinite(scale) or scale <= 0.:
        scale = np.median(r[np.isfinite(r)]) if np.any(np.isfinite(r)) else 1.
        scale = max(float(scale), 1.0e-12)
    z = r / scale
    if loss == 'soft_l1':
        return 1.0 / np.sqrt(1.0 + z ** 2)
    if loss == 'huber':
        w = np.ones_like(z)
        big = z > 1.0
        w[big] = 1.0 / z[big]
        return w
    return np.ones_like(resid)


def solve_scale(basis, target, space='linear', weights=None,
                loss='l2', irls_iters=4):
    """Closed-form optimal scale ``s`` for ``q = s * basis`` against ``target``.

    Returns ``nan`` when the pair carries no usable information -- an all-NaN
    basis, a degenerate target, or (in log space) no positive pairs.  A robust
    ``loss`` re-solves the same closed form a few times with IRLS weights;
    each iteration is two dot products, so this stays far cheaper than a
    numerical optimiser.
    """
    basis = _as_1d(basis)
    target = _as_1d(target)
    if basis.size != target.size or basis.size == 0:
        return np.nan

    if weights is None:
        w0 = np.ones_like(basis)
    else:
        w0 = _as_1d(weights)
        if w0.size != basis.size:
            return np.nan

    if space == 'log':
        good = (
            np.isfinite(basis) & np.isfinite(target) & np.isfinite(w0)
            & (basis > 0.) & (target > 0.) & (w0 > 0.)
        )
        if not np.any(good):
            return np.nan
        lb = np.log(basis[good])
        ly = np.log(target[good])
        w = w0[good]
        d = ly - lb
        log_s = float(np.sum(w * d) / np.sum(w))
        for _ in range(irls_iters if loss not in (None, 'l2', 'linear') else 0):
            wr = residual_weights(loss, (log_s + lb) - ly) * w
            denom = np.sum(wr)
            if denom <= 0.:
                break
            log_s = float(np.sum(wr * d) / denom)
        if not np.isfinite(log_s):
            return np.nan
        # exp() of a large residual overflows to inf; the caller's bound clip
        # would then hide a fit that is in fact meaningless.
        if log_s > 200. or log_s < -200.:
            return np.nan
        return float(np.exp(log_s))

    good = (
        np.isfinite(basis) & np.isfinite(target) & np.isfinite(w0) & (w0 > 0.)
    )
    if not np.any(good):
        return np.nan
    b = basis[good]
    y = target[good]
    w = w0[good]
    denom = float(np.sum(w * b * b))
    if not np.isfinite(denom) or denom <= 0.:
        return np.nan
    s = float(np.sum(w * b * y) / denom)
    for _ in range(irls_iters if loss not in (None, 'l2', 'linear') else 0):
        wr = residual_weights(loss, s * b - y) * w
        denom = float(np.sum(wr * b * b))
        if not np.isfinite(denom) or denom <= 0.:
            break
        s = float(np.sum(wr * b * y) / denom)
    return s if np.isfinite(s) else np.nan


def series_cost(q, target, space='linear', weights=None, loss='l2'):
    """Scalar cost of one candidate hydrograph against the target series."""
    q = _as_1d(q)
    target = _as_1d(target)
    if q.size != target.size or q.size == 0:
        return np.inf
    if weights is None:
        w = np.ones_like(q)
    else:
        w = _as_1d(weights)
        if w.size != q.size:
            return np.inf

    if space == 'log':
        good = (
            np.isfinite(q) & np.isfinite(target) & np.isfinite(w)
            & (q > 0.) & (target > 0.) & (w > 0.)
        )
        if not np.any(good):
            return np.inf
        r = np.log(q[good]) - np.log(target[good])
        ww = w[good]
    else:
        good = np.isfinite(q) & np.isfinite(target) & np.isfinite(w) & (w > 0.)
        if not np.any(good):
            return np.inf
        r = q[good] - target[good]
        ww = w[good]

    if loss not in (None, 'l2', 'linear'):
        ww = ww * residual_weights(loss, r)
    denom = float(np.sum(ww))
    if denom <= 0.:
        return np.inf
    cost = float(np.sum(ww * r ** 2) / denom)
    return cost if np.isfinite(cost) else np.inf


def nrmse_pair(q, target):
    """(linear nRMSE, log nRMSE) of a hydrograph against its target.

    Both are reported for every fit, always.  Linear nRMSE is what the canary
    matrix already measures; log nRMSE is what a multiplicative error structure
    actually cares about, and the two disagree exactly where the fit is being
    dragged by high flows.  Recording both is what lets that question be
    settled from a global run instead of argued from priors.
    """
    q = _as_1d(q)
    target = _as_1d(target)
    if q.size != target.size or q.size == 0:
        return np.nan, np.nan

    lin = np.nan
    good = np.isfinite(q) & np.isfinite(target)
    if np.any(good):
        denom = float(np.mean(target[good]))
        if np.isfinite(denom) and denom > 0.:
            lin = float(np.sqrt(np.mean((q[good] - target[good]) ** 2)) / denom)

    log = np.nan
    goodl = good & (q > 0.) & (target > 0.)
    if np.any(goodl):
        r = np.log(q[goodl]) - np.log(target[goodl])
        log = float(np.sqrt(np.mean(r ** 2)))

    return lin, log


def bound_mask(params, bounds, tol=BOUND_TOL):
    """Bitmask of which parameters ended up pinned against a bound.

    A parameter on its bound is not a fitted parameter, it is a censored one.
    Exporting it unmarked is how an unidentifiable reach ends up looking
    identical to a well-constrained one.
    """
    mask = 0
    for i, (value, (lo, hi)) in enumerate(zip(params, bounds)):
        if not np.isfinite(value):
            mask |= (1 << i)
            continue
        for edge in (lo, hi):
            if edge is None or not np.isfinite(edge):
                continue
            if abs(value - edge) <= tol * max(1.0, abs(edge)):
                mask |= (1 << i)
                break
    return mask


# ---------------------------------------------------------------------------
# observation quality
# ---------------------------------------------------------------------------

def observation_weights(obs, mode='none'):
    """Per-timestep weights from SWOT reach quality flags, or all ones.

    ``mode='none'`` is the default so that turning the new fitter on does not
    silently also change the weighting -- the two are separate questions and
    the canary matrix should be able to separate them.
    """
    nt = int(obs.get('nt', 0) or 0)
    if nt <= 0:
        return None
    if mode in (None, 'none', 'equal'):
        return None

    w = np.ones(nt, dtype=float)
    if mode == 'obs_quality':
        for key, penalty in (('reach_q', 0.5), ('xovr_cal_q', 0.5)):
            flag = obs.get(key)
            if flag is None:
                continue
            flag = _as_1d(flag)
            if flag.size != nt:
                continue
            # SWOT quality flags are 0 good / 1 suspect / 2 degraded / 3 bad.
            with np.errstate(invalid='ignore'):
                bad = np.where(np.isfinite(flag), np.clip(flag, 0., 3.), 0.)
            w *= penalty ** bad
    return w


def da_valid_fraction(obs):
    """Fraction of retained timesteps whose dA is finite."""
    dA = _as_1d(obs.get('dA', []))
    if dA.size == 0:
        return np.nan
    return float(np.mean(np.isfinite(dA)))


def is_identifiable(obs, target, min_obs=MIN_VALID_OBS):
    """Whether this reach carries enough signal to constrain a flow law.

    Low dA variance is an identifiability condition, not corrupt data: the
    reach is real, the geometry just did not move enough during the record to
    pin down A0.  The correct response is to say so and keep the prior, not to
    drop the reach and not to export a confidently wrong A0.
    """
    target = _as_1d(target)
    n_valid = int(np.sum(np.isfinite(target)))
    if n_valid < min_obs:
        return False, n_valid
    dA = _as_1d(obs.get('dA', []))
    if dA.size:
        finite = dA[np.isfinite(dA)]
        if finite.size >= 2:
            spread = float(np.nanmax(finite) - np.nanmin(finite))
            if not np.isfinite(spread) or spread <= 0.:
                return False, n_valid
    return True, n_valid


# ---------------------------------------------------------------------------
# search grids
# ---------------------------------------------------------------------------

def a0_grid(obs, prior_a0, a0_min, n_points=48):
    """Log-spaced A0 candidates, bounded by geometry rather than by guesswork.

    A0 is a cross-sectional area, so ``width * depth`` bounds it far better
    than an arbitrary numeric range does.  The grid spans roughly 0.05 m to
    100 m of mean depth above the minimum admissible area, log-spaced because
    A0 varies over orders of magnitude between a headwater and a trunk reach.

    The FLPE prior is appended as an extra candidate whenever it is admissible,
    so the grid can never do worse than the old single-start optimiser's
    starting point.
    """
    dA = _as_1d(obs.get('dA', []))
    w = _as_1d(obs.get('w', []))

    w_fin = w[np.isfinite(w) & (w > 0.)]
    w_med = float(np.median(w_fin)) if w_fin.size else 100.0

    dA_fin = dA[np.isfinite(dA)]
    dA_span = float(np.nanmax(dA_fin) - np.nanmin(dA_fin)) if dA_fin.size >= 2 else 0.

    lo_off = max(0.05 * w_med, 0.5)
    hi_off = max(100.0 * w_med, 20.0 * dA_span, 500.0)
    if hi_off <= lo_off:
        hi_off = lo_off * 100.0

    grid = a0_min + np.geomspace(lo_off, hi_off, int(n_points))

    if np.isfinite(prior_a0) and prior_a0 > a0_min:
        grid = np.append(grid, float(prior_a0))

    return np.unique(grid[np.isfinite(grid)])


def exponent_grid(prior_value, lo=-2.0, hi=2.0, n_points=17):
    """Candidates for MetroMan's ``x1`` / HiVDI's ``beta``.

    Deliberately narrower than the ``(-10, 10)`` optimiser bounds: those bounds
    exist to stop a runaway, not because a depth exponent of 8 is physical.
    The local refine that follows the grid still has the full bounds, so a
    genuinely extreme reach is not locked out -- it just has to earn it.
    """
    grid = np.linspace(lo, hi, int(n_points))
    if np.isfinite(prior_value) and lo <= prior_value <= hi:
        grid = np.append(grid, float(prior_value))
    return np.unique(grid[np.isfinite(grid)])


def momma_grid(obs, prior_B, prior_H, b_points=14, depth_points=10):
    """(B, H) candidates for MOMMA, parameterised by depth rather than by H.

    MOMMA's two parameters are a zero-flow stage and a bankfull stage, both in
    WSE units, and only their difference matters to the roughness term.
    Gridding ``(B, H)`` directly wastes most of the rectangle on
    ``H <= B``; gridding ``(B, H - B)`` does not.
    """
    h = _as_1d(obs.get('h', []))
    h_fin = h[np.isfinite(h)]
    if h_fin.size == 0:
        return np.empty((0, 2))

    h_min = float(np.min(h_fin))
    h_max = float(np.max(h_fin))
    b_max = h_min - 0.1
    h_span = max(h_max - h_min, 0.5)

    b_lo = max(0.1, b_max - max(5.0 * h_span, 10.0))
    if b_lo >= b_max:
        b_lo = b_max - 1.0
    b_values = np.linspace(b_lo, b_max, int(b_points))
    if np.isfinite(prior_B) and b_lo <= prior_B <= b_max:
        b_values = np.append(b_values, float(prior_B))
    b_values = np.unique(b_values)

    depth_lo = 0.2
    depth_hi = max(3.0 * h_span, 20.0)
    depths = np.geomspace(depth_lo, depth_hi, int(depth_points))

    pairs = []
    for b in b_values:
        for d in depths:
            H = b + d
            if H >= b_max + 0.1:
                pairs.append((b, H))
        if np.isfinite(prior_H) and prior_H >= b_max + 0.1 and prior_H > b:
            pairs.append((b, float(prior_H)))

    if not pairs:
        return np.empty((0, 2))
    return np.unique(np.asarray(pairs, dtype=float), axis=0)


# ---------------------------------------------------------------------------
# flow-law bases
#
# Each entry says how one algorithm's flow law separates into
# ``q = scale * basis(theta)``, and how to put a solved scale back into the
# parameter order that moi.Integrate's own ``*_flowlaw`` methods expect.
# ---------------------------------------------------------------------------

SCALED_ALGS = ('busboi', 'hivdi', 'metroman', 'sad', 'sic4dvar')
ALL_ALGS = SCALED_ALGS + ('momma',)


def _base_term(dA, w, S, a0):
    """Manning geometry term shared by every dA-driven flow law."""
    with np.errstate(all='ignore'):
        return (dA + a0) ** (5. / 3.) * w ** (-2. / 3.) * np.sqrt(S)


def _depth_term(dA, w, a0):
    with np.errstate(all='ignore'):
        return (dA + a0) / w


def momma_basis(B, H, obs, save):
    """Vectorised MOMMA flow law.

    Numerically identical to ``Integrate.momma_flowlaw`` -- including its NaN
    behaviour when the stage drops below ``B`` and its ``inf`` when the
    bankfull stage collapses onto the zero-flow stage -- but without the
    per-timestep Python loop, because the grid evaluates it a few hundred
    times per reach instead of a few dozen.  ``tests`` pins the two together.
    """
    h = _as_1d(obs['h'])
    w = _as_1d(obs['w'])
    S = _as_1d(obs['S'])

    if H <= B + 0.1:
        return np.full(h.shape, np.inf)

    n_b = 0.11 * save ** 0.18
    r = 2.
    with np.errstate(all='ignore'):
        log_factor = np.log10((H - B) / (h - B))
        n = n_b * np.where(h <= H, 1. + log_factor, 1. - log_factor)
        q = (((h - B) * (r / (1. + r))) ** (5. / 3.) * w * np.sqrt(S)) / n
    return q


def scale_bounds_from_param(kind, lo, hi):
    """Bounds on the scale ``s`` implied by the bounds on the raw parameter."""
    if kind == 'direct':
        return (lo, hi)
    # inverse: s = 1 / p, so the bounds swap and invert.
    s_lo = 0. if (hi is None or not np.isfinite(hi)) else 1. / hi
    s_hi = np.inf if (lo is None or lo <= 0.) else 1. / lo
    return (s_lo, s_hi)


def _scaled_candidates(alg, obs, priors, a0_min, grid_points, exponent_points):
    """Yield ``(basis, make_params)`` over the non-scale parameter grid.

    ``make_params(scale)`` returns the parameter tuple in the same order the
    corresponding ``Integrate.*_flowlaw`` expects, so the caller never has to
    know which slot the scale occupies.
    """
    dA = _as_1d(obs['dA'])
    w = _as_1d(obs['w'])
    S = _as_1d(obs['S'])

    a0_values = a0_grid(obs, priors.get('a0', np.nan), a0_min, grid_points)

    if alg in ('busboi', 'sad', 'sic4dvar'):
        # q = base / n  ->  scale = 1/n, one-dimensional search over A0.
        for a0 in a0_values:
            basis = _base_term(dA, w, S, a0)
            yield basis, (lambda s, a0=a0: (1. / s, a0))
        return

    if alg == 'metroman':
        # n = na * ((dA+A0)/w)**x1, q = base / n  ->  scale = 1/na.
        x_values = exponent_grid(priors.get('x1', np.nan), n_points=exponent_points)
        for a0 in a0_values:
            base = _base_term(dA, w, S, a0)
            depth = _depth_term(dA, w, a0)
            for x1 in x_values:
                with np.errstate(all='ignore'):
                    basis = base * depth ** (-x1)
                yield basis, (lambda s, x1=x1, a0=a0: (1. / s, x1, a0))
        return

    if alg == 'hivdi':
        # n_inv = alpha * ((dA+A0)/w)**beta, q = base * n_inv -> scale = alpha.
        b_values = exponent_grid(priors.get('beta', np.nan), n_points=exponent_points)
        for a0 in a0_values:
            base = _base_term(dA, w, S, a0)
            depth = _depth_term(dA, w, a0)
            for beta in b_values:
                with np.errstate(all='ignore'):
                    basis = base * depth ** beta
                yield basis, (lambda s, beta=beta, a0=a0: (s, beta, a0))
        return

    raise ValueError('no separable form for algorithm %r' % (alg,))


def fit_scaled_law(alg, obs, target, priors, param_bounds, a0_min,
                   space='linear', loss='l2', weights=None,
                   grid_points=48, exponent_points=17):
    """Exhaustive bounded search for a flow law that is linear in one parameter.

    Returns ``(params, cost, n_evals)`` with ``params`` in the flow law's own
    order, or ``(None, inf, n_evals)`` when nothing on the grid produced a
    finite cost.
    """
    scale_index = 0  # every separable law here carries its scale first
    kind = 'direct' if alg == 'hivdi' else 'inverse'
    s_lo, s_hi = scale_bounds_from_param(kind, *param_bounds[scale_index])

    best_params = None
    best_cost = np.inf
    n_evals = 0

    for basis, make_params in _scaled_candidates(
        alg, obs, priors, a0_min, grid_points, exponent_points
    ):
        n_evals += 1
        s = solve_scale(basis, target, space=space, weights=weights, loss=loss)
        if not np.isfinite(s):
            continue
        s = float(np.clip(s, s_lo, s_hi))
        if s <= 0.:
            continue
        cost = series_cost(s * basis, target, space=space,
                           weights=weights, loss=loss)
        if cost < best_cost:
            best_cost = cost
            best_params = make_params(s)

    return best_params, best_cost, n_evals


def fit_momma_law(obs, target, priors, save, param_bounds,
                  space='linear', loss='l2', weights=None,
                  b_points=14, depth_points=10):
    """Two-dimensional grid for MOMMA, which has no scale parameter.

    ``n_b = 0.11 * Save**0.18`` is pinned by slope, so there is nothing to
    solve in closed form and the grid has to carry the whole search.  That is
    also why MOMMA cannot be rescued the way the dA-driven laws can: its level
    is fixed by geometry, not fitted.
    """
    grid = momma_grid(obs, priors.get('B', np.nan), priors.get('H', np.nan),
                      b_points=b_points, depth_points=depth_points)

    (b_lo, b_hi), (h_lo, _h_hi) = param_bounds
    best_params = None
    best_cost = np.inf
    n_evals = 0

    for B, H in grid:
        if B < b_lo or B > b_hi or H < h_lo:
            continue
        n_evals += 1
        q = momma_basis(B, H, obs, save)
        cost = series_cost(q, target, space=space, weights=weights, loss=loss)
        if cost < best_cost:
            best_cost = cost
            best_params = (float(B), float(H))

    return best_params, best_cost, n_evals


# ---------------------------------------------------------------------------
# local refine
# ---------------------------------------------------------------------------

def refine_local(flowlaw, obs, target, params, param_bounds, extra=(),
                 space='linear', loss='l2', weights=None, penalty=None):
    """Polish a grid optimum with a bounded local solve.

    The grid answers "which basin of attraction"; this answers "where in it".
    Started from the grid optimum, so it is still deterministic, and the result
    is kept only if it actually improves the cost -- a local solver that wanders
    off is discarded rather than trusted, which is precisely what the previous
    single-start code failed to do.

    SciPy is imported here rather than at module scope so the grid machinery
    stays importable (and unit-testable) with numpy alone.
    """
    try:
        from scipy import optimize
    except Exception:
        return params, series_cost(
            _safe_flowlaw(flowlaw, params, obs, extra), target,
            space=space, weights=weights, loss=loss)

    def cost(theta):
        q = _safe_flowlaw(flowlaw, theta, obs, extra)
        if q is None:
            return 1.0e12
        value = series_cost(q, target, space=space, weights=weights, loss=loss)
        if not np.isfinite(value):
            return 1.0e12
        if penalty is not None:
            value *= penalty(theta)
        return value

    start_cost = cost(np.asarray(params, dtype=float))
    try:
        res = optimize.minimize(
            cost, np.asarray(params, dtype=float), bounds=param_bounds,
            method='L-BFGS-B',
        )
    except Exception:
        return params, start_cost

    if not getattr(res, 'success', False):
        return params, start_cost
    refined = tuple(float(v) for v in res.x)
    refined_cost = cost(np.asarray(refined, dtype=float))
    if np.isfinite(refined_cost) and refined_cost < start_cost:
        return refined, refined_cost
    return params, start_cost


def as_row(q, nt, fill=np.nan):
    """Coerce a hydrograph to the (1, nt) storage shape MOI uses everywhere.

    Every ``Integrate.*_flowlaw`` ends with ``np.reshape(q, (1, nt))``, and the
    rest of the pipeline quietly depends on that: ``Output.write_output`` calls
    ``np.insert(q, iInsert, fillvalue, 1)`` to put the dropped timesteps back,
    which raises ``AxisError`` on a 1-D array and takes the whole basin's output
    with it.  The contract was never written down anywhere, so it was easy to
    break by raveling a hydrograph for a cost function and storing the result.

    This is that contract, in one place.  It accepts a scalar, a 1-D series or a
    (1, nt) row, and pads or truncates to ``nt`` rather than raising -- a
    malformed hydrograph for one algorithm must not cost the other five their
    output file.  Callers that care whether a repair happened can compare
    ``np.size(q)`` against ``nt`` before calling.
    """
    # Keep zero valid timesteps as (1, 0).  Output later re-inserts every
    # dropped timestep; forcing a synthetic value here would make that final
    # record one sample too long.
    nt = max(int(nt), 0)
    out = np.full((1, nt), fill, dtype=float)
    if q is None:
        return out
    try:
        arr = np.asarray(q, dtype=float).ravel()
    except (TypeError, ValueError):
        return out
    k = min(nt, arr.size)
    if k:
        out[0, :k] = arr[:k]
    return out


def _safe_flowlaw(flowlaw, params, obs, extra=()):
    try:
        q = np.asarray(flowlaw(params, obs, *extra), dtype=float).ravel()
    except Exception:
        return None
    return q


# ---------------------------------------------------------------------------
# top-level entry point
# ---------------------------------------------------------------------------

def fit_flow_law(alg, obs, target, priors, param_bounds, flowlaw,
                 fallback_params, a0_min=None, save=None, extra=(),
                 space='linear', loss='l2', weight_mode='none',
                 grid_points=48, exponent_points=17, refine=True,
                 penalty=None, min_obs=MIN_VALID_OBS):
    """Fit one reach's flow-law parameters to the integrator hydrograph.

    Always returns a ``FitOutcome`` -- never raises, never returns ``None``.
    A reach that cannot be fitted comes back carrying ``fallback_params`` and a
    ``status`` that says why, because a global run needs every reach to produce
    *something* and needs that something to be honestly labelled.  Silently
    exporting a stalled optimiser's output is the failure mode this replaces.
    """
    target = _as_1d(target)
    weights = observation_weights(obs, weight_mode)

    identifiable, n_valid = is_identifiable(obs, target, min_obs=min_obs)
    if not identifiable:
        return _outcome_from_params(
            fallback_params, FIT_UNIDENTIFIABLE, flowlaw, obs, target,
            param_bounds, extra, n_valid, 0, np.inf, 'varpro')

    try:
        if alg == 'momma':
            params, cost, n_evals = fit_momma_law(
                obs, target, priors, save, param_bounds,
                space=space, loss=loss, weights=weights)
        else:
            params, cost, n_evals = fit_scaled_law(
                alg, obs, target, priors, param_bounds, a0_min,
                space=space, loss=loss, weights=weights,
                grid_points=grid_points, exponent_points=exponent_points)
    except Exception:
        return _outcome_from_params(
            fallback_params, FIT_FAILED, flowlaw, obs, target,
            param_bounds, extra, n_valid, 0, np.inf, 'varpro')

    if params is None or not np.isfinite(cost):
        return _outcome_from_params(
            fallback_params, FIT_FAILED, flowlaw, obs, target,
            param_bounds, extra, n_valid, n_evals, np.inf, 'varpro')

    if refine:
        params, cost = refine_local(
            flowlaw, obs, target, params, param_bounds, extra=extra,
            space=space, loss=loss, weights=weights, penalty=penalty)

    return _outcome_from_params(
        params, FIT_GOOD, flowlaw, obs, target, param_bounds, extra,
        n_valid, n_evals, cost, 'varpro')


def _outcome_from_params(params, status, flowlaw, obs, target, param_bounds,
                         extra, n_valid, n_evals, cost, method):
    params = tuple(
        float(p) if p is not None and np.isfinite(p) else np.nan
        for p in params
    )
    lin = log = np.nan
    if all(np.isfinite(p) for p in params):
        q = _safe_flowlaw(flowlaw, params, obs, extra)
        if q is not None and q.size == target.size:
            lin, log = nrmse_pair(q, target)
    return FitOutcome(
        params=params, status=status, nrmse_lin=lin, nrmse_log=log,
        param_at_bound=bound_mask(params, param_bounds),
        n_valid_obs=n_valid, n_grid_evals=n_evals, objective=cost,
        method=method,
    )


# ---------------------------------------------------------------------------
# hydrograph rescaling
# ---------------------------------------------------------------------------

def powerlaw_rescale(q, qbar_target, q33_target, b_lo=0.2, b_hi=3.0, iters=64):
    """Solve ``q' = a * q**b`` matching both the integrated mean and q33.

    The current rescale is a single multiplicative level shift: it matches the
    mean exactly and leaves the shape untouched, which also means it cannot
    correct an FLPE hydrograph whose *spread* disagrees with the integrator.
    Two constraints buy one shape parameter.

    Because ``q**b`` is monotone for ``b > 0``, the 33rd percentile of the
    transformed series is the transform of the 33rd percentile, so the whole
    system reduces to a scalar root in ``b`` and a bisection is exact enough.

    Returns ``(a, b)``, or ``None`` when the targets do not admit a solution in
    range -- in which case the caller should fall back to the mean rescale
    rather than force a fit.
    """
    q = _as_1d(q)
    good = np.isfinite(q) & (q > 0.)
    if np.count_nonzero(good) < 3:
        return None
    qq = q[good]
    if not (np.isfinite(qbar_target) and qbar_target > 0.):
        return None
    if not (np.isfinite(q33_target) and q33_target > 0.):
        return None

    q33_src = float(np.quantile(qq, 0.33))
    if not np.isfinite(q33_src) or q33_src <= 0.:
        return None

    def residual(b):
        with np.errstate(all='ignore'):
            powered = qq ** b
            mean_p = float(np.mean(powered))
            if not np.isfinite(mean_p) or mean_p <= 0.:
                return np.nan
            a = qbar_target / mean_p
            return a * (q33_src ** b) - q33_target

    f_lo = residual(b_lo)
    f_hi = residual(b_hi)
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0.:
        return None

    for _ in range(iters):
        b_mid = 0.5 * (b_lo + b_hi)
        f_mid = residual(b_mid)
        if not np.isfinite(f_mid):
            return None
        if f_lo * f_mid <= 0.:
            b_hi = b_mid
            f_hi = f_mid
        else:
            b_lo = b_mid
            f_lo = f_mid

    b = 0.5 * (b_lo + b_hi)
    with np.errstate(all='ignore'):
        mean_p = float(np.mean(qq ** b))
    if not np.isfinite(mean_p) or mean_p <= 0.:
        return None
    a = qbar_target / mean_p
    if not np.isfinite(a) or a <= 0.:
        return None
    return float(a), float(b)

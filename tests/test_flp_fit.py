"""Tests for the deterministic flow-law fitter.

The claims worth pinning down are:

  * the closed-form scale really is the optimum, not an approximation
  * the search is deterministic -- identical inputs give identical parameters
  * the vectorised MOMMA law is numerically identical to the one it replaces
  * a reach that cannot be fitted comes back labelled, not silently wrong
"""

import numpy as np
import pytest

from moi import flp_fit
from moi.Integrate import Integrate


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def synthetic_obs(nt=40, seed=7):
    rng = np.random.default_rng(seed)
    dA = np.sort(rng.normal(0., 120., nt))
    w = 200. + rng.normal(0., 10., nt)
    S = np.full(nt, 1.0e-4) * rng.uniform(0.9, 1.1, nt)
    h = 10. + (dA - dA.min()) / 200.
    return {'nt': nt, 'dA': dA, 'w': w, 'S': S, 'h': h, 't': np.arange(nt)}


def bam_q(params, obs):
    n, a0 = params
    return (obs['dA'] + a0) ** (5. / 3.) * obs['w'] ** (-2. / 3.) * np.sqrt(obs['S']) / n


def metroman_q(params, obs):
    na, x1, a0 = params
    base = (obs['dA'] + a0) ** (5. / 3.) * obs['w'] ** (-2. / 3.) * np.sqrt(obs['S'])
    return base / (na * ((obs['dA'] + a0) / obs['w']) ** x1)


def hivdi_q(params, obs):
    alpha, beta, a0 = params
    base = (obs['dA'] + a0) ** (5. / 3.) * obs['w'] ** (-2. / 3.) * np.sqrt(obs['S'])
    return base * alpha * ((obs['dA'] + a0) / obs['w']) ** beta


# ---------------------------------------------------------------------------
# closed-form scale
# ---------------------------------------------------------------------------

def test_linear_scale_matches_numerical_optimum():
    rng = np.random.default_rng(1)
    basis = rng.uniform(1., 50., 60)
    target = 4.2 * basis + rng.normal(0., 2., 60)

    analytic = flp_fit.solve_scale(basis, target, space='linear')

    candidates = np.linspace(analytic * 0.5, analytic * 1.5, 20001)
    costs = [flp_fit.series_cost(s * basis, target) for s in candidates]
    numerical = candidates[int(np.argmin(costs))]

    assert analytic == pytest.approx(numerical, rel=1e-3)


def test_log_scale_matches_numerical_optimum():
    rng = np.random.default_rng(2)
    basis = rng.uniform(1., 50., 60)
    target = 4.2 * basis * rng.lognormal(0., 0.2, 60)

    analytic = flp_fit.solve_scale(basis, target, space='log')

    candidates = np.linspace(analytic * 0.5, analytic * 1.5, 20001)
    costs = [flp_fit.series_cost(s * basis, target, space='log') for s in candidates]
    numerical = candidates[int(np.argmin(costs))]

    assert analytic == pytest.approx(numerical, rel=1e-3)


def test_scale_returns_nan_rather_than_guessing():
    assert np.isnan(flp_fit.solve_scale(np.full(5, np.nan), np.ones(5)))
    assert np.isnan(flp_fit.solve_scale(np.ones(5), np.full(5, np.nan)))
    # log space with no positive pairs has no solution, and must say so.
    assert np.isnan(flp_fit.solve_scale(np.ones(5), -np.ones(5), space='log'))


def test_robust_loss_resists_a_single_bad_timestep():
    basis = np.linspace(1., 20., 30)
    target = 3.0 * basis.copy()
    target[10] = 500.0  # one dA jump

    plain = flp_fit.solve_scale(basis, target, loss='l2')
    robust = flp_fit.solve_scale(basis, target, loss='soft_l1')

    assert abs(robust - 3.0) < abs(plain - 3.0)


# ---------------------------------------------------------------------------
# parameter recovery and determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('alg,truth,flowlaw', [
    ('busboi', (0.035, 900.), bam_q),
    ('sad', (0.028, 1500.), bam_q),
    ('sic4dvar', (0.041, 700.), bam_q),
    ('metroman', (0.030, -0.25, 1100.), metroman_q),
    ('hivdi', (28.0, -0.4, 1300.), hivdi_q),
])
def test_recovers_known_parameters(alg, truth, flowlaw):
    obs = synthetic_obs()
    target = flowlaw(truth, obs)
    a0_min = -float(np.min(obs['dA'])) + 1.
    n_hi = 10. if alg == 'sic4dvar' else np.inf
    if len(truth) == 2:
        bounds = ((0.001, n_hi), (a0_min, np.inf))
    else:
        bounds = ((0.001, np.inf), (-1e1, 1e1), (a0_min, np.inf))

    params, cost, n_evals = flp_fit.fit_scaled_law(
        alg, obs, target, priors={}, param_bounds=bounds, a0_min=a0_min)

    assert params is not None
    assert n_evals > 0
    lin, _log = flp_fit.nrmse_pair(flowlaw(params, obs), target)
    assert lin < 0.05


def test_search_is_deterministic():
    obs = synthetic_obs()
    target = bam_q((0.035, 900.), obs)
    a0_min = -float(np.min(obs['dA'])) + 1.
    bounds = ((0.001, np.inf), (a0_min, np.inf))

    first = flp_fit.fit_scaled_law('busboi', obs, target, {}, bounds, a0_min)[0]
    second = flp_fit.fit_scaled_law('busboi', obs, target, {}, bounds, a0_min)[0]

    assert first == second


def test_grid_beats_a_badly_started_local_solve():
    """The failure mode this whole change exists to remove.

    A single-start bounded solve started far from the answer converges to a
    wrong optimum *and reports success while doing it* -- so nothing downstream
    can tell the difference.  Four of the six algorithms used to accept that
    result unconditionally.  The grid does not depend on where it starts.
    """
    optimize = pytest.importorskip('scipy.optimize')
    obs = synthetic_obs()
    truth = (0.035, 900.)
    target = bam_q(truth, obs)
    a0_min = -float(np.min(obs['dA'])) + 1.
    bounds = ((0.001, np.inf), (a0_min, np.inf))

    outcome = flp_fit.fit_flow_law(
        alg='busboi', obs=obs, target=target, priors={},
        param_bounds=bounds, flowlaw=bam_q,
        fallback_params=(0.03, a0_min + 10.), a0_min=a0_min)

    bad_start = (0.001, 1.0e6)
    res = optimize.minimize(
        lambda p: flp_fit.series_cost(bam_q(p, obs), target),
        bad_start, bounds=bounds)
    local_nrmse, _ = flp_fit.nrmse_pair(bam_q(res.x, obs), target)

    # The stalled solve claims to have succeeded -- that is the problem.
    assert res.success
    assert local_nrmse > 0.1

    assert outcome.status == flp_fit.FIT_GOOD
    assert outcome.nrmse_lin < 1e-3
    assert outcome.nrmse_lin < local_nrmse


def test_grid_alone_is_already_close_and_refine_finishes_the_job():
    obs = synthetic_obs()
    target = bam_q((0.035, 900.), obs)
    a0_min = -float(np.min(obs['dA'])) + 1.
    bounds = ((0.001, np.inf), (a0_min, np.inf))

    grid_params, _cost, _n = flp_fit.fit_scaled_law(
        'busboi', obs, target, {}, bounds, a0_min)
    grid_nrmse, _ = flp_fit.nrmse_pair(bam_q(grid_params, obs), target)

    refined = flp_fit.fit_flow_law(
        alg='busboi', obs=obs, target=target, priors={},
        param_bounds=bounds, flowlaw=bam_q,
        fallback_params=(0.03, a0_min + 10.), a0_min=a0_min)

    assert grid_nrmse < 0.02          # the grid alone is already usable
    assert refined.nrmse_lin <= grid_nrmse


# ---------------------------------------------------------------------------
# MOMMA
# ---------------------------------------------------------------------------

def test_vectorised_momma_matches_the_looped_flow_law():
    obs = synthetic_obs()
    integrator = Integrate.__new__(Integrate)
    save = 2.0e-4

    for B, H in ((8.0, 14.0), (9.5, 12.0), (5.0, 30.0)):
        looped = np.asarray(
            integrator.momma_flowlaw((B, H), obs, save), dtype=float).ravel()
        vector = flp_fit.momma_basis(B, H, obs, save)
        np.testing.assert_allclose(vector, looped, rtol=1e-12, equal_nan=True)


def test_momma_degenerate_bankfull_matches_legacy_infinity():
    obs = synthetic_obs()
    # H <= B + 0.1 is the case momma_flowlaw answers with inf.
    assert np.all(np.isinf(flp_fit.momma_basis(10.0, 10.05, obs, 2.0e-4)))


def test_momma_grid_recovers_its_own_parameters():
    obs = synthetic_obs()
    integrator = Integrate.__new__(Integrate)
    save = 2.0e-4
    truth = (8.0, 14.0)
    target = np.asarray(
        integrator.momma_flowlaw(truth, obs, save), dtype=float).ravel()

    bounds = flp_fit.momma_param_bounds(obs['h'])
    params, cost, n_evals = flp_fit.fit_momma_law(
        obs, target, {}, save, bounds)

    assert params is not None and n_evals > 0
    lin, _log = flp_fit.nrmse_pair(
        flp_fit.momma_basis(params[0], params[1], obs, save), target)
    assert lin < 0.25


def _shifted_obs(shift, nt=40, seed=7):
    obs = synthetic_obs(nt=nt, seed=seed)
    obs['h'] = obs['h'] + shift
    return obs


def _legacy_momma_b_search_range(h):
    """The grid range before negative stages were allowed, for regression."""
    h_min, h_max = float(np.min(h)), float(np.max(h))
    b_max = h_min - 0.1
    h_span = max(h_max - h_min, 0.5)
    b_lo = max(0.1, b_max - max(5.0 * h_span, 10.0))
    if b_lo >= b_max:
        b_lo = b_max - 1.0
    return b_lo, b_max


def test_momma_bounds_admit_negative_stages():
    # Noatak 81340100031-like: every WSE below the geoid.
    h = np.linspace(-1.3684, -0.3684, 30)
    (b_lo, b_hi), (h_lo, h_hi) = flp_fit.momma_param_bounds(h)

    assert b_lo == -np.inf
    assert b_hi == pytest.approx(-1.4684)
    assert h_lo == pytest.approx(-1.3684)
    assert h_hi == np.inf
    assert b_lo < b_hi < h_lo

    s_lo, s_hi = flp_fit.momma_b_search_range(h)
    assert s_hi == b_hi
    assert s_lo == pytest.approx(b_hi - 10.0)


@pytest.mark.parametrize('h_min', [-50.0, -1.3684, 0.05, 0.15, 1.0, 9.0, 12.0, 500.0])
def test_momma_grid_is_nonempty_and_inside_the_bounds(h_min):
    obs = synthetic_obs()
    obs['h'] = obs['h'] - float(np.min(obs['h'])) + h_min
    bounds = flp_fit.momma_param_bounds(obs['h'])
    (b_lo, b_hi), (h_lo, _h_hi) = bounds

    grid = flp_fit.momma_grid(obs, np.nan, np.nan)

    assert grid.shape[0] > 0
    assert np.all(grid[:, 0] >= b_lo)
    assert np.all(grid[:, 0] <= b_hi)
    assert np.all(grid[:, 0] < float(np.min(obs['h'])))
    assert np.all(grid[:, 1] >= h_lo)
    assert np.all(grid[:, 1] > grid[:, 0])

    _params, _cost, n_evals = flp_fit.fit_momma_law(
        obs, np.ones(obs['nt']), {}, 2.0e-4, bounds)
    assert n_evals == grid.shape[0]


@pytest.mark.parametrize('margin', [0.001, 0.01, 5.0, 500.0])
def test_momma_search_range_unchanged_where_the_old_floor_never_bound(margin):
    obs = synthetic_obs()
    h = obs['h'] - float(np.min(obs['h']))
    span = max(float(np.max(h)), 0.5)
    # The old floor was inactive exactly when min(h) - 0.1 - depth >= 0.1.
    h = h + 0.2 + max(5.0 * span, 10.0) + margin
    assert flp_fit.momma_b_search_range(h) == _legacy_momma_b_search_range(h)


def test_momma_search_range_widens_where_the_old_floor_bound():
    for h_min in (-1.3684, 0.05, 1.0, 9.0):
        obs = synthetic_obs()
        h = obs['h'] - float(np.min(obs['h'])) + h_min
        new_lo, new_hi = flp_fit.momma_b_search_range(h)
        old_lo, old_hi = _legacy_momma_b_search_range(h)
        assert new_hi == old_hi
        assert new_lo < old_lo


def test_momma_bounds_ignore_non_finite_stages():
    h = np.array([np.nan, -2.0, -1.0, np.inf])
    (b_lo, b_hi), (h_lo, _h_hi) = flp_fit.momma_param_bounds(h)
    assert b_hi == pytest.approx(-2.1)
    assert h_lo == pytest.approx(-2.0)

    (b_lo, b_hi), (h_lo, _h_hi) = flp_fit.momma_param_bounds([np.nan, np.nan])
    assert np.isnan(b_hi) and np.isnan(h_lo)
    assert np.all(np.isnan(flp_fit.momma_b_search_range([np.nan])))


def test_momma_fit_recovers_parameters_below_the_geoid():
    obs = _shifted_obs(-12.0)           # h spans roughly -2 .. -0.5 m
    integrator = Integrate.__new__(Integrate)
    save = 2.0e-4
    truth = (-5.0, 1.0)
    assert truth[0] < np.min(obs['h'])
    target = np.asarray(
        integrator.momma_flowlaw(truth, obs, save), dtype=float).ravel()

    outcome = flp_fit.fit_flow_law(
        alg='momma', obs=obs, target=target, priors={},
        param_bounds=flp_fit.momma_param_bounds(obs['h']),
        flowlaw=integrator.momma_flowlaw, fallback_params=(np.nan, np.nan),
        save=save, extra=(save,), penalty=Integrate.momma_shape_penalty)

    assert outcome.status == flp_fit.FIT_GOOD
    assert outcome.n_grid_evals > 0
    B, H = outcome.params
    assert B < np.min(obs['h'])
    assert H > B
    assert outcome.nrmse_lin < 0.05


@pytest.mark.parametrize('shift', [-12.0, -10.5, -9.5, -5.0, 100.0])
def test_momma_fit_is_invariant_to_the_vertical_datum(shift):
    # The flow law depends on stage only through h - B and H - B, so moving
    # the datum must move B and H by the same amount and change nothing else.
    # The reference reach sits high enough that the old B >= 0.1 floor never
    # applied to it; the shifts cover negative, near-zero and low stages.
    integrator = Integrate.__new__(Integrate)
    save = 2.0e-4
    truth = (7.0, 13.0)

    def fit(offset):
        obs = _shifted_obs(offset)
        target = np.asarray(integrator.momma_flowlaw(
            (truth[0] + offset, truth[1] + offset), obs, save),
            dtype=float).ravel()
        return flp_fit.fit_flow_law(
            alg='momma', obs=obs, target=target,
            priors={'B': 8.0 + offset, 'H': 14.0 + offset},
            param_bounds=flp_fit.momma_param_bounds(obs['h']),
            flowlaw=integrator.momma_flowlaw,
            fallback_params=(8.0 + offset, 14.0 + offset),
            save=save, extra=(save,), penalty=Integrate.momma_shape_penalty)

    reference = fit(0.0)
    shifted = fit(shift)

    assert reference.status == shifted.status == flp_fit.FIT_GOOD
    assert shifted.n_grid_evals == reference.n_grid_evals
    assert shifted.params[0] - shift == pytest.approx(reference.params[0], abs=1e-4)
    assert shifted.params[1] - shift == pytest.approx(reference.params[1], abs=1e-4)
    assert shifted.nrmse_lin == pytest.approx(reference.nrmse_lin, rel=1e-4, abs=1e-8)


# ---------------------------------------------------------------------------
# identifiability and failure handling
# ---------------------------------------------------------------------------

def test_constant_da_is_unidentifiable_not_fitted():
    obs = synthetic_obs()
    obs['dA'] = np.zeros(obs['nt'])
    ok, _n = flp_fit.is_identifiable(obs, np.ones(obs['nt']))
    assert not ok


def test_too_few_observations_is_refused():
    obs = synthetic_obs(nt=4)
    ok, n_valid = flp_fit.is_identifiable(obs, np.ones(4))
    assert not ok and n_valid == 4


def test_fit_flow_law_never_raises_and_always_labels():
    obs = synthetic_obs(nt=4)  # too short to fit
    fallback = (0.03, 500.)
    outcome = flp_fit.fit_flow_law(
        alg='busboi', obs=obs, target=np.ones(4), priors={},
        param_bounds=((0.001, np.inf), (1., np.inf)), flowlaw=bam_q,
        fallback_params=fallback, a0_min=1.)

    assert outcome.status == flp_fit.FIT_UNIDENTIFIABLE
    assert outcome.params == fallback


def test_fit_flow_law_survives_a_flow_law_that_explodes():
    obs = synthetic_obs()

    def exploding(params, obs, *extra):
        raise ZeroDivisionError('boom')

    outcome = flp_fit.fit_flow_law(
        alg='busboi', obs=obs, target=np.ones(obs['nt']), priors={},
        param_bounds=((0.001, np.inf), (1., np.inf)), flowlaw=exploding,
        fallback_params=(0.03, 500.), a0_min=1., refine=False)

    assert outcome.status in (flp_fit.FIT_GOOD, flp_fit.FIT_FAILED)
    assert np.isnan(outcome.nrmse_lin)


def test_bound_mask_flags_pinned_parameters():
    bounds = ((0.001, 10.), (5., np.inf))
    assert flp_fit.bound_mask((0.001, 100.), bounds) == 0b01
    assert flp_fit.bound_mask((10., 5.), bounds) == 0b11
    assert flp_fit.bound_mask((0.5, 100.), bounds) == 0b00
    assert flp_fit.bound_mask((np.nan, 100.), bounds) == 0b01


# ---------------------------------------------------------------------------
# rescaling
# ---------------------------------------------------------------------------

def test_powerlaw_rescale_matches_both_targets():
    rng = np.random.default_rng(11)
    q = rng.lognormal(2.0, 0.6, 200)
    qbar_target = 40.0
    q33_target = 24.0

    solved = flp_fit.powerlaw_rescale(q, qbar_target, q33_target)
    assert solved is not None
    a, b = solved
    q_new = a * q ** b

    assert float(np.mean(q_new)) == pytest.approx(qbar_target, rel=1e-6)
    assert float(np.quantile(q_new, 0.33)) == pytest.approx(q33_target, rel=1e-5)


def test_powerlaw_rescale_refuses_impossible_targets():
    q = np.linspace(1., 10., 50)
    # q33 above the mean cannot be produced by a monotone power transform here.
    assert flp_fit.powerlaw_rescale(q, 10.0, 1e6) is None
    assert flp_fit.powerlaw_rescale(q, np.nan, 5.0) is None
    assert flp_fit.powerlaw_rescale(np.array([np.nan, np.nan]), 5.0, 3.0) is None


def test_mean_rescale_identity_is_exact():
    """The property the canary run measured at 1.4e-16: mean(q_out) == target."""
    rng = np.random.default_rng(3)
    q = rng.lognormal(1.0, 0.5, 80)
    target = 123.456
    factor = target / float(np.mean(q))
    assert float(np.mean(q * factor)) == pytest.approx(target, rel=1e-12)


# ---------------------------------------------------------------------------
# observation weighting
# ---------------------------------------------------------------------------

def test_quality_weighting_downweights_flagged_timesteps():
    obs = synthetic_obs(nt=6)
    obs['reach_q'] = np.array([0., 0., 1., 2., 3., 0.])
    obs['xovr_cal_q'] = np.zeros(6)

    assert flp_fit.observation_weights(obs, 'none') is None
    w = flp_fit.observation_weights(obs, 'obs_quality')
    assert w[0] == pytest.approx(1.0)
    assert w[2] == pytest.approx(0.5)
    assert w[4] == pytest.approx(0.125)
    assert np.all(np.diff(w[:5]) <= 0)


def test_da_valid_fraction():
    obs = {'dA': np.array([1., np.nan, 3., 4.])}
    assert flp_fit.da_valid_fraction(obs) == pytest.approx(0.75)

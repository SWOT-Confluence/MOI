"""End-to-end behaviour of the rewritten compute_FLPs().

Six near-identical loops became one, so the thing worth testing is that the one
loop still does what all six did -- for every algorithm, including MOMMA, which
is the odd one out -- and that it now also records why each fit turned out the
way it did.

Built on a hand-made Integrate instance rather than real basin data: the FLP
refit only needs geometry, a hydrograph and an integrated mean, and constructing
those directly keeps the test independent of SoS, SWORD and netCDF.
"""

import numpy as np
import pytest

from moi import flp_fit
from moi.Integrate import Integrate

ALGS = ('busboi', 'hivdi', 'metroman', 'momma', 'sad', 'sic4dvar')


def make_obs(nt=40, seed=5, constant_da=False):
    rng = np.random.default_rng(seed)
    dA = np.zeros(nt) if constant_da else np.sort(rng.normal(0., 150., nt))
    return {
        'nt': nt,
        'dA': dA,
        'w': 180. + rng.normal(0., 8., nt),
        'S': np.full(nt, 1.2e-4),
        'h': 12. + (dA - dA.min()) / 250. + rng.uniform(0., 0.05, nt),
        't': np.arange(nt, dtype=float),
        'iDelete': np.where(np.zeros(nt, dtype=bool)),
        'reach_q': np.zeros(nt),
        'xovr_cal_q': np.zeros(nt),
    }


def make_integrator(reach='74100', nt=40, qbar=250., q33=180.,
                    constant_da=False, params=None):
    obs = make_obs(nt=nt, constant_da=constant_da)
    rng = np.random.default_rng(9)
    # A plausible FLPE hydrograph: right shape, wrong level.  That is the case
    # the rescale path exists for.
    flpe_q = 40. + 30. * (obs['dA'] - obs['dA'].min()) / 100. + rng.uniform(0., 3., nt)

    priors = {
        'busboi': {'n': 0.033, 'a0': 800.},
        'hivdi': {'alpha': 30., 'beta': -0.3, 'a0': 800.},
        'metroman': {'na': 0.032, 'x1': -0.3, 'a0': 800.},
        'momma': {'B': 9.0, 'H': 15.0, 'Save': 1.2e-4},
        'sad': {'n': 0.031, 'a0': 800.},
        'sic4dvar': {'n': 0.034, 'a0': 800.},
    }

    alg_dict = {}
    for alg in ALGS:
        entry = dict(priors[alg])
        entry.update({
            's1-flpe-exists': True,
            'q': flpe_q.copy(),
            'qbar': float(np.mean(flpe_q)),
            'qbar_source': 'FLPE',
            'q33_source': 'FLPE',
            'integrator': {'qbar': qbar, 'q33': q33, 'sbQ_rel': 0.2},
        })
        alg_dict[alg] = {reach: entry}

    integrator = Integrate.__new__(Integrate)
    integrator.alg_dict = alg_dict
    integrator.obs_dict = {reach: obs}
    integrator.basin_dict = {'reach_ids': [reach], 'reach_ids_all': [reach]}
    integrator.params_dict = {
        'Integrator_Hydrograph_Method': 'rescale',
        'FLP_Fit_Method': 'series',
        'FLP_Optimizer': 'varpro',
    }
    if params:
        integrator.params_dict.update(params)
    integrator.VerboseFlag = False
    integrator.flp_fit_report = {}
    integrator.topology_report = {}
    integrator.unconverged_solves = []
    return integrator, reach


def run(integrator):
    integrator.calc_integrator_hydrographs()
    integrator.compute_FLPs()


# ---------------------------------------------------------------------------

def test_every_algorithm_produces_a_labelled_fit():
    integrator, reach = make_integrator()
    run(integrator)

    for alg in ALGS:
        integ = integrator.alg_dict[alg][reach]['integrator']
        assert integ['flp_fit_status'] in flp_fit.FIT_STATUS_CODE, alg
        assert integ['flp_fit_method'] == 'varpro', alg
        assert np.shape(integ['q']) == (1, 40), alg
        # Provenance that used to be tracked and never written.
        assert integ['qbar_source'] == 'FLPE', alg
        assert integ['q33_source'] == 'FLPE', alg
        assert integ['rescale_method'] == 'mean', alg
        assert np.isfinite(integ['rescale_factor']), alg


def test_parameters_land_under_the_expected_names():
    integrator, reach = make_integrator()
    run(integrator)

    for alg, keys in Integrate.FLP_PARAM_KEYS.items():
        integ = integrator.alg_dict[alg][reach]['integrator']
        for key in keys:
            assert key in integ, (alg, key)
            assert np.isfinite(integ[key]), (alg, key)
    assert np.isfinite(integrator.alg_dict['momma'][reach]['integrator']['Save'])


def test_rescaled_hydrograph_hits_the_integrated_mean():
    integrator, reach = make_integrator(qbar=250.)
    run(integrator)

    for alg in ALGS:
        integ = integrator.alg_dict[alg][reach]['integrator']
        assert integ['q_source'] == 'rescaled', alg
        assert float(np.nanmean(integ['q'])) == pytest.approx(250., rel=1e-10)


def test_diagnostics_are_recorded_for_every_reach():
    integrator, reach = make_integrator()
    run(integrator)

    for alg in ALGS:
        integ = integrator.alg_dict[alg][reach]['integrator']
        assert np.isfinite(integ['flp_fit_nrmse']), alg
        assert 'flp_fit_nrmse_log' in integ, alg
        assert isinstance(integ['flp_param_at_bound'], int), alg
        assert integ['flp_n_valid_obs'] == 40, alg
        assert integ['dA_valid_frac'] == pytest.approx(1.0), alg


def test_repeat_runs_give_identical_parameters():
    first, reach = make_integrator()
    run(first)
    second, _ = make_integrator()
    run(second)

    for alg, keys in Integrate.FLP_PARAM_KEYS.items():
        for key in keys:
            assert (first.alg_dict[alg][reach]['integrator'][key]
                    == second.alg_dict[alg][reach]['integrator'][key]), (alg, key)


def test_unidentifiable_reach_keeps_its_prior_and_says_so():
    integrator, reach = make_integrator(constant_da=True)
    run(integrator)

    for alg in ('busboi', 'sad', 'sic4dvar', 'metroman', 'hivdi'):
        integ = integrator.alg_dict[alg][reach]['integrator']
        assert integ['flp_fit_status'] == flp_fit.FIT_UNIDENTIFIABLE, alg


def test_legacy_optimizer_still_runs_and_is_labelled():
    integrator, reach = make_integrator(params={'FLP_Optimizer': 'legacy'})
    run(integrator)

    for alg in ALGS:
        integ = integrator.alg_dict[alg][reach]['integrator']
        assert integ['flp_fit_method'] == 'legacy', alg
        assert integ['flp_fit_status'] in flp_fit.FIT_STATUS_CODE, alg


def test_powerlaw_rescale_matches_mean_and_q33():
    integrator, reach = make_integrator(
        qbar=250., q33=170., params={'Rescale_Transform': 'powerlaw'})
    run(integrator)

    integ = integrator.alg_dict['busboi'][reach]['integrator']
    assert integ['rescale_method'] == 'powerlaw'
    q = np.asarray(integ['q'], dtype=float).ravel()
    assert float(np.mean(q)) == pytest.approx(250., rel=1e-6)
    assert float(np.quantile(q, 0.33)) == pytest.approx(170., rel=1e-4)


def test_log_space_objective_runs_and_changes_the_fit():
    linear, reach = make_integrator()
    run(linear)
    log, _ = make_integrator(params={'FLP_Fit_Space': 'log'})
    run(log)

    lin_params = linear.alg_dict['sic4dvar'][reach]['integrator']['n']
    log_params = log.alg_dict['sic4dvar'][reach]['integrator']['n']
    assert np.isfinite(lin_params) and np.isfinite(log_params)
    # Both are valid fits; they should not be the same fit.
    assert lin_params != log_params


def test_report_counts_every_fit():
    integrator, _reach = make_integrator()
    run(integrator)

    report = integrator.run_report()
    assert set(report['flp_fit_status_counts']) == set(ALGS)
    assert sum(sum(c.values())
               for c in report['flp_fit_status_counts'].values()) == len(ALGS)


def test_one_broken_reach_does_not_take_down_the_algorithm():
    integrator, good = make_integrator()
    bad = '99999'
    for alg in ALGS:
        entry = dict(integrator.alg_dict[alg][good])
        entry['integrator'] = dict(entry['integrator'])
        integrator.alg_dict[alg][bad] = entry
    integrator.basin_dict['reach_ids'].append(bad)
    integrator.basin_dict['reach_ids_all'].append(bad)
    # Geometry that will make the flow law misbehave rather than merely fit badly.
    broken = make_obs()
    broken['w'] = np.zeros(broken['nt'])
    broken['S'] = np.full(broken['nt'], np.nan)
    integrator.obs_dict[bad] = broken

    run(integrator)

    for alg in ALGS:
        assert 'flp_fit_status' in integrator.alg_dict[alg][good]['integrator'], alg
        assert np.isfinite(
            integrator.alg_dict[alg][good]['integrator']['flp_fit_nrmse']), alg

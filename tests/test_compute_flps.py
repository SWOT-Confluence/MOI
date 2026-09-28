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
        'flowlaw_valid': np.ones(nt, dtype=bool),
        'flowlaw_valid_da': np.ones(nt, dtype=bool),
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


# ---------------------------------------------------------------------------
# MOMMA stage datum
#
# B and H are absolute stages.  A B >= 0.1 m floor made the bound interval
# empty wherever min(h) < 0.2 m (e.g. Noatak 81340100031, WSE ~ -1.4 .. -0.4 m):
# the grid evaluated nothing, the fit came back 'failed', and the fallback
# (B, H) went on to offline discharge.
# ---------------------------------------------------------------------------

NOATAK_LIKE_SHIFT = -13.4   # moves make_obs() stages to about -1.4 .. +0.7 m


def make_shifted_integrator(shift, params=None, momma_prior=None):
    integrator, reach = make_integrator(params=params)
    integrator.obs_dict[reach]['h'] = integrator.obs_dict[reach]['h'] + shift
    entry = integrator.alg_dict['momma'][reach]
    if momma_prior is None:
        entry['B'] = entry['B'] + shift
        entry['H'] = entry['H'] + shift
    else:
        entry['B'], entry['H'] = momma_prior
    return integrator, reach


def test_momma_fits_a_reach_below_the_geoid():
    integrator, reach = make_shifted_integrator(NOATAK_LIKE_SHIFT)
    h = integrator.obs_dict[reach]['h']
    assert np.min(h) < 0.2   # the old B >= 0.1 interval was empty here
    run(integrator)

    integ = integrator.alg_dict['momma'][reach]['integrator']
    assert integ['flp_fit_status'] == flp_fit.FIT_GOOD
    assert integ['flp_grid_evals'] > 0
    assert integ['B'] < np.min(h)
    assert integ['H'] > integ['B']
    assert np.isfinite(integ['flp_fit_nrmse'])


def test_momma_fit_without_flpe_priors_below_the_geoid():
    # 81340100031 had no usable reach-scale B/H from the MOMMA module.
    integrator, reach = make_shifted_integrator(
        NOATAK_LIKE_SHIFT, momma_prior=(np.nan, np.nan))
    run(integrator)

    integ = integrator.alg_dict['momma'][reach]['integrator']
    assert integ['flp_fit_status'] == flp_fit.FIT_GOOD
    assert integ['flp_grid_evals'] > 0
    assert integ['B'] < np.min(integrator.obs_dict[reach]['h'])


@pytest.mark.parametrize('shift', [NOATAK_LIKE_SHIFT, -11.0, -6.0])
def test_compute_flps_momma_is_invariant_to_the_vertical_datum(shift):
    reference, reach = make_integrator()
    run(reference)
    shifted, _ = make_shifted_integrator(shift)
    run(shifted)

    ref = reference.alg_dict['momma'][reach]['integrator']
    new = shifted.alg_dict['momma'][reach]['integrator']
    assert new['flp_fit_status'] == ref['flp_fit_status'] == flp_fit.FIT_GOOD
    assert new['flp_grid_evals'] == ref['flp_grid_evals']
    assert new['B'] - shift == pytest.approx(ref['B'], abs=1e-4)
    assert new['H'] - shift == pytest.approx(ref['H'], abs=1e-4)
    assert new['flp_fit_nrmse'] == pytest.approx(
        ref['flp_fit_nrmse'], rel=1e-4, abs=1e-8)
    np.testing.assert_allclose(new['q'], ref['q'], rtol=1e-12)
    # The shift must not leak into any other algorithm.
    for alg in ALGS:
        if alg == 'momma':
            continue
        for key in Integrate.FLP_PARAM_KEYS[alg]:
            assert (shifted.alg_dict[alg][reach]['integrator'][key]
                    == reference.alg_dict[alg][reach]['integrator'][key]), (alg, key)


@pytest.mark.parametrize('params', [
    {'FLP_Optimizer': 'legacy'},
    {'FLP_Fit_Method': 'moments'},
])
def test_legacy_momma_paths_accept_a_negative_stage(params):
    integrator, reach = make_shifted_integrator(NOATAK_LIKE_SHIFT, params=params)
    spec = integrator._flp_reach_spec('momma', reach)
    (b_lo, b_hi), (h_lo, _h_hi) = spec['bounds']
    assert b_lo < b_hi < h_lo
    run(integrator)

    integ = integrator.alg_dict['momma'][reach]['integrator']
    assert integ['flp_fit_method'] == 'legacy'
    assert integ['flp_fit_status'] != flp_fit.FIT_FAILED


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


# ---------------------------------------------------------------------------
# Shape contract
#
# Regression tests for the AxisError that took down write_output on basin 1254:
# Output.write_output does np.insert(q, iInsert, fillvalue, axis=1) to put the
# dropped timesteps back, which needs q to be (1, nt).  Storing a raveled
# hydrograph made it 1-D and lost the output file for the whole basin.
# ---------------------------------------------------------------------------

def test_stored_hydrograph_is_always_a_row():
    integrator, reach = make_integrator()
    run(integrator)

    for alg in ALGS:
        q = integrator.alg_dict[alg][reach]['integrator']['q']
        assert isinstance(q, np.ndarray), alg
        assert q.ndim == 2, (alg, np.shape(q))
        assert q.shape == (1, 40), (alg, q.shape)
        # The shape np.insert(..., axis=1) needs.
        assert np.insert(q, [0], -999., axis=1).shape == (1, 41), alg


def test_flow_law_sourced_hydrograph_is_also_a_row():
    """The rescale path always produced (1, nt); the flow-law path did not.

    Only reaches that fall back to the flow-law hydrograph could reach
    write_output as 1-D, which is why this went unnoticed on the canary basins
    where nearly everything rescales.
    """
    integrator, reach = make_integrator(
        params={'Integrator_Hydrograph_Method': 'flowlaw'})
    run(integrator)

    for alg in ALGS:
        integ = integrator.alg_dict[alg][reach]['integrator']
        assert integ['q_source'] == 'flowlaw', alg
        assert np.shape(integ['q']) == (1, 40), (alg, np.shape(integ['q']))


def test_moments_objective_hydrograph_is_also_a_row():
    integrator, reach = make_integrator(
        params={'FLP_Fit_Method': 'moments',
                'Integrator_Hydrograph_Method': 'flowlaw'})
    run(integrator)

    for alg in ALGS:
        assert np.shape(
            integrator.alg_dict[alg][reach]['integrator']['q']) == (1, 40), alg


def test_as_row_normalises_every_shape_it_can_receive():
    assert flp_fit.as_row(None, 5).shape == (1, 5)
    assert np.all(np.isnan(flp_fit.as_row(None, 5)))
    assert flp_fit.as_row(np.arange(5.), 5).shape == (1, 5)
    assert flp_fit.as_row(np.arange(5.).reshape(1, 5), 5).shape == (1, 5)
    # momma_flowlaw returns a bare inf when bankfull collapses onto zero-flow.
    padded = flp_fit.as_row(np.inf, 4)
    assert padded.shape == (1, 4)
    assert np.isinf(padded[0, 0]) and np.isnan(padded[0, 1])
    # Too long is truncated, too short is padded -- neither raises.
    assert flp_fit.as_row(np.arange(9.), 4).shape == (1, 4)
    short = flp_fit.as_row(np.arange(2.), 4)
    assert short.shape == (1, 4) and np.isnan(short[0, 3])
    np.testing.assert_array_equal(flp_fit.as_row(np.arange(5.), 5)[0], np.arange(5.))
    assert flp_fit.as_row(np.array([], dtype=float), 0).shape == (1, 0)
    assert np.all(np.isnan(flp_fit.as_row('not-a-number', 3)))


def test_length_repair_is_counted_not_silent():
    """A hydrograph that disagrees with the record is repaired AND reported.

    Silently truncating would make a partly-missing series indistinguishable
    from a complete one, which is the whole thing the diagnostics exist to stop.
    """
    integrator, reach = make_integrator()
    run(integrator)
    assert integrator.run_report()['n_hydrograph_repairs'] == 0

    # A reach outside basin_dict['reach_ids'] is skipped by the fitter, so
    # whatever is already in its integrator dict is what the normalisation pass
    # sees -- the same situation as an algorithm that produced a short series.
    stray = '88888'
    for alg in ALGS:
        integrator.alg_dict[alg][stray] = {'integrator': {'q': np.arange(7.)}}
    integrator.obs_dict[stray] = make_obs()
    integrator.basin_dict['reach_ids_all'].append(stray)

    integrator.compute_FLPs()

    assert integrator.run_report()['n_hydrograph_repairs'] == len(ALGS)
    for alg in ALGS:
        q = integrator.alg_dict[alg][stray]['integrator']['q']
        assert q.shape == (1, 40), alg
        assert np.isnan(q[0, 7])


# ---------------------------------------------------------------------------
# fit quality, q_flowlaw and geometry status
def test_shape_diagnostics_and_classification():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    good = flp_fit.shape_diagnostics(target * 1.1, target)
    assert good['flp_fit_pearson_r'] > 0.999
    assert np.isclose(good['flp_fit_nbias'], 0.1)
    assert flp_fit.classify_fit_quality(flp_fit.FIT_GOOD, good, 0) == flp_fit.FIT_GOOD
    # At a bound is poor even when the shape matches.
    assert flp_fit.classify_fit_quality(flp_fit.FIT_GOOD, good, 1) == flp_fit.FIT_POOR

    reversed_shape = flp_fit.shape_diagnostics(target[::-1], target)
    assert reversed_shape['flp_fit_pearson_r'] < 0
    assert flp_fit.classify_fit_quality(
        flp_fit.FIT_GOOD, reversed_shape, 0) == flp_fit.FIT_POOR
    # Only 'good' is ever downgraded.
    assert flp_fit.classify_fit_quality(
        flp_fit.FIT_NO_SERIES, reversed_shape, 0) == flp_fit.FIT_NO_SERIES


def test_flowlaw_series_and_diagnostics_are_stored():
    integrator, reach = make_integrator()
    run(integrator)
    for alg in ALGS:
        integ = integrator.alg_dict[alg][reach]['integrator']
        assert np.shape(integ['q_flowlaw']) == (1, 40), alg
        assert np.isfinite(integ['flp_fit_pearson_r']), alg
        assert np.isfinite(integ['flp_fit_nse']), alg
        assert integ['flp_fit_status'] in (flp_fit.FIT_GOOD, flp_fit.FIT_POOR), alg
        if integ['flp_fit_status'] == flp_fit.FIT_GOOD:
            assert integ['flp_fit_pearson_r'] > 0 and integ['flp_fit_nse'] > 0


def test_sign_reversed_area_marks_da_laws_invalid():
    integrator, reach = make_integrator()
    integrator.obs_dict[reach]['geometry_status'] = 'dA_not_increasing_with_wse'
    run(integrator)
    for alg in ALGS:
        status = integrator.alg_dict[alg][reach]['integrator']['flp_fit_status']
        if alg == 'momma':
            assert status != flp_fit.FIT_INVALID_GEOMETRY
        else:
            assert status == flp_fit.FIT_INVALID_GEOMETRY, alg
    assert integrator.flp_fit_report['busboi'] == {flp_fit.FIT_INVALID_GEOMETRY: 1}


def test_a0_lower_bound_is_admissible_offline():
    integrator, reach = make_integrator()
    integrator.obs_dict[reach]['dA'] = integrator.obs_dict[reach]['dA'] + 1000.
    for alg in ('busboi', 'sad', 'sic4dvar', 'hivdi', 'metroman'):
        spec = integrator._flp_reach_spec(alg, reach)
        assert spec['a0_min'] >= 1.0, alg


def test_sic4dvar_file_values_are_strickler_and_bed_elevation():
    n, a0 = Integrate._sic4dvar_priors(30.0, 150.2)
    assert np.isclose(n, 1.0 / 30.0) and np.isnan(a0)
    assert np.isclose(Integrate._sic4dvar_priors(0.03, 5.0)[0], 0.03)
    assert np.isnan(Integrate._sic4dvar_priors(np.nan, 5.0)[0])


def test_offline_mask_removes_timesteps_from_the_fit_target():
    integrator, reach = make_integrator()
    keep = np.ones(40, dtype=bool)
    keep[[3, 7, 11]] = False
    integrator.obs_dict[reach]['flowlaw_valid'] = keep
    integrator.obs_dict[reach]['flowlaw_valid_da'] = keep
    integrator.calc_integrator_hydrographs()

    target = integrator.flp_fit_target('busboi', reach)
    assert np.isnan(target[[3, 7, 11]]).all()
    assert np.isfinite(target[keep]).all()
    # The retained MOI hydrograph itself is untouched.
    q = integrator.alg_dict['busboi'][reach]['integrator']['q']
    assert np.isfinite(np.ravel(q)).all()

    integrator.params_dict['FLP_Use_Offline_Mask'] = False
    assert np.isfinite(integrator.flp_fit_target('busboi', reach)).all()


# ---------------------------------------------------------------------------
# export gating: what offline and SWORD receive
# ---------------------------------------------------------------------------

def _exported(integrator, reach, alg):
    return integrator.alg_dict[alg][reach]['integrator']


def test_invalid_geometry_masks_da_laws_but_keeps_candidates():
    integrator, reach = make_integrator()
    integrator.obs_dict[reach]['geometry_status'] = 'dA_not_increasing_with_wse'
    run(integrator)
    for alg in ALGS:
        integ = _exported(integrator, reach, alg)
        keys = Integrate.FLP_PARAM_KEYS[alg]
        if alg == 'momma':
            assert integ['flp_exported'] == 1
            continue
        assert integ['flp_exported'] == 0, alg
        assert integ['flp_export_reason'] == flp_fit.EXPORT_REASON_INVALID_GEOMETRY
        assert all(np.isnan(integ[k]) for k in keys), alg
        assert all(np.isfinite(integ['flp_candidate_' + k]) for k in keys), alg
        assert np.isnan(integ['q_flowlaw']).all()
        assert np.isfinite(integ['q_flowlaw_candidate']).any()
        # The MOI hydrograph itself is not touched.
        assert np.isfinite(np.ravel(integ['q'])).all()
    coverage = integrator.run_report()['flp_export_coverage']['busboi']
    assert coverage['exported'] == 0
    assert coverage['masked'] == {flp_fit.EXPORT_REASON_INVALID_GEOMETRY: 1}


def test_report_only_exports_everything():
    integrator, reach = make_integrator(
        params={'FLP_Export_Policy': flp_fit.EXPORT_REPORT_ONLY})
    integrator.obs_dict[reach]['geometry_status'] = 'dA_not_increasing_with_wse'
    run(integrator)
    for alg in ALGS:
        integ = _exported(integrator, reach, alg)
        assert integ['flp_exported'] == 1, alg
        assert integ['flp_fit_status'] in (flp_fit.FIT_INVALID_GEOMETRY,
                                           flp_fit.FIT_GOOD, flp_fit.FIT_POOR)


def test_export_decision_rules():
    good_q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    anti = {'flp_fit_pearson_r': -0.4, 'flp_fit_n_paired': 20}
    weak = {'flp_fit_pearson_r': 0.3, 'flp_fit_n_paired': 20}
    undefined = {'flp_fit_pearson_r': np.nan, 'flp_fit_n_paired': 20}
    few = {'flp_fit_pearson_r': -0.4, 'flp_fit_n_paired': 3}
    decide = flp_fit.export_decision
    assert decide(flp_fit.FIT_POOR, (1., 2.), good_q, anti) == \
        flp_fit.EXPORT_REASON_ANTI_CORRELATED
    assert decide(flp_fit.FIT_POOR, (1., 2.), good_q, few) == \
        flp_fit.EXPORT_REASON_NONE
    assert decide(flp_fit.FIT_POOR, (1., 2.), good_q, weak) == \
        flp_fit.EXPORT_REASON_NONE
    assert decide(flp_fit.FIT_FAILED, (1., 2.), good_q, weak) == \
        flp_fit.EXPORT_REASON_FIT_FAILED
    assert decide(flp_fit.FIT_POOR, (1., 2.), good_q, undefined) == \
        flp_fit.EXPORT_REASON_UNDEFINED_CORRELATION
    assert decide(flp_fit.FIT_POOR, (1., 2.), good_q, weak,
                  flp_fit.EXPORT_MASK_ALL_POOR) == flp_fit.EXPORT_REASON_POOR_FIT
    assert decide(flp_fit.FIT_GOOD, (1., np.nan), good_q, weak) == \
        flp_fit.EXPORT_REASON_NON_FINITE
    assert decide(flp_fit.FIT_GOOD, (1., 2.), good_q * np.nan, weak) == \
        flp_fit.EXPORT_REASON_NON_FINITE
    assert decide(flp_fit.FIT_INVALID_GEOMETRY, (1., 2.), good_q, anti,
                  flp_fit.EXPORT_REPORT_ONLY) == flp_fit.EXPORT_REASON_NONE
    with pytest.raises(ValueError):
        decide(flp_fit.FIT_GOOD, (1., 2.), good_q, weak, 'mask_everything')


def test_sic4dvar_fallback_exports_a_usable_a0():
    # Constant dA -> unidentifiable -> the fallback (init) parameters are used.
    integrator, reach = make_integrator(constant_da=True)
    integrator.alg_dict['sic4dvar'][reach]['n'] = 30.0   # Strickler K in file
    run(integrator)
    integ = _exported(integrator, reach, 'sic4dvar')
    assert integ['flp_fit_status'] == flp_fit.FIT_UNIDENTIFIABLE
    assert np.isclose(integ['flp_candidate_n'], 1.0 / 30.0)
    assert np.isfinite(integ['flp_candidate_a0'])
    assert integ['flp_candidate_a0'] >= 1.0


def test_prior_or_respects_bounds():
    assert Integrate._prior_or(np.nan, 0.03, 0.001) == 0.03
    assert Integrate._prior_or(5.0, 0.03, 0.001, 1.0) == 0.03
    assert Integrate._prior_or(0.5, 0.03, 0.001, 1.0) == 0.5

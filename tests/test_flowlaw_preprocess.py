"""moi/flowlaw_preprocess.py: observation filter, width and slope policy.

The same file ships in offline; tools/check_offline_parity.py runs offline's
real code against MOI.  These tests pin the rules themselves, including the
counter-examples raised in review.
"""
import netCDF4
import numpy as np
import pytest

from moi import flowlaw_preprocess as fp
from moi.Input import Input

NT = 30
H = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
W = np.array([100.0, 120.0, 140.0, 160.0, 170.0])
DEFAULT_BREAKS = [-1500.0, 49000.0, 99500.0, 150000.0]


def _fit(h_break, slopes, intercepts):
    coeffs = np.zeros((2, 3, 1))
    coeffs[0, :, 0] = slopes
    coeffs[1, :, 0] = intercepts
    return {'h_break': np.asarray(h_break, dtype=float).reshape(4, 1),
            'fit_coeffs': coeffs}


def _series():
    # Evenly spaced values have no Tukey outliers of their own.
    h = np.linspace(19.8, 20.2, NT)
    w = np.linspace(115., 125., NT)
    s = np.linspace(0.9e-4, 1.1e-4, NT)
    q = {
        'time': np.arange(NT, dtype=float), 'xtrk_dist': np.full(NT, 30e3),
        'ice_clim_f': np.zeros(NT), 'dark_frac': np.zeros(NT),
        'obs_frac_n': np.ones(NT), 'xovr_cal_q': np.zeros(NT),
        'n_good_nod': np.full(NT, 50.), 'p_width': np.full(NT, 100.),
        'p_length': np.full(NT, 1e4), 'reach_q_b': np.zeros(NT),
    }
    return h, w, s, q


# ---------------------------------------------------------------- width ----

def test_valid_fit_uses_the_three_segments():
    area_fit = _fit([9.0, 11.5, 13.5, 15.0], [10.0] * 3, [0.0, 5.0, 10.0])
    width, source = fp.constrain_width(H, W, area_fit)
    assert source == fp.WIDTH_SOURCE_HYPSOMETRIC
    np.testing.assert_allclose(width, [100.0, 110.0, 125.0, 135.0, 150.0])


@pytest.mark.parametrize('slopes, intercepts', [
    ([0, 0, 0], [-5869.1] * 3),            # 73150600241: negative fallback width
    ([0, 0, 0], [686.2] * 3),              # 73150600741: constant 686 m
    ([120.0] * 3, [-1194.2] * 3),          # 73150600751: 5.8-485 m
])
def test_failed_or_implausible_fits_use_the_median_width(slopes, intercepts):
    area_fit = _fit(DEFAULT_BREAKS if slopes == [0, 0, 0] else [9, 11.5, 13.5, 15],
                    slopes, intercepts)
    width, source = fp.constrain_width(H, W, area_fit)
    assert source == fp.WIDTH_SOURCE_MEDIAN
    np.testing.assert_allclose(width, np.median(W))


def test_too_few_observed_widths_cannot_validate_a_fit():
    area_fit = _fit(DEFAULT_BREAKS, [0, 0, 0], [1e9] * 3)
    width, source = fp.constrain_width(H[:2], W[:2], area_fit)
    assert source == fp.WIDTH_SOURCE_MEDIAN
    np.testing.assert_allclose(width, 110.0)


def test_width_checks_ignore_filtered_observations():
    area_fit = _fit(DEFAULT_BREAKS, [0, 0, 0], [686.0] * 3)
    w = np.array([100., 110., 120., 130., 900.])
    keep = np.array([True, True, True, True, False])
    assert fp.constrain_width(H, w, area_fit)[1] == fp.WIDTH_SOURCE_HYPSOMETRIC
    width, source = fp.constrain_width(H, w, area_fit, keep)
    assert source == fp.WIDTH_SOURCE_MEDIAN
    np.testing.assert_allclose(width, 115.0)


def test_missing_fit_and_missing_widths():
    no_fit = {'h_break': np.full(4, np.nan), 'fit_coeffs': None}
    assert fp.constrain_width(H, W, no_fit)[1] == fp.WIDTH_SOURCE_MEDIAN
    width, source = fp.constrain_width(H, np.full(5, np.nan), no_fit)
    assert source == fp.WIDTH_SOURCE_NONE and np.isnan(width).all()


# ---------------------------------------------------------------- slope ----

def test_legacy_safe_slope_policy():
    used, valid, status = fp.prepare_slope(
        [-1.0, 0.0, 1e-6, 2e-5, np.nan], fp.SLOPE_LEGACY_SAFE)
    np.testing.assert_array_equal(valid, [False, False, True, True, False])
    np.testing.assert_array_equal(status, [
        fp.SLOPE_NONPOSITIVE, fp.SLOPE_NONPOSITIVE, fp.SLOPE_FLOORED,
        fp.SLOPE_OK, fp.SLOPE_MISSING])
    assert np.isnan(used[[0, 1, 4]]).all()
    np.testing.assert_allclose(used[2:4], [fp.SLOPE_FLOOR, 2e-5])


def test_l2_3p4_slope_policy():
    used, valid, _status = fp.prepare_slope([-1.0, 3e-5, 3.4e-5, 1e-4],
                                            fp.SLOPE_L2_3P4)
    np.testing.assert_array_equal(valid, [False, False, True, True])
    np.testing.assert_allclose(used[2:], [3.4e-5, 1e-4])


def test_unknown_options_are_rejected():
    with pytest.raises(ValueError):
        fp.prepare_slope([1e-4], 'algorithm_specific')
    h, w, s, q = _series()
    with pytest.raises(ValueError):
        fp.observation_keep_mask(h, w, s, q, 'robust_v3')


# ------------------------------------------------- observation filter ----

@pytest.mark.parametrize('version', fp.OBSERVATION_FILTERS)
def test_clean_series_is_kept(version):
    h, w, s, q = _series()
    assert fp.observation_keep_mask(h, w, s, q, version)[0].all()


@pytest.mark.parametrize('version', fp.OBSERVATION_FILTERS)
def test_each_nrt_check_drops_its_timestep(version):
    h, w, s, q = _series()
    q['xtrk_dist'][1] = 5e3
    q['xtrk_dist'][2] = -70e3
    q['ice_clim_f'][3] = 2
    q['dark_frac'][4] = 0.7
    q['obs_frac_n'][5] = 0.3
    q['xovr_cal_q'][6] = 2
    q['n_good_nod'][7] = 9
    q['p_width'][8] = 50
    q['p_length'][9] = 4000
    q['reach_q_b'][10] = 507510785
    q['time'][11] = np.nan
    keep, info = fp.observation_keep_mask(h, w, s, q, version)
    assert not keep[1:12].any()
    assert keep[13:29].all()
    assert info['n_nrt_bad'] == 11


def test_legacy_and_robust_v2_differ_on_unknown_quality():
    h, w, s, q = _series()
    q['dark_frac'][3] = np.nan
    assert fp.observation_keep_mask(h, w, s, q, fp.FILTER_LEGACY)[0][3]
    keep, info = fp.observation_keep_mask(h, w, s, q, fp.FILTER_ROBUST_V2)
    assert not keep[3] and info['n_quality_unknown'] == 1


def test_one_gap_no_longer_disables_outlier_filtering():
    h, w, s, q = _series()
    w[5] = 1e9
    w[7] = np.nan
    # legacy: one NaN width switches the whole series to min/max limits
    assert fp.observation_keep_mask(h, w, s, q, fp.FILTER_LEGACY)[0][5]
    keep, info = fp.observation_keep_mask(h, w, s, q, fp.FILTER_ROBUST_V2)
    assert not keep[5] and keep[7]
    assert info['n_outlier'] == 1


def test_robust_v2_skips_only_the_undersampled_variable():
    h, w, s, q = _series()
    w[:] = np.nan
    w[:3] = [100., 110., 5000.]
    h[10] = 99.0
    keep, info = fp.observation_keep_mask(h, w, s, q, fp.FILTER_ROBUST_V2)
    assert info['outlier_test_skipped'] == 'width'
    assert keep[2]            # width outlier not judged: too few widths
    assert not keep[10]       # height outlier still removed


def test_missing_quality_variables_keep_everything():
    h, w, s, _q = _series()
    keep, info = fp.observation_keep_mask(h, w, s, None)
    assert keep.all() and info['outlier_test_skipped'] == 'no_quality_variables'


# ------------------------------------------------------------ together ----

def test_prepare_flowlaw_inputs_combines_filter_width_and_slope():
    h, w, s, q = _series()
    s = np.linspace(1e-5, 5e-5, NT)
    s[4] = 0.0          # inside the Tukey limits, invalid under legacy_safe
    q['dark_frac'][6] = 0.9
    area_fit = _fit([19.0, 19.9, 20.1, 21.0], [0.0] * 3, [120.0] * 3)
    prep = fp.prepare_flowlaw_inputs(h, w, s, area_fit, q)
    assert prep['width_source'] == fp.WIDTH_SOURCE_HYPSOMETRIC
    assert not prep['flowlaw_valid'][4] and prep['keep'][4]   # slope only
    assert not prep['flowlaw_valid'][6] and not prep['keep'][6]
    assert prep['flowlaw_valid'].sum() == NT - 2
    assert np.isnan(prep['slope_used'][4])


def test_read_config_defaults_and_values(tmp_path):
    path = tmp_path / 'r_integrator.nc'
    with netCDF4.Dataset(path, 'w') as ds:
        ds.setncattr(fp.CONFIG_ATTRIBUTES['slope_policy'], fp.SLOPE_L2_3P4)
    with netCDF4.Dataset(path) as ds:
        assert fp.read_config(ds) == {
            'observation_filter': fp.DEFAULT_OBSERVATION_FILTER,
            'slope_policy': fp.SLOPE_L2_3P4}


def test_geometry_status_flags_sign_reversed_area():
    corr, status = Input.geometry_status(H, -(H - H.mean()) * 150.0)
    assert status == Input.GEOMETRY_DA_NOT_INCREASING and corr < 0
    corr, status = Input.geometry_status(H, (H - H.mean()) * 150.0)
    assert status == Input.GEOMETRY_OK and corr > 0.99
    assert Input.geometry_status(H[:2], H[:2])[1] == Input.GEOMETRY_TOO_FEW_OBS


# ------------------------------------------------- Input.extract_swot ----

def _write_swot(path, h, w, s, dA, q, hwfit=True):
    nt = h.size
    with netCDF4.Dataset(path, 'w') as ds:
        ds.createDimension('nt', nt)
        reach = ds.createGroup('reach')
        series = {'wse': h, 'width': w, 'slope2': s, 'd_x_area': dA,
                  'reach_q': np.zeros(nt), **q}
        for name, values in series.items():
            reach.createVariable(name, 'f8', ('nt',), fill_value=-1e12)[:] = \
                np.where(np.isfinite(values), values, -1e12)
        if hwfit:
            fit = reach.createGroup('hwfit')
            for dim, size in (('b', 4), ('c', 2), ('s', 3), ('o', 1)):
                fit.createDimension(dim, size)
            fit.createVariable('h_break', 'f8', ('b', 'o'))[:] = np.full((4, 1), np.nan)
            fit.createVariable('fit_coeffs', 'f8', ('c', 's', 'o'))[:] = np.zeros((2, 3, 1))


def test_extract_swot_uses_offline_observations_for_flowlaw_geometry(tmp_path):
    h, w, s, q = _series()
    dA = 150.0 * (h - h.mean())
    dA[3] = -1e6                 # extreme area at an observation offline drops
    q['dark_frac'][3] = 0.9
    s[8] = -2e-5                 # negative slope: no discharge, but kept in MOI
    w[12] = np.nan               # no width at all -> removed from every product
    h[12] = np.nan
    _write_swot(tmp_path / '1_SWOT.nc', h, w, s, dA, q)

    obj = Input(None, None, tmp_path, None, {'reach_ids': ['1']},
                'constrained', False)
    obj.extract_swot()
    obs = obj.obs_dict['1']

    assert obs['nt'] == NT - 1
    for key in ('h', 'w', 'S', 'dA', 'reach_q', 'xovr_cal_q', 'flowlaw_valid'):
        assert np.size(obs[key]) == NT - 1, key
    # index 3 (NRT-bad) and 8 (negative slope) stay in MOI but carry no
    # flow-law geometry, so they cannot influence bounds or fits.
    assert not obs['flowlaw_valid'][3] and not obs['flowlaw_valid'][8]
    assert np.isnan(obs['dA'][3]) and np.isnan(obs['S'][8])
    assert np.nanmin(obs['dA']) > -1e3
    assert obs['S_raw'][8] == pytest.approx(-2e-5)
    assert obs['geometry_status'] == Input.GEOMETRY_OK
    assert obs['width_source'] == fp.WIDTH_SOURCE_MEDIAN


def test_extract_swot_without_hwfit_or_quality_variables(tmp_path):
    h, w, s, _q = _series()
    q = {'time': np.arange(NT, dtype=float), 'xovr_cal_q': np.zeros(NT)}
    _write_swot(tmp_path / '1_SWOT.nc', h, w, s, 150.0 * (h - h.mean()), q,
                hwfit=False)
    obj = Input(None, None, tmp_path, None, {'reach_ids': ['1']},
                'constrained', False)
    obj.extract_swot()
    obs = obj.obs_dict['1']
    assert obs['width_source'] == fp.WIDTH_SOURCE_MEDIAN
    assert obs['flowlaw_valid'].all()
    assert obs['observation_filter_counts']['outlier_test_skipped'] == \
        'no_quality_variables'

import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

from netCDF4 import Dataset
import numpy as np
import pytest
import scipy.sparse as sp

from moi.gage_statistics import observation_days, study_period
from moi.Input import Input
from moi.Integrate import Integrate
from moi.Output import Output
from moi.sfoi_math_core import adjust_lsq_mult_sparse
from run_MOI import set_moi_params


def ordinal(day):
    return dt.date.fromisoformat(day).toordinal()


def seconds(*days):
    epoch = dt.datetime(2000, 1, 1)
    return np.array([(dt.datetime.fromisoformat(day) - epoch).total_seconds()
                     for day in days])


def gage(days, values, **extra):
    return {'source': 'SVS', 'group': 'calibration',
            't': np.array([ordinal(day) for day in days]),
            'Q': np.array(values, dtype=float), **extra}


def prepare(gages, obs=None, **params):
    integrator = Integrate.__new__(Integrate)
    integrator.gage_dict = gages
    integrator.obs_dict = obs if obs is not None else {
        '10': {'t_full': seconds('2024-01-01', '2024-01-10'),
               't': seconds('2024-01-05')}}
    integrator._use_sos_gage_fallback = False
    integrator.sos_dict = {}
    integrator.params_dict = {'Gage_Statistic_Mode': 'period', **params}
    integrator.VerboseFlag = False
    integrator.prepare_gage_constraints()
    return integrator


def test_period_includes_non_swot_days_and_gages_without_geometry():
    obs = {'10': {'t_full': seconds('2024-01-01', '2024-01-05'),
                  't': np.array([])},
           '20': {'t_full': seconds('2024-01-02', '2024-01-04'),
                  't': seconds('2024-01-03')}}
    record = gage(['2023-12-31', '2024-01-01', '2024-01-02', '2024-01-03',
                   '2024-01-04', '2024-01-05', '2024-01-06'],
                  [1000, 10, 20, 30, 40, 50, 1000])
    integrator = prepare({'10': record, '20': record, '30': record}, obs)
    assert set(integrator.gage_dict) == {'10', '20', '30'}
    for selected in integrator.gage_dict.values():
        assert selected['Qbar'] == 30
        assert selected['q33'] == pytest.approx(np.quantile([10, 20, 30, 40, 50], .33))
        assert selected['n_matched'] == 5
    report = integrator.integ_dict['gage_preparation']
    assert report['period_start'] == '2024-01-01'
    assert report['period_end'] == '2024-01-05'
    assert report['period_days'] == 5


def test_twenty_percent_boundary_and_duplicate_days():
    integrator = prepare({
        '20': gage(['2024-01-01', '2024-01-10'], [10, 30]),
        '30': gage(['2024-01-01', '2024-01-01'], [10, 30]),
        '40': gage(['2023-12-31', '2024-01-11'], [10, 30]),
    })
    assert set(integrator.gage_dict) == {'20'}
    diagnostics = integrator.integ_dict['gage_preparation']['reaches']
    assert diagnostics['20']['period_coverage'] == .2
    assert diagnostics['30']['period_coverage'] == .1
    assert diagnostics['30']['status'] == 'insufficient_period_coverage'
    assert diagnostics['40']['status'] == 'no_gage_data_in_period'
    # Low coverage never falls back to data outside the scientific target.
    assert '40' not in integrator.gage_dict


def test_manual_period_overrides_raw_dates_and_handles_leap_day():
    integrator = prepare({
        '20': gage(['2024-02-28', '2024-02-29', '2024-03-01', '2024-03-02'],
                   [10, 20, 30, 1000])
    }, obs={}, Gage_Period_Start='2024-02-28', Gage_Period_End='2024-03-01')
    assert integrator.gage_dict['20']['Qbar'] == 20
    assert integrator.integ_dict['gage_preparation']['period_days'] == 3


@pytest.mark.parametrize('overrides', [
    {'Gage_Min_Period_Coverage': -.1},
    {'Gage_Min_Period_Coverage': 1.1},
    {'Gage_Min_Period_Coverage': np.nan},
    {'Gage_Period_Start': '2024-02-30'},
    {'Gage_Period_Start': '2024-01-11', 'Gage_Period_End': '2024-01-01'},
])
def test_invalid_period_configuration_fails(overrides):
    with pytest.raises(ValueError):
        prepare({'20': gage(['2024-01-01'], [10])}, **overrides)


def test_no_raw_dates_requires_explicit_period():
    with pytest.raises(ValueError, match='Cannot determine'):
        prepare({'20': gage(['2024-01-01'], [10])}, obs={})


def test_full_time_string_fallback_uses_utc_dates_and_ignores_bad_values():
    obs = {'time_str': ['bad', '2023-12-31T23:30:00-02:00',
                        '2024-01-03T00:00:00Z'],
           't': seconds('2024-01-02')}
    assert observation_days(obs).tolist() == [ordinal('2024-01-01'), ordinal('2024-01-03')]
    assert study_period({'10': obs}) == (ordinal('2024-01-01'), ordinal('2024-01-03'))


def test_zero_discharge_is_in_mean_quantile_and_solver_rows():
    integrator = prepare({
        '20': gage(['2024-01-01', '2024-01-02', '2024-01-03'], [0, 0, 30]),
        '30': gage(['2024-01-01', '2024-01-02'], [0, 0]),
    })
    assert integrator.gage_dict['20']['Qbar'] == 10
    assert integrator.gage_dict['20']['q33'] == 0
    assert integrator.gage_dict['30']['Qbar'] == 0
    integrator.Branch = 'constrained'
    integrator.basin_dict = {'reach_ids_all': ['20', '30']}
    rows = integrator.build_gage_observation_rows({'A_obs': sp.eye(2)}, 'q33')
    np.testing.assert_array_equal(rows['L'], [0, 0])
    result = adjust_lsq_mult_sparse(
        A_obs=sp.csr_matrix([[1.], [1.]]), L=np.array([10., 0.]),
        cov=np.array([1., .1]), lb=np.array([0.]), x0=np.array([5.]),
        fixed_weight_mask=np.array([False, True]),
        robust_eligible_mask=np.array([True, False]),
    )
    assert np.all(np.isfinite(result.W))
    assert result.x[0] < 1


def test_pseudo_gages_keep_swot_day_matching_without_period_coverage_gate():
    integrator = prepare({
        '20': gage(['2024-01-04', '2024-01-05'], [1000, 10],
                   source='corridors', group=''),
    }, Gage_Min_Period_Coverage=1.0)
    assert integrator.gage_dict['20']['Qbar'] == 10
    assert integrator.gage_dict['20']['n_matched'] == 1


def test_validation_and_bad_groups_cannot_become_constraints():
    integrator = prepare({
        group: gage(['2024-01-01', '2024-01-10'], [10, 30], group=group)
        for group in ['calibration', 'validation', 'bad']})
    assert set(integrator.gage_dict) == {'calibration'}


def test_preparation_diagnostics_survive_json_and_netcdf(tmp_path):
    integrator = prepare({
        '20': gage(['2024-01-01', '2024-01-10'], [10, 30]),
        '30': gage(['2024-01-01'], [10]),
    })
    report = integrator.integ_dict['gage_preparation']
    json.dumps(report, allow_nan=False)
    writer = Output({}, tmp_path, integrator.integ_dict, {}, {}, tmp_path, {},
                    gage_groups={'20': 'calibration', '30': 'calibration'},
                    gage_dict=integrator.gage_dict)
    for reach in ['20', '30']:
        path = tmp_path / f'{reach}.nc'
        with Dataset(path, 'w') as dataset:
            writer._write_gage_metadata(dataset, reach)
        with Dataset(path) as dataset:
            g = dataset['gage']
            assert g.period_start == '2024-01-01'
            assert g.period_end == '2024-01-10'
            assert g.min_period_coverage == .2
            assert g.status == report['reaches'][reach]['status']
            assert g.constraint_source == ('gage' if reach == '20' else 'none')
    with Dataset(tmp_path / '20.nc') as dataset:
        assert dataset['gage'].Qbar == 20
        assert dataset['gage'].period_coverage == .2


def test_svs_loader_retains_zero_discharge(tmp_path):
    path = tmp_path / 'SVS.nc'
    with Dataset(path, 'w') as dataset:
        for name, size in [('station', 1), ('time', 3), ('ymd', 3)]:
            dataset.createDimension(name, size)
        dataset.createVariable('reach_id_v17c', 'i8', ('station',))[:] = [1001]
        dataset.createVariable('Q', 'f8', ('station', 'time'))[:] = [[0, -1, 10]]
        dataset.createVariable('date_ymd', 'i4', ('ymd', 'time'))[:] = [
            [2024, 2024, 2024], [1, 1, 1], [1, 2, 3]]
    inp = Input(tmp_path, tmp_path, tmp_path, tmp_path,
                {'reach_ids_all': ['1001']}, 'constrained', False)
    selected = inp.extract_svs(path)
    assert selected['1001']['Q'].tolist() == [0, 10]


def test_downloaded_basin_files_outside_reach_selection_define_period(tmp_path):
    inp = Input(tmp_path, tmp_path, tmp_path, tmp_path,
                {'basin_id': '7426'}, 'constrained', False)
    inp.obs_dict = {'74260000101': {'t_full': seconds('2024-02-01', '2024-02-02')}}
    for reach, dates in [
        ('74260000201', ['2024-01-01', '2024-03-01']),
        ('12340000201', ['2020-01-01', '2030-01-01']),
    ]:
        with Dataset(tmp_path / f'{reach}_SWOT.nc', 'w') as dataset:
            dataset.createDimension('nt', 2)
            group = dataset.createGroup('reach')
            group.createVariable('time', 'f8', ('nt',))[:] = seconds(*dates)
    assert inp.gage_study_period() == (ordinal('2024-01-01'), ordinal('2024-03-01'))
    assert inp.gage_study_period('2024-01-10', '2024-01-20') == (
        ordinal('2024-01-10'), ordinal('2024-01-20'))


def test_station_selection_counts_days_in_target_period(tmp_path):
    path = tmp_path / 'SVS.nc'
    with Dataset(path, 'w') as dataset:
        for name, size in [('station', 2), ('time', 5), ('ymd', 3)]:
            dataset.createDimension(name, size)
        dataset.createVariable('reach_id_v17c', 'i8', ('station',))[:] = [1001, 1001]
        dataset.createVariable('Q', 'f8', ('station', 'time'))[:] = [
            [100, 100, 100, np.nan, np.nan],
            [np.nan, np.nan, np.nan, 20, 30]]
        dataset.createVariable('date_ymd', 'i4', ('ymd', 'time'))[:] = [
            [2023, 2023, 2023, 2024, 2024], [1, 1, 1, 1, 1], [1, 2, 3, 1, 2]]
    inp = Input(tmp_path, tmp_path, tmp_path, tmp_path,
                {'reach_ids_all': ['1001']}, 'constrained', False)
    selected = inp.extract_svs(path, gage_period=(ordinal('2024-01-01'), ordinal('2024-01-02')))
    assert selected['1001']['station_index'] == 1
    assert selected['1001']['Q'].tolist() == [20, 30]


def test_configured_period_is_shared_by_station_selection_and_integrator(tmp_path):
    inp = Input(tmp_path, tmp_path, tmp_path, tmp_path,
                {'basin_id': '7426'}, 'constrained', False)
    inp.obs_dict = {'10': {'t_full': seconds('2024-01-01', '2024-01-10')}}
    params = set_moi_params()
    assert inp.configure_gage_period(params) == (ordinal('2024-01-01'), ordinal('2024-01-10'))
    assert params['Gage_Period_Start'] == '2024-01-01'
    assert params['Gage_Period_End'] == '2024-01-10'
    assert params['Gage_Period_Source'] == 'basin_raw_swot_with_optional_overrides'
    integrator = prepare({'20': gage(['2024-01-01', '2024-01-10'], [10, 30])},
                         obs={}, **params)
    assert integrator.gage_dict['20']['Qbar'] == 20
    assert inp.configure_gage_period({'Gage_Statistic_Mode': 'swot_days'}) is None


def test_geometry_filter_cannot_shorten_automatic_period(tmp_path):
    from test_time_str_output import _write_swot_file
    _write_swot_file(tmp_path / '1001_SWOT.nc')
    with Dataset(tmp_path / '1001_SWOT.nc', 'a') as dataset:
        dataset['reach/wse'][:] = [np.nan, 2., np.nan]
    inp = Input(tmp_path, tmp_path, tmp_path, tmp_path,
                {'reach_ids': ['1001']}, 'constrained', False)
    inp.extract_swot()
    assert inp.obs_dict['1001']['nt'] == 1
    assert inp.gage_study_period() == (ordinal('2024-01-01'), ordinal('2024-01-21'))


def test_period_defaults_and_cli_options():
    params = set_moi_params()
    assert params['Gage_Statistic_Mode'] == 'period'
    assert params['Gage_Min_Period_Coverage'] == .2
    result = subprocess.run([sys.executable, str(Path(__file__).parents[1] / 'run_MOI.py'),
                             '--help'], capture_output=True, text=True, check=True)
    assert '--gage-period-start' in result.stdout
    assert '--gage-period-end' in result.stdout
    assert '--gage-min-period-coverage' in result.stdout

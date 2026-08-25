import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp

from moi.Input import Input
from moi.Integrate import Integrate
from moi.sfoi_math_core import adjust_lsq_mult_sparse


def test_default_calval_file_is_shipped_with_moi_module():
    calval_file = Input.default_calval_file()

    assert calval_file == (
        Path(__file__).resolve().parents[1]
        / 'CalValSeparation_basin_stratified_v2.csv'
    )
    assert calval_file.is_file()


def test_augmented_features_require_a_calibration_gage():
    integrator = object.__new__(Integrate)
    integrator.Branch = 'constrained'
    integrator.params_dict = {
        'SFOI_Bias_Augmentation': True,
        'SFOI_Correlation_Enabled': True,
        'SFOI_Bias_Flow_Levels': ('Mean',),
    }

    assert integrator._augmented_solver_features('Mean', 0) == (False, False)
    assert integrator._augmented_solver_features('Mean', 1) == (True, True)
    assert integrator._augmented_solver_features('q33', 1) == (False, False)


def test_augmented_features_require_a_genuine_flpe_row():
    integrator = object.__new__(Integrate)
    integrator.Branch = 'constrained'
    integrator.params_dict = {
        'SFOI_Bias_Augmentation': True,
        'SFOI_Correlation_Enabled': True,
        'SFOI_Bias_Flow_Levels': ('Mean',),
    }

    assert integrator._augmented_solver_features('Mean', 1, 0) == (False, False)
    assert integrator._augmented_solver_features('Mean', 1, 1) == (True, True)


def test_identical_prior_only_algorithms_reuse_one_solver_result(monkeypatch):
    solver_calls = {'count': 0}

    def ordinary_solver(**kwargs):
        solver_calls['count'] += 1
        return SimpleNamespace(
            x=np.array([100.0]),
            P=np.array([1.0]),
            W=np.array([1.0]),
            V=np.array([0.0]),
            So=1.0,
            index=np.array([True]),
            delta=[0.0],
            status='success_converged',
            outer_iterations=1,
            inner_iterations=1,
            diagnostics=[],
            converged=True,
        )

    monkeypatch.setattr('moi.Integrate.adjust_lsq_mult_sparse', ordinary_solver)
    monkeypatch.setattr(
        'moi.Integrate.estimate_q_uncertainty_rel',
        lambda *args, **kwargs: np.array([0.6]),
    )

    integrator = object.__new__(Integrate)
    integrator.alg_dict = {
        alg: {'1': {'integrator': {}}} for alg in ('hivdi', 'sad')
    }
    integrator.basin_dict = {'reach_ids_all': ['1']}
    integrator.params_dict = {}
    integrator.Branch = 'constrained'
    integrator.VerboseFlag = False
    integrator.junctions_valid = True
    integrator.GoodFLPE = {}
    integrator.reach_epsilons = {}
    integrator.gage_diagnostics = {}
    integrator.integ_dict = {
        'bias_correction': {},
        'gage_constraints': {},
        'solver_reuse': {},
    }
    prior_problem = {
        'A_obs': sp.eye(1, format='csr'),
        'L_obs': np.array([100.0]),
        'cov_obs': np.array([0.6]),
        'A_sfoi': sp.eye(1, format='csr'),
        'L_sfoi': np.array([100.0]),
        'cov_sfoi': np.array([0.6]),
        'A_eq': sp.csr_matrix((0, 1)),
        'b_eq': np.array([]),
        'lb': np.array([0.0]),
        'ub': None,
        'x0': np.array([100.0]),
        'K_regions': 0,
        'reach_regions': np.array([0]),
    }
    integrator.initialize_integration_vars = (
        lambda alg, flow, previous, n: (
            np.array([100.0]),
            np.array([60.0]),
            True,
            np.array([1000.0]),
            ['Prior'],
            np.array([False]),
        )
    )
    integrator.build_soft_sfoi_system = (
        lambda n, qbar, sigq, facc, conversion: prior_problem
    )
    integrator.build_gage_observation_rows = lambda problem, flow: {
        'A': sp.csr_matrix((0, 1)),
        'L': np.array([]),
        'cov': np.array([]),
        'reach_ids': [],
        'station_ids': [],
    }
    integrator._print_initial_state = lambda *args, **kwargs: None
    integrator._print_reach_results = lambda *args, **kwargs: None
    integrator.compute_mass_conservation_metrics = lambda *args, **kwargs: {'1': 0.0}
    previous = {
        'hivdi': np.array([np.nan]),
        'sad': np.array([np.nan]),
    }

    integrator.integrator_optimization_calcs(0, 1, 'Mean', previous)

    assert solver_calls['count'] == 1
    assert integrator.integ_dict['solver_reuse']['Mean']['sad'] == 'hivdi'
    assert integrator.alg_dict['hivdi']['1']['integrator']['qbar'] == 100.0
    assert integrator.alg_dict['sad']['1']['integrator']['qbar'] == 100.0


def test_missing_algorithm_data_preserves_prior_provenance():
    integrator = object.__new__(Integrate)
    integrator.alg_dict = {
        'hivdi': {
            '1': {'s1-flpe-exists': True, 'q': np.array([120.0, 120.0])},
            '2': {'s1-flpe-exists': False, 'q': np.nan},
        }
    }
    integrator.sos_dict = {
        '1': {'Qbar': 100.0, 'q33': 90.0},
        '2': {'Qbar': 200.0, 'q33': 180.0},
    }
    integrator.basin_dict = {'reach_ids_all': ['1', '2']}
    integrator.sword_dict = {'reach_id': np.array([1, 2], dtype=np.int64)}
    integrator.params_dict = {
        'FLPE_Uncertainty': 0.67,
        'Prior_Uncertainty': 0.60,
        'Fill_Uncertainty': 1.0,
    }
    integrator.GoodFLPE = {}
    integrator.pull_sword_attributes_for_reach = lambda index: {'facc': 1000.0}
    integrator.get_pre_mean_q()

    qbar, sigq, ok, _, source, real_mask = (
        integrator.initialize_integration_vars(
            'hivdi',
            'Mean',
            {'hivdi': np.array([np.nan, np.nan])},
            2,
        )
    )

    np.testing.assert_allclose(qbar, [120.0, 200.0])
    assert source == ['FLPE', 'Prior']
    np.testing.assert_array_equal(real_mask, [True, False])
    assert sigq[1] == 120.0
    assert ok


def test_filled_array_handles_masked_integer_netcdf_values():
    values = np.ma.array([1001, 0], mask=[False, True], dtype=np.int64)

    np.testing.assert_array_equal(Input._filled_array(values), [1001, 0])


def test_build_gage_rows_keeps_a_separate_observation():
    integrator = object.__new__(Integrate)
    integrator.Branch = 'constrained'
    integrator.params_dict = {'Gage_Uncertainty': 0.10}
    integrator.basin_dict = {'reach_ids_all': ['10', '20']}
    integrator.gage_dict = {
        '20': {
            'Qbar': 125.0,
            'q33': 95.0,
            'station_id': 'test-station',
            'n_matched': 3,
        }
    }
    problem = {'A_obs': sp.csr_matrix((3, 3))}

    mean_gage = integrator.build_gage_observation_rows(problem, 'Mean')
    q33_gage = integrator.build_gage_observation_rows(problem, 'q33')

    assert mean_gage['A'].shape == (1, 3)
    np.testing.assert_allclose(mean_gage['A'].toarray(), [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(mean_gage['L'], [125.0])
    np.testing.assert_allclose(mean_gage['cov'], [0.10])
    np.testing.assert_allclose(q33_gage['A'].toarray(), [[0.0, 1.0, 0.0]])
    np.testing.assert_allclose(q33_gage['L'], [95.0])
    np.testing.assert_allclose(q33_gage['cov'], [0.10])


def test_prepare_gage_constraint_matches_swot_sampling_days():
    integrator = object.__new__(Integrate)
    integrator.sos_dict = {}
    integrator.params_dict = {
        'Gage_Min_Matched_Samples': 2,
        'Gage_Match_SWOT_Days': True,
        'Gage_Allow_Full_Record_Fallback': False,
    }
    epoch = datetime.datetime(2000, 1, 1)
    day_1 = datetime.date(2024, 1, 1).toordinal()
    day_2 = datetime.date(2024, 1, 2).toordinal()
    day_3 = datetime.date(2024, 1, 3).toordinal()
    integrator.obs_dict = {
        '20': {
            't': np.array([
                (datetime.datetime(2024, 1, 1) - epoch).total_seconds(),
                (datetime.datetime(2024, 1, 3) - epoch).total_seconds(),
            ])
        }
    }
    integrator.gage_dict = {
        '20': {'Q': np.array([10.0, 1000.0, 30.0]), 't': np.array([day_1, day_2, day_3])}
    }
    integrator.VerboseFlag = False

    integrator.prepare_gage_constraints()

    assert integrator.gage_dict['20']['n_matched'] == 2
    assert integrator.gage_dict['20']['Qbar'] == 20.0
    assert np.isclose(integrator.gage_dict['20']['q33'], 16.6)


def test_protected_gage_is_not_rejected_and_anchors_solution():
    # One deliberately poor FLPE observation and one precise gage observe the
    # same physical state. The gage is fixed-scale and robust-protected.
    A = sp.csr_matrix([[1.0], [1.0]])
    result = adjust_lsq_mult_sparse(
        A_obs=A,
        L=np.array([200.0, 100.0]),
        cov=np.array([0.50, 0.01]),
        lb=np.array([0.0]),
        x0=np.array([150.0]),
        maxiter=8,
        change_thresh=1.0e-6,
        inner_itermax=6,
        outlier_limit=2.5,
        theta_floor=1.0,
        fixed_weight_mask=np.array([False, True]),
        robust_eligible_mask=np.array([True, False]),
    )

    assert result.index[1]
    assert abs(result.x[0] - 100.0) < abs(result.x[0] - 200.0)


def test_extract_svs_selects_longest_station_record(tmp_path):
    from netCDF4 import Dataset

    svs_path = tmp_path / 'SVS_test.nc'
    with Dataset(svs_path, 'w') as dataset:
        dataset.createDimension('station', 2)
        dataset.createDimension('time', 3)
        dataset.createDimension('num_rchs', 2)
        dataset.createDimension('ymd', 3)

        reach = dataset.createVariable('reach_id_v17b', 'i8', ('station', 'num_rchs'))
        reach[:] = np.array([[74260000101, 0], [74260000101, 0]], dtype=np.int64)

        q = dataset.createVariable('Q', 'f8', ('station', 'time'), fill_value=-9999.0)
        q[:] = np.ma.array(
            [[10.0, -9999.0, -9999.0], [20.0, 21.0, 22.0]],
            mask=[[False, True, True], [False, False, False]],
        )

        ymd = dataset.createVariable('date_ymd', 'i4', ('ymd', 'time'))
        ymd[:] = np.array([[2024, 2024, 2024], [1, 1, 1], [1, 2, 3]])

    input_obj = Input(
        alg_dir=Path('.'),
        sos_dir=Path('.'),
        swot_dir=Path('.'),
        sword_dir=Path('.'),
        basin_data={'reach_ids_all': ['74260000101']},
        branch='constrained',
        verbose=False,
    )
    gages = input_obj.extract_svs(svs_path)

    assert set(gages) == {'74260000101'}
    np.testing.assert_allclose(gages['74260000101']['Q'], [20.0, 21.0, 22.0])
    assert gages['74260000101']['t'][0] == datetime.date(2024, 1, 1).toordinal()


def test_extract_svs_uses_only_calibration_group(tmp_path):
    from netCDF4 import Dataset

    svs_path = tmp_path / 'SVS_test.nc'
    with Dataset(svs_path, 'w') as dataset:
        dataset.createDimension('station', 2)
        dataset.createDimension('time', 2)
        dataset.createDimension('num_rchs', 1)
        dataset.createDimension('ymd', 3)

        reach = dataset.createVariable('reach_id_v17b', 'i8', ('station', 'num_rchs'))
        reach[:] = np.array([[1001], [2002]], dtype=np.int64)

        q = dataset.createVariable('Q', 'f8', ('station', 'time'))
        q[:] = np.array([[10.0, 12.0], [20.0, 22.0]])

        ymd = dataset.createVariable('date_ymd', 'i4', ('ymd', 'time'))
        ymd[:] = np.array([[2024, 2024], [1, 1], [1, 2]])

    calval_path = tmp_path / 'CalValSeparation.csv'
    calval_path.write_text(
        'reach_id_v17b,group\n'
        '1001,calibration\n'
        '2002,validation\n',
        encoding='utf-8',
    )

    input_obj = Input(
        alg_dir=Path('.'),
        sos_dir=Path('.'),
        swot_dir=Path('.'),
        sword_dir=Path('.'),
        basin_data={'reach_ids_all': ['1001', '2002']},
        branch='constrained',
        verbose=False,
    )
    gages = input_obj.extract_svs(svs_path, calval_file=calval_path)

    assert set(gages) == {'1001'}
    assert gages['1001']['group'] == 'calibration'


def test_explicit_svs_selection_does_not_refill_validation_gages_from_sos():
    integrator = object.__new__(Integrate)
    integrator._use_sos_gage_fallback = False
    integrator.sos_dict = {
        '2002': {
            'gage': {
                'source': 'SoS',
                't': np.array([datetime.date(2024, 1, 1).toordinal()]),
                'Q': np.array([20.0]),
            }
        }
    }
    integrator.gage_dict = {
        '1001': {
            'source': 'SVS',
            'group': 'calibration',
            't': np.array([datetime.date(2024, 1, 1).toordinal()]),
            'Q': np.array([10.0]),
        }
    }
    integrator.obs_dict = {}
    integrator.params_dict = {
        'Gage_Match_SWOT_Days': False,
        'Gage_Min_Matched_Samples': 1,
    }
    integrator.VerboseFlag = False

    integrator.prepare_gage_constraints()

    assert set(integrator.gage_dict) == {'1001'}

from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from moi.Output import Output


def test_bias_correlation_output_records_convergence(tmp_path):
    diagnostic = {
        'estimated_bias_fraction': 0.25,
        'bias_std_fraction': 0.05,
        'correlation_rho': 0.20,
        'last_delta': 8.0e-6,
        'last_physical_rms_delta': 7.0e-4,
        'last_physical_p95_delta': 1.2e-3,
        'last_raw_delta': 9.0e-4,
        'last_robust_delta': 2.0e-4,
        'final_So': 1.1,
        'enabled': True,
        'converged': True,
        'outer_iterations': 18,
        'status': 'success_bias_correlated_converged_success_osqp_optimal',
        'correlation_effects': [0.1, -0.1],
        'n_real_flpe_rows': 12,
        'oscillation_events': 2,
        'relaxation_recoveries': 1,
        'convergence_thresholds': {
            'physical_rms': 1.0e-2,
            'physical_p95': 2.0e-2,
            'bias': 1.0e-3,
            'effect': 1.0e-2,
            'robust': 1.0e-3,
        },
    }
    writer = Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={'bias_correction': {'metroman': {'Mean': diagnostic}}},
        alg_dict={},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
    )
    output_path = tmp_path / 'diagnostics.nc'

    with Dataset(output_path, 'w') as dataset:
        writer._write_bias_correlation_diagnostics(dataset)

    with Dataset(output_path, 'r') as dataset:
        group = dataset.groups['moi_bias_correlation'].groups['metroman']
        assert group.getncattr('mean_converged') == 1
        assert group.getncattr('mean_outer_iterations') == 18
        assert '_converged_' in group.getncattr('mean_solver_status')
        assert np.isclose(group.variables['mean_last_delta'].getValue(), 8.0e-6)
        assert np.isclose(
            group.variables['mean_last_physical_p95_delta'].getValue(),
            1.2e-3,
        )
        assert np.isclose(
            group.variables['mean_final_reduced_chi_square'].getValue(), 1.1
        )
        assert group.getncattr('mean_n_real_flpe_rows') == 12
        assert group.getncattr('mean_oscillation_events') == 2
        assert np.isclose(
            group.getncattr('mean_physical_rms_threshold'), 1.0e-2
        )


def test_gage_output_reports_presence_and_calval_group(tmp_path):
    writer = Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
        gage_groups={'1001': 'calibration', '2002': 'validation'},
    )

    output_path = tmp_path / 'gage_metadata.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '1001')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 1
        # Catalogued but not handed to the integrator: nothing constrained it.
        assert gage.getncattr('constraint_source') == 'none'
        assert gage.getncattr('group') == 'calibration'

    output_path = tmp_path / 'validation_gage_metadata.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '2002')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 1
        assert gage.getncattr('group') == 'validation'

    output_path = tmp_path / 'no_gage_metadata.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '3003')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 0
        assert gage.getncattr('has_corridors') == 0
        assert gage.getncattr('constraint_source') == 'none'
        assert gage.getncattr('group') == 'none'


def test_corridors_pseudo_gage_is_not_reported_as_a_gage(tmp_path):
    """A CORRIDORS reach is constrained, but it is not a gaged reach.

    merge_corridors_and_gages folds the pseudo-gages into gage_dict, so without
    the corridors_reaches set they would be written out as has_gage=1.
    """
    writer = Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
        gage_groups={'1001': 'calibration'},
        gage_dict={
            '1001': {'source': 'SVS', 'group': 'calibration'},
            '4004': {'source': 'corridors', 'station_id': None},
        },
        corridors_reaches={'4004'},
    )

    output_path = tmp_path / 'corridors_metadata.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '4004')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 0
        assert gage.getncattr('has_corridors') == 1
        assert gage.getncattr('constraint_source') == 'corridors'
        # 'group' is a Cal/Val label for real stations only.
        assert gage.getncattr('group') == 'none'

    output_path = tmp_path / 'real_gage_metadata.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '1001')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 1
        assert gage.getncattr('has_corridors') == 0
        assert gage.getncattr('constraint_source') == 'gage'


def test_validation_reach_standing_in_on_corridors(tmp_path):
    """A withheld validation gage is catalogued but did not constrain anything.

    has_gage and constraint_source answer different questions, and this reach
    is where they diverge: it is in the gage catalogue, its gage was
    deliberately kept out of constrained integration, and a CORRIDORS
    pseudo-gage stood in for it.  Scoring it as gage-constrained would leak the
    validation gage into the assessment.
    """
    writer = Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
        gage_groups={'2002': 'validation', '3003': 'validation'},
        # The validation gages are absent from gage_dict, as intended; only the
        # pseudo-gage for 2002 was actually available to the integrator.
        gage_dict={'2002': {'source': 'corridors'}},
        corridors_reaches={'2002'},
    )

    output_path = tmp_path / 'validation_with_corridors.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '2002')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 1
        assert gage.getncattr('has_corridors') == 1
        assert gage.getncattr('constraint_source') == 'corridors'
        assert gage.getncattr('group') == 'validation'

    # The other validation reach had no stand-in at all.
    output_path = tmp_path / 'validation_unconstrained.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '3003')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 1
        assert gage.getncattr('has_corridors') == 0
        assert gage.getncattr('constraint_source') == 'none'
        assert gage.getncattr('group') == 'validation'


def test_reach_with_both_reports_both_but_credits_the_gage(tmp_path):
    """A reach can have CORRIDORS data and still be constrained by its gage.

    merge_corridors_and_gages records availability for the pseudo-gage even
    when the real station wins, so has_corridors must say the data exists while
    constraint_source credits what actually entered the solver.
    """
    writer = Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
        gage_groups={'1001': 'calibration'},
        gage_dict={'1001': {'source': 'SVS', 'group': 'calibration'}},
        # A pseudo-gage was built for 1001 but deferred to the station.
        corridors_reaches={'1001'},
    )

    output_path = tmp_path / 'both.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '1001')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 1
        assert gage.getncattr('has_corridors') == 1
        assert gage.getncattr('constraint_source') == 'gage'
        assert gage.getncattr('group') == 'calibration'


def test_corridors_source_is_honoured_without_the_explicit_set(tmp_path):
    """The entry's own 'source' is enough when corridors_reaches is not given."""
    writer = Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
        gage_dict={'4004': {'source': 'corridors'}},
    )

    output_path = tmp_path / 'inferred_corridors.nc'
    with Dataset(output_path, 'w') as dataset:
        writer._write_gage_metadata(dataset, '4004')

    with Dataset(output_path, 'r') as dataset:
        gage = dataset.groups['gage']
        assert gage.getncattr('has_gage') == 0
        assert gage.getncattr('constraint_source') == 'corridors'

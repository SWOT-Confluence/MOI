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

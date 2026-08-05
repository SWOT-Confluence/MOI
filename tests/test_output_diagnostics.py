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
        'enabled': True,
        'converged': True,
        'outer_iterations': 18,
        'status': 'success_bias_correlated_converged_success_osqp_optimal',
        'correlation_effects': [0.1, -0.1],
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

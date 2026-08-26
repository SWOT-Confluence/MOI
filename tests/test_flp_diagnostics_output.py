"""The FLP fit diagnostics have to survive the trip to disk.

Recording a fit status in memory and not writing it out is the failure this
guards against -- exactly what happened to ``qbar_source`` and ``q33_source``,
which the solve tracked and used but Output.py never wrote, leaving consumers
unable to tell a real FLPE estimate from a prior reconstruction.
"""

from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

from moi.Output import Output

FILL = -999999999999


def make_writer(tmp_path, integrator):
    return Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={'sic4dvar': {'12345': {'integrator': integrator}}},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
    )


def test_diagnostics_round_trip(tmp_path):
    integrator = {
        'flp_fit_nrmse': 0.183,
        'flp_fit_nrmse_log': 0.211,
        'flp_fit_status': 'good',
        'flp_fit_status_code': 0,
        'flp_param_at_bound': 0b10,
        'flp_n_valid_obs': 37,
        'dA_valid_frac': 0.92,
        'rescale_factor': 1.47,
        'rescale_b': 1.0,
        'rescale_method': 'mean',
        'flp_fit_method': 'varpro',
        'q_source': 'rescaled',
        'qbar_source': 'FLPE',
        'q33_source': 'Prior',
    }
    writer = make_writer(tmp_path, integrator)

    path = tmp_path / 'diag.nc'
    with Dataset(path, 'w', format='NETCDF4') as out:
        group = out.createGroup('sic4dvar')
        writer._write_flp_diagnostics(group, 'sic4dvar', '12345', FILL)

    with Dataset(path) as ds:
        group = ds['sic4dvar']
        assert float(group['flp_fit_nrmse'][:]) == pytest.approx(0.183)
        assert float(group['flp_fit_nrmse_log'][:]) == pytest.approx(0.211)
        assert int(group['flp_fit_status_code'][:]) == 0
        assert int(group['flp_param_at_bound'][:]) == 2
        assert int(group['flp_n_valid_obs'][:]) == 37
        assert float(group['dA_valid_frac'][:]) == pytest.approx(0.92)
        assert float(group['rescale_factor'][:]) == pytest.approx(1.47)
        assert group.flp_fit_status == 'good'
        assert group.flp_fit_method == 'varpro'
        assert group.rescale_method == 'mean'
        assert group.hydrograph_source == 'rescaled'
        # The provenance gap the canary run identified.
        assert group.qbar_source == 'FLPE'
        assert group.q33_source == 'Prior'


def test_missing_diagnostics_become_fill_not_a_crash(tmp_path):
    """A reach the fitter never reached still has to produce a valid file."""
    writer = make_writer(tmp_path, {})

    path = tmp_path / 'empty.nc'
    with Dataset(path, 'w', format='NETCDF4') as out:
        group = out.createGroup('sic4dvar')
        writer._write_flp_diagnostics(group, 'sic4dvar', '12345', FILL)

    with Dataset(path) as ds:
        group = ds['sic4dvar']
        assert group['flp_fit_nrmse'][:].mask.all()
        assert group.flp_fit_status == ''
        assert group.qbar_source == ''

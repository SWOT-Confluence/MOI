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

from moi import flp_fit
from moi.Output import Output, _as_row

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


# ---------------------------------------------------------------------------
# Re-inserting the dropped timesteps
#
# Regression tests for the AxisError on basin 1254: np.insert(q, iInsert,
# fillvalue, 1) assumed q was (1, nt) and raised on a 1-D array, taking
# write_output down for every reach and every algorithm in the basin.
# ---------------------------------------------------------------------------

def make_expander(tmp_path, q, nt_full):
    writer = Output(
        basin_dict={},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={'sad': {'12345': {'integrator': {'q': q}}}},
        obs_dict={'12345': {'nt': nt_full}},
        sword_dir=Path(tmp_path),
        params_dict={},
    )
    return writer


@pytest.mark.parametrize('q', [
    np.arange(5, dtype=float),                    # 1-D: the shape that crashed
    np.arange(5, dtype=float).reshape(1, 5),      # (1, nt): the shape assumed
])
def test_expand_accepts_either_shape(tmp_path, q):
    # 5 valid timesteps, 2 dropped -> a 7-step record.
    writer = make_expander(tmp_path, q, nt_full=7)
    iInsert = np.array([2, 4])

    out = writer._expand_q_to_full_record('sad', '12345', iInsert, 2, FILL)

    assert out.shape == (1, 7)
    assert out[0, 2] == FILL and out[0, 5] == FILL
    assert writer.q_shape_repairs == []


def test_expand_repairs_a_short_series_instead_of_raising(tmp_path):
    writer = make_expander(tmp_path, np.arange(3, dtype=float), nt_full=7)

    with pytest.warns(UserWarning):
        out = writer._expand_q_to_full_record('sad', '12345', np.array([2, 4]), 2, FILL)

    assert out.shape == (1, 7)
    assert len(writer.q_shape_repairs) == 1
    assert writer.q_shape_repairs[0]['size'] == 3
    assert writer.q_shape_repairs[0]['expected'] == 5


def test_expand_handles_a_missing_hydrograph(tmp_path):
    writer = make_expander(tmp_path, None, nt_full=6)

    with pytest.warns(UserWarning):
        out = writer._expand_q_to_full_record('sad', '12345', np.array([0]), 1, FILL)

    assert out.shape == (1, 6)
    assert len(writer.q_shape_repairs) == 1


def test_expand_with_nothing_dropped_is_a_passthrough(tmp_path):
    writer = make_expander(tmp_path, np.arange(4, dtype=float), nt_full=4)

    out = writer._expand_q_to_full_record('sad', '12345', np.array([], dtype=int), 0, FILL)

    assert out.shape == (1, 4)
    np.testing.assert_array_equal(out[0], np.arange(4, dtype=float))


def test_expand_zero_valid_timesteps_has_no_off_by_one(tmp_path):
    """All geometry-filtered timesteps must round-trip to the full record."""
    writer = make_expander(tmp_path, np.array([], dtype=float), nt_full=2)

    out = writer._expand_q_to_full_record(
        'sad', '12345', np.array([0, 0]), 2, FILL)

    assert out.shape == (1, 2)
    np.testing.assert_array_equal(out[0], np.array([FILL, FILL]))
    assert writer.q_shape_repairs == []


def test_output_row_helper_does_not_drift_from_flp_fit():
    """Output._as_row is duplicated from flp_fit.as_row on purpose.

    Output.py is the last-mile stage: a failure to import or resolve anything
    here costs an entire basin every output file for every algorithm. It had no
    dependency on moi.flp_fit before, and adding one so five lines of numpy
    could be shared created a new way for a partially-deployed tree to fail --
    which is how commit 423d9d0 failed, with Output.py updated and flp_fit.py
    left behind.

    Duplication is only acceptable if it cannot drift. This is that guarantee.
    """
    cases = [
        (None, 5),
        (np.arange(5, dtype=float), 5),
        (np.arange(5, dtype=float).reshape(1, 5), 5),
        (np.arange(9, dtype=float), 4),      # too long -> truncated
        (np.arange(2, dtype=float), 4),      # too short -> padded
        (np.inf, 4),                         # momma's degenerate bankfull
        (np.array([]), 3),
        (7.5, 1),
        (np.array([], dtype=float), 0),      # no valid geometry timesteps
        ('not-a-number', 3),                 # malformed value -> fill series
    ]
    for q, nt in cases:
        mine = _as_row(q, nt)
        theirs = flp_fit.as_row(q, nt)
        assert mine.shape == theirs.shape, (q, nt)
        np.testing.assert_array_equal(mine, theirs)

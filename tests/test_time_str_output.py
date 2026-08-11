from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from moi.Input import Input
from moi.Output import Output


TIME_STRINGS = np.asarray(
    [
        '2024-01-01T00:00:00Z',
        '2024-01-11T00:00:00Z',
        '2024-01-21T00:00:00Z',
    ],
    dtype=str,
)


def _write_swot_file(path):
    with Dataset(path, 'w') as dataset:
        dataset.createDimension('nt', 3)
        dataset.createDimension('num_time_chars', 20)
        reach = dataset.createGroup('reach')

        for name in ('wse', 'width', 'slope2', 'd_x_area'):
            reach.createVariable(name, 'f8', ('nt',))[:] = [1.0, 2.0, 3.0]
        reach['wse'][1] = np.nan
        reach.createVariable('time', 'f8', ('nt',))[:] = [
            757382400.0,
            758246400.0,
            759110400.0,
        ]
        reach.createVariable('reach_q', 'i4', ('nt',))[:] = 0
        reach.createVariable('xovr_cal_q', 'i4', ('nt',))[:] = 0

        chars = np.full((3, 20), b' ', dtype='S1')
        for index, value in enumerate(TIME_STRINGS):
            chars[index] = np.frombuffer(value.encode('utf-8'), dtype='S1')
        reach.createVariable(
            'time_str',
            'S1',
            ('nt', 'num_time_chars'),
        )[:] = chars


def _algorithm_data(reach):
    q = np.asarray([[10.0, 30.0]])
    common = {'q': q.copy(), 'qbar': 20.0, 'q33': 15.0, 'sbQ_rel': 0.2}
    return {
        'busboi': {
            reach: {
                'qbar': 20.0,
                'integrator': {**common, 'n': 0.03, 'a0': 10.0},
            }
        },
        'hivdi': {
            reach: {
                'qbar': 20.0,
                'integrator': {
                    **common,
                    'Abar': 10.0,
                    'alpha': 1.0,
                    'beta': 2.0,
                },
            }
        },
        'metroman': {
            reach: {
                'qbar': 20.0,
                'integrator': {**common, 'a0': 10.0, 'na': 0.03, 'x1': 0.1},
            }
        },
        'momma': {
            reach: {
                'qbar': 20.0,
                'integrator': {**common, 'B': 1.0, 'H': 2.0, 'Save': 3.0},
            }
        },
        'sad': {
            reach: {
                'qbar': 20.0,
                'integrator': {**common, 'n': 0.03, 'a0': 10.0},
            }
        },
        'sic4dvar': {
            reach: {
                'qbar': 20.0,
                'integrator': {**common, 'n': 0.03, 'a0': 10.0},
            }
        },
    }


def test_extract_swot_keeps_full_time_strings_across_qc_deletions(tmp_path):
    reach = '1001'
    _write_swot_file(tmp_path / f'{reach}_SWOT.nc')
    input_obj = Input(
        alg_dir=Path('.'),
        sos_dir=Path('.'),
        swot_dir=tmp_path,
        sword_dir=Path('.'),
        basin_data={'reach_ids': [reach]},
        branch='constrained',
        verbose=False,
    )

    input_obj.extract_swot()

    assert input_obj.obs_dict[reach]['nt'] == 2
    np.testing.assert_array_equal(
        input_obj.obs_dict[reach]['time_str'],
        TIME_STRINGS,
    )
    assert input_obj.obs_dict[reach]['time_str_source'] == 'SWOT reach/time_str'


def test_output_writes_root_time_str_aligned_with_restored_q_axis(tmp_path):
    reach = '1001'
    obs_dict = {
        reach: {
            'nt': 2,
            'iDelete': (np.asarray([1]),),
            'time_str': TIME_STRINGS.copy(),
            'time_str_source': 'SWOT reach/time_str',
        }
    }
    writer = Output(
        basin_dict={'reach_ids': [reach], 'reach_ids_all': [reach]},
        out_dir=tmp_path,
        integ_dict={},
        alg_dict=_algorithm_data(reach),
        obs_dict=obs_dict,
        sword_dir=tmp_path,
        params_dict={'write_fill_only': False},
    )

    writer.write_output()

    with Dataset(tmp_path / f'{reach}_integrator.nc', 'r') as dataset:
        assert len(dataset.dimensions['nt']) == 3
        assert dataset.variables['time_str'][:].tolist() == TIME_STRINGS.tolist()
        assert dataset.variables['time_str'].source == 'SWOT reach/time_str'
        q = dataset.groups['metroman'].variables['q'][:]
        np.testing.assert_allclose(q[[0, 2]], [10.0, 30.0])
        assert np.ma.getmaskarray(q)[1]

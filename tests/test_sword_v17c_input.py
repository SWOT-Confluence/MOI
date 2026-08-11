from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from moi.Input import Input


def test_extract_sword_reads_v17c_facc_and_quality(tmp_path):
    sword_path = tmp_path / 'na_sword_v17c.nc'
    with Dataset(sword_path, 'w') as dataset:
        reaches = dataset.createGroup('reaches')
        reaches.createDimension('num_domains', 1)
        reaches.createDimension('num_reaches', 2)
        reaches.createDimension('num_neighbors', 4)
        reaches.createDimension('num_orbits', 1)

        reaches.createVariable('reach_id', 'i8', ('num_reaches',))[:] = [100, 110]
        reaches.createVariable('facc', 'f8', ('num_reaches',))[:] = [1000.0, 600.0]
        quality = reaches.createVariable(
            'facc_quality',
            'i4',
            ('num_reaches',),
            fill_value=-9999,
        )
        quality[:] = np.ma.array([-9999, 1], mask=[True, False])
        reaches.createVariable('n_rch_up', 'i4', ('num_reaches',))[:] = [0, 1]
        reaches.createVariable('n_rch_down', 'i4', ('num_reaches',))[:] = [1, 0]
        reaches.createVariable(
            'rch_id_up',
            'i8',
            ('num_neighbors', 'num_reaches'),
        )[:] = np.array(
            [[0, 100], [0, 0], [0, 0], [0, 0]],
            dtype=np.int64,
        )
        reaches.createVariable(
            'rch_id_dn',
            'i8',
            ('num_neighbors', 'num_reaches'),
        )[:] = np.array(
            [[110, 0], [0, 0], [0, 0], [0, 0]],
            dtype=np.int64,
        )
        reaches.createVariable('swot_obs', 'i4', ('num_reaches',))[:] = [1, 1]
        reaches.createVariable('width', 'f8', ('num_reaches',))[:] = [100.0, 60.0]

    input_obj = Input(
        alg_dir=Path('.'),
        sos_dir=Path('.'),
        swot_dir=Path('.'),
        sword_dir=tmp_path,
        basin_data={'sword': sword_path.name},
        branch='constrained',
        verbose=False,
    )

    input_obj.extract_sword()

    np.testing.assert_allclose(input_obj.sword_dict['facc'], [1000.0, 600.0])
    np.testing.assert_array_equal(
        input_obj.sword_dict['facc_quality'],
        [-9999, 1],
    )
    assert input_obj.sword_dict['facc'].dtype.kind == 'f'


def test_extract_sword_keeps_older_versions_without_facc_quality(tmp_path):
    sword_path = tmp_path / 'na_sword_v17b.nc'
    with Dataset(sword_path, 'w') as dataset:
        reaches = dataset.createGroup('reaches')
        reaches.createDimension('num_domains', 1)
        reaches.createDimension('num_reaches', 1)
        reaches.createDimension('num_neighbors', 4)
        reaches.createDimension('num_orbits', 1)

        reaches.createVariable('reach_id', 'i8', ('num_reaches',))[:] = [100]
        reaches.createVariable('facc', 'f8', ('num_reaches',))[:] = [1000.0]
        reaches.createVariable('n_rch_up', 'i4', ('num_reaches',))[:] = [0]
        reaches.createVariable('n_rch_down', 'i4', ('num_reaches',))[:] = [0]
        reaches.createVariable(
            'rch_id_up',
            'i8',
            ('num_neighbors', 'num_reaches'),
        )[:] = 0
        reaches.createVariable(
            'rch_id_dn',
            'i8',
            ('num_neighbors', 'num_reaches'),
        )[:] = 0
        reaches.createVariable('swot_obs', 'i4', ('num_reaches',))[:] = [1]

    input_obj = Input(
        alg_dir=Path('.'),
        sos_dir=Path('.'),
        swot_dir=Path('.'),
        sword_dir=tmp_path,
        basin_data={'sword': sword_path.name},
        branch='constrained',
        verbose=False,
    )

    input_obj.extract_sword()

    assert 'facc_quality' not in input_obj.sword_dict

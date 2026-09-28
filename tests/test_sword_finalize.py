import json

from netCDF4 import Dataset
import numpy as np
import pytest

from moi import flowlaw_preprocess, sword_finalize

BRANCHES = ('constrained', 'unconstrained')


def _make_sword(path, reach_ids=(1001, 1002, 1003)):
    with Dataset(path, 'w') as ds:
        reaches = ds.createGroup('reaches')
        reaches.createDimension('num_reaches', len(reach_ids))
        reaches.createVariable('reach_id', 'i8', ('num_reaches',))[:] = reach_ids
        models = reaches.createGroup('discharge_models')
        for branch in BRANCHES:
            group = models.createGroup(branch)
            for model, variables in sword_finalize.PARAMETERS.values():
                sub = group.createGroup(model)
                for name in variables.values():
                    sub.createVariable(name, 'f8', ('num_reaches',))[:] = -1.0


def _make_integrator(path, value, drop_group=None, masked=(), nan=(),
                     preprocess=None):
    preprocess = ({
        'flowlaw_preprocess_version': flowlaw_preprocess.PREPROCESS_VERSION,
        'flowlaw_observation_filter': flowlaw_preprocess.FILTER_ROBUST_V2,
        'flowlaw_slope_policy': flowlaw_preprocess.SLOPE_L2_3P4,
    } if preprocess is None else preprocess)
    with Dataset(path, 'w') as ds:
        for name, setting in preprocess.items():
            ds.setncattr(name, setting)
        for group_name, (_model, variables) in sword_finalize.PARAMETERS.items():
            if group_name == drop_group:
                continue
            group = ds.createGroup(group_name)
            group.createVariable('flp_exported', 'i8')[...] = 0 if group_name in masked else 1
            for offset, name in enumerate(variables):
                blank = group_name in masked and name in sword_finalize._FLP[group_name]
                group.createVariable(name, 'f8')[...] = (
                    np.nan if blank or (group_name, name) in nan else value + offset)


@pytest.fixture
def layout(tmp_path):
    source = tmp_path / 'input' / 'na_sword_v17.nc'
    source.parent.mkdir()
    _make_sword(source)
    dirs = {branch: tmp_path / branch for branch in BRANCHES}
    for directory in dirs.values():
        directory.mkdir()
    basin_json = tmp_path / 'basin.json'
    basin_json.write_text(json.dumps([
        {'basin_id': 1, 'reach_id': ['1001'], 'sword': 'na_sword_v17.nc'},
        {'basin_id': 2, 'reach_id': ['1002'], 'sword': 'na_sword_v17.nc'},
        {'basin_id': 3, 'reach_id': ['5005'], 'sword': 'eu_sword_v17.nc'},
    ]))
    output = tmp_path / 'output' / 'na_sword_v17_moi.nc'
    return source, output, basin_json, dirs


def _fill(dirs, **kwargs):
    for offset, branch in ((0, 'constrained'), (100, 'unconstrained')):
        _make_integrator(dirs[branch] / '1001_integrator.nc', 10.0 + offset, **kwargs)
        _make_integrator(dirs[branch] / '1002_integrator.nc', 20.0 + offset)


def _abar(path, branch):
    with Dataset(path) as ds:
        return ds['reaches']['discharge_models'][branch]['BAM']['Abar'][:]


def test_both_branches_are_written_in_one_pass(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs)
    report = sword_finalize.finalize(source, output, basin_json, dirs)

    assert report['published'] and report['problems'] == []
    np.testing.assert_allclose(_abar(output, 'constrained'), [10.0, 20.0, -1.0])
    np.testing.assert_allclose(_abar(output, 'unconstrained'), [110.0, 120.0, -1.0])
    np.testing.assert_allclose(_abar(source, 'constrained'), [-1.0] * 3)
    with Dataset(output) as ds:
        for branch in BRANCHES:
            group = ds['reaches']['discharge_models'][branch]
            assert group.flowlaw_preprocess_version == \
                flowlaw_preprocess.PREPROCESS_VERSION
            assert group.flowlaw_observation_filter == 'robust_v2'
            assert group.flowlaw_slope_policy == 'l2_3p4'
    assert sorted(p.name for p in output.parent.iterdir()) == [output.name]


def test_mixed_preprocessing_config_blocks_publication(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs)
    config = {
        'flowlaw_preprocess_version': flowlaw_preprocess.PREPROCESS_VERSION,
        'flowlaw_observation_filter': flowlaw_preprocess.FILTER_LEGACY,
        'flowlaw_slope_policy': flowlaw_preprocess.SLOPE_L2_3P4,
    }
    _make_integrator(dirs['constrained'] / '1002_integrator.nc', 20.0,
                     preprocess=config)

    report = sword_finalize.finalize(source, output, basin_json, dirs)
    assert not report['published'] and not output.exists()
    assert any('preprocessing config differs' in p for p in report['problems'])


def test_missing_preprocessing_config_blocks_publication(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs)
    _make_integrator(dirs['constrained'] / '1001_integrator.nc', 10.0,
                     preprocess={})

    report = sword_finalize.finalize(source, output, basin_json, dirs)
    assert not report['published'] and not output.exists()
    assert any('flowlaw_preprocess_version' in p for p in report['problems'])


def test_missing_integrator_file_blocks_publication(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs)
    (dirs['unconstrained'] / '1002_integrator.nc').unlink()

    report = sword_finalize.finalize(source, output, basin_json, dirs)
    assert not report['published'] and not output.exists()
    assert report['missing_integrator_files'] == ['unconstrained/1002']

    report = sword_finalize.finalize(source, output, basin_json, dirs,
                                     allow_missing=True)
    assert report['published']
    np.testing.assert_allclose(_abar(output, 'unconstrained'), [110.0, -1.0, -1.0])


def test_failed_validation_leaves_an_existing_output_untouched(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs)
    sword_finalize.finalize(source, output, basin_json, dirs)
    before = output.read_bytes()

    _fill(dirs, drop_group='sad')          # 1001 now lacks a group
    report = sword_finalize.finalize(source, output, basin_json, dirs)
    assert not report['published']
    assert any('no sad group' in p for p in report['problems'])
    assert output.read_bytes() == before


def test_masked_parameters_are_accepted_unmasked_nan_is_not(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs, masked=('busboi',))
    report = sword_finalize.finalize(source, output, basin_json, dirs)
    assert report['published']
    assert np.isnan(_abar(output, 'constrained')[0])

    _fill(dirs, nan=(('busboi', 'a0'),))
    report = sword_finalize.finalize(source, output, basin_json, dirs)
    assert not report['published']
    assert any('busboi/a0 is not finite and was not masked' in p
               for p in report['problems'])


def test_nan_uncertainty_is_only_a_warning(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs, nan=(('momma', 'Save'),))
    report = sword_finalize.finalize(source, output, basin_json, dirs)
    assert report['published']
    assert any('momma/Save' in w for w in report['warnings'])


def test_source_is_never_the_output(layout):
    source, _output, basin_json, dirs = layout
    with pytest.raises(ValueError):
        sword_finalize.finalize(source, source, basin_json, dirs)


def test_cli_exit_codes(layout):
    source, output, basin_json, dirs = layout
    args = ['--source-sword', str(source), '--output-sword', str(output),
            '--basin-json', str(basin_json),
            '--constrained-integrator-dir', str(dirs['constrained'])]
    assert sword_finalize.main(args) == 2          # no integrator files yet
    _fill(dirs)
    assert sword_finalize.main(args) == 0
    assert output.is_file()


def test_version_and_attribute_names_match_the_shared_module():
    # sword_finalize.py is standalone (no moi imports), so it repeats these.
    from moi import flowlaw_preprocess
    assert sword_finalize.PREPROCESS_VERSION == flowlaw_preprocess.PREPROCESS_VERSION
    assert set(sword_finalize.PREPROCESS_ATTRIBUTES) == set(
        flowlaw_preprocess.CONFIG_ATTRIBUTES.values())


def test_branch_without_any_integrator_file_blocks_even_with_allow_missing(layout):
    source, output, basin_json, dirs = layout
    _fill(dirs)
    for path in dirs['unconstrained'].iterdir():
        path.unlink()
    report = sword_finalize.finalize(source, output, basin_json, dirs,
                                     allow_missing=True)
    assert not report['published'] and not output.exists()
    assert any('unconstrained: no usable integrator file' in p
               for p in report['problems'])


def test_per_basin_sword_write_records_the_preprocessing_config(tmp_path):
    from moi import flowlaw_preprocess
    from moi.Output import Output

    _make_sword(tmp_path / 'na_sword_v17.nc')
    out_dir = tmp_path / 'out'
    out_dir.mkdir()
    Output(basin_dict={'sword': 'na_sword_v17.nc', 'reach_ids': []},
           out_dir=out_dir, integ_dict={}, alg_dict={}, obs_dict={},
           sword_dir=tmp_path,
           params_dict={'Slope_Policy': flowlaw_preprocess.SLOPE_L2_3P4},
           ).write_sword_output('constrained')

    with Dataset(out_dir / 'na_sword_v17_moi.nc') as ds:
        branch = ds['reaches']['discharge_models']['constrained']
        assert flowlaw_preprocess.read_config(branch) == {
            'observation_filter': flowlaw_preprocess.DEFAULT_OBSERVATION_FILTER,
            'slope_policy': flowlaw_preprocess.SLOPE_L2_3P4}

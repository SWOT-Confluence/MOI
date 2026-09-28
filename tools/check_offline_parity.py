#!/usr/bin/env python3
"""Check that offline reconstructs exactly what MOI exports.

Runs the real code of both repositories on synthetic SWOT files.

Part 1 -- flow-law parity, all six algorithms and every combination of
observation filter x slope policy (the options travel to offline through the
integrator file attributes, as in production):
    MOI  Input.extract_swot + Integrate.*_flowlaw with fixed parameters
    vs   offline run_offline.main (flp_source=integrator)

Part 2 -- round trip:
    MOI  Input.extract_swot -> Integrate.compute_FLPs -> Output.write_output
    ->   offline run_offline.main on the written *_integrator.nc
    and  q_flowlaw in the integrator file == offline discharge.  One reach has
    sign-reversed dA, so its dA-driven FLPs must be masked and produce no L2.

Every comparison requires identical valid-timestep masks and a maximum
relative difference below 1e-9.

Usage:
    python3 tools/check_offline_parity.py --offline-repo ../offline-NRTfilter
"""

import argparse
import contextlib
import importlib
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import types

from netCDF4 import Dataset
import numpy as np

MOI_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MOI_REPO))

from moi import flowlaw_preprocess as fp  # noqa: E402
from moi import flp_fit  # noqa: E402
from moi.Input import Input  # noqa: E402
from moi.Integrate import Integrate  # noqa: E402
from moi.Output import Output  # noqa: E402

FILL = -999999999999.
# MOI group -> (offline variable, MOI flow law, integrator parameter variables)
ALGORITHMS = {
    'busboi': ('dschg_gb', 'bam_flowlaw', ('n', 'a0')),
    'hivdi': ('dschg_gh', 'hivdi_flowlaw', ('alpha', 'beta', 'Abar')),
    'metroman': ('dschg_gm', 'metroman_flowlaw', ('na', 'x1', 'Abar')),
    'momma': ('dschg_go', 'momma_flowlaw', ('B', 'H')),
    'sad': ('dschg_gs', 'sad_flowlaw', ('n', 'a0')),
    'sic4dvar': ('dschg_gi', 'sic4dvar_flowlaw', ('n', 'a0')),
}
NT = 60


# ---------------------------------------------------------------------------
# synthetic inputs
# ---------------------------------------------------------------------------
def make_swot(path, reach_id, seed, hwfit, gaps=False, reverse_da=False):
    """SWOT series with quality flags, outliers, gaps and bad slopes."""
    rng = np.random.default_rng(seed)
    h = 20. + np.cumsum(rng.normal(0., 0.15, NT))
    w_obs = 120. + 25. * (h - h.mean()) + rng.normal(0., 6., NT)
    slope = np.abs(rng.normal(8e-5, 3e-5, NT))
    slope[[3, 17, 21]] = [-2e-5, 5e-6, 2.5e-5]   # negative, tiny, below 3.4e-5
    w_obs[9] = 900.                                 # width outlier
    h[25] = 45.                                     # height outlier
    dA = 130. * (h - np.median(h)) * (-1. if reverse_da else 1.)
    time = 7.0e8 + 86400. * 10. * np.arange(NT)

    q = {'xtrk_dist': np.full(NT, 30e3), 'ice_clim_f': np.zeros(NT),
         'dark_frac': np.zeros(NT), 'obs_frac_n': np.ones(NT),
         'xovr_cal_q': np.zeros(NT), 'n_good_nod': np.full(NT, 50.),
         'p_width': np.full(NT, 120.), 'p_length': np.full(NT, 10e3),
         'reach_q_b': np.zeros(NT)}
    q['xtrk_dist'][[5, 40]] = [5e3, 70e3]
    q['dark_frac'][12] = 0.9
    q['n_good_nod'][30] = 3
    q['xovr_cal_q'][33] = 2
    q['obs_frac_n'][36] = np.nan                    # unknown quality
    if gaps:
        w_obs[[44, 45]] = np.nan
        time[50] = np.nan

    with Dataset(path, 'w') as ds:
        ds.createDimension('nt', NT)
        ds.createDimension('nchar', 1)
        ds.createVariable('observations', 'i4', ('nt',))[:] = np.arange(NT)
        reach = ds.createGroup('reach')

        def var(name, values):
            v = reach.createVariable(name, 'f8', ('nt',), fill_value=FILL)
            v[:] = np.where(np.isfinite(values), values, FILL)

        reach.createVariable('reach_id', 'i8')[...] = int(reach_id)
        for name, values in (('wse', h), ('width', w_obs), ('slope2', slope),
                             ('d_x_area', dA), ('time', time),
                             ('reach_q', np.zeros(NT))):
            var(name, values)
        for name in ('wse_u', 'width_u', 'slope2_u', 'd_x_area_u'):
            var(name, np.full(NT, 0.1))
        for name, values in q.items():
            var(name, values)
        fit = reach.createGroup('hwfit')
        for dim, size in (('nbreak', 4), ('ncoef', 2), ('nseg', 3), ('one', 1)):
            fit.createDimension(dim, size)
        fit.createVariable('h_break', 'f8', ('nbreak', 'one'))[:] = \
            np.asarray(hwfit[0], dtype=float).reshape(4, 1)
        coeffs = np.zeros((2, 3, 1))
        coeffs[0, :, 0], coeffs[1, :, 0] = hwfit[1], hwfit[2]
        fit.createVariable('fit_coeffs', 'f8', ('ncoef', 'nseg', 'one'))[:] = coeffs
    return h, dA


def fixed_parameters(dA):
    a0 = max(1., 1. - np.nanmin(dA)) + 400.
    return {
        'busboi': {'n': 0.031, 'a0': a0},
        'sad': {'n': 0.028, 'a0': a0 * 1.3},
        'sic4dvar': {'n': 0.035, 'a0': a0 * 0.8},
        'hivdi': {'alpha': 30., 'beta': -0.2, 'Abar': a0},
        'metroman': {'na': 0.03, 'x1': -0.2, 'Abar': a0},
        'momma': {'B': 10., 'H': 25., 'Save': 1e-4},
    }


def write_integrator(path, params, config):
    with Dataset(path, 'w') as ds:
        for key, attribute in fp.CONFIG_ATTRIBUTES.items():
            ds.setncattr(attribute, config.get(key, fp.PREPROCESS_VERSION))
        for group_name, values in params.items():
            group = ds.createGroup(group_name)
            for key, value in {**values, 'sbQ_rel': 0.2}.items():
                group.createVariable(key, 'f8')[...] = value


# ---------------------------------------------------------------------------
# running both sides
# ---------------------------------------------------------------------------
def import_offline(offline_repo):
    try:
        import geopandas  # noqa: F401  (only needed for single-pass inputs)
    except ImportError:
        sys.modules['geopandas'] = types.ModuleType('geopandas')
    sys.path.insert(0, str(offline_repo))
    module = importlib.import_module('run_offline')
    if sys.version_info >= (3, 13):
        # offline.WriteQ.write_q assigns through locals(), which no longer
        # works from Python 3.13 (PEP 667); the offline container runs 3.12.
        write_q_module = importlib.import_module('offline.WriteQ')

        def write_q(output_dir, data_dict):
            path = Path(output_dir) / f"{data_dict['reach_id']}_offline.nc"
            with Dataset(path, 'w', format='NETCDF4') as out:
                out.createDimension('nt', data_dict['nt'])
                for key in write_q_module.DSCHG_KEYS:
                    out.createVariable(
                        key, 'f8', ('nt',), fill_value=write_q_module.FILL_VALUE,
                    )[:] = np.nan_to_num(data_dict[key], copy=True,
                                         nan=write_q_module.FILL_VALUE)
        module.write_q = write_q
    return module


def run_offline(module, mnt, reach_ids):
    reaches = [{'reach_id': int(r), 'swot': f'{r}_SWOT.nc',
                'sword': 'na_sword_v17.nc', 'sos': 'na_sos.nc'} for r in reach_ids]
    (mnt / 'input' / 'reaches.json').write_text(json.dumps(reaches))
    module.INPUT = str(mnt / 'input')
    module.FLPE_DIR = str(mnt / 'moi')
    out = mnt / 'offline'
    out.mkdir(exist_ok=True)
    argv = sys.argv
    try:
        for index in range(len(reaches)):
            sys.argv = ['run_offline.py', 'constrained', 'timeseries',
                        'integrator', 'reaches.json', str(index)]
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(Path(mnt / 'input'), str(out), index)
    finally:
        sys.argv = argv
    return out


def extract(swot_dir, reach_ids, config):
    obj = Input(None, None, swot_dir, None, {'reach_ids': list(reach_ids)},
                'constrained', False)
    obj.observation_filter = config['observation_filter']
    obj.slope_policy = config['slope_policy']
    with contextlib.redirect_stdout(io.StringIO()):
        obj.extract_swot()
    return obj.obs_dict


def to_full_grid(values, obs):
    full = np.full(obs['nt'] + np.shape(obs['iDelete'])[1], np.nan)
    valid = np.setdiff1d(np.arange(full.size), obs['iDelete'][0])
    full[valid] = np.ravel(values)
    return full


def compare(label, q_moi, q_off):
    produced = np.isfinite(q_off)
    moi_valid = np.isfinite(q_moi)
    both = produced & moi_valid
    same_mask = np.array_equal(produced, moi_valid)
    max_rel = (float(np.max(np.abs(q_off[both] / q_moi[both] - 1.)))
               if both.any() else 0.)
    ok = same_mask and max_rel < 1e-9
    detail = '' if same_mask else (
        f' offline_only={int((produced & ~moi_valid).sum())}'
        f' moi_only={int((moi_valid & ~produced).sum())}')
    print(f"{'OK  ' if ok else 'FAIL'} {label:58s} n={int(both.sum()):2d} "
          f"max_rel_diff={max_rel:.1e}{detail}")
    return ok


def offline_series(out, reach, alg):
    with Dataset(out / f'{reach}_offline.nc') as ds:
        return np.ma.filled(ds[ALGORITHMS[alg][0]][:].astype(float), np.nan)


# ---------------------------------------------------------------------------
# part 1: fixed parameters, every option combination
# ---------------------------------------------------------------------------
CASES = {  # reach -> (seed, (h_break, slopes, intercepts), gaps)
    '11111111111': (1, ([18., 19.5, 20.5, 23.], [20., 25., 30.],
                        [-280., -375., -475.]), False),
    '22222222222': (2, ([-1500., 49000., 99500., 150000.], [0., 0., 0.],
                        [-5869.] * 3), False),
    '33333333333': (3, ([18., 19.5, 20.5, 23.], [20., 25., 30.],
                        [-280., -375., -475.]), True),
}


def part_fixed_parameters(offline_module):
    ok = True
    for filt, slope in itertools.product(fp.OBSERVATION_FILTERS, fp.SLOPE_POLICIES):
        config = {'observation_filter': filt, 'slope_policy': slope}
        with tempfile.TemporaryDirectory() as tmp:
            mnt = Path(tmp)
            (mnt / 'input/swot').mkdir(parents=True)
            (mnt / 'moi').mkdir()
            params = {}
            for reach, (seed, hwfit, gaps) in CASES.items():
                _h, dA = make_swot(mnt / 'input/swot' / f'{reach}_SWOT.nc',
                                   reach, seed, hwfit, gaps)
                params[reach] = fixed_parameters(dA)
                write_integrator(mnt / 'moi' / f'{reach}_integrator.nc',
                                 params[reach], config)
            out = run_offline(offline_module, mnt, CASES)
            obs_dict = extract(mnt / 'input/swot', CASES, config)
            for reach in CASES:
                obs = obs_dict[reach]
                for alg, (_var, law, keys) in ALGORITHMS.items():
                    p = params[reach][alg]
                    args = (tuple(p[k] for k in keys), obs)
                    if alg == 'momma':
                        args += (p['Save'],)
                    with np.errstate(all='ignore'):
                        q_moi = to_full_grid(getattr(Integrate, law)(None, *args), obs)
                    ok &= compare(f'{filt}/{slope} {reach} {alg} '
                                  f'width={obs["width_source"]}',
                                  q_moi, offline_series(out, reach, alg))
    return ok


# ---------------------------------------------------------------------------
# part 2: MOI fit -> integrator file -> offline
# ---------------------------------------------------------------------------
def build_integrator(obs_dict):
    """Integrate instance around real extracted observations."""
    alg_dict = {}
    for alg in ALGORITHMS:
        alg_dict[alg] = {}
        for reach, obs in obs_dict.items():
            # A plausible FLPE hydrograph that rises with WSE.
            h = np.where(np.isfinite(obs['h']), obs['h'], np.nanmedian(obs['h']))
            flpe_q = 50. + 40. * (h - np.nanmin(h))
            entry = {'s1-flpe-exists': True, 'q': flpe_q,
                     'qbar': float(np.mean(flpe_q)), 'qbar_source': 'FLPE',
                     'q33_source': 'FLPE',
                     'integrator': {'qbar': 1.2 * float(np.mean(flpe_q)),
                                    'q33': 1.2 * float(np.quantile(flpe_q, 0.33)),
                                    'sbQ_rel': 0.2}}
            if alg == 'momma':
                entry.update({'B': np.nanmin(obs['h']) - 3., 'H': np.nanmax(obs['h']) + 3.,
                              'Save': 1e-4})
            alg_dict[alg][reach] = entry
    integrator = Integrate.__new__(Integrate)
    integrator.alg_dict = alg_dict
    integrator.obs_dict = obs_dict
    reach_ids = list(obs_dict)
    integrator.basin_dict = {'reach_ids': reach_ids, 'reach_ids_all': reach_ids}
    integrator.params_dict = {'Integrator_Hydrograph_Method': 'rescale',
                              'FLP_Fit_Method': 'series', 'FLP_Optimizer': 'varpro'}
    integrator.VerboseFlag = False
    integrator.flp_fit_report = {}
    integrator.topology_report = {}
    integrator.unconverged_solves = []
    return integrator


def part_round_trip(offline_module):
    ok = True
    config = {'observation_filter': fp.FILTER_ROBUST_V2,
              'slope_policy': fp.SLOPE_LEGACY_SAFE}
    reaches = {'44444444444': False, '55555555555': True}   # reach -> reverse dA
    with tempfile.TemporaryDirectory() as tmp:
        mnt = Path(tmp)
        (mnt / 'input/swot').mkdir(parents=True)
        (mnt / 'moi').mkdir()
        for i, (reach, reverse) in enumerate(reaches.items()):
            make_swot(mnt / 'input/swot' / f'{reach}_SWOT.nc', reach, 10 + i,
                      ([18., 19.5, 20.5, 23.], [20., 25., 30.],
                       [-280., -375., -475.]), reverse_da=reverse)
        obs_dict = extract(mnt / 'input/swot', reaches, config)
        integrator = build_integrator(obs_dict)
        with contextlib.redirect_stdout(io.StringIO()):
            integrator.calc_integrator_hydrographs()
            integrator.compute_FLPs()
            Output(basin_dict=integrator.basin_dict, out_dir=mnt / 'moi',
                   integ_dict={}, alg_dict=integrator.alg_dict,
                   obs_dict=obs_dict, sword_dir=mnt,
                   params_dict={'write_fill_only': False,
                                'Observation_Filter_Version': config['observation_filter'],
                                'Slope_Policy': config['slope_policy']},
                   ).write_output()
        out = run_offline(offline_module, mnt, reaches)

        for reach, reverse in reaches.items():
            with Dataset(mnt / 'moi' / f'{reach}_integrator.nc') as ds:
                for alg in ALGORITHMS:
                    group = ds[alg]
                    q_flowlaw = np.ma.filled(group['q_flowlaw'][:].astype(float), np.nan)
                    exported = int(group['flp_exported'][:])
                    label = (f'round trip {reach} {alg} status={group.flp_fit_status} '
                             f'exported={exported}')
                    ok &= compare(label, q_flowlaw, offline_series(out, reach, alg))
                    if reverse and alg != 'momma':
                        masked = (exported == 0 and group.flp_export_reason
                                  == flp_fit.EXPORT_REASON_INVALID_GEOMETRY)
                        if not masked:
                            print(f'FAIL {reach} {alg}: reversed dA was not masked')
                            ok = False
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--offline-repo', type=Path, required=True)
    args = parser.parse_args()
    offline_module = import_offline(args.offline_repo.resolve())
    with np.errstate(all='ignore'):
        ok = part_fixed_parameters(offline_module)
        ok &= part_round_trip(offline_module)
    print('PARITY PASSED' if ok else 'PARITY FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

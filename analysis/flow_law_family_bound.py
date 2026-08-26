"""Measure the flow-law family's residual floor.

The canary matrix left one question open: after the series refit, median
``flp_fit_nrmse`` is still 0.17-0.32.  Two very different things produce that
number, and they call for opposite responses:

  * the optimiser is stalling               -> better search pays off
  * the flow law cannot represent the shape -> better search pays nothing, and
                                               the effort belongs in dA quality
                                               and the rescale transform

This script separates them.  For every reach and algorithm it refits with a
search far denser than production would ever afford -- a large grid, both
objective spaces, no time limit -- and reports the best achievable nRMSE.  That
number is the ceiling on any future optimiser work.  If the floor is ~0.15,
0.17-0.32 is mostly irreducible and the remaining effort is misdirected.  If the
floor is ~0.05, the gap is search or identifiability and the production fitter
is worth pushing on.

It reads finished MOI output rather than rerunning the integrator, so it costs
one pass over a canary run's outputs and touches nothing.

Usage
-----
    python analysis/flow_law_family_bound.py \\
        --output-dir  /path/to/moi/output \\
        --swot-dir    /path/to/input/swot \\
        --csv         family_bound.csv

The target series is each algorithm group's ``q``, i.e. the hydrograph MOI
actually exported, which for a rescaled reach is the FLPE shape at the
integrated level.  Reaches whose hydrograph is a flow-law reconstruction rather
than a rescaled FLPE series are skipped: refitting a flow law to its own output
measures nothing.
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moi import flp_fit  # noqa: E402

FILL = -999999999999
SMIN = 1.7e-5

# Bounds mirror moi/Integrate.py's _flp_reach_spec.  a0_min is filled in per
# reach from the observed dA.
ALG_PARAMS = {
    'busboi': dict(n_hi=np.inf, n_params=2),
    'hivdi': dict(n_hi=np.inf, n_params=3),
    'metroman': dict(n_hi=np.inf, n_params=3),
    'sad': dict(n_hi=np.inf, n_params=2),
    'sic4dvar': dict(n_hi=10., n_params=2),
}


def load_swot_obs(swot_dir, reach):
    """Geometry for one reach, filtered exactly as Input.extract_swot does."""
    path = Path(swot_dir) / ('%s_SWOT.nc' % reach)
    if not path.is_file():
        return None
    with Dataset(path) as ds:
        nt = ds.dimensions['nt'].size
        h = ds['reach/wse'][0:nt].filled(np.nan)
        w = ds['reach/width'][0:nt].filled(np.nan)
        S = ds['reach/slope2'][0:nt].filled(np.nan)
        dA = ds['reach/d_x_area'][0:nt].filled(np.nan)

    keep = ~(np.isnan(h) | np.isnan(w) | np.isnan(S) | np.isnan(dA))
    h, w, S, dA = h[keep], w[keep], S[keep], dA[keep]
    np.putmask(S, S < SMIN, SMIN)
    return {'nt': int(keep.sum()), 'h': h, 'w': w, 'S': S, 'dA': dA}


def load_target(out_file, alg):
    """Exported hydrograph and its provenance for one algorithm group."""
    with Dataset(out_file) as ds:
        if alg not in ds.groups:
            return None, ''
        group = ds[alg]
        if 'q' not in group.variables:
            return None, ''
        q = np.asarray(group['q'][:], dtype=float).ravel()
        source = getattr(group, 'hydrograph_source', '')
    q[q == FILL] = np.nan
    return q, source


def dense_fit(alg, obs, target, space):
    """Best parameters this flow law can reach, given an unlimited search."""
    a0_min = -float(np.min(obs['dA'])) + 1.
    if alg == 'sic4dvar':
        bounds = ((0.001, 10.), (a0_min, np.inf))
    elif alg in ('busboi', 'sad'):
        bounds = ((0.001, np.inf), (a0_min, np.inf))
    else:
        bounds = ((0.001, np.inf), (-1e1, 1e1), (a0_min, np.inf))

    params, cost, n_evals = flp_fit.fit_scaled_law(
        alg, obs, target, priors={}, param_bounds=bounds, a0_min=a0_min,
        space=space, grid_points=400, exponent_points=81,
    )
    if params is None:
        return None, np.nan, np.nan, n_evals

    flowlaw = FLOWLAWS[alg]
    q = flp_fit._safe_flowlaw(flowlaw, params, obs)
    lin, log = flp_fit.nrmse_pair(q, target) if q is not None else (np.nan, np.nan)
    return params, lin, log, n_evals


# Standalone copies of the flow laws, so this script does not need to build an
# Integrate instance (which would need SoS, SWORD and a basin definition).
def _base(dA, w, S, a0):
    with np.errstate(all='ignore'):
        return (dA + a0) ** (5. / 3.) * w ** (-2. / 3.) * np.sqrt(S)


def _bam(params, obs):
    return _base(obs['dA'], obs['w'], obs['S'], params[1]) / params[0]


def _hivdi(params, obs):
    alpha, beta, a0 = params
    with np.errstate(all='ignore'):
        return _base(obs['dA'], obs['w'], obs['S'], a0) * (
            alpha * ((obs['dA'] + a0) / obs['w']) ** beta)


def _metroman(params, obs):
    na, x1, a0 = params
    with np.errstate(all='ignore'):
        n = na * ((obs['dA'] + a0) / obs['w']) ** x1
        return _base(obs['dA'], obs['w'], obs['S'], a0) / n


FLOWLAWS = {
    'busboi': _bam, 'sad': _bam, 'sic4dvar': _bam,
    'hivdi': _hivdi, 'metroman': _metroman,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--swot-dir', required=True, type=Path)
    parser.add_argument('--csv', type=Path, default=Path('family_bound.csv'))
    parser.add_argument('--limit', type=int, default=None,
                        help='Stop after this many reaches (for a quick look).')
    args = parser.parse_args()

    out_files = sorted(args.output_dir.glob('*_integrator.nc'))
    if args.limit:
        out_files = out_files[:args.limit]
    print('%d output files' % len(out_files))

    rows = []
    for i, out_file in enumerate(out_files):
        reach = out_file.name.split('_')[0]
        obs = load_swot_obs(args.swot_dir, reach)
        if obs is None or obs['nt'] < flp_fit.MIN_VALID_OBS:
            continue

        for alg in ALG_PARAMS:
            target, source = load_target(out_file, alg)
            if target is None or target.size != obs['nt']:
                continue
            if source and source != 'rescaled':
                # Fitting a flow law to a hydrograph that flow law generated
                # would report a floor of zero and mean nothing.
                continue
            if np.count_nonzero(np.isfinite(target)) < flp_fit.MIN_VALID_OBS:
                continue

            row = {'reach_id': reach, 'algorithm': alg, 'nt': obs['nt']}
            for space in ('linear', 'log'):
                params, lin, log, n_evals = dense_fit(alg, obs, target, space)
                row['floor_%s_nrmse_lin' % space] = lin
                row['floor_%s_nrmse_log' % space] = log
                row['floor_%s_evals' % space] = n_evals
            rows.append(row)

        if (i + 1) % 25 == 0:
            print('  %d/%d reaches' % (i + 1, len(out_files)))

    if not rows:
        print('no comparable reaches found')
        return

    fieldnames = list(rows[0].keys())
    with open(args.csv, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print('wrote %d rows to %s' % (len(rows), args.csv))

    print('\nmedian achievable nRMSE (the floor):')
    print('  %-10s %8s %8s %8s' % ('alg', 'n', 'linear', 'log'))
    for alg in sorted(ALG_PARAMS):
        subset = [r for r in rows if r['algorithm'] == alg]
        if not subset:
            continue
        lin = np.nanmedian([r['floor_linear_nrmse_lin'] for r in subset])
        log = np.nanmedian([r['floor_log_nrmse_log'] for r in subset])
        print('  %-10s %8d %8.3f %8.3f' % (alg, len(subset), lin, log))
    print('\nCompare against flp_fit_nrmse in the same outputs.  A production '
          'residual close to this floor means the flow-law family, not the '
          'optimiser, is the binding constraint.')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Publish one continent's *_sword_v17_moi.nc from the MOI integrator files.

run_MOI.py normally updates the continental SWORD copy from every basin task:
hundreds of parallel tasks append to one NetCDF file (with random-wait retries
on the lock), the copy is only made when missing so values from an earlier run
survive, and every task needs write access to the SWORD directory.

With ``run_MOI.py --defer-sword-output`` the basin tasks skip that step and
this program runs once per continent after all basin tasks (both branches)
have finished:

1. validate: every expected reach exists in the source SWORD and has an
   integrator file per branch with every algorithm group and parameter; a
   non-finite parameter is accepted only where MOI deliberately masked it
   (``flp_exported == 0``);
2. stop without creating or replacing anything if validation failed;
3. copy the read-only source SWORD to a temporary file next to the output,
   write the constrained and unconstrained parameters, close it;
4. reopen it and check every written value;
5. atomically rename it to the output path.

The output path is explicit: the source SWORD directory is treated as
read-only.  Exposing the output at the path JPL requires
(/mnt/data/input/sword/<continent>_sword_v17_moi.nc) is a job for the
container bind/DAG.  Standalone (no moi imports) so it runs as
``python3 /app/moi/sword_finalize.py`` inside the MOI container.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from netCDF4 import Dataset
import numpy as np

# integrator group -> (SWORD discharge model, {integrator var: SWORD var}).
# Same parameters Output.write_sword_output writes.
PARAMETERS = {
    'busboi': ('BAM', {'a0': 'Abar', 'n': 'n', 'sbQ_rel': 'sbQ_rel'}),
    'hivdi': ('HiVDI', {'Abar': 'Abar', 'alpha': 'alpha', 'beta': 'beta',
                        'sbQ_rel': 'sbQ_rel'}),
    'metroman': ('MetroMan', {'Abar': 'Abar', 'na': 'ninf', 'x1': 'p',
                              'sbQ_rel': 'sbQ_rel'}),
    'momma': ('MOMMA', {'B': 'B', 'H': 'H', 'Save': 'Save'}),
    'sad': ('SADS', {'a0': 'Abar', 'n': 'n', 'sbQ_rel': 'sbQ_rel'}),
    'sic4dvar': ('SIC4DVar', {'a0': 'Abar', 'n': 'n', 'sbQ_rel': 'sbQ_rel'}),
}
# Flow-law parameters: a non-finite value blocks publication unless MOI masked
# the export (flp_exported == 0).  sbQ_rel and Save are only reported.
_FLP = {group: set(v) - {'sbQ_rel', 'Save'}
        for group, (_model, v) in PARAMETERS.items()}


def expected_reaches(basin_json, sword_name):
    """Reach ids that basin.json assigns to ``sword_name``, in order."""
    with open(basin_json) as stream:
        basins = json.load(stream)
    if isinstance(basins, dict):
        basins = [basins] if 'reach_id' in basins else list(basins.values())
    reaches = []
    for basin in basins:
        if str(basin.get('sword')) != sword_name:
            continue
        ids = basin.get('reach_id', [])
        reaches.extend(str(int(r)) for r in (ids if isinstance(ids, list) else [ids]))
    return list(dict.fromkeys(reaches))


def _scalar(variable):
    value = np.ma.filled(np.ma.asarray(variable[...], dtype=float), np.nan)
    return float(np.ravel(value)[0]) if np.size(value) else np.nan


def _read_reach(path, reach):
    """``(values, problems, warnings)`` for one integrator file."""
    values, problems, warnings = {}, [], []
    try:
        with Dataset(path) as integrator:
            for group, (_model, variables) in PARAMETERS.items():
                if group not in integrator.groups:
                    problems.append(f'{reach}: {path.name} has no {group} group')
                    continue
                g = integrator[group]
                masked = ('flp_exported' in g.variables
                          and _scalar(g['flp_exported']) == 0)
                for name in variables:
                    if name not in g.variables:
                        problems.append(f'{reach}: {group}/{name} missing')
                        continue
                    value = _scalar(g[name])
                    if not np.isfinite(value):
                        if name not in _FLP[group]:
                            warnings.append(f'{reach}: {group}/{name} is not finite')
                        elif not masked:
                            problems.append(f'{reach}: {group}/{name} is not '
                                            'finite and was not masked by MOI')
                    values[(group, name)] = value
    except OSError as exc:
        problems.append(f'{reach}: cannot read {path}: {exc}')
    return values, problems, warnings


def plan(source_sword, basin_json, integrator_dirs):
    """Everything to write, plus the validation report; touches no output."""
    sword_name = Path(source_sword).name
    reaches = expected_reaches(basin_json, sword_name)
    with Dataset(source_sword) as sword:
        index = {int(r): i for i, r in enumerate(sword['reaches']['reach_id'][:])}
        branches = sword['reaches']['discharge_models'].groups
        missing_branches = [b for b in integrator_dirs if b not in branches]

    report = {'source_sword': str(source_sword), 'expected_reaches': len(reaches),
              'missing_integrator_files': [], 'problems': [], 'warnings': [],
              'reaches_missing_from_sword': [], 'values_to_write': 0}
    report['problems'] += [f'SWORD has no discharge_models/{b}'
                           for b in missing_branches]
    writes = []
    for reach in reaches:
        if int(reach) not in index:
            report['reaches_missing_from_sword'].append(reach)
            continue
        for branch, directory in integrator_dirs.items():
            path = Path(directory) / f'{reach}_integrator.nc'
            if not path.is_file():
                report['missing_integrator_files'].append(f'{branch}/{reach}')
                continue
            values, problems, warnings = _read_reach(path, reach)
            report['problems'] += [f'{branch} {p}' for p in problems]
            report['warnings'] += [f'{branch} {w}' for w in warnings]
            if problems:
                continue
            for (group, name), value in values.items():
                model, variables = PARAMETERS[group]
                writes.append((branch, model, variables[name], index[int(reach)], value))
    report['values_to_write'] = len(writes)
    return writes, report


def _write_and_verify(path, writes):
    with Dataset(path, 'a') as sword:
        models = sword['reaches']['discharge_models']
        for branch, model, variable, i, value in writes:
            models[branch][model][variable][i] = value
    with Dataset(path) as sword:
        models = sword['reaches']['discharge_models']
        for branch, model, variable, i, value in writes:
            stored = float(np.ma.filled(np.ma.asarray(
                models[branch][model][variable][i], dtype=float), np.nan))
            if not (stored == value or (np.isnan(stored) and np.isnan(value))):
                raise RuntimeError(f'verification failed for {branch}/{model}/'
                                   f'{variable}[{i}]: {stored} != {value}')


def finalize(source_sword, output_sword, basin_json, integrator_dirs,
             allow_missing=False):
    """Validate, then publish ``output_sword`` atomically.  Returns the report.

    Nothing is created or replaced unless validation passes.  With
    ``allow_missing`` a reach whose integrator file is absent (MOI skipped its
    basin) keeps the source SWORD values; every other problem still blocks.
    """
    source_sword, output_sword = Path(source_sword), Path(output_sword)
    if source_sword.resolve() == output_sword.resolve():
        raise ValueError('output SWORD must differ from the source SWORD')
    writes, report = plan(source_sword, basin_json, integrator_dirs)
    blocked = (report['problems'] or report['reaches_missing_from_sword']
               or (report['missing_integrator_files'] and not allow_missing))
    report['published'] = False
    if blocked:
        return report

    output_sword.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_sword.with_name(f'{output_sword.name}.tmp-{os.getpid()}')
    try:
        with open(source_sword, 'rb') as src, open(temporary, 'wb') as dst:
            while chunk := src.read(1 << 24):
                dst.write(chunk)
        _write_and_verify(temporary, writes)
        os.replace(temporary, output_sword)
    finally:
        if temporary.exists():
            temporary.unlink()
    report['published'] = True
    report['output_sword'] = str(output_sword)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--source-sword', required=True, type=Path)
    parser.add_argument('--output-sword', required=True, type=Path)
    parser.add_argument('--basin-json', required=True, type=Path)
    parser.add_argument('--constrained-integrator-dir', type=Path)
    parser.add_argument('--unconstrained-integrator-dir', type=Path)
    parser.add_argument('--allow-missing', action='store_true',
                        help='Publish even if some reaches have no integrator '
                             'file (they keep the source SWORD values).')
    args = parser.parse_args(argv)

    dirs = {branch: path for branch, path in (
        ('constrained', args.constrained_integrator_dir),
        ('unconstrained', args.unconstrained_integrator_dir)) if path}
    if not dirs:
        parser.error('give at least one --*-integrator-dir')

    report = finalize(args.source_sword, args.output_sword, args.basin_json,
                      dirs, allow_missing=args.allow_missing)
    print(json.dumps(report, indent=2))
    return 0 if report['published'] else 2


if __name__ == '__main__':
    sys.exit(main())

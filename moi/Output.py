# Standard imports
from datetime import datetime
from pathlib import Path
import time
import random
import os
import warnings

# Third-party imports
from netCDF4 import Dataset
import numpy as np
import shutil

from moi import flowlaw_preprocess


def _finite_or_fill(values, fillvalue):
    """``values`` with NaN and +/-inf replaced by the NetCDF fill value.

    np.nan_to_num(values, nan=fill) turned inf into +/-1.8e308, which was then
    written as if it were a discharge.
    """
    values = np.asarray(values, dtype=float)
    return np.where(np.isfinite(values), values, fillvalue)


def _as_row(q, nt, fill=np.nan):
    """Coerce a hydrograph to the (1, nt) shape write_output needs.

    Deliberately duplicated from ``flp_fit.as_row`` rather than imported.
    Output.py is the last-mile stage: if anything here fails to import or
    resolve, an entire basin loses every output file for every algorithm. This
    module previously had no dependency on ``moi.flp_fit``, and adding one so
    that five lines of numpy could be shared bought nothing while creating a
    new way for a partially-deployed tree to fail at runtime -- which is
    exactly how it did fail.

    ``tests/test_flp_diagnostics_output.py`` pins this against
    ``flp_fit.as_row`` so the two cannot drift apart silently.
    """
    # A reach can have no valid SWOT timesteps after geometry filtering.  The
    # caller will then re-insert every dropped timestep, so a (1, 0) row is
    # the only shape that preserves the final record length.
    nt = max(int(nt), 0)
    out = np.full((1, nt), fill, dtype=float)
    if q is None:
        return out
    try:
        arr = np.asarray(q, dtype=float).ravel()
    except (TypeError, ValueError):
        # Output is the last mile: a malformed value for one algorithm must
        # become a visible fill series, not abort all reaches in the basin.
        return out
    k = min(nt, arr.size)
    if k:
        out[0, :k] = arr[:k]
    return out

def wait_random(min_seconds=1, max_seconds=10):
    """Wait for a random amount of time between min_seconds and max_seconds."""
    random_wait_time = random.uniform(min_seconds, max_seconds)
    print(f"Waiting for {random_wait_time:.2f} seconds...")
    time.sleep(random_wait_time)
    print("Done waiting!")

class Output:
    """Writes integration results stored in integ_dict to NetCDF file.
    
    Attributes
    ----------
    basin_dict: dict
        dict of reach_ids and SoS file needed to process entire basin of data
    FILL_VALUE: float
        Float fill value for missing data
    out_dir: Path
        path to output dir
    stage_estimate: dict
        dict of integrator estimate data

    Methods
    -------
    write_output()
        Write data stored to NetCDF file labelled with basin id
    write_sword_output(branch)
        Write integrator FLPs back to a copy of the SWORD file
    """

    def __init__(
        self,
        basin_dict,
        out_dir,
        integ_dict,
        alg_dict,
        obs_dict,
        sword_dir,
        params_dict,
        gage_groups=None,
        gage_dict=None,
        corridors_reaches=None,
        corridors_diagnostics=None,
    ):
        self.basin_dict = basin_dict
        self.out_dir = out_dir
        self.stage_estimate = integ_dict
        self.alg_dict = alg_dict
        self.obs_dict = obs_dict
        self.sword_dir = sword_dir
        self.params_dict = params_dict
        # ``gage_groups`` comes from the complete Cal/Val CSV, whereas
        # ``gage_dict`` contains the gages made available to the integrator.
        # The former is needed to label validation gages, which intentionally
        # are not included in constrained integration.
        #
        # ``corridors_reaches`` names the reaches whose gage_dict entry is a
        # CORRIDORS pseudo-gage rather than a real station.  The two are folded
        # into one dict by Input.merge_corridors_and_gages, so without this set
        # a pseudo-gage would be indistinguishable from a station here.
        #
        # Two questions are deliberately kept apart, because a validation reach
        # answers them differently: whether a reach HAS a gage (it is in the
        # catalogue) and what actually CONSTRAINED it (what reached gage_dict).
        # A validation gage is withheld from constrained integration on purpose,
        # so it is has_gage=1 with constraint_source='none' -- or 'corridors'
        # when a pseudo-gage stood in for it.
        self.corridors_reaches = {
            str(reach) for reach in (corridors_reaches or set())
        }
        # Per-reach fit diagnostics for the pseudo-gages, from Input.  Written
        # to the gage group and used to gate nothing -- see
        # Corridors.CORRIDORS_DIAGNOSTIC_KEYS for why they are recorded rather
        # than acted on.
        self.corridors_diagnostics = {
            str(reach): dict(diagnostics)
            for reach, diagnostics in (corridors_diagnostics or {}).items()
        }
        self.gage_groups = {
            str(reach): str(group).strip().lower()
            for reach, group in (gage_groups or {}).items()
            if group is not None and str(group).strip()
        }

        # Catalogued gages: the Cal/Val CSV plus any real station handed to the
        # integrator.  Validation reaches belong here even though they never
        # constrain anything.
        self.gage_reaches = set(self.gage_groups)
        # What actually constrained each reach, keyed by reach id.
        self.constraint_sources = {}

        for reach, gage in (gage_dict or {}).items():
            reach = str(reach)
            source = gage.get('source') if isinstance(gage, dict) else None
            # constraint_source follows the winning entry alone.  Reading it
            # off corridors_reaches instead would mislabel a reach that has
            # CORRIDORS data but was constrained by its real gage.
            if source == 'corridors':
                # Belt and braces: honour the entry's own source even when the
                # caller did not pass corridors_reaches.
                self.corridors_reaches.add(reach)
                self.constraint_sources[reach] = 'corridors'
                continue
            self.constraint_sources[reach] = 'gage'
            self.gage_reaches.add(reach)
            if reach not in self.gage_groups:
                group = gage.get('group') if isinstance(gage, dict) else None
                self.gage_groups[reach] = (
                    str(group).strip().lower()
                    if group is not None and str(group).strip()
                    else 'unclassified'
                )
        # Hydrographs whose length did not match the record, repaired rather
        # than raised on.  Empty is the expected state; anything here means an
        # algorithm produced a malformed series for a reach.
        self.q_shape_repairs = []

    def _write_bias_correlation_diagnostics(self, out):
        """Persist basin-scale augmentation diagnostics in each reach file."""
        all_diagnostics = self.stage_estimate.get('bias_correction', {})
        if not all_diagnostics:
            return

        root = out.createGroup('moi_bias_correlation')
        for algorithm, flow_diagnostics in all_diagnostics.items():
            algorithm_group = root.createGroup(str(algorithm))
            for flow_level, diagnostic in flow_diagnostics.items():
                prefix = str(flow_level).lower()
                values = {
                    f'{prefix}_bias_fraction': diagnostic.get(
                        'estimated_bias_fraction', np.nan
                    ),
                    f'{prefix}_bias_std_fraction': diagnostic.get(
                        'bias_std_fraction', np.nan
                    ),
                    f'{prefix}_correlation_rho': diagnostic.get(
                        'correlation_rho', np.nan
                    ),
                    f'{prefix}_last_delta': diagnostic.get(
                        'last_delta', np.nan
                    ),
                    f'{prefix}_last_physical_rms_delta': diagnostic.get(
                        'last_physical_rms_delta', np.nan
                    ),
                    f'{prefix}_last_physical_p95_delta': diagnostic.get(
                        'last_physical_p95_delta', np.nan
                    ),
                    f'{prefix}_last_raw_delta': diagnostic.get(
                        'last_raw_delta', np.nan
                    ),
                    f'{prefix}_last_robust_delta': diagnostic.get(
                        'last_robust_delta', np.nan
                    ),
                    f'{prefix}_final_reduced_chi_square': diagnostic.get(
                        'final_So', np.nan
                    ),
                }
                for variable_name, value in values.items():
                    variable = algorithm_group.createVariable(variable_name, 'f8')
                    variable.assignValue(float(value))

                algorithm_group.setncattr(
                    f'{prefix}_bias_enabled',
                    int(bool(diagnostic.get('enabled', False))),
                )
                algorithm_group.setncattr(
                    f'{prefix}_solver_status',
                    str(diagnostic.get('status', 'unknown')),
                )
                algorithm_group.setncattr(
                    f'{prefix}_converged',
                    int(bool(diagnostic.get('converged', False))),
                )
                algorithm_group.setncattr(
                    f'{prefix}_outer_iterations',
                    int(diagnostic.get('outer_iterations', 0)),
                )
                algorithm_group.setncattr(
                    f'{prefix}_n_real_flpe_rows',
                    int(diagnostic.get('n_real_flpe_rows', 0)),
                )
                algorithm_group.setncattr(
                    f'{prefix}_oscillation_events',
                    int(diagnostic.get('oscillation_events', 0)),
                )
                algorithm_group.setncattr(
                    f'{prefix}_relaxation_recoveries',
                    int(diagnostic.get('relaxation_recoveries', 0)),
                )
                thresholds = diagnostic.get('convergence_thresholds', {})
                for threshold_name, threshold_value in thresholds.items():
                    algorithm_group.setncattr(
                        f'{prefix}_{threshold_name}_threshold',
                        float(threshold_value),
                    )
                effects = diagnostic.get('correlation_effects', [])
                algorithm_group.setncattr(
                    f'{prefix}_correlation_effects',
                    ','.join(f'{float(value):.12g}' for value in effects),
                )

    def _write_time_strings(self, out, reach):
        """Write root-level SWOT time strings aligned with the restored nt axis."""
        nt = len(out.dimensions['nt'])
        values = self.obs_dict.get(str(reach), {}).get('time_str')
        if values is None:
            warnings.warn(
                f'Reach {reach} has no time_str metadata; writing empty strings.',
                RuntimeWarning,
            )
            values = np.full(nt, '', dtype=str)
        else:
            values = np.asarray(values, dtype=str).ravel()

        if values.size != nt:
            raise ValueError(
                f'Reach {reach} time_str length does not match output nt: '
                f'{values.size} != {nt}'
            )

        time_str = out.createVariable('time_str', str, ('nt',))
        time_str.long_name = 'SWOT reach time string in UTC.'
        time_str.source = self.obs_dict.get(str(reach), {}).get(
            'time_str_source',
            'unavailable',
        )
        time_str[:] = values

    def _write_gage_metadata(self, out, reach):
        """Write what in-situ discharge this reach has, and what constrained it.

        has_gage
            The reach is in the real gage catalogue.  Independent of whether
            that gage was used: a validation gage is catalogued but withheld.
        has_corridors
            A CORRIDORS pseudo-gage was built for the reach.  Also independent
            of whether it was used: a reach with both keeps its real gage by
            default, and is then has_gage=1, has_corridors=1, source='gage'.
        constraint_source
            What actually reached the integrator -- 'gage', 'corridors' or
            'none'.  This is the field to score against, because it alone
            says whether the estimate saw in-situ discharge.
        group
            Cal/Val label for the catalogued gage.
        """
        reach = str(reach)
        gage = out.createGroup('gage')
        has_gage = reach in self.gage_reaches
        has_corridors = reach in self.corridors_reaches

        gage.setncattr('has_gage', int(has_gage))
        gage.setncattr('has_corridors', int(has_corridors))
        gage.setncattr(
            'constraint_source', self.constraint_sources.get(reach, 'none')
        )
        gage.setncattr(
            'group',
            self.gage_groups.get(reach, 'none') if has_gage else 'none',
        )

        # Written only where a pseudo-gage was actually built, so their absence
        # is itself the signal that CORRIDORS had nothing for the reach.
        diagnostics = self.corridors_diagnostics.get(reach)
        if has_corridors and diagnostics:
            for attribute, key in self.CORRIDORS_DIAGNOSTIC_ATTRS:
                if key not in diagnostics:
                    continue
                value = diagnostics[key]
                gage.setncattr(
                    attribute,
                    int(value) if isinstance(value, (int, np.integer))
                    else float(value),
                )
        
    # Diagnostic fields describing HOW the flow-law parameters in this group
    # were obtained, and how well they reproduce the hydrograph.  Written for
    # every reach and used to gate nothing: thresholds chosen before the global
    # distribution is known are guesses, so the first global run should record
    # the distribution and the thresholds should be set from it afterwards.
    FLP_DIAGNOSTIC_FIELDS = (
        # (netCDF name, integrator key, dtype)
        ('flp_fit_nrmse', 'flp_fit_nrmse', 'f8'),
        ('flp_fit_nrmse_log', 'flp_fit_nrmse_log', 'f8'),
        # i8, not i4: the file-wide fill value (-999999999999) does not fit in
        # a 32-bit integer, so an i4 variable could not represent "missing".
        ('flp_fit_status_code', 'flp_fit_status_code', 'i8'),
        ('flp_param_at_bound', 'flp_param_at_bound', 'i8'),
        ('flp_n_valid_obs', 'flp_n_valid_obs', 'i8'),
        ('dA_valid_frac', 'dA_valid_frac', 'f8'),
        ('rescale_factor', 'rescale_factor', 'f8'),
        ('rescale_exponent', 'rescale_b', 'f8'),
        # How well q_flowlaw (what offline can reconstruct from the exported
        # parameters) reproduces the MOI hydrograph it was fitted to.
        ('flp_fit_pearson_r', 'flp_fit_pearson_r', 'f8'),
        ('flp_fit_nse', 'flp_fit_nse', 'f8'),
        ('flp_fit_nbias', 'flp_fit_nbias', 'f8'),
        ('corr_wse_dA', 'corr_wse_dA', 'f8'),
        ('flp_fit_n_paired', 'flp_fit_n_paired', 'i8'),
        ('flowlaw_valid_obs', 'flowlaw_valid_obs', 'i8'),
        # 1 if the parameters in this group are the ones offline/SWORD get;
        # 0 if they were masked (NaN) for flp_export_reason.
        ('flp_exported', 'flp_exported', 'i8'),
    )

    # CORRIDORS pseudo-gage fit diagnostics: (netCDF attribute, entry key).
    # Quality reporting only.  A pseudo-gage's weight is fixed, so none of
    # these changes what the integrator did -- they exist so a global run can
    # measure how the pseudo-gages actually performed.
    CORRIDORS_DIAGNOSTIC_ATTRS = (
        # Matched measurement/overpass pairs that entered the flow-law fit.
        ('corridors_n_measurements', 'n_corridors_measurements'),
        # Distinct SWOT overpasses among those pairs.  Lower than the pair
        # count wherever one overpass matched several daily measurements.
        ('corridors_n_overpasses', 'n_corridors_overpasses'),
        # In-sample relative RMSE of the fit.  Zero by construction when the
        # reach was fitted to a single measurement.
        ('corridors_fit_relative_rmse', 'corridors_fit_relative_rmse'),
        # What the integrator actually used to weight it.
        ('corridors_relative_uncertainty', 'relative_uncertainty'),
    )

    # String provenance, written as group attributes rather than variables so
    # they cost nothing and stay readable with ncdump -h.
    FLP_DIAGNOSTIC_ATTRS = (
        ('flp_fit_status', 'flp_fit_status'),
        ('flp_fit_method', 'flp_fit_method'),
        ('rescale_method', 'rescale_method'),
        ('hydrograph_source', 'q_source'),
        ('qbar_source', 'qbar_source'),
        ('q33_source', 'q33_source'),
        ('geometry_status', 'geometry_status'),
        ('width_source', 'width_source'),
        ('flp_export_reason', 'flp_export_reason'),
    )

    def _expand_q_to_full_record(self, algorithm, reach, iInsert, nDelete,
                                 fillvalue, key='q'):
        """Put back the timesteps extract_swot() dropped, whatever shape q is in.

        This used to be ``np.insert(q, iInsert, fillvalue, 1)``, which assumed q
        was always ``(1, nt)``.  That held only because every ``*_flowlaw`` ends
        with a reshape; anything reaching here as a 1-D array raised
        ``AxisError`` and killed ``write_output`` for the entire basin -- all
        six algorithms, every reach -- over one malformed hydrograph.

        The shape is normalised rather than validated.  Raising here would be
        the same failure with a friendlier message: the point of writing output
        is that a partly-bad basin still produces files, with the bad parts
        marked.  A repair is recorded in ``self.q_shape_repairs`` so it is
        visible rather than silent.

        Note the helper is local to this module on purpose -- see ``_as_row``.
        """
        integ = self.alg_dict[algorithm][reach]['integrator']
        nt_valid = int(self.obs_dict[reach]['nt']) - int(nDelete)
        if nt_valid < 0:
            nt_valid = 0

        q = integ.get(key)
        if key == 'q' and (q is None or np.size(q) != nt_valid):
            self.q_shape_repairs.append({
                'algorithm': algorithm,
                'reach': reach,
                'size': 0 if q is None else int(np.size(q)),
                'expected': nt_valid,
            })
            warnings.warn(
                'integrator q for %s/%s had %s values, expected %d; padded'
                % (algorithm, reach,
                   'none' if q is None else np.size(q), nt_valid)
            )

        row = _as_row(q, nt_valid)
        if nDelete == 0:
            return row
        return np.insert(row, iInsert, fillvalue, axis=1)

    def _write_flp_diagnostics(self, group, algorithm, reach, fillvalue):
        """Write the FLP fit diagnostics for one algorithm group.

        ``qbar_source`` and ``q33_source`` were already tracked in Integrate.py
        and already drove prior-vs-FLPE uncertainty inside the solve, but were
        never written out -- so a consumer could not tell a real FLPE estimate
        from a prior reconstruction wearing the same variable names.  They are
        written here.
        """
        integ = self.alg_dict.get(algorithm, {}).get(reach, {}).get('integrator', {})

        for name, key, dtype in self.FLP_DIAGNOSTIC_FIELDS:
            variable = group.createVariable(name, dtype, fill_value=fillvalue)
            value = integ.get(key, np.nan)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = np.nan
            if not np.isfinite(value):
                value = fillvalue
            variable[:] = int(value) if dtype.startswith('i') else value

        for key in ('q_flowlaw', 'q_flowlaw_candidate'):
            if key in integ:
                variable = group.createVariable(
                    key, 'f8', ('nt',), fill_value=fillvalue)
                variable[:] = _finite_or_fill(np.ravel(integ[key]), fillvalue)

        # The fitted parameters, whether or not they were exported.
        for key in sorted(k for k in integ if k.startswith('flp_candidate_')):
            variable = group.createVariable(key, 'f8', fill_value=fillvalue)
            variable[:] = _finite_or_fill(integ[key], fillvalue)

        for attr_name, key in self.FLP_DIAGNOSTIC_ATTRS:
            value = integ.get(key, '')
            setattr(group, attr_name, str(value) if value is not None else '')

    def _write_preprocess_config(self, out):
        """Record how observations were prepared; offline reads it back."""
        values = {
            'preprocess_version': flowlaw_preprocess.PREPROCESS_VERSION,
            'observation_filter': self.params_dict.get(
                'Observation_Filter_Version',
                flowlaw_preprocess.DEFAULT_OBSERVATION_FILTER),
            'slope_policy': self.params_dict.get(
                'Slope_Policy', flowlaw_preprocess.DEFAULT_SLOPE_POLICY),
        }
        for key, attribute in flowlaw_preprocess.CONFIG_ATTRIBUTES.items():
            out.setncattr(attribute, values[key])

    def write_output(self):
        """Write data stored to NetCDF files for each reach"""
        fillvalue = -999999999999
        self.q_shape_repairs = []

        if self.out_dir == Path('/mnt/data/output'):
            reaches_to_write = self.basin_dict['reach_ids']
        else:
            reaches_to_write = self.basin_dict['reach_ids_all']

        for reach in reaches_to_write:
             if self.params_dict['write_fill_only']:
                 for algo in self.alg_dict.keys():
                     if 'q' in self.alg_dict[algo][reach]['integrator'].keys():
                         self.alg_dict[algo][reach]['integrator']['q'][:] = np.nan
                     self.alg_dict[algo][reach]['integrator']['qbar'] = np.nan
                     self.alg_dict[algo][reach]['integrator']['q33'] = np.nan
                     self.alg_dict[algo][reach]['integrator']['sbQ_rel'] = np.nan
                     if 'qbar' in self.alg_dict[algo][reach].keys():
                         self.alg_dict[algo][reach]['qbar'] = np.nan

             not_obs = False
             try:
                tmpdata = self.obs_dict[reach]
             except:
                not_obs = True
             if reach not in self.basin_dict['reach_ids']: 
                not_obs = True
             
             if not_obs:
                # NetCDF file creation for unobserved reaches
                out_file = self.out_dir / f"{reach}_integrator.nc"
                out = Dataset(out_file, 'w', format="NETCDF4")
                out.production_date = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
                self._write_preprocess_config(out)
                self._write_bias_correlation_diagnostics(out)
                self._write_gage_metadata(out, reach)

                # 1 busboi
                gb = out.createGroup("busboi")
                gb_qbar_stage2 = gb.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                gb_qbar_stage2[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['integrator']['qbar'], fillvalue)
                gb_sbQ_rel = gb.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                gb_sbQ_rel[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['integrator']['sbQ_rel'], fillvalue)

                # 2 hivdi
                hv = out.createGroup("hivdi")
                hv_qbar_stage2 = hv.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                hv_qbar_stage2[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['qbar'], fillvalue)
                hv_sbQ_rel = hv.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                hv_sbQ_rel[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel'], fillvalue)

                # 3 metroman
                mm = out.createGroup("metroman")
                mm_qbar_stage2 = mm.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                mm_qbar_stage2[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['qbar'], fillvalue)
                mm_sbQ_rel = mm.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                mm_sbQ_rel[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['sbQ_rel'], fillvalue)

                # 4 momma
                mo = out.createGroup("momma")
                mo_qbar_stage2 = mo.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                mo_qbar_stage2[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['qbar'], fillvalue)
                mo_sbQ_rel = mo.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                mo_sbQ_rel[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['sbQ_rel'], fillvalue)

                # 5 sad
                sad = out.createGroup("sad")
                sad_qbar_stage2 = sad.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                sad_qbar_stage2[:] = _finite_or_fill(self.alg_dict['sad'][reach]['integrator']['qbar'], fillvalue)
                sad_sbQ_rel = sad.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                sad_sbQ_rel[:] = _finite_or_fill(self.alg_dict['sad'][reach]['integrator']['sbQ_rel'], fillvalue)

                # 6 sic4dvar
                sic = out.createGroup("sic4dvar")
                sic_qbar_stage2 = sic.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                sic_qbar_stage2[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['integrator']['qbar'], fillvalue)
                sic_sbQ_rel = sic.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                sic_sbQ_rel[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel'], fillvalue)
                
                out.close()
                continue
             
             iDelete = self.obs_dict[reach]['iDelete']
             shape_iDelete = np.shape(iDelete)
             nDelete = shape_iDelete[1]
             iInsert = iDelete - np.arange(nDelete)
             iInsert = np.reshape(iInsert,[nDelete,]) 
             self.obs_dict[reach]['nt'] += nDelete

             for algo in ['busboi', 'hivdi', 'metroman', 'momma', 'sad', 'sic4dvar']:
                 integ = self.alg_dict[algo][reach]['integrator']
                 integ['q'] = self._expand_q_to_full_record(
                     algo, reach, iInsert, nDelete, fillvalue)
                 for key in ('q_flowlaw', 'q_flowlaw_candidate'):
                     if key in integ:
                         integ[key] = self._expand_q_to_full_record(
                             algo, reach, iInsert, nDelete, fillvalue, key=key)

             # NetCDF file creation for observed reaches
             out_file = self.out_dir / f"{reach}_integrator.nc"
             out = Dataset(out_file, 'w', format="NETCDF4")
             out.production_date = datetime.now().strftime('%d-%b-%Y %H:%M:%S')
             self._write_preprocess_config(out)
             self._write_bias_correlation_diagnostics(out)
             self._write_gage_metadata(out, reach)

             out.createDimension("nt", self.obs_dict[reach]['nt'])
             nt = out.createVariable("nt", "i4", ("nt",))
             nt.units = "time steps"
             nt[:] = range(self.obs_dict[reach]['nt'])
             self._write_time_strings(out, reach)

             # 1 busboi
             gb = out.createGroup("busboi")
             gbq = gb.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             gbq[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['integrator']['q'], fillvalue)
             gb_a0 = gb.createVariable("a0", "f8", fill_value=fillvalue)
             gb_a0[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['integrator']['a0'], fillvalue)
             gb_n = gb.createVariable("n", "f8", fill_value=fillvalue)
             gb_n[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['integrator']['n'], fillvalue)
             gb_qbar_s1 = gb.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 gb_qbar_s1[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['qbar'], fillvalue)
             except:
                 gb_qbar_s1[:] = np.nan
             gb_qbar_s2 = gb.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             gb_qbar_s2[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['integrator']['qbar'], fillvalue)
             gb_sbQ = gb.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             gb_sbQ[:] = _finite_or_fill(self.alg_dict['busboi'][reach]['integrator']['sbQ_rel'], fillvalue)
             self._write_flp_diagnostics(gb, 'busboi', reach, fillvalue)

             # 2 hivdi
             hv = out.createGroup("hivdi")
             hvq = hv.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             hvq[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['q'], fillvalue)
             hv_Abar = hv.createVariable("Abar", "f8", fill_value=fillvalue)
             hv_Abar[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['Abar'], fillvalue)
             hv_alpha = hv.createVariable("alpha", "f8", fill_value=fillvalue)
             hv_alpha[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['alpha'], fillvalue)
             hv_beta = hv.createVariable("beta", "f8", fill_value=fillvalue)
             hv_beta[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['beta'], fillvalue)
             hv_qbar_s1 = hv.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 hv_qbar_s1[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['qbar'], fillvalue)
             except:
                 hv_qbar_s1[:] = np.nan
             hv_qbar_s2 = hv.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             hv_qbar_s2[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['qbar'], fillvalue)
             hv_sbQ = hv.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             hv_sbQ[:] = _finite_or_fill(self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel'], fillvalue)
             self._write_flp_diagnostics(hv, 'hivdi', reach, fillvalue)

             # 3 metroman
             mm = out.createGroup("metroman")
             mmq = mm.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             mmq[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['q'], fillvalue)
             mm_Abar = mm.createVariable("Abar", "f8", fill_value=fillvalue)
             mm_Abar[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['a0'], fillvalue)
             mm_na = mm.createVariable("na", "f8", fill_value=fillvalue)
             mm_na[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['na'], fillvalue)
             mm_x1 = mm.createVariable("x1", "f8", fill_value=fillvalue)
             mm_x1[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['x1'], fillvalue)
             mm_qbar_s1 = mm.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 mm_qbar_s1[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['qbar'], fillvalue)
             except:
                 mm_qbar_s1[:] = np.nan
             mm_qbar_s2 = mm.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             mm_qbar_s2[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['qbar'], fillvalue)
             mm_q33 = mm.createVariable("q33_basinScale", "f8", fill_value=fillvalue)
             mm_q33[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['q33'], fillvalue)
             mm_sbQ = mm.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             mm_sbQ[:] = _finite_or_fill(self.alg_dict['metroman'][reach]['integrator']['sbQ_rel'], fillvalue)
             self._write_flp_diagnostics(mm, 'metroman', reach, fillvalue)

             # 4 momma
             mo = out.createGroup("momma")
             moq = mo.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             moq[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['q'], fillvalue)
             mo_B = mo.createVariable("B", "f8", fill_value=fillvalue)
             mo_B[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['B'], fillvalue)
             mo_H = mo.createVariable("H", "f8", fill_value=fillvalue)
             mo_H[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['H'], fillvalue)
             mo_S = mo.createVariable("Save", "f8", fill_value=fillvalue)
             mo_S[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['Save'], fillvalue)
             mo_qbar_s1 = mo.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 mo_qbar_s1[:] = _finite_or_fill(self.alg_dict['momma'][reach]['qbar'], fillvalue)
             except:
                 mo_qbar_s1[:] = np.nan
             mo_qbar_s2 = mo.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             mo_qbar_s2[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['qbar'], fillvalue)
             mo_sbQ = mo.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             mo_sbQ[:] = _finite_or_fill(self.alg_dict['momma'][reach]['integrator']['sbQ_rel'], fillvalue)
             self._write_flp_diagnostics(mo, 'momma', reach, fillvalue)

             # 5 sad
             sd = out.createGroup("sad")
             sdq = sd.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             sdq[:] = _finite_or_fill(self.alg_dict['sad'][reach]['integrator']['q'], fillvalue)
             sd_n = sd.createVariable("n", "f8", fill_value=fillvalue)
             sd_n[:] = _finite_or_fill(self.alg_dict['sad'][reach]['integrator']['n'], fillvalue)
             sd_a0 = sd.createVariable("a0", "f8", fill_value=fillvalue)
             sd_a0[:] = _finite_or_fill(self.alg_dict['sad'][reach]['integrator']['a0'], fillvalue)
             sd_qbar_s1 = sd.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 sd_qbar_s1[:] = _finite_or_fill(self.alg_dict['sad'][reach]['qbar'], fillvalue)
             except:
                 sd_qbar_s1[:] = np.nan
             sd_qbar_s2 = sd.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             sd_qbar_s2[:] = _finite_or_fill(self.alg_dict['sad'][reach]['integrator']['qbar'], fillvalue)
             sd_sbQ = sd.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             sd_sbQ[:] = _finite_or_fill(self.alg_dict['sad'][reach]['integrator']['sbQ_rel'], fillvalue)
             self._write_flp_diagnostics(sd, 'sad', reach, fillvalue)

             # 6 sic4dvar
             sic = out.createGroup("sic4dvar")
             sicq = sic.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             sicq[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['integrator']['q'], fillvalue)
             sic_n = sic.createVariable("n", "f8", fill_value=fillvalue)
             sic_n[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['integrator']['n'], fillvalue)
             sic_a0 = sic.createVariable("a0", "f8", fill_value=fillvalue)
             sic_a0[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['integrator']['a0'], fillvalue)
             sic_qbar_s1 = sic.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 sic_qbar_s1[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['qbar'], fillvalue)
             except:
                 sic_qbar_s1[:] = np.nan
             sic_qbar_s2 = sic.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             sic_qbar_s2[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['integrator']['qbar'], fillvalue)
             sic_sbQ = sic.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             sic_sbQ[:] = _finite_or_fill(self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel'], fillvalue)
             self._write_flp_diagnostics(sic, 'sic4dvar', reach, fillvalue)

             out.close()

    def write_sword_output(self, branch):
        """Make a new copy of the SWORD file, and write the Confluence estimates into the file."""
        sword_src_file = self.sword_dir.joinpath(self.basin_dict['sword'])
        
        if self.out_dir == Path('/mnt/data/output'):
            sword_dest_file = self.sword_dir.joinpath(self.basin_dict['sword'].replace('.nc', '_moi.nc'))
        else:
            print('Debug mode: writing to output directory')
            sword_dest_file = self.out_dir.joinpath(self.basin_dict['sword'].replace('.nc', '_moi.nc'))

        if not os.path.exists(sword_dest_file):
            shutil.copy(sword_src_file, sword_dest_file)

        sword_dataset = None
        try_cnt = 0
        while try_cnt < 20:
            try:
                sword_dataset = Dataset(sword_dest_file, 'a')
                break
            except Exception as e:
                print(e, 'waiting...')
                wait_random(2, 30)
                try_cnt += 1

        if sword_dataset is None:
            raise RuntimeError(f'Could not open SWORD file: {sword_dest_file}')

        try:
            reaches = sword_dataset['reaches']['reach_id'][:]
            for reach in self.basin_dict['reach_ids']:
                reach_ind = np.where(reaches == np.int64(reach))
                if reach_ind[0].size != 1:
                    raise RuntimeError(
                        f'Expected one SWORD match for reach {reach}, '
                        f'found {reach_ind[0].size}'
                    )
                try:
                    # 1) BAM branch (using busboi outputs)
                    sword_dataset['reaches']['discharge_models'][branch]['BAM']['Abar'][reach_ind] = \
                        self.alg_dict['busboi'][reach]['integrator']['a0']
                    sword_dataset['reaches']['discharge_models'][branch]['BAM']['n'][reach_ind] = \
                        self.alg_dict['busboi'][reach]['integrator']['n']
                    sword_dataset['reaches']['discharge_models'][branch]['BAM']['sbQ_rel'][reach_ind] = \
                        self.alg_dict['busboi'][reach]['integrator']['sbQ_rel']

                    # 2) HiVDI
                    sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['Abar'][reach_ind] = \
                        self.alg_dict['hivdi'][reach]['integrator']['Abar']
                    sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['alpha'][reach_ind] = \
                        self.alg_dict['hivdi'][reach]['integrator']['alpha']
                    sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['beta'][reach_ind] = \
                        self.alg_dict['hivdi'][reach]['integrator']['beta']
                    sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['sbQ_rel'][reach_ind] = \
                        self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel']

                    # 3) MetroMan
                    sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['Abar'][reach_ind] = \
                        self.alg_dict['metroman'][reach]['integrator']['a0']
                    sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['ninf'][reach_ind] = \
                        self.alg_dict['metroman'][reach]['integrator']['na']
                    sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['p'][reach_ind] = \
                        self.alg_dict['metroman'][reach]['integrator']['x1']
                    sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['sbQ_rel'][reach_ind] = \
                        self.alg_dict['metroman'][reach]['integrator']['sbQ_rel']

                    # 4) MOMMA
                    sword_dataset['reaches']['discharge_models'][branch]['MOMMA']['B'][reach_ind] = \
                        self.alg_dict['momma'][reach]['integrator']['B']
                    sword_dataset['reaches']['discharge_models'][branch]['MOMMA']['H'][reach_ind] = \
                        self.alg_dict['momma'][reach]['integrator']['H']
                    sword_dataset['reaches']['discharge_models'][branch]['MOMMA']['Save'][reach_ind] = \
                        self.alg_dict['momma'][reach]['integrator']['Save']

                    # 5) SADS
                    sword_dataset['reaches']['discharge_models'][branch]['SADS']['Abar'][reach_ind] = \
                        self.alg_dict['sad'][reach]['integrator']['a0']
                    sword_dataset['reaches']['discharge_models'][branch]['SADS']['n'][reach_ind] = \
                        self.alg_dict['sad'][reach]['integrator']['n']
                    sword_dataset['reaches']['discharge_models'][branch]['SADS']['sbQ_rel'][reach_ind] = \
                        self.alg_dict['sad'][reach]['integrator']['sbQ_rel']

                    # 6) SIC4DVar
                    sword_dataset['reaches']['discharge_models'][branch]['SIC4DVar']['Abar'][reach_ind] = \
                        self.alg_dict['sic4dvar'][reach]['integrator']['a0']
                    sword_dataset['reaches']['discharge_models'][branch]['SIC4DVar']['n'][reach_ind] = \
                        self.alg_dict['sic4dvar'][reach]['integrator']['n']
                    sword_dataset['reaches']['discharge_models'][branch]['SIC4DVar']['sbQ_rel'][reach_ind] = \
                        self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel']
                except Exception as e:
                    print(reach, 'data not found for sword...', e)
        except Exception as e:
            raise RuntimeError(f'Error during SWORD writing: {e}') from e
        finally:
            if sword_dataset:
                sword_dataset.close()

# Standard imports
import warnings
import sys
import datetime
import csv
import copy

# Third-party imports
import numpy as np
import pandas as pd
from scipy import optimize
import scipy.sparse as sp

from moi.sfoi_math_core import (
    adjust_lsq_bias_correlated_sparse,
    adjust_lsq_mult_sparse,
    build_sparse_sfoi_problem,
    estimate_q_uncertainty_rel,
)

class Integrate:
    """Integrates reach-level FLPE algorithm data using Fast Sparse SFOI core.
    Maintains 100% downstream compatibility with Output.py expectations.
    """

    def __init__(
        self,
        alg_dict,
        basin_dict,
        sos_dict,
        sword_dict,
        obs_dict,
        params_dict,
        Branch,
        VerboseFlag,
        gage_dict=None,
    ):
        self.alg_dict = alg_dict
        self.basin_dict = basin_dict
        self.obs_dict = obs_dict
        self.sword_dict = sword_dict
        self.params_dict = params_dict
        self.sos_dict = sos_dict
        self._use_sos_gage_fallback = gage_dict is None
        self.gage_dict = {} if gage_dict is None else dict(gage_dict)
        self.Branch = Branch
        self.VerboseFlag = VerboseFlag
        self.junctions = []
        self.GoodFLPE = {}
        self.mass_diag_static = None
        self.reach_epsilons = {}
        self.gage_diagnostics = {}
        
        self.integ_dict = {
            "pre_q_mean": np.array([]),
            "q_mean": np.array([]),
            "gage_constraints": {},
            "bias_correction": {},
            "solver_reuse": {},
            "flpe": {
                "busboi" : np.array([]),
                "hivdi" : np.array([]),
                "metroman" : np.array([]),
                "momma" : np.array([]),
                "sad" : np.array([]),
                "sic4dvar" : np.array([])
            }
        }

        self.patch_input_deficiencies()

        # if self.VerboseFlag: print('getting pre mean q')
        self.get_pre_mean_q()

        if self.Branch == 'constrained':
            self.prepare_gage_constraints()

    def patch_input_deficiencies(self):
        """
        [CRITICAL] Silently fixes dict structures so Output.py won't throw KeyErrors.
        Output.py HARD requires the 'integrator' dict and specific parameter keys 
        for ALL reaches (even unobserved ones), which Input.py fails to create.
        """
        for alg in self.alg_dict:
            for reach in self.basin_dict['reach_ids_all']:
                if reach not in self.alg_dict[alg]:
                    self.alg_dict[alg][reach] = {'s1-flpe-exists': False, 'qbar': np.nan, 'q33': np.nan}
                
                # Pre-build the integrator dict with NaNs to satisfy Output.py
                if 'integrator' not in self.alg_dict[alg][reach]:
                    # Determine a safe 'nt' for the dummy q array
                    nt = self.obs_dict.get(reach, {}).get('nt', 1)
                    if not isinstance(nt, int) or nt < 1: nt = 1
                    
                    self.alg_dict[alg][reach]['integrator'] = {
                        'qbar': np.nan,
                        'q33': np.nan,
                        'sbQ_rel': np.nan,
                        'q': np.full((1, nt), np.nan)
                    }
                    
                    # Fill algorithm-specific parameter requirements for Output.py
                    if alg == 'busboi':
                        self.alg_dict[alg][reach]['integrator'].update({'n': np.nan, 'a0': np.nan})
                    elif alg == 'hivdi':
                        self.alg_dict[alg][reach]['integrator'].update({'Abar': np.nan, 'alpha': np.nan, 'beta': np.nan})
                    elif alg == 'metroman':
                        self.alg_dict[alg][reach]['integrator'].update({'a0': np.nan, 'na': np.nan, 'x1': np.nan})
                    elif alg == 'momma':
                        self.alg_dict[alg][reach]['integrator'].update({'B': np.nan, 'H': np.nan, 'Save': np.nan})
                    elif alg == 'sad':
                        self.alg_dict[alg][reach]['integrator'].update({'n': np.nan, 'a0': np.nan})
                    elif alg == 'sic4dvar':
                        self.alg_dict[alg][reach]['integrator'].update({'n': np.nan, 'a0': np.nan})

    def get_pre_mean_q(self):
        for alg in self.alg_dict:
            for reach in self.alg_dict[alg]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    reach_data = self.alg_dict[alg][reach]
                    has_flpe = bool(reach_data.get('s1-flpe-exists', False))
                    qbar = np.nan
                    q33 = np.nan
                    if has_flpe:
                        q_val = reach_data.get('q', np.nan)
                        qbar = np.nanmean(q_val)
                        q33 = np.nanquantile(q_val, .33)

                    qbar_is_flpe = bool(has_flpe and np.isfinite(qbar))
                    q33_is_flpe = bool(has_flpe and np.isfinite(q33))
                    if not qbar_is_flpe:
                        qbar = self.sos_dict[str(reach)].get('Qbar', np.nan)
                    if not q33_is_flpe:
                        q33 = self.sos_dict[str(reach)].get('q33', np.nan)

                    reach_data['qbar'] = qbar
                    reach_data['q33'] = q33
                    # Preserve provenance after numerical fallback. Without
                    # these fields, a finite SoS prior is indistinguishable
                    # from a genuine FLPE estimate later in system assembly.
                    reach_data['qbar_source'] = (
                        'FLPE' if qbar_is_flpe else 'Prior'
                    )
                    reach_data['q33_source'] = (
                        'FLPE' if q33_is_flpe else 'Prior'
                    )

    @staticmethod
    def _swot_ordinal_days(times):
        epoch = datetime.datetime(2000, 1, 1)
        ordinals = []
        for value in np.asarray(times, dtype=float).ravel():
            if not np.isfinite(value):
                continue
            try:
                ordinals.append((epoch + datetime.timedelta(seconds=float(value))).toordinal())
            except (OverflowError, ValueError):
                continue
        return np.asarray(ordinals, dtype=np.int64)

    def _merge_sos_gages(self):
        """Use legacy SoS gages only where no explicit SVS gage was loaded."""
        if not getattr(self, '_use_sos_gage_fallback', True):
            return
        for reach, reach_data in self.sos_dict.items():
            reach = str(reach)
            if reach in self.gage_dict:
                continue
            gage = reach_data.get('gage') if isinstance(reach_data, dict) else None
            if not isinstance(gage, dict) or 'Q' not in gage or 't' not in gage:
                continue
            self.gage_dict[reach] = {
                'source': gage.get('source', 'SoS'),
                'station_id': gage.get('station_id', ''),
                't': np.asarray(gage['t']),
                'Q': np.asarray(gage['Q']),
            }

    def prepare_gage_constraints(self):
        """Match gage data to SWOT sampling days and compute mean/q33 constraints."""
        self._merge_sos_gages()
        all_swot_days = []
        for obs in self.obs_dict.values():
            all_swot_days.extend(self._swot_ordinal_days(obs.get('t', [])).tolist())
        all_swot_days = np.unique(np.asarray(all_swot_days, dtype=np.int64))

        min_samples = int(self.params_dict.get('Gage_Min_Matched_Samples', 1))
        match_swot_days = bool(self.params_dict.get('Gage_Match_SWOT_Days', True))
        allow_full_record = bool(self.params_dict.get('Gage_Allow_Full_Record_Fallback', False))
        prepared = {}

        for reach, gage in self.gage_dict.items():
            reach = str(reach)
            q_raw = gage.get('Q', [])
            qt_raw = gage.get('t', [])
            if hasattr(q_raw, 'filled'):
                q_raw = q_raw.filled(np.nan)
            if hasattr(qt_raw, 'filled'):
                qt_raw = qt_raw.filled(np.nan)
            q = np.asarray(q_raw, dtype=float).ravel()
            qt = np.asarray(qt_raw, dtype=float).ravel()
            count = min(q.size, qt.size)
            q = q[:count]
            qt = qt[:count]
            valid = np.isfinite(q) & (q > 0) & np.isfinite(qt)
            q = q[valid]
            qt = qt[valid].astype(np.int64)
            if q.size == 0:
                continue

            if match_swot_days:
                if reach in self.obs_dict:
                    target_days = self._swot_ordinal_days(self.obs_dict[reach].get('t', []))
                else:
                    target_days = all_swot_days
                target_days = np.unique(target_days)
                matched = []
                for day in target_days:
                    index = np.where(qt == day)[0]
                    if index.size:
                        matched.append(float(q[index[0]]))
                matched_q = np.asarray(matched, dtype=float)
            else:
                matched_q = q

            if matched_q.size < min_samples and allow_full_record:
                matched_q = q
            if matched_q.size < min_samples:
                continue

            prepared_gage = dict(gage)
            prepared_gage['Qbar'] = float(np.nanmean(matched_q))
            prepared_gage['q33'] = float(np.nanquantile(matched_q, 0.33))
            prepared_gage['n_matched'] = int(matched_q.size)
            prepared[reach] = prepared_gage

        self.gage_dict = prepared
        if self.VerboseFlag:
            print(f'Prepared {len(self.gage_dict)} independent gage constraints')

    def get_gage_mean_q(self):
        """Backward-compatible alias for the new independent-gage preparation."""
        self.prepare_gage_constraints()

    def build_gage_observation_rows(self, problem, FlowLevel):
        """Build H_g rows selecting physical Q states for gage flow statistics."""
        nvar = problem['A_obs'].shape[1]
        if self.Branch != 'constrained' or FlowLevel not in ('Mean', 'q33'):
            return {
                'A': sp.csr_matrix((0, nvar)),
                'L': np.array([], dtype=float),
                'cov': np.array([], dtype=float),
                'reach_ids': [],
                'station_ids': [],
            }

        reach_index = {
            str(reach): i for i, reach in enumerate(self.basin_dict['reach_ids_all'])
        }
        rows, cols, data = [], [], []
        values, covariances, reaches, stations = [], [], [], []
        default_cov = float(self.params_dict.get('Gage_Uncertainty', 0.10))
        value_key = 'Qbar' if FlowLevel == 'Mean' else 'q33'

        for reach, gage in self.gage_dict.items():
            index = reach_index.get(str(reach))
            value = float(gage.get(value_key, np.nan))
            if index is None or not np.isfinite(value) or value <= 0:
                continue
            row = len(values)
            rows.append(row)
            cols.append(index)
            data.append(1.0)
            values.append(value)
            covariance = float(gage.get('relative_uncertainty', default_cov))
            covariances.append(max(covariance, 1.0e-6))
            reaches.append(str(reach))
            stations.append(str(gage.get('station_id', '')))

        return {
            'A': sp.csr_matrix((data, (rows, cols)), shape=(len(values), nvar)),
            'L': np.asarray(values, dtype=float),
            'cov': np.asarray(covariances, dtype=float),
            'reach_ids': reaches,
            'station_ids': stations,
        }

    def pull_sword_attributes_for_reach(self, k):
         sword_data_reach = {}
         for key in self.sword_dict:
             if np.shape(self.sword_dict[key]) == (self.sword_dict['num_reaches'],):
                 sword_data_reach[key] = self.sword_dict[key][k]
         for key in self.sword_dict:            
             if key == 'rch_id_up':
                 sword_data_reach[key] = self.sword_dict[key][0:sword_data_reach['n_rch_up'], k]
             elif key == 'rch_id_dn':
                 sword_data_reach[key] = self.sword_dict[key][0:sword_data_reach['n_rch_down'], k]
             elif key == 'swot_orbits':
                 sword_data_reach[key] = self.sword_dict[key][0:sword_data_reach['swot_obs'], k]
         return sword_data_reach

    def ChecksPriorToAddingJunction(self, junction_to_check):
         AlreadyExists = False
         for junction in self.junctions:
             if junction['upflows'] == junction_to_check['upflows'] and junction['downflows'] == junction_to_check['downflows']: 
                 AlreadyExists = True    
         AllReachesInReachFile = True
         for r in junction_to_check['upflows']:
             if str(r) not in self.basin_dict['reach_ids_all']:
                 AllReachesInReachFile = False
         for r in junction_to_check['downflows']:
             if str(r) not in self.basin_dict['reach_ids_all']:
                 AllReachesInReachFile = False            
         return AlreadyExists, AllReachesInReachFile

    def CreateJunctionList(self):
         self.junctions = list()
         self.junctions_valid = True

         for reach in self.basin_dict['reach_ids_all']:
             reach = np.int64(reach)
             k = np.argwhere(self.sword_dict['reach_id'] == reach)[0, 0]
             sword_data_reach = self.pull_sword_attributes_for_reach(k) 

             # Upstream
             junction_up = {'originating_reach_id': reach, 'upflows': list()}
             for i in range(sword_data_reach['n_rch_up']):
                 junction_up['upflows'].append(sword_data_reach['rch_id_up'][i])
       
             if len(junction_up['upflows']) > 0:
                 junction_up['downflows'] = list()
                 if not any(junction_up['upflows']):
                    self.junctions_valid = False
                    continue
                 kup = np.argwhere(self.sword_dict['reach_id'] == junction_up['upflows'][0])[0, 0]
                 sword_data_reach_up = self.pull_sword_attributes_for_reach(kup)
                 for j in range(sword_data_reach_up['n_rch_down']):
                     junction_up['downflows'].append(sword_data_reach_up['rch_id_dn'][j])
                 AlreadyExists, AllReachesInReachFile = self.ChecksPriorToAddingJunction(junction_up)
                 if not AlreadyExists and AllReachesInReachFile:
                     self.junctions.append(junction_up)

             # Downstream
             junction_dn = {'originating_reach_id': reach, 'downflows': list()}
             for i in range(sword_data_reach['n_rch_down']):
                 junction_dn['downflows'].append(sword_data_reach['rch_id_dn'][i])            

             if len(junction_dn['downflows']) > 0:
                 junction_dn['upflows'] = list()
                 if not any(junction_dn['downflows']):
                    self.junctions_valid = False
                    continue
                 kdn = np.argwhere(self.sword_dict['reach_id'] == junction_dn['downflows'][0])[0, 0]
                 sword_data_reach_dn = self.pull_sword_attributes_for_reach(kdn)
                 for j in range(sword_data_reach_dn['n_rch_up']):
                     junction_dn['upflows'].append(sword_data_reach_dn['rch_id_up'][j])
                 AlreadyExists, AllReachesInReachFile = self.ChecksPriorToAddingJunction(junction_dn)
                 if not AlreadyExists and AllReachesInReachFile:
                     self.junctions.append(junction_dn) 


    def calcG(self, m, n):
        G = np.zeros((m, n))
        for junction in self.junctions:
            row = junction['row_num']
            for upflow in junction['upflows']:
                try:
                     kup = self.basin_dict['reach_ids_all'].index(str(upflow))
                     G[row, kup] = 1
                except ValueError: pass
            for downflow in junction['downflows']:
                try:
                     kdn = self.basin_dict['reach_ids_all'].index(str(downflow))
                     G[row, kdn] = -1
                except ValueError: pass
        return G

    def _summary(self, name, values):
        arr = np.asarray(values, dtype=float).ravel()
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return f'{name}: count={arr.size}, finite=0'
        return (
            f'{name}: count={arr.size}, finite={finite.size}, '
            f'min={np.nanmin(finite):.6g}, p05={np.nanpercentile(finite, 5):.6g}, '
            f'median={np.nanmedian(finite):.6g}, p95={np.nanpercentile(finite, 95):.6g}, '
            f'max={np.nanmax(finite):.6g}'
        )

    @staticmethod
    def _prior_only_solver_signature(
        A,
        L,
        cov,
        A_eq,
        b_eq,
        lb,
        ub,
        x0,
        idxbad,
        fixed_weight,
        robust_eligible,
    ):
        """Return an exact signature for safe reuse of a prior-only solve.

        The signature includes every numerical problem input. Two algorithm
        labels therefore share a solver result only when they have no genuine
        FLPE observations and their ordinary WLS systems are identical.
        """

        def dense_signature(values):
            if values is None:
                return None
            array = np.ascontiguousarray(np.asarray(values))
            return array.shape, array.dtype.str, array.tobytes()

        def sparse_signature(matrix):
            if matrix is None:
                return None
            matrix = sp.csr_matrix(matrix)
            return (
                matrix.shape,
                matrix.indptr.tobytes(),
                matrix.indices.tobytes(),
                matrix.data.tobytes(),
            )

        return (
            sparse_signature(A),
            dense_signature(L),
            dense_signature(cov),
            sparse_signature(A_eq),
            dense_signature(b_eq),
            dense_signature(lb),
            dense_signature(ub),
            dense_signature(x0),
            dense_signature(idxbad),
            dense_signature(fixed_weight),
            dense_signature(robust_eligible),
        )

    def _print_initial_state(self, alg, FlowLevel, Qbar, sigQ, facc, datasource, problem):
        if not self.VerboseFlag:
            return
        sources, counts = np.unique(np.asarray(datasource, dtype=str), return_counts=True)
        source_msg = ', '.join([f'{src}={cnt}' for src, cnt in zip(sources, counts)])
        print(f'      [input] {alg} {FlowLevel}: {source_msg}')
        print(f'      [input] {self._summary("Qbar_input", Qbar)}')
        print(f'      [input] {self._summary("sigQ_input", sigQ)}')
        print(f'      [input] {self._summary("facc", facc)}')
        print(f'      [input] {self._summary("R_prior", problem.get("R_prior", []))}')

    def _print_reach_results(self, alg, FlowLevel, Qbar, Qintegrator, residual, stdQc_rel, datasource, result=None):
        if not self.VerboseFlag:
            return
        print(f'      [result] {alg} {FlowLevel} reach-level values begin')
        print('      [result] reach_id,source,Q_input,Q_integrator,residual,sbQ_rel')
        for i, reach in enumerate(self.basin_dict['reach_ids_all']):
            sbq = stdQc_rel[i] if FlowLevel == 'Mean' and i < len(stdQc_rel) else np.nan
            print(
                f'      [result] {reach},{datasource[i]},'
                f'{Qbar[i]:.12g},{Qintegrator[i]:.12g},{residual[i]:.12g},{sbq:.12g}'
            )
        print(f'      [result] {alg} {FlowLevel} reach-level values end')
        if result is not None:
            print(f'      [result] {self._summary("final_V", result.V)}')
            print(f'      [result] {self._summary("final_W", result.W)}')



    def initialize_integration_vars(self, alg, FlowLevel, PreviousResiduals, n):
        Qbar = np.full(n, np.nan, dtype=float)
        sigQ = np.full(n, np.nan, dtype=float)
        facc = np.full(n, np.nan, dtype=float)
        runoff = np.full(n, np.nan, dtype=float)
        datasource = []
        prior_uncertainty = float(
            self.params_dict.get(
                'Prior_Uncertainty',
                self.params_dict.get('FLPE_Uncertainty', 0.6),
            )
        )

        value_key = 'qbar' if FlowLevel == 'Mean' else 'q33'
        source_key = 'qbar_source' if FlowLevel == 'Mean' else 'q33_source'
        sos_key = 'Qbar' if FlowLevel == 'Mean' else 'q33'

        for i, reach in enumerate(self.basin_dict['reach_ids_all']):
            k = np.argwhere(self.sword_dict['reach_id'] == np.int64(reach))[0, 0]
            facc[i] = self.pull_sword_attributes_for_reach(k)['facc']
            reach_data = self.alg_dict[alg].get(reach, {})
            value = reach_data.get(value_key, np.nan)
            if np.ma.is_masked(value):
                value = np.nan
            source = reach_data.get(source_key)
            if source is None:
                source = (
                    'FLPE'
                    if reach_data.get('s1-flpe-exists', False)
                    and np.isfinite(value)
                    else 'Prior'
                )

            prior_value = self.sos_dict.get(str(reach), {}).get(sos_key, np.nan)
            if np.ma.is_masked(prior_value):
                prior_value = np.nan

            # Screen only genuine FLPE estimates. If one fails, fall back to
            # the named SoS prior and preserve that provenance explicitly.
            if source == 'FLPE':
                nstdev = 10.0
                if (
                    not np.isfinite(value)
                    or (
                        np.isfinite(prior_value)
                        and abs(value - prior_value)
                        > abs(prior_value)
                        * self.params_dict.get('FLPE_Uncertainty', 0.6)
                        * nstdev
                    )
                ):
                    value = prior_value
                    source = 'Prior'
            elif not np.isfinite(value):
                value = prior_value

            Qbar[i] = value
            datasource.append(source if np.isfinite(value) else 'None')

            if source == 'FLPE' and np.isfinite(value):
                if facc[i] > 5000:
                    dynamic_unc = 0.20
                elif facc[i] > 500:
                    dynamic_unc = 0.35
                else:
                    dynamic_unc = 0.75

                if (
                    self.params_dict.get('use_previous_residual_weighting', False)
                    and not np.isnan(PreviousResiduals[alg][i])
                ):
                    raw_sig = max(
                        abs(PreviousResiduals[alg][i]), abs(Qbar[i]) * 0.01
                    ) ** (-(self.params_dict.get('norm', 2.0) - 2.0))
                    sigQ[i] = min(raw_sig, abs(Qbar[i]) * dynamic_unc * 2.0)
                else:
                    sigQ[i] = abs(Qbar[i]) * dynamic_unc
            elif np.isfinite(value):
                sigQ[i] = abs(Qbar[i]) * prior_uncertainty

        valid = np.isfinite(Qbar) & np.isfinite(facc) & (facc > 0)
        runoff[valid] = Qbar[valid] / facc[valid] / 1000**2 * 86400 * 365
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            runoff_avg = np.nanmean(runoff)
        if not np.isfinite(runoff_avg):
            runoff_avg = 315.36

        for i in range(n):
            if not np.isfinite(Qbar[i]):
                Qbar[i] = runoff_avg * facc[i] * 1000**2 / 86400 / 365
                sigQ[i] = Qbar[i] * self.params_dict.get('Fill_Uncertainty', 1.0)
                datasource[i] = 'Fill'

        bignumber = 1.0e9
        sigQmin = 10.0
        for i in range(n):
            if Qbar[i] == 0.0 and np.isnan(PreviousResiduals[alg][i]):
                sigQ[i] = bignumber
            if sigQ[i] < sigQmin:
                sigQ[i] = sigQmin
            if np.isinf(PreviousResiduals[alg][i]):
                Qbar[i] = 5.0
                sigQ[i] = 0.1

        real_flpe_mask = np.asarray(datasource, dtype=str) == 'FLPE'
        self.GoodFLPE[alg] = bool(np.any(real_flpe_mask))
        input_data_ok = bool(np.all(np.isfinite(Qbar)) and np.all(np.isfinite(sigQ)))
        return (
            Qbar,
            sigQ,
            input_data_ok,
            facc,
            datasource,
            real_flpe_mask,
        )

    @staticmethod
    def _require_converged_augmented_result(result):
        """Reject an augmented result that exhausted its outer iterations."""
        if not result.converged:
            raise RuntimeError(
                'bias-correlated outer solver did not converge: '
                f'{result.status}; iterations={result.outer_iterations}; '
                f'last_delta={result.delta[-1] if result.delta else np.nan}'
            )

    def _augmented_solver_features(
        self, FlowLevel, n_gage_rows, n_real_flpe_rows=None
    ):
        """Enable bias/correlation only where calibration gages identify them.

        An ungaged basin has no independent observation that can distinguish a
        systematic FLPE bias or a regional correlation effect from the physical
        discharge state. Such basins therefore use the ordinary MOI solver.
        """
        configured_levels = self.params_dict.get(
            'SFOI_Bias_Flow_Levels', ('Mean',)
        )
        if isinstance(configured_levels, str):
            configured_levels = {
                value.strip()
                for value in configured_levels.split(',')
                if value.strip()
            }
        else:
            configured_levels = set(configured_levels)

        has_identifying_gage = (
            self.Branch == 'constrained'
            and int(n_gage_rows) > 0
            and FlowLevel in configured_levels
            and (
                n_real_flpe_rows is None
                or int(n_real_flpe_rows) > 0
            )
        )
        bias_enabled = (
            has_identifying_gage
            and bool(self.params_dict.get('SFOI_Bias_Augmentation', True))
        )
        correlation_enabled = (
            has_identifying_gage
            and bool(self.params_dict.get('SFOI_Correlation_Enabled', True))
        )
        return bias_enabled, correlation_enabled

    def integrator_optimization_calcs(self, m, n, FlowLevel, PreviousResiduals):
        """Run sparse paper-style SFOI for each reach-scale FLPE algorithm.

        Rebuilt model:
            x = [Q_all, R_regions]

        Observations are Q_all and regional runoff pseudo-observations. Mass
        conservation is imposed only for dependent reaches, so source/headwater
        reaches keep the independent discharge degrees of freedom used by the
        paper's Q_ind parameters. The solver applies the paper-style
        multiplicative-error outer loop and robust outlier inner loop.
        """
        residuals = {}
        prior_only_solver_cache = {}
        for alg in self.alg_dict:
            if self.VerboseFlag:
                print(f'    RUNNING SPARSE PAPER-SFOI for {alg} ({FlowLevel})')

            (
                Qbar,
                sigQ,
                input_data_ok,
                facc,
                datasource,
                real_flpe_mask,
            ) = self.initialize_integration_vars(
                alg, FlowLevel, PreviousResiduals, n
            )
            u_conversion = 1000.0 / (365.25 * 24 * 3600)

            problem = self.build_soft_sfoi_system(n, Qbar, sigQ, facc, u_conversion)
            gage_obs = self.build_gage_observation_rows(problem, FlowLevel)
            self._print_initial_state(alg, FlowLevel, Qbar, sigQ, facc, datasource, problem)

            Success = False
            Qintegrator = np.full((n,), np.nan)
            stdQc_rel = np.full((n,), self.params_dict.get('FLPE_Uncertainty', 0.6), dtype=float)
            x_hat_saved = None
            result = None
            bias_enabled = False
            correlation_enabled = False

            if input_data_ok and self.junctions_valid:
                try:
                    use_hard_mass = bool(self.params_dict.get('SFOI_Hard_Mass', False))
                    if use_hard_mass:
                        solver_A = problem['A_obs']
                        solver_L = problem['L_obs']
                        solver_cov = problem['cov_obs']
                        solver_A_eq = problem['A_eq']
                        solver_b_eq = problem['b_eq']
                        n_soft_mass_rows = 0
                    else:
                        solver_A = problem['A_sfoi']
                        solver_L = problem['L_sfoi']
                        solver_cov = problem['cov_sfoi']
                        solver_A_eq = None
                        solver_b_eq = None
                        n_soft_mass_rows = problem['A_eq'].shape[0]

                    n_rows_before_gages = solver_A.shape[0]
                    if gage_obs['A'].shape[0] > 0:
                        solver_A = sp.vstack([solver_A, gage_obs['A']], format='csr')
                        solver_L = np.concatenate([solver_L, gage_obs['L']])
                        solver_cov = np.concatenate([solver_cov, gage_obs['cov']])

                    # FLPE and runoff observations remain robust-eligible. Gage
                    # rows are fixed-scale and protected from outlier rejection.
                    # Soft mass rows are also protected and fixed-scale so the
                    # inner loop cannot erase the physical coupling.
                    robust_eligible = np.ones(solver_L.size, dtype=bool)
                    fixed_weight = np.zeros(solver_L.size, dtype=bool)
                    if n_soft_mass_rows:
                        mass_start = problem['A_obs'].shape[0]
                        mass_stop = mass_start + n_soft_mass_rows
                        robust_eligible[mass_start:mass_stop] = False
                        fixed_weight[mass_start:mass_stop] = True
                    if gage_obs['A'].shape[0] > 0:
                        robust_eligible[n_rows_before_gages:] = False
                        fixed_weight[n_rows_before_gages:] = True

                    if self.VerboseFlag:
                        mode = 'hard A_eq' if use_hard_mass else 'Mike-style soft sparse A'
                        print(
                            f'      [solver] mode={mode}, rows={solver_A.shape[0]}, '
                            f'variables={solver_A.shape[1]}, soft_mass_rows={n_soft_mass_rows}, '
                            f'gage_rows={gage_obs["A"].shape[0]}'
                        )

                    bias_enabled, correlation_enabled = (
                        self._augmented_solver_features(
                            FlowLevel,
                            gage_obs['A'].shape[0],
                            int(np.sum(real_flpe_mask)),
                        )
                    )

                    cache_key = None
                    reused_from_algorithm = None
                    if not np.any(real_flpe_mask):
                        cache_key = self._prior_only_solver_signature(
                            solver_A,
                            solver_L,
                            solver_cov,
                            solver_A_eq,
                            solver_b_eq,
                            problem['lb'],
                            problem.get('ub', None),
                            problem['x0'],
                            problem.get('idxbad', None),
                            fixed_weight,
                            robust_eligible,
                        )
                        cached = prior_only_solver_cache.get(cache_key)
                        if cached is not None:
                            result = copy.deepcopy(cached['result'])
                            reused_from_algorithm = cached['algorithm']
                            result.reused_from_algorithm = reused_from_algorithm
                            self.integ_dict.setdefault('solver_reuse', {}).setdefault(
                                FlowLevel, {}
                            )[alg] = reused_from_algorithm
                            if self.VerboseFlag:
                                print(
                                    '      [solver] prior-only inputs match '
                                    f'{reused_from_algorithm}; reusing its result for {alg}'
                                )

                    if result is None and (bias_enabled or correlation_enabled):
                        result = adjust_lsq_bias_correlated_sparse(
                            A_obs=solver_A,
                            L=solver_L,
                            cov=solver_cov,
                            n_reaches=n,
                            n_regions=problem['K_regions'],
                            flpe_eligible_mask=real_flpe_mask,
                            correlation_groups=problem['reach_regions'],
                            correlation_rho=(
                                self.params_dict.get('SFOI_Correlation_Rho', 0.20)
                                if correlation_enabled
                                else 0.0
                            ),
                            bias_enabled=bias_enabled,
                            bias_prior_std=self.params_dict.get(
                                'SFOI_Bias_Prior_Std', 0.50
                            ),
                            bias_initial=self.params_dict.get(
                                'SFOI_Bias_Initial', 0.0
                            ),
                            bias_min=self.params_dict.get('SFOI_Bias_Min', -0.80),
                            bias_max=self.params_dict.get('SFOI_Bias_Max', 2.0),
                            correlation_effect_bound=self.params_dict.get(
                                'SFOI_Correlation_Effect_Bound', 8.0
                            ),
                            A_eq=solver_A_eq,
                            b_eq=solver_b_eq,
                            lb=problem['lb'],
                            ub=problem.get('ub', None),
                            x0=problem['x0'],
                            maxiter=self.params_dict.get(
                                'SFOI_Augmented_Maxiter', 40
                            ),
                            change_thresh=self.params_dict.get(
                                'SFOI_Augmented_Change_Thresh', 1.0e-2
                            ),
                            physical_rms_thresh=self.params_dict.get(
                                'SFOI_Augmented_Physical_RMS_Thresh', 1.0e-2
                            ),
                            physical_p95_thresh=self.params_dict.get(
                                'SFOI_Augmented_Physical_P95_Thresh', 2.0e-2
                            ),
                            bias_thresh=self.params_dict.get(
                                'SFOI_Augmented_Bias_Thresh', 1.0e-3
                            ),
                            effect_thresh=self.params_dict.get(
                                'SFOI_Augmented_Effect_Thresh', 1.0e-2
                            ),
                            robust_thresh=self.params_dict.get(
                                'SFOI_Augmented_Robust_Thresh', 1.0e-3
                            ),
                            robust_limit=self.params_dict.get(
                                'SFOI_Outlier_Limit', 2.5
                            ),
                            alpha=self.params_dict.get('SFOI_Alpha', 0.05),
                            robust_damping=self.params_dict.get(
                                'SFOI_Augmented_Robust_Damping', 0.5
                            ),
                            oscillation_damping=self.params_dict.get(
                                'SFOI_Augmented_Oscillation_Damping', 0.5
                            ),
                            oscillation_direction_threshold=self.params_dict.get(
                                'SFOI_Augmented_Oscillation_Direction_Threshold',
                                -0.5,
                            ),
                            minimum_step_relaxation=self.params_dict.get(
                                'SFOI_Augmented_Minimum_Step_Relaxation', 0.05
                            ),
                            relaxation_recovery=self.params_dict.get(
                                'SFOI_Augmented_Relaxation_Recovery', 1.25
                            ),
                            relaxation_recovery_patience=self.params_dict.get(
                                'SFOI_Augmented_Relaxation_Recovery_Patience', 3
                            ),
                            theta_floor=self.params_dict.get(
                                'SFOI_Theta_Floor', 5.0
                            ),
                            w_max=self.params_dict.get('SFOI_W_Max', 1.0e6),
                            fixed_weight_mask=fixed_weight,
                            robust_eligible_mask=robust_eligible,
                            verbose=self.VerboseFlag,
                        )
                    elif result is None:
                        result = adjust_lsq_mult_sparse(
                            A_obs=solver_A,
                            L=solver_L,
                            cov=solver_cov,
                            A_eq=solver_A_eq,
                            b_eq=solver_b_eq,
                            lb=problem['lb'],
                            ub=problem.get('ub', None),
                            x0=problem['x0'],
                            idxbad=problem.get('idxbad', None),
                            maxiter=self.params_dict.get('SFOI_Maxiter', 20),
                            change_thresh=self.params_dict.get(
                                'SFOI_Change_Thresh', 0.05
                            ),
                            inner_itermax=self.params_dict.get(
                                'SFOI_Inner_Maxiter', 15
                            ),
                            outlier_limit=self.params_dict.get(
                                'SFOI_Outlier_Limit', 2.5
                            ),
                            alpha=self.params_dict.get('SFOI_Alpha', 0.05),
                            verbose=self.VerboseFlag,
                            theta_floor=self.params_dict.get(
                                'SFOI_Theta_Floor', 5.0
                            ),
                            w_max=self.params_dict.get('SFOI_W_Max', 1.0e6),
                            n_reaches=n,
                            n_regions=problem['K_regions'],
                            n_soft_mass_rows=n_soft_mass_rows,
                            n_gage_rows=gage_obs['A'].shape[0],
                            fixed_weight_mask=fixed_weight,
                            robust_eligible_mask=robust_eligible,
                        )

                    if cache_key is not None and reused_from_algorithm is None:
                        prior_only_solver_cache[cache_key] = {
                            'algorithm': alg,
                            'result': copy.deepcopy(result),
                        }

                    if (
                        (bias_enabled or correlation_enabled)
                    ):
                        self._require_converged_augmented_result(result)

                    x_hat_saved = result.x
                    if x_hat_saved is not None and np.all(np.isfinite(x_hat_saved[:n])):
                        Qintegrator = np.clip(x_hat_saved[:n], 0.1, np.inf)
                        stdQc_rel = estimate_q_uncertainty_rel(
                            result,
                            n_reaches=n,
                            Qhat=Qintegrator,
                            fallback_rel=self.params_dict.get('FLPE_Uncertainty', 0.6),
                        )
                        Success = True
                        if hasattr(result, 'bias'):
                            final_iteration = (
                                result.diagnostics[-1]
                                if result.diagnostics
                                else {}
                            )
                            bias_diagnostic = {
                                'flow_level': FlowLevel,
                                'enabled': bool(result.bias_enabled),
                                'estimated_bias_fraction': float(result.bias),
                                'estimated_bias_percent': 100.0 * float(result.bias),
                                'bias_std_fraction': float(result.bias_std),
                                'correlation_rho': float(result.correlation_rho),
                                'correlation_effects': np.asarray(
                                    result.correlation_effects, dtype=float
                                ).tolist(),
                                'status': result.status,
                                'converged': bool(result.converged),
                                'outer_iterations': result.outer_iterations,
                                'last_delta': (
                                    float(result.delta[-1])
                                    if result.delta
                                    else np.nan
                                ),
                                'last_physical_rms_delta': float(
                                    final_iteration.get('physical_delta', np.nan)
                                ),
                                'last_physical_p95_delta': float(
                                    final_iteration.get(
                                        'physical_p95_delta', np.nan
                                    )
                                ),
                                'last_raw_delta': float(
                                    final_iteration.get('raw_delta', np.nan)
                                ),
                                'last_robust_delta': float(
                                    final_iteration.get('robust_delta', np.nan)
                                ),
                                'final_So': float(result.So),
                                'n_real_flpe_rows': int(
                                    np.sum(real_flpe_mask)
                                ),
                                'convergence_thresholds': dict(
                                    getattr(result, 'convergence_thresholds', {})
                                ),
                                'oscillation_events': int(
                                    getattr(result, 'oscillation_events', 0)
                                ),
                                'relaxation_recoveries': int(
                                    getattr(result, 'relaxation_recoveries', 0)
                                ),
                                'final_step_relaxation': float(
                                    getattr(result, 'step_relaxation', 1.0)
                                ),
                            }
                            self.integ_dict['bias_correction'].setdefault(alg, {})[
                                FlowLevel
                            ] = bias_diagnostic
                            if self.VerboseFlag:
                                print(
                                    '      [bias-correlation] '
                                    f'{alg} {FlowLevel}: '
                                    f'bias={bias_diagnostic["estimated_bias_percent"]:.3f}% '
                                    f'+/- {100.0 * bias_diagnostic["bias_std_fraction"]:.3f}%, '
                                    f'rho={bias_diagnostic["correlation_rho"]:.3f}'
                                )
                        if gage_obs['A'].shape[0] > 0:
                            gage_hat = np.asarray(gage_obs['A'] @ x_hat_saved).ravel()
                            diagnostics = []
                            observed_field = (
                                'observed_q_mean' if FlowLevel == 'Mean' else 'observed_q33'
                            )
                            estimated_field = (
                                'estimated_q_mean' if FlowLevel == 'Mean' else 'estimated_q33'
                            )
                            for j, reach in enumerate(gage_obs['reach_ids']):
                                observed = float(gage_obs['L'][j])
                                estimated = float(gage_hat[j])
                                diagnostic = {
                                    'reach_id': reach,
                                    'station_id': gage_obs['station_ids'][j],
                                    'flow_level': FlowLevel,
                                    'observed_value': observed,
                                    'estimated_value': estimated,
                                    'residual': observed - estimated,
                                    'residual_percent': 100.0 * (estimated - observed) / max(abs(observed), 1.0e-12),
                                    'relative_uncertainty': float(gage_obs['cov'][j]),
                                }
                                diagnostic[observed_field] = observed
                                diagnostic[estimated_field] = estimated
                                diagnostics.append(diagnostic)
                            self.gage_diagnostics.setdefault(alg, {})[FlowLevel] = diagnostics
                            self.integ_dict['gage_constraints'].setdefault(alg, {})[
                                FlowLevel
                            ] = diagnostics
                        if self.VerboseFlag:
                            print(f"      SFOI status={result.status}, outer_iters={result.outer_iterations}, delta_last={result.delta[-1] if result.delta else np.nan}")
                except Exception as e:
                    print(f"      Sparse paper-SFOI solver failed: {e}")
                    if bias_enabled or correlation_enabled:
                        raise RuntimeError(
                            f'{alg} {FlowLevel} augmented solve failed: {e}'
                        ) from e

            if Success and FlowLevel == 'Mean':
                if not hasattr(self, 'reach_epsilons'):
                    self.reach_epsilons = {}
                self.reach_epsilons[alg] = self.compute_mass_conservation_metrics(x_hat_saved, problem)

            residuals[alg] = Qbar - Qintegrator if Success else np.full((n,), np.nan)
            if Success:
                residuals[alg][Qintegrator < 0.] = np.inf
            self._print_reach_results(alg, FlowLevel, Qbar, Qintegrator, residuals[alg], stdQc_rel, datasource, result if Success else None)

            # Save data into dictionaries expected by Output.py.
            for i, reach in enumerate(self.basin_dict['reach_ids_all']):
                if reach in self.alg_dict[alg]:
                    if FlowLevel == 'Mean':
                        self.alg_dict[alg][reach]['integrator']['qbar'] = float(Qintegrator[i]) if Success else np.nan
                        self.alg_dict[alg][reach]['integrator']['sbQ_rel'] = float(stdQc_rel[i]) if Success else self.params_dict.get('FLPE_Uncertainty', 0.6)
                    elif FlowLevel == 'q33':
                        self.alg_dict[alg][reach]['integrator']['q33'] = float(Qintegrator[i]) if Success else np.nan

        return residuals

    def bam_objfun(self,params,obs,qbar_target,q33_target): 
        qbam=self.bam_flowlaw(params,obs)
        qbam_bar=np.nanmean(qbam)
        y=(qbam_bar-qbar_target)**2
        if not np.isnan(q33_target):
            qbam_33=np.nanquantile(qbam,.33)
            y+=(qbam_33-q33_target)**2 
        return y

    def bam_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        bam_n=params[0]
        bam_Abar=params[1]
        qbam = ((d_x_area+bam_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / bam_n
        qbam=np.reshape(qbam,(1,len(d_x_area)))
        return qbam

    def hivdi_objfun(self,params,obs,qbar_target,q33_target): 
        q=self.hivdi_flowlaw(params,obs)
        qbar=np.nanmean(q)
        y=(qbar-qbar_target)**2
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(q,.33)
            y+=(q33_alg-q33_target)**2 
        return y

    def hivdi_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        hivdi_alpha=params[0]
        hivdi_beta=params[1]
        hivdi_Abar=params[2]
        hivdi_n_inv = hivdi_alpha * ((d_x_area+hivdi_Abar)/reach_width)**hivdi_beta
        qhivdi = ((d_x_area+hivdi_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) * hivdi_n_inv
        qhivdi=np.reshape(qhivdi,(1,len(d_x_area)))
        return qhivdi

    def metroman_objfun(self,params,obs,qbar_target,q33_target): 
        q=self.metroman_flowlaw(params,obs)
        qbar=np.nanmean(q)
        y=abs(qbar-qbar_target)
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(q,.33)
            y+=abs(q33_alg-q33_target)
        return y

    def metroman_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        metro_ninf=params[0]
        metro_p=params[1]
        metro_Abar=params[2]
        metro_n = metro_ninf * ((d_x_area+metro_Abar) / reach_width)**metro_p
        metro_q = ((d_x_area+metro_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / metro_n
        metro_q=np.reshape(metro_q,(1,len(d_x_area)))
        return metro_q

    def momma_objfun(self,params,obs,qbar_target,q33_target,aux_var): 
        q=self.momma_flowlaw(params,obs,aux_var)
        if np.all(np.isnan(q)): return 1e9
        qbar=np.nanmean(q)
        y=(qbar-qbar_target)**2
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(q,.33)
            y+=(q33_alg-q33_target)**2 
        B=params[0]
        H=params[1]
        Db=H-B
        if Db<0.2 and Db >= 0.1: yfac=2.
        elif Db < 0.1: yfac=10.
        else: yfac=1.
        y*=yfac
        return y

    def momma_flowlaw(self,params,obs,aux_var):
        reach_height=obs['h']
        reach_width=obs['w']
        reach_slope=obs['S']
        momma_B=params[0]
        momma_H=params[1]
        momma_Save=aux_var
        momma_r = 2
        momma_nb = 0.11 * momma_Save**0.18
        momma_q=np.empty( (obs['nt'],)) 

        if momma_H <= momma_B+0.1:
             momma_q=np.inf
        else:
             for t in range(obs['nt']):
                  log_factor = np.log10((momma_H-momma_B)/(reach_height[t]-momma_B))
                  if reach_height[t] <= momma_H:
                       momma_n = momma_nb*(1+log_factor)
                  else:
                       momma_n = momma_nb*(1-log_factor)
                  momma_q[t] = (((reach_height[t] - momma_B)*(momma_r/(1+momma_r)))**(5/3) *
                       reach_width[t] * reach_slope[t]**(1/2)) / momma_n
             momma_q=np.reshape(momma_q,(1,len(reach_height)))
        return momma_q

    def sad_objfun(self,params,obs,qbar_target,q33_target): 
        qsad=self.sad_flowlaw(params,obs)
        if np.all(np.isnan(qsad)): return 1e9
        qsad_bar=np.nanmean(qsad)
        y=(qsad_bar-qbar_target)**2
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(qsad,.33)
            y+=(q33_alg-q33_target)**2 
        return y

    def sad_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        sad_n=params[0]
        sad_Abar=params[1]
        qsad = ((d_x_area+sad_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / sad_n
        qsad=np.reshape(qsad,(1,len(d_x_area)))
        return qsad

    def sic4dvar_objfun(self,params,obs,qbar_target): 
        qsic4dvar=self.sic4dvar_flowlaw(params,obs)
        qsic4dvar_bar=np.nanmean(qsic4dvar)
        y=abs(qsic4dvar_bar-qbar_target)
        return y

    def sic4dvar_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        sic4dvar_n=params[0]
        sic4dvar_Abar=params[1]
        qsic4dvar = ((d_x_area+sic4dvar_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / sic4dvar_n
        qsic4dvar=np.reshape(qsic4dvar,(1,len(d_x_area)))
        return qsic4dvar

    def compute_FLPs(self):         
        if self.VerboseFlag: print('CALCULATING BUSBOI surrogate FLPs')
        for reach in self.alg_dict['busboi']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nhat = np.nanmean(self.alg_dict['busboi'][reach].get('n', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (nhat, np.nanmean(self.alg_dict['busboi'][reach].get('a0', np.nan))) if not np.isnan(nhat) else (0.03, Abar_min+10.)
                
            param_bounds = ((0.001, np.inf), (Abar_min, np.inf))
            qbar = self.alg_dict['busboi'][reach]['integrator']['qbar'] 
            q33 = self.alg_dict['busboi'][reach]['integrator'].get('q33', np.nan) 

            try:
                res = optimize.minimize(fun=self.bam_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
                if res.success:
                    param_est = res.x
                else:
                    param_est = init_params
            except Exception:
                param_est = init_params
            
            self.alg_dict['busboi'][reach]['integrator']['n'] = param_est[0]
            self.alg_dict['busboi'][reach]['integrator']['a0'] = param_est[1]
            self.alg_dict['busboi'][reach]['integrator']['q'] = self.bam_flowlaw(param_est, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING HiVDI FLPs')
        for reach in self.alg_dict['hivdi']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alphaflpe = np.nanmean(self.alg_dict['hivdi'][reach].get('alpha', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['hivdi'][reach].get('alpha', np.nan)), np.nanmean(self.alg_dict['hivdi'][reach].get('beta', np.nan)), np.nanmean(self.alg_dict['hivdi'][reach].get('a0', np.nan))) if not np.isnan(alphaflpe) else (33.3, 1.0, Abar_min+10.)
                
            param_bounds = ((0.001, np.inf), (-1e1, 1.e1), (Abar_min, np.inf))
            qbar = self.alg_dict['hivdi'][reach]['integrator']['qbar']
            q33 = self.alg_dict['hivdi'][reach]['integrator'].get('q33', np.nan) 
            res = optimize.minimize(fun=self.hivdi_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
            
            self.alg_dict['hivdi'][reach]['integrator']['alpha'] = res.x[0]
            self.alg_dict['hivdi'][reach]['integrator']['beta'] = res.x[1]
            self.alg_dict['hivdi'][reach]['integrator']['Abar'] = res.x[2]
            self.alg_dict['hivdi'][reach]['integrator']['q'] = self.hivdi_flowlaw(res.x, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING MetroMan FLPs')
        for reach in self.alg_dict['metroman']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                naflpe = np.nanmean(self.alg_dict['metroman'][reach].get('na', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['metroman'][reach].get('na', np.nan)), np.nanmean(self.alg_dict['metroman'][reach].get('x1', np.nan)), np.nanmean(self.alg_dict['metroman'][reach].get('a0', np.nan))) if not np.isnan(naflpe) else (0.03, -1., Abar_min+10.)

            param_bounds = ((0.001, np.inf), (-1e1, 1e1), (Abar_min, np.inf))
            qbar = self.alg_dict['metroman'][reach]['integrator']['qbar']
            q33 = self.alg_dict['metroman'][reach]['integrator'].get('q33', np.nan)
            res = optimize.minimize(fun=self.metroman_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
            
            self.alg_dict['metroman'][reach]['integrator']['na'] = res.x[0]
            self.alg_dict['metroman'][reach]['integrator']['x1'] = res.x[1]
            self.alg_dict['metroman'][reach]['integrator']['a0'] = res.x[2]
            self.alg_dict['metroman'][reach]['integrator']['q'] = self.metroman_flowlaw(res.x, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING MOMMA FLPs')
        for reach in self.alg_dict['momma']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Bflpe = np.nanmean(self.alg_dict['momma'][reach].get('B', np.nan))
                Bmax = np.min(self.obs_dict[reach]['h']) - 0.1
                init_params = (np.nanmean(self.alg_dict['momma'][reach].get('B', np.nan)), np.nanmean(self.alg_dict['momma'][reach].get('H', np.nan))) if not np.isnan(Bflpe) else (Bmax-1.0, Bmax+1.0)
                
            min_H_obs = np.min(self.obs_dict[reach]['h'])
            if min_H_obs - init_params[0] > 10.: init_params = (min_H_obs - 10., min_H_obs)
            param_bounds = ((0.1, Bmax), (Bmax+0.1, np.inf))
            aux_var = self.alg_dict['momma'][reach].get('Save', np.nan)
            if np.isnan(aux_var): aux_var = 20e-5
            qbar = self.alg_dict['momma'][reach]['integrator']['qbar']
            q33 = self.alg_dict['momma'][reach]['integrator'].get('q33', np.nan) 

            try: res = optimize.minimize(fun=self.momma_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33, aux_var), bounds=param_bounds)
            except Exception: res = lambda: None; res.success = False

            if not res.success: param_est = (self.alg_dict['momma'][reach].get('B', np.nan), self.alg_dict['momma'][reach].get('H', np.nan))
            else: param_est = res.x
            
            self.alg_dict['momma'][reach]['integrator']['B'] = param_est[0]
            self.alg_dict['momma'][reach]['integrator']['H'] = param_est[1]
            self.alg_dict['momma'][reach]['integrator']['Save'] = aux_var
            self.alg_dict['momma'][reach]['integrator']['q'] = self.momma_flowlaw(param_est, self.obs_dict[reach], aux_var)

        if self.VerboseFlag: print('CALCULATING SAD FLPs')
        for reach in self.alg_dict['sad']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nflpe = np.nanmean(self.alg_dict['sad'][reach].get('n', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['sad'][reach].get('n', np.nan)), np.nanmean(self.alg_dict['sad'][reach].get('a0', np.nan))) if not np.isnan(nflpe) else (0.03, Abar_min+10.)

            param_bounds = ((0.001, np.inf), (Abar_min, np.inf))
            qbar = self.alg_dict['sad'][reach]['integrator']['qbar']
            q33 = self.alg_dict['sad'][reach]['integrator'].get('q33', np.nan) 
            res = optimize.minimize(fun=self.sad_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
            
            self.alg_dict['sad'][reach]['integrator']['n'] = res.x[0]
            self.alg_dict['sad'][reach]['integrator']['a0'] = res.x[1]
            self.alg_dict['sad'][reach]['integrator']['q'] = self.sad_flowlaw(res.x, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING SIC4DVar FLPs')
        for reach in self.alg_dict['sic4dvar']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nflpe = np.nanmean(self.alg_dict['sic4dvar'][reach].get('n', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['sic4dvar'][reach].get('n', np.nan)), np.nanmean(self.alg_dict['sic4dvar'][reach].get('a0', np.nan))) if not np.isnan(nflpe) else (0.03, Abar_min+10.)

            param_bounds = ((0.001, 10.), (Abar_min, np.inf))
            qbar = self.alg_dict['sic4dvar'][reach]['integrator']['qbar']
            res = optimize.minimize(fun=self.sic4dvar_objfun, x0=init_params, args=(self.obs_dict[reach], qbar), bounds=param_bounds)
            
            self.alg_dict['sic4dvar'][reach]['integrator']['n'] = res.x[0]
            self.alg_dict['sic4dvar'][reach]['integrator']['a0'] = res.x[1]
            self.alg_dict['sic4dvar'][reach]['integrator']['q'] = self.sic4dvar_flowlaw(res.x, self.obs_dict[reach])
    
        # if self.VerboseFlag: print('Enforcing strict array shapes for Output.py compatibility...')
        for alg in self.alg_dict:
            for reach in self.basin_dict['reach_ids_all']:
                if reach in self.obs_dict and 'integrator' in self.alg_dict[alg].get(reach, {}):
                    expected_nt = self.obs_dict[reach].get('nt', 1)
                    if expected_nt < 0: expected_nt = 1
                    
                    q_arr = self.alg_dict[alg][reach]['integrator'].get('q')

                    if q_arr is not None:
                        q_arr = np.atleast_2d(q_arr)
                        if q_arr.shape != (1, expected_nt):
                            new_q = np.full((1, expected_nt), np.nan)
                            
                            copy_len = min(expected_nt, q_arr.shape[1])
                            new_q[0, :copy_len] = q_arr[0, :copy_len]
                                
                            self.alg_dict[alg][reach]['integrator']['q'] = new_q
    

    def build_soft_sfoi_system(self, n, Qbar, sigQ, facc, u_conversion):
        """Build sparse paper-style SFOI system.

        Name retained for backward compatibility with older Integrate.py calls.
        Internally this now builds the rebuilt statistical model, not the old
        Huber/strict-mass stack.
        """
        prefix_len = int(self.params_dict.get('Runoff_Region_Prefix_Length', 6))
        problem = build_sparse_sfoi_problem(
            reach_ids=self.basin_dict['reach_ids_all'],
            sword_dict=self.sword_dict,
            junctions=self.junctions,
            Qbar=Qbar,
            sigQ=sigQ,
            facc=facc,
            u_conversion=u_conversion,
            params_dict=self.params_dict,
            prefix_len=prefix_len,
        )
        if self.VerboseFlag:
            current_basin = str(self.basin_dict['reach_ids_all'][0])[:4] if self.basin_dict['reach_ids_all'] else 'the'
            print(f"      [Topology] Divided {current_basin} basin into {problem['K_regions']} runoff regions using reach_id prefix length {prefix_len}.")
            print(f"      [SFOI] Observation rows={problem['A_obs'].shape[0]}, variables={problem['A_obs'].shape[1]}, dependent mass rows={problem['A_eq'].shape[0]}, Mike-style rows={problem['A_sfoi'].shape[0]}")
        return problem

    def build_mass_diagnostics(self, n, u_conversion):
        """Build static dependent-reach mass diagnostics using the rebuilt model."""
        facc = np.zeros(n, dtype=float)
        for i, reach in enumerate(self.basin_dict['reach_ids_all']):
            try:
                k = np.argwhere(self.sword_dict['reach_id'] == np.int64(reach))[0, 0]
                facc[i] = self.pull_sword_attributes_for_reach(k)['facc']
            except Exception:
                facc[i] = 0.0
        dummy_Q = np.maximum(facc * u_conversion * 315.36, 1.0)
        dummy_sig = np.maximum(dummy_Q * self.params_dict.get('FLPE_Uncertainty', 0.6), 10.0)
        return self.build_soft_sfoi_system(n, dummy_Q, dummy_sig, facc, u_conversion)

    def compute_mass_conservation_metrics(self, x_hat, diag, qmin_local=5.0):
        """Return normalized mass residuals for dependent reaches.

        Independent/source reaches do not have mass equations in the rebuilt
        statistical model, so they are reported as 0.0 in this diagnostic.
        """
        n = diag['n_reaches']
        x_hat = np.asarray(x_hat).ravel()
        Qintegrator = x_hat[:n]

        eps_local = np.zeros(n, dtype=float)
        if diag['A_eq'].shape[0] > 0:
            mass_resid = np.asarray(diag['A_eq'] @ x_hat - diag['b_eq']).ravel()
            eq_idx = np.asarray(diag['eq_reach_indices'], dtype=int)
            denom = np.maximum(np.abs(Qintegrator[eq_idx]), qmin_local)
            eps_local[eq_idx] = np.abs(mass_resid / denom)

        reach_list = [str(r) for r in self.basin_dict['reach_ids_all']]
        return {reach_list[i]: float(eps_local[i]) for i in range(n)}



    def integrate(self):
          """Integrate reach-level FLPE data. (Main Runner)"""
          # if self.VerboseFlag: print('creating junction list')
          self.CreateJunctionList()       

          FlowLevels = ['Mean', 'q33'] 
          m = len(self.junctions)
          for i, junction in enumerate(self.junctions): junction['row_num'] = i    
          n = len(self.basin_dict['reach_ids_all'])
          
          u_conversion = 1000.0 / (365.25 * 24 * 3600)
          self.mass_diag_static = self.build_mass_diagnostics(n, u_conversion)

          if self.VerboseFlag:
              print(f'Number of junctions = {m}, Number of reaches = {n}')

          for FlowLevel in FlowLevels:
              if self.VerboseFlag:
                  print(f'\nRunning flow level {FlowLevel}')
              residuals = {} 
              for alg in self.alg_dict: residuals[alg] = np.full((n,), np.nan)
              
              n_outer_driver = self.params_dict.get('niter', 1) if self.params_dict.get('use_driver_residual_iterations', False) else 1
              for i in range(n_outer_driver):
                  if self.VerboseFlag:
                      print(f'  Running driver iteration {i+1} / {n_outer_driver}')
                  residuals = self.integrator_optimization_calcs(m, n, FlowLevel, residuals)

          if self.params_dict.get('quit_before_flpe', False):
              sys.exit('done with integration... exiting')

          if self.VerboseFlag:
              print('Computing all FLPs (Final Parameter Estimation)')
          self.compute_FLPs()

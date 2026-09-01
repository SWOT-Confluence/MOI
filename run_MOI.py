#!/usr/bin/env python3
import sys
import argparse
import json
import os
import traceback
from pathlib import Path
import warnings

# This will now pull your modified Integrate.py
from moi.Input import Input
from moi.Integrate import Integrate
from moi.Output import Output
from moi.Corridors import Corridors


def get_basin_data(basin_json, index_to_run):
    """Extract basin data by index and return a normalized dictionary."""
    with open(basin_json, 'r') as f:
        data = json.load(f)

    try:
        item = data[index_to_run]
    except (KeyError, IndexError, TypeError):
        item = data

    return {
        "basin_id": item.get("basin_id", "Unknown"),
        "reach_ids": [str(i) for i in item.get("reach_id", [])],
        "sos": item.get("sos"),
        "sword": item.get("sword"),
    }


def get_all_sword_reach_in_basin(input_obj, verbose):
    """Identify all reaches belonging to the current basin from SWORD."""
    basin_level = len(str(input_obj.basin_dict['basin_id']))
    basin_reach_list_all = []

    for reachid in input_obj.sword_dict['reach_id']:
        reachidstr = str(reachid)
        if reachidstr[0:basin_level] == str(input_obj.basin_dict['basin_id']):
            basin_reach_list_all.append(reachidstr)

    if verbose:
        print(f'Total reaches in SWORD for this basin: {len(basin_reach_list_all)}')

    input_obj.basin_dict['reach_ids_all'] = [str(rid) for rid in basin_reach_list_all]
    return input_obj


def set_moi_params():
    """Initialize standard MOI parameters for SFOI."""
    return {
        'FLPE_Uncertainty': 0.67,
        # SoS discharge used when an algorithm has no genuine FLPE estimate.
        # It remains a prior row and is never bias/correlation augmented.
        'Prior_Uncertainty': 0.60,
        'Gage_Uncertainty': 0.10,
        'Gage_Min_Matched_Samples': 1,
        'Gage_Match_SWOT_Days': True,
        'Gage_Allow_Full_Record_Fallback': False,
        'Fill_Uncertainty': 1.0,
        # Soft mass rows target zero, so configure their uncertainty as an
        # absolute discharge residual rather than a coefficient of variation.
        'SFOI_Soft_Mass_Sigma': 5.0,
        # Bias-aware sparse SFOI. Bias is estimated independently for each
        # algorithm and applies only to FLPE discharge rows, never runoff,
        # mass-balance, or gage rows.
        'SFOI_Bias_Augmentation': True,
        'SFOI_Bias_Flow_Levels': ('Mean',),
        'SFOI_Bias_Prior_Std': 0.50,
        'SFOI_Bias_Initial': 0.0,
        'SFOI_Bias_Min': -0.80,
        'SFOI_Bias_Max': 2.0,
        # Within-runoff-region random correlation represented with sparse
        # latent effects. Keep rho modest until calibrated against real errors.
        'SFOI_Correlation_Enabled': True,
        'SFOI_Correlation_Rho': 0.20,
        'SFOI_Correlation_Effect_Bound': 8.0,
        # Below this many genuine FLPE rows, bias_prior_std and
        # correlation_rho are scaled down proportionally (see
        # Integrate.py's dispatch), pulling the estimate toward the
        # bias-free, uncorrelated model as real evidence gets thinner
        # instead of letting a handful of points drive a full-strength
        # per-region correlation structure to its box bounds.
        'SFOI_Augmented_Identification_Reference': 25.0,
        'SFOI_Augmented_Maxiter': 50,
        # Component-specific convergence tolerances. The legacy aggregate
        # threshold remains only as a backward-compatible fallback.
        'SFOI_Augmented_Change_Thresh': 1.0e-2,
        'SFOI_Augmented_Physical_RMS_Thresh': 1.0e-2,
        'SFOI_Augmented_Physical_P95_Thresh': 2.0e-2,
        'SFOI_Augmented_Bias_Thresh': 1.0e-3,
        'SFOI_Augmented_Effect_Thresh': 1.0e-2,
        'SFOI_Augmented_Robust_Thresh': 1.0e-3,
        # The applied physical (Q, R) step must stay under its RMS/p95
        # threshold for this many consecutive iterations, with So under
        # SFOI_Augmented_So_Max, to accept a result -- see the
        # "convergence gates on the physical state only" note in
        # adjust_lsq_bias_correlated_sparse.
        'SFOI_Augmented_Convergence_Hold_Iters': 3,
        'SFOI_Augmented_So_Max': 25.0,
        'SFOI_Augmented_Robust_Damping': 0.50,
        # Progressively shorten a Gauss-Newton step when its direction strongly
        # opposes the preceding step (the signature of a period-two cycle).
        'SFOI_Augmented_Oscillation_Damping': 0.50,
        'SFOI_Augmented_Oscillation_Direction_Threshold': -0.50,
        'SFOI_Augmented_Minimum_Step_Relaxation': 0.05,
        'SFOI_Augmented_Relaxation_Recovery': 1.25,
        'SFOI_Augmented_Relaxation_Recovery_Patience': 3,
        'norm': 0.5,
        'rho': 0.7,
        'niter': 1,
        'method': 'linear',
        # 'rescale': keep the FLPE hydrograph's shape and shift its level to
        # the integrated mean.  'flowlaw': regenerate it from the refitted
        # flow-law parameters (previous behaviour, kept for reproducibility).
        'Integrator_Hydrograph_Method': 'rescale',
        # 'series': fit the flow-law parameters to the integrator hydrograph.
        # 'moments': match qbar and q33 only (previous behaviour).
        'FLP_Fit_Method': 'series',
        # --- flow-law parameter refit ---------------------------------------
        # 'varpro': solve the scale parameter analytically and scan the rest on
        # a bounded grid -- deterministic, restart-free, and able to tell a
        # stalled optimiser apart from a flow law that genuinely cannot fit.
        # 'legacy': the previous single-start L-BFGS-B solve.
        'FLP_Optimizer': 'varpro',
        # 'linear' or 'log'.  Linear least squares on discharge is dominated by
        # high flows; hydrologic error is multiplicative and the rescale step is
        # itself multiplicative, so log space is the arm worth testing.  Default
        # stays linear so the canary matrix keeps a fixed reference.
        'FLP_Fit_Space': 'linear',
        # 'l2', 'soft_l1' or 'huber'.  A robust loss suppresses individual bad
        # d_x_area timesteps -- a geometry dropout, a jump after a missing pass
        # -- without discarding the reach.
        'FLP_Fit_Loss': 'l2',
        # 'none' or 'obs_quality' (weight by SWOT reach_q / xovr_cal_q).
        'FLP_Fit_Weighting': 'none',
        'FLP_Grid_Points': 48,
        'FLP_Exponent_Points': 17,
        'FLP_Local_Refine': True,
        # Below this many usable timesteps a two-parameter fit is interpolation,
        # not estimation; the reach keeps its prior and is flagged.
        'FLP_Min_Valid_Obs': 5,
        # 'mean': shift the FLPE hydrograph's level to the integrated mean.
        # 'powerlaw': q' = a*q**b, matching the integrated mean AND q33, which
        # can also correct a hydrograph whose spread disagrees.
        'Rescale_Transform': 'mean',
        # --- SFOI weighting -------------------------------------------------
        # 'step' reproduces the historical facc thresholds exactly; 'continuous'
        # replaces them with a calibrated power law.  Default 'step' so this
        # cannot move qbar_basinScale until asked.
        'SFOI_Uncertainty_Model': 'step',
        'SFOI_Use_Algorithm_Uncertainty': False,
        'FLPE_Outlier_Nstdev': 10.0,
        # Record non-convergence of the plain solver always; refuse its output
        # only when this is on, so a global run can measure the cost first.
        'SFOI_Require_Plain_Convergence': False,
        'quit_before_flpe': False,
        'apply_patches': False,
        'write_fill_only': False,
        # CORRIDORS field discharge as an extra integrator constraint.
        'UseCORRIDORS': False,
        # Matched field measurements below this leave the reach without a
        # pseudo-gage.  A floor, not a defensible minimum: a one-parameter law
        # fitted to three points is barely constrained.
        'Corridors_Min_Observations': 3,
        # Relative-uncertainty floor for a pseudo-gage.  Matches
        # Gage_Uncertainty, so a well-fitting pseudo-gage carries the same
        # weight as a station and a poorly fitting one is downweighted by its
        # own residual.
        'Corridors_Min_Uncertainty': 0.10,
        # What wins when a reach has both a real gage and a pseudo-gage.
        # False keeps the station, which is the better measurement; True
        # reproduces the original CORRIDORS branch, where the pseudo-gage
        # replaced it.
        'Corridors_Override_Gage': False,
    }


def resolve_index(cli_index):
    """
    Resolve basin index from:
    1) CLI -i/--index
    2) OFFSET + SLURM_ARRAY_TASK_ID
    """
    if cli_index is not None:
        return cli_index

    offset = int(os.environ.get("OFFSET", "0"))
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))

    if "SLURM_ARRAY_TASK_ID" in os.environ or "OFFSET" in os.environ:
        return offset + task_id

    sys.exit("Error: No index found. Please use -i or provide SLURM_ARRAY_TASK_ID/OFFSET.")


def main():
    # Targeted rather than blanket.  A plain filterwarnings("ignore") hid the
    # pipeline's own warnings too -- a CORRIDORS resource being skipped, a
    # reach dropped for an ambiguous SWORD translation -- which are exactly
    # what an operator needs to see.  numpy's per-reach divide-by-zero and
    # invalid-value noise stays suppressed; it would otherwise bury them.
    warnings.simplefilter('ignore', RuntimeWarning)
    warnings.simplefilter('ignore', DeprecationWarning)
    warnings.simplefilter('ignore', FutureWarning)
    warnings.simplefilter('once', UserWarning)

    parser = argparse.ArgumentParser(description='Run SFOI Integration Pipeline')
    parser.add_argument('-i', '--index', type=int, default=None, help='Index of basin in JSON')
    parser.add_argument('-j', '--basinjson', type=str, default='basin.json', help='Name of the basin.json')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    parser.add_argument(
        '--svs-file',
        type=Path,
        default=None,
        help='SVS NetCDF file used for independent Mean and q33 gage constraints.',
    )
    parser.add_argument(
        '--svs-reach-id-col',
        default='reach_id_v17b',
        help='SVS reach-ID variable; compatible v17 alternatives are auto-detected.',
    )
    parser.add_argument(
        '--gage-calval-csv',
        type=Path,
        default=None,
        help=(
            'CSV assigning reach IDs to calibration/validation groups. '
            'Defaults to CalValSeparation_basin_stratified_v2.csv in the '
            'cloned MOI module.'
        ),
    )
    parser.add_argument(
        '--gage-include-json',
        type=Path,
        default=None,
        help='Optional JSON list/dict limiting which reach IDs become gage constraints.',
    )
    parser.add_argument(
        '--gage-exclude-json',
        type=Path,
        default=None,
        help='Optional JSON list/dict of reach IDs reserved from gage constraints.',
    )
    parser.add_argument(
        '--correlation-rho',
        type=float,
        default=None,
        help=(
            'Within-runoff-region FLPE error correlation in [0, 1). '
            'Defaults to the configured value (0.20).'
        ),
    )
    parser.add_argument(
        '--bias-prior-std',
        type=float,
        default=None,
        help='Weak prior standard deviation for the multiplicative FLPE bias.',
    )
    parser.add_argument(
        '--disable-correlation',
        action='store_true',
        help='Disable region-correlated FLPE errors and retain bias augmentation.',
    )
    parser.add_argument(
        '--disable-bias-augmentation',
        action='store_true',
        help='Disable the systematic FLPE bias state.',
    )
    parser.add_argument(
        '--use-corridors',
        action='store_true',
        help='Enable the integration of CORRIDORS data.',
    )
    parser.add_argument(
        '--corridors-dir',
        type=Path,
        # Resolved against this file, not the working directory: the container
        # runs from an arbitrary cwd, and a relative default would silently
        # find nothing there.
        default=Path(__file__).resolve().parent / 'corridors',
        help=(
            'Directory holding the CORRIDORS resource CSVs and the SWORD '
            'v16-v17 translation table. Defaults to the corridors/ directory '
            'beside run_MOI.py, which is where confluence-offline bind-mounts '
            'them in the container.'
        ),
    )
    parser.add_argument(
        '--corridors-override-gage',
        action='store_true',
        help=(
            'Let a CORRIDORS pseudo-gage replace a real gage on the same '
            'reach, as the original CORRIDORS branch did. By default the real '
            'gage is kept, being the better measurement.'
        ),
    )
    parser.add_argument(
        '--corridors-timezone',
        type=str,
        default=None,
        help=(
            'IANA timezone used to read CORRIDORS measurement dates, which are '
            'local calendar dates. Defaults to America/Anchorage; every '
            'resource released so far is Alaskan, but that will not hold.'
        ),
    )
    parser.add_argument(
        '--integrator-hydrograph',
        type=str,
        default=None,
        choices=['rescale', 'flowlaw'],
        help=(
            'How the integrator hydrograph is built. rescale: keep the FLPE '
            "hydrograph's shape and shift its level to the integrated mean. "
            'flowlaw: regenerate it from the refitted flow-law parameters '
            '(behaviour before the rescale change).'
        ),
    )
    parser.add_argument(
        '--flp-fit',
        type=str,
        default=None,
        choices=['series', 'series_log', 'moments'],
        help=(
            'How final flow-law parameters are fitted. series: least squares '
            'against the integrator hydrograph. series_log: the same in log '
            'space, which stops high flows dominating the fit. moments: match '
            'qbar and q33 only (behaviour before the series fit change).'
        ),
    )
    parser.add_argument(
        '--flp-optimizer',
        type=str,
        default=None,
        choices=['varpro', 'legacy'],
        help=(
            'How the flow-law fit is searched. varpro: analytic scale plus a '
            'bounded global scan over the remaining parameters -- '
            'deterministic and restart-free. legacy: single-start L-BFGS-B '
            'from the FLPE prior.'
        ),
    )
    parser.add_argument(
        '--flp-loss',
        type=str,
        default=None,
        choices=['l2', 'soft_l1', 'huber'],
        help='Loss for the flow-law fit. A robust loss down-weights bad dA timesteps.',
    )
    parser.add_argument(
        '--flp-weighting',
        type=str,
        default=None,
        choices=['none', 'obs_quality'],
        help='Per-timestep weighting for the flow-law fit, from SWOT quality flags.',
    )
    parser.add_argument(
        '--no-flp-refine',
        action='store_true',
        help='Skip the local polish after the grid search (grid optimum only).',
    )
    parser.add_argument(
        '--rescale-transform',
        type=str,
        default=None,
        choices=['mean', 'powerlaw'],
        help=(
            'How the FLPE hydrograph is rescaled. mean: multiplicative shift to '
            'the integrated mean. powerlaw: q = a*q**b matching the mean and q33.'
        ),
    )
    parser.add_argument(
        '--sfoi-uncertainty-model',
        type=str,
        default=None,
        choices=['step', 'continuous'],
        help=(
            'FLPE uncertainty as a function of drainage area. step: the '
            'historical facc thresholds. continuous: a calibrated power law '
            'without the discontinuities.'
        ),
    )
    parser.add_argument(
        '--flpe-outlier-nstdev',
        type=float,
        default=None,
        help=(
            'How many sigma an FLPE estimate may sit from the SoS climatology '
            'before it is replaced by the prior (default 10).'
        ),
    )
    parser.add_argument(
        '--require-plain-convergence',
        action='store_true',
        help=(
            'Refuse output from a plain solver run that exhausted its '
            'iterations, instead of only recording it.'
        ),
    )
    parser.add_argument(
        '-b',
        '--branch',
        type=str,
        default='constrained',
        choices=['constrained', 'unconstrained'],
    )
    args = parser.parse_args()

    index_to_run = resolve_index(args.index)

    # Path Configuration
    input_dir = Path("/mnt/data/input")
    flpe_dir = Path("/mnt/data/flpe")
    basin_json = input_dir / args.basinjson
    sword_dir = input_dir / "sword"

    # Output Directories
    output_dir = Path("/mnt/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    basin_data = get_basin_data(basin_json, index_to_run)
    basin_id = basin_data["basin_id"]

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"sfoi_pipeline_{basin_id}.log"

    print("\n=========================================================")
    print(f" Starting Unified Pipeline for Basin {basin_id} (Task {index_to_run})")
    print("=========================================================")

    try:
        with open(log_file, 'w') as log:
            log.write(f"Starting SFOI Pipeline Index {index_to_run}, Basin {basin_id}\n")

            params_dict = set_moi_params()
            if args.use_corridors:
                params_dict['UseCORRIDORS'] = True
            if args.corridors_override_gage:
                params_dict['Corridors_Override_Gage'] = True
            if args.correlation_rho is not None:
                if not 0.0 <= args.correlation_rho < 1.0:
                    raise ValueError('--correlation-rho must satisfy 0 <= rho < 1')
                params_dict['SFOI_Correlation_Rho'] = args.correlation_rho
            if args.bias_prior_std is not None:
                if args.bias_prior_std <= 0:
                    raise ValueError('--bias-prior-std must be positive')
                params_dict['SFOI_Bias_Prior_Std'] = args.bias_prior_std
            if args.disable_correlation:
                params_dict['SFOI_Correlation_Enabled'] = False
            if args.disable_bias_augmentation:
                params_dict['SFOI_Bias_Augmentation'] = False
            if args.integrator_hydrograph is not None:
                params_dict['Integrator_Hydrograph_Method'] = args.integrator_hydrograph
            if args.flp_fit is not None:
                # series_log is the series objective evaluated in log space;
                # it selects a space, not a different fitting method.
                if args.flp_fit == 'series_log':
                    params_dict['FLP_Fit_Method'] = 'series'
                    params_dict['FLP_Fit_Space'] = 'log'
                else:
                    params_dict['FLP_Fit_Method'] = args.flp_fit
            if args.flp_optimizer is not None:
                params_dict['FLP_Optimizer'] = args.flp_optimizer
            if args.flp_loss is not None:
                params_dict['FLP_Fit_Loss'] = args.flp_loss
            if args.flp_weighting is not None:
                params_dict['FLP_Fit_Weighting'] = args.flp_weighting
            if args.no_flp_refine:
                params_dict['FLP_Local_Refine'] = False
            if args.rescale_transform is not None:
                params_dict['Rescale_Transform'] = args.rescale_transform
            if args.sfoi_uncertainty_model is not None:
                params_dict['SFOI_Uncertainty_Model'] = args.sfoi_uncertainty_model
            if args.flpe_outlier_nstdev is not None:
                if args.flpe_outlier_nstdev <= 0:
                    raise ValueError('--flpe-outlier-nstdev must be positive')
                params_dict['FLPE_Outlier_Nstdev'] = args.flpe_outlier_nstdev
            if args.require_plain_convergence:
                params_dict['SFOI_Require_Plain_Convergence'] = True

            # ---------------------------------------------------------
            # 1. INPUT EXTRACTION
            # ---------------------------------------------------------
            print(f"[{basin_id}] Extracting Input Data...")
            input_obj = Input(
                flpe_dir,
                input_dir / "sos",
                input_dir / "swot",
                sword_dir,
                basin_data,
                args.branch,
                args.verbose,
            )
            input_obj.extract_sword()
            input_obj = get_all_sword_reach_in_basin(input_obj, False)
            svs_loaded = False

            try:
                input_obj.extract_swot()
                input_obj.extract_sos()
                input_obj.extract_alg()

                if args.branch == 'constrained':
                    svs_file = args.svs_file
                    if svs_file is None:
                        svs_candidates = sorted((input_dir / 'svs').glob('*SVS*.nc'))
                        if len(svs_candidates) == 1:
                            svs_file = svs_candidates[0]
                        elif len(svs_candidates) > 1:
                            raise RuntimeError(
                                'Multiple SVS files found; select one with --svs-file: '
                                + ', '.join(str(path) for path in svs_candidates)
                            )

                    if svs_file is not None:
                        calval_file = args.gage_calval_csv
                        if calval_file is None:
                            calval_file = Input.default_calval_file()
                        if not calval_file.is_file():
                            raise FileNotFoundError(
                                'Calibration/validation CSV not found. Expected '
                                f'{calval_file}; select another file with '
                                '--gage-calval-csv.'
                            )

                        input_obj.extract_svs(
                            svs_file,
                            reach_id_col=args.svs_reach_id_col,
                            include_reach_ids=args.gage_include_json,
                            exclude_reach_ids=args.gage_exclude_json,
                            calval_file=calval_file,
                        )
                        svs_loaded = True
                    else:
                        warnings.warn(
                            'No SVS file found for constrained MOI; falling back to '
                            'eligible gages embedded in the SoS file.'
                        )
            except Exception as e:
                log.write(f"Data Missing: {str(e)}. Skipping basin.\n")
                print(f"[{basin_id}] SKIP: Essential data missing ({str(e)})")
                sys.exit(1)

            # Optionally extract CORRIDORS data.  A basin with no CORRIDORS
            # resource is the normal case, so nothing here may be fatal.
            corridors_dict = None
            if params_dict.get('UseCORRIDORS') and args.branch == 'constrained':
                if args.corridors_dir and args.corridors_dir.is_dir():
                    print(f"[{basin_id}] Extracting CORRIDORS Data...")
                    try:
                        corridors_kwargs = {}
                        if args.corridors_timezone:
                            corridors_kwargs['timezone'] = args.corridors_timezone
                        corridors_obj = Corridors(
                            args.corridors_dir,
                            input_obj.basin_dict,
                            input_obj.obs_dict,
                            verbose=args.verbose,
                            min_observations=params_dict['Corridors_Min_Observations'],
                            min_uncertainty=params_dict['Corridors_Min_Uncertainty'],
                            **corridors_kwargs,
                        )
                        corridors_dict = corridors_obj.integrate_corridors_data()
                        input_obj.merge_corridors_and_gages(
                            corridors_dict,
                            override_gage=params_dict['Corridors_Override_Gage'],
                        )
                    except Exception as e:
                        corridors_dict = None
                        warnings.warn(
                            f'CORRIDORS extraction failed ({e}); continuing '
                            'without it.'
                        )
                        log.write(f"CORRIDORS extraction failed: {e}\n")
                else:
                    warnings.warn(
                        'UseCORRIDORS is true, but no valid --corridors-dir was '
                        'provided. Proceeding without CORRIDORS.'
                    )

            # ---------------------------------------------------------
            # 2. INTEGRATION
            # ---------------------------------------------------------
            print(f"[{basin_id}] Running Integration Math...")
            obs_dict = getattr(input_obj, 'obs_dict', {})
            integrate_obj = Integrate(
                input_obj.alg_dict,
                input_obj.basin_dict,
                input_obj.sos_dict,
                input_obj.sword_dict,
                obs_dict,
                params_dict,
                args.branch,
                args.verbose,
                # An explicit (even empty) SVS selection must not be refilled
                # from SoS, because that could reintroduce validation gages.
                # CORRIDORS pseudo-gages live in the same dict, so passing it
                # is equally required once any were merged in.
                gage_dict=(
                    getattr(input_obj, 'gage_dict', {})
                    if (svs_loaded or corridors_dict) else None
                ),
                corridors_dict=corridors_dict,
            )

            try:
                integrate_obj.integrate()
                integration_success = True
            except Exception as e:
                integration_success = False
                log.write(f"Integration failed: {e}\n")
                log.write(traceback.format_exc())
                print(f"[{basin_id}] ERROR: Integration failed. Check log.")
                sys.exit(1)

            if integration_success:
                # ---------------------------------------------------------
                # 3. EXPORT SFOI NETCDF
                # ---------------------------------------------------------
                print(f"[{basin_id}] Generating SFOI NetCDF files...")
                params_dict['write_fill_only'] = False
                output_obj = Output(
                    input_obj.basin_dict,
                    output_dir,
                    integrate_obj.integ_dict,
                    input_obj.alg_dict,
                    input_obj.obs_dict,
                    sword_dir,
                    params_dict,
                    gage_groups=getattr(input_obj, 'calval_groups', {}),
                    gage_dict=getattr(integrate_obj, 'gage_dict', {}),
                    corridors_reaches=getattr(input_obj, 'corridors_reaches', set()),
                )
                output_obj.write_output()
                output_obj.write_sword_output(args.branch)
                print(f"[{basin_id}] ✓ NetCDF export complete.")

    except Exception:
        with open(log_file, 'a') as log:
            log.write("\n!!! FATAL ERROR !!!\n")
            log.write(traceback.format_exc())
        print(f"[{basin_id}] FATAL CRASH. Details in: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()

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
        'SFOI_Augmented_Maxiter': 40,
        # Component-specific convergence tolerances. The legacy aggregate
        # threshold remains only as a backward-compatible fallback.
        'SFOI_Augmented_Change_Thresh': 1.0e-2,
        'SFOI_Augmented_Physical_RMS_Thresh': 1.0e-2,
        'SFOI_Augmented_Physical_P95_Thresh': 2.0e-2,
        'SFOI_Augmented_Bias_Thresh': 1.0e-3,
        'SFOI_Augmented_Effect_Thresh': 1.0e-2,
        'SFOI_Augmented_Robust_Thresh': 1.0e-3,
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
        'quit_before_flpe': False,
        'apply_patches': False,
        'write_fill_only': False,
        # parameters to handle Corridors data
        'UseCORRIDORS': False,  
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
    warnings.filterwarnings("ignore")

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
        help='Enable the integration of CORRIDORS data.'
    )
    parser.add_argument(
        '--corridors-dir',
        type=Path,
        default=Path('./corridors/'),
        help='Directory containing multiple CSV files for CORRIDORS data.'
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
    #input_dir=Path('/fs/ess/PAS1926/mike/noatak/confluence_run17dev/run17dev_mnt/input')
    #flpe_dir=Path('/fs/ess/PAS1926/mike/noatak/confluence_run17dev/run17dev_mnt/flpe')
    basin_json = input_dir / args.basinjson
    sword_dir = input_dir / "sword"

    # Output Directories
    output_dir = Path("/mnt/data/output")
    #output_dir=Path('/fs/ess/PAS1926/mike/noatak/confluence_run17dev/run17dev_mnt/moi')
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

            # Optionally extract CORRIDORS data
            corridors_dict = None
            if params_dict.get('UseCORRIDORS') and args.branch == 'constrained':
                if args.corridors_dir and args.corridors_dir.is_dir():
                    print(f"[{basin_id}] Extracting CORRIDORS Data...")
                    corridors_obj = Corridors(
                            args.corridors_dir, 
                            input_obj.basin_dict,
                            input_obj.obs_dict,
                            verbose=args.verbose
                            )
                    corridors_dict = corridors_obj.integrate_corridors_data()
                    input_obj.merge_corridors_and_gages(corridors_dict)
                else:
                    warnings.warn("UseCORRIDORS is true, but no valid --corridors-dir was provided. Proceeding without CORRIDORS.")

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
                gage_dict=getattr(input_obj, 'gage_dict', {}) if svs_loaded else None,
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

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
        'Gage_Uncertainty': 0.10,
        'Fill_Uncertainty': 1.0,
        'norm': 0.5,
        'rho': 0.7,
        'niter': 1,
        'method': 'linear',
        'quit_before_flpe': False,
        'apply_patches': False,
        'write_fill_only': False,
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

            try:
                input_obj.extract_swot()
                input_obj.extract_sos()
                input_obj.extract_alg()
            except Exception as e:
                log.write(f"Data Missing: {str(e)}. Skipping basin.\n")
                print(f"[{basin_id}] SKIP: Essential data missing ({str(e)})")
                sys.exit(0)

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

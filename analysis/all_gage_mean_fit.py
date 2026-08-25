"""All-gage mean-flow fit experiment for selected MOI basins.

This module supports the companion Jupyter notebook
``all_gage_mean_fit_target_basins.ipynb``.  It deliberately uses the gage
diagnostics produced inside :class:`moi.Integrate.Integrate`, so the observed
means in the residual table are exactly the means supplied to the solver.
"""

from __future__ import annotations

from collections.abc import Iterable
import json
from pathlib import Path
import re
import sys
import traceback
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_RESULT_PATTERN = "*_sword_v17_SOS_results_*.nc"


def normalize_algorithm_name(value: Any) -> str:
    """Normalize algorithm labels used in SOS validation products."""
    value = str(value).replace("\x00", "").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"_+", "_", value)
    value = re.sub(r"^(flpe|moi)_+", "", value)
    value = re.sub(r"_+(flpe|moi)$", "", value)
    return value.strip("_")


def normalize_reach_id(value: Any) -> str:
    """Convert an integer-like reach ID to a stable string."""
    if value is None or np.ma.is_masked(value):
        return ""
    text = str(value).replace("\x00", "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(number) or number <= 0:
        return ""
    return str(int(round(number)))


def _decode_char_variable(variable) -> np.ndarray:
    """Decode a NetCDF character or VLEN string variable."""
    from netCDF4 import chartostring

    raw = np.ma.filled(variable[:], b"")
    values = np.asarray(raw)
    if (
        values.dtype.kind in {"S", "U"}
        and values.ndim >= 2
        and values.dtype.itemsize in {1, 4}
    ):
        try:
            values = np.asarray(chartostring(values, encoding="utf-8"))
        except (AttributeError, TypeError, ValueError):
            pass

    def decode_one(value: Any) -> str:
        if isinstance(value, (bytes, np.bytes_)):
            return bytes(value).decode("utf-8", errors="ignore").replace("\x00", "").strip()
        return str(value).replace("\x00", "").strip()

    return np.vectorize(decode_one, otypes=[str])(values)


def _read_numeric_variable(variable) -> np.ndarray:
    """Read a NetCDF numeric variable and convert fill values to NaN."""
    values = np.asarray(np.ma.filled(variable[:], np.nan), dtype=float)
    fill_value = getattr(variable, "_FillValue", None)
    if fill_value is not None:
        try:
            values[np.isclose(values, float(fill_value))] = np.nan
        except (TypeError, ValueError):
            pass
    values[np.abs(values) > 1.0e8] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def _find_reach_ids(dataset: Dataset, expected_size: int) -> np.ndarray | None:
    """Find a one-dimensional reach-ID variable matching validation rows."""
    candidates = []
    if "reaches" in dataset.groups:
        for name in ("reach_id", "reach_ids"):
            if name in dataset.groups["reaches"].variables:
                candidates.append(dataset.groups["reaches"].variables[name])
    for name in ("reach_id", "reach_ids"):
        if name in dataset.variables:
            candidates.append(dataset.variables[name])

    for variable in candidates:
        values = np.asarray(np.ma.filled(variable[:], -1)).squeeze()
        if values.ndim == 1 and values.size == expected_size:
            return values
    return None


def load_moi_validation_samples(
    sos_result_dir: Path,
    file_pattern: str = DEFAULT_RESULT_PATTERN,
    basin_id_digits: int = 4,
) -> pd.DataFrame:
    """Load final-MOI validation nBias samples from continental SOS files."""
    from netCDF4 import Dataset

    sos_result_dir = Path(sos_result_dir)
    paths = sorted(sos_result_dir.glob(file_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No SOS result files matching {file_pattern!r} in {sos_result_dir}"
        )

    tables: list[pd.DataFrame] = []
    for path in paths:
        print(f"Reading final-MOI validation metrics: {path.name}")
        with Dataset(path, mode="r") as dataset:
            if "validation" not in dataset.groups:
                raise KeyError(f"{path.name}: /validation group not found")
            validation = dataset.groups["validation"]
            if "moi" not in validation.groups:
                raise KeyError(f"{path.name}: /validation/moi group not found")
            group = validation.groups["moi"]

            for name in ("nbias", "algo_names", "has_validation"):
                if name not in group.variables:
                    raise KeyError(f"{path.name}: /validation/moi/{name} not found")

            nbias = _read_numeric_variable(group.variables["nbias"])
            algorithms = _decode_char_variable(group.variables["algo_names"])
            algorithms = np.vectorize(normalize_algorithm_name, otypes=[str])(
                algorithms
            )
            has_validation = np.asarray(
                np.ma.filled(group.variables["has_validation"][:], 0),
                dtype=int,
            ).reshape(-1)

            if algorithms.shape != nbias.shape:
                raise ValueError(
                    f"{path.name}: algo_names shape {algorithms.shape} does not "
                    f"match nBias shape {nbias.shape}"
                )
            n_reaches, n_algorithms = nbias.shape
            if has_validation.size != n_reaches:
                raise ValueError(f"{path.name}: has_validation length mismatch")

            reach_ids = _find_reach_ids(dataset, n_reaches)
            if reach_ids is None:
                raise KeyError(f"{path.name}: validation reach IDs not found")

            if "gageid" in group.variables:
                gage_ids = _decode_char_variable(group.variables["gageid"]).reshape(-1)
            else:
                gage_ids = np.full(n_reaches, "", dtype=object)

        row_index = np.repeat(np.arange(n_reaches), n_algorithms)
        reach_id = np.repeat(reach_ids, n_algorithms)
        gage_id = np.repeat(gage_ids, n_algorithms)
        algorithm = algorithms.reshape(-1)
        nbias_flat = nbias.reshape(-1)
        valid = np.repeat(has_validation == 1, n_algorithms) & np.isfinite(nbias_flat)

        table = pd.DataFrame(
            {
                "source_file": path.name,
                "row_index": row_index[valid],
                "reach_id": [normalize_reach_id(v) for v in reach_id[valid]],
                "gage_id": gage_id[valid],
                "algorithm": algorithm[valid],
                "nbias_moi": nbias_flat[valid],
            }
        )
        table = table.loc[table["reach_id"] != ""].copy()
        table["abs_nbias_moi"] = table["nbias_moi"].abs()
        table["basin_id"] = table["reach_id"].str[:basin_id_digits]
        table["validation_gage_key"] = (
            table["source_file"].astype(str)
            + ":"
            + table["row_index"].astype(str)
        )
        tables.append(table)

    result = pd.concat(tables, ignore_index=True)
    if result.empty:
        raise ValueError("No finite final-MOI validation nBias samples were found")
    return result


def read_calval_table(path: Path) -> pd.DataFrame:
    """Read Cal/Val metadata without losing leading zeros in station IDs."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"reach_id_v17b", "basin_id", "group"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Cal/Val CSV is missing columns: {sorted(missing)}")
    table["basin_id"] = table["basin_id"].str.strip()
    table["group"] = table["group"].str.strip().str.lower()
    table["reach_id_v17b"] = table["reach_id_v17b"].map(normalize_reach_id)
    return table


def build_basin_ranking(
    validation_samples: pd.DataFrame,
    calval_table: pd.DataFrame,
    fixed_basins: Iterable[str],
    n_extra_basins: int = 3,
    min_all_gages: int = 10,
    min_validation_gages: int = 3,
    rank_metric: str = "median_abs_nbias_moi",
) -> tuple[pd.DataFrame, list[str]]:
    """Rank gage-rich basins by final-MOI validation |nBias|."""
    fixed_basins = [str(value) for value in fixed_basins]

    calval_counts = (
        calval_table.groupby("basin_id", as_index=True)
        .agg(
            all_gages=("reach_id_v17b", "nunique"),
            calibration_gages=("group", lambda s: int((s == "calibration").sum())),
            validation_gages_calval=("group", lambda s: int((s == "validation").sum())),
        )
    )

    grouped = validation_samples.groupby("basin_id", as_index=True)
    ranking = grouped.agg(
        validation_samples=("abs_nbias_moi", "size"),
        validation_gages=("validation_gage_key", "nunique"),
        algorithms=("algorithm", "nunique"),
        median_abs_nbias_moi=("abs_nbias_moi", "median"),
        mean_abs_nbias_moi=("abs_nbias_moi", "mean"),
        p90_abs_nbias_moi=("abs_nbias_moi", lambda s: float(s.quantile(0.90))),
        max_abs_nbias_moi=("abs_nbias_moi", "max"),
    )
    ranking = ranking.join(calval_counts, how="left")
    for name in (
        "all_gages",
        "calibration_gages",
        "validation_gages_calval",
    ):
        ranking[name] = ranking[name].fillna(0).astype(int)

    if rank_metric not in ranking.columns:
        raise ValueError(
            f"Unknown rank metric {rank_metric!r}; choices: {list(ranking.columns)}"
        )

    ranking["gage_rich_eligible"] = (
        (ranking["all_gages"] >= int(min_all_gages))
        & (ranking["validation_gages"] >= int(min_validation_gages))
    )
    ranking["fixed_basin"] = ranking.index.isin(fixed_basins)
    ranking["selected_extra"] = False

    candidates = ranking.loc[
        ranking["gage_rich_eligible"] & ~ranking["fixed_basin"]
    ].sort_values(
        [rank_metric, "all_gages"],
        ascending=[False, False],
    )
    if len(candidates) < n_extra_basins:
        raise ValueError(
            f"Only {len(candidates)} non-fixed basins meet the gage thresholds; "
            f"need {n_extra_basins}. Lower MIN_ALL_GAGES or MIN_VALIDATION_GAGES."
        )
    extras = candidates.head(n_extra_basins).index.astype(str).tolist()
    ranking.loc[extras, "selected_extra"] = True
    ranking["selected"] = ranking["fixed_basin"] | ranking["selected_extra"]

    ranking = ranking.reset_index().sort_values(
        ["selected", rank_metric, "all_gages"],
        ascending=[False, False, False],
    )
    return ranking.reset_index(drop=True), fixed_basins + extras


def write_all_gage_calval_csv(
    source_csv: Path,
    destination_csv: Path,
    target_basins: Iterable[str],
) -> pd.DataFrame:
    """Write an experiment CSV with every listed target-basin gage in calibration."""
    source_csv = Path(source_csv)
    destination_csv = Path(destination_csv)
    targets = {str(value) for value in target_basins}
    table = read_calval_table(source_csv)
    selected = table["basin_id"].isin(targets)
    missing = targets.difference(table.loc[selected, "basin_id"].unique())
    if missing:
        raise ValueError(f"Target basins missing from Cal/Val CSV: {sorted(missing)}")

    before = (
        table.loc[selected]
        .groupby(["basin_id", "group"])
        .size()
        .rename("count_before")
        .reset_index()
    )
    table.loc[selected, "group"] = "calibration"
    destination_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination_csv, index=False)

    after = (
        table.loc[selected]
        .groupby(["basin_id", "group"])
        .size()
        .rename("count_after")
        .reset_index()
    )
    audit = before.merge(after, on=["basin_id", "group"], how="outer").fillna(0)
    audit["count_before"] = audit["count_before"].astype(int)
    audit["count_after"] = audit["count_after"].astype(int)
    return audit.sort_values(["basin_id", "group"]).reset_index(drop=True)


def discover_svs_file(svs_dir: Path, pattern: str = "*SVS*.nc") -> Path:
    """Return the single SVS file used by the experiment."""
    candidates = sorted(Path(svs_dir).glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one SVS file matching {pattern!r} in {svs_dir}; "
            f"found {len(candidates)}: {[p.name for p in candidates]}"
        )
    return candidates[0]


def _walk_basin_records(value: Any):
    """Yield basin records from common list/dictionary JSON layouts."""
    if isinstance(value, list):
        for item in value:
            yield from _walk_basin_records(item)
        return
    if not isinstance(value, dict):
        return
    if "basin_id" in value and ("reach_id" in value or "reach_ids" in value):
        yield value
        return
    for item in value.values():
        if isinstance(item, (dict, list)):
            yield from _walk_basin_records(item)


def discover_basin_catalog(input_dir: Path) -> dict[str, dict[str, Any]]:
    """Find basin records across JSON files below the run input directory."""
    input_dir = Path(input_dir)
    catalog: dict[str, dict[str, Any]] = {}
    json_paths = sorted(input_dir.rglob("*.json"))
    for path in json_paths:
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue

        for raw in _walk_basin_records(payload):
            basin_id = str(raw.get("basin_id", "")).strip()
            reaches = raw.get("reach_id", raw.get("reach_ids", []))
            if not basin_id or not isinstance(reaches, list):
                continue
            if not raw.get("sos") or not raw.get("sword"):
                continue
            record = {
                "basin_id": basin_id,
                "reach_ids": [normalize_reach_id(value) for value in reaches],
                "sos": raw["sos"],
                "sword": raw["sword"],
                "source_json": str(path),
            }
            record["reach_ids"] = [value for value in record["reach_ids"] if value]
            if basin_id in catalog:
                old = catalog[basin_id]
                comparable = ("reach_ids", "sos", "sword")
                if any(old[key] != record[key] for key in comparable):
                    raise ValueError(
                        f"Conflicting basin records for {basin_id}: "
                        f"{old['source_json']} and {path}"
                    )
            else:
                catalog[basin_id] = record

    if not catalog:
        raise FileNotFoundError(f"No basin records found in JSON files under {input_dir}")
    return catalog


def _load_moi_api(moi_repo: Path):
    """Import the checked-out MOI implementation selected by the notebook."""
    moi_repo = Path(moi_repo).resolve()
    if not (moi_repo / "run_MOI.py").is_file():
        raise FileNotFoundError(f"run_MOI.py not found in MOI_REPO={moi_repo}")
    if str(moi_repo) not in sys.path:
        sys.path.insert(0, str(moi_repo))
    from moi.Input import Input
    from moi.Integrate import Integrate
    from moi.Output import Output
    from run_MOI import get_all_sword_reach_in_basin, set_moi_params

    return Input, Integrate, Output, get_all_sword_reach_in_basin, set_moi_params


def _standardize_gage_diagnostics(
    basin_id: str,
    integrator,
) -> pd.DataFrame:
    """Convert internal mean-flow diagnostics to an analysis-ready table."""
    records: list[dict[str, Any]] = []
    matched_counts = {
        str(reach): int(gage.get("n_matched", 0))
        for reach, gage in integrator.gage_dict.items()
    }
    groups = {
        str(reach): str(gage.get("group", ""))
        for reach, gage in integrator.gage_dict.items()
    }

    for algorithm, flow_levels in integrator.gage_diagnostics.items():
        for diagnostic in flow_levels.get("Mean", []):
            observed = float(diagnostic["observed_value"])
            fitted = float(diagnostic["estimated_value"])
            reach_id = normalize_reach_id(diagnostic["reach_id"])
            residual = fitted - observed
            nbias = residual / observed if observed > 0 else np.nan
            records.append(
                {
                    "basin_id": str(basin_id),
                    "algorithm": str(algorithm),
                    "reach_id": reach_id,
                    "station_id": str(diagnostic.get("station_id", "")),
                    "group": groups.get(reach_id, ""),
                    "n_matched_dates": matched_counts.get(reach_id, 0),
                    "observed_mean_cms": observed,
                    "fitted_mean_cms": fitted,
                    "residual_cms": residual,
                    "nbias": nbias,
                    "abs_nbias": abs(nbias),
                    "residual_percent": 100.0 * nbias,
                    "relative_uncertainty": float(
                        diagnostic.get("relative_uncertainty", np.nan)
                    ),
                }
            )
    columns = [
        "basin_id",
        "algorithm",
        "reach_id",
        "station_id",
        "group",
        "n_matched_dates",
        "observed_mean_cms",
        "fitted_mean_cms",
        "residual_cms",
        "nbias",
        "abs_nbias",
        "residual_percent",
        "relative_uncertainty",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def _solver_diagnostics(basin_id: str, integrator) -> pd.DataFrame:
    """Flatten the mean-flow bias/correlation solver diagnostics."""
    columns = [
        "basin_id",
        "algorithm",
        "status",
        "converged",
        "outer_iterations",
        "final_reduced_chi_square",
        "estimated_bias_fraction",
        "bias_std_fraction",
        "correlation_rho",
        "n_real_flpe_rows",
    ]
    records = []
    for algorithm, levels in integrator.integ_dict.get("bias_correction", {}).items():
        diagnostic = levels.get("Mean")
        if not diagnostic:
            continue
        records.append(
            {
                "basin_id": str(basin_id),
                "algorithm": str(algorithm),
                "status": diagnostic.get("status", "unknown"),
                "converged": bool(diagnostic.get("converged", False)),
                "outer_iterations": diagnostic.get("outer_iterations", np.nan),
                "final_reduced_chi_square": diagnostic.get("final_So", np.nan),
                "estimated_bias_fraction": diagnostic.get(
                    "estimated_bias_fraction", np.nan
                ),
                "bias_std_fraction": diagnostic.get("bias_std_fraction", np.nan),
                "correlation_rho": diagnostic.get("correlation_rho", np.nan),
                "n_real_flpe_rows": diagnostic.get("n_real_flpe_rows", np.nan),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def run_one_basin(
    basin_id: str,
    basin_record: dict[str, Any],
    *,
    moi_repo: Path,
    run_root: Path,
    svs_file: Path,
    all_gage_calval_csv: Path,
    result_root: Path,
    write_reach_netcdf: bool = False,
    verbose: bool = True,
    parameter_overrides: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run constrained MOI for one basin with every listed gage in calibration."""
    (
        Input,
        Integrate,
        Output,
        get_all_sword_reach_in_basin,
        set_moi_params,
    ) = _load_moi_api(moi_repo)

    run_root = Path(run_root)
    result_root = Path(result_root)
    basin_id = str(basin_id)
    basin_data = {
        "basin_id": basin_id,
        "reach_ids": list(basin_record["reach_ids"]),
        "sos": basin_record["sos"],
        "sword": basin_record["sword"],
    }
    params = set_moi_params()
    if parameter_overrides:
        params.update(parameter_overrides)

    input_obj = Input(
        run_root / "flpe",
        run_root / "input" / "sos",
        run_root / "input" / "swot",
        run_root / "input" / "sword",
        basin_data,
        "constrained",
        verbose,
    )
    input_obj.extract_sword()
    input_obj = get_all_sword_reach_in_basin(input_obj, verbose)
    input_obj.extract_swot()
    input_obj.extract_sos()
    input_obj.extract_alg()
    input_obj.extract_svs(
        svs_file,
        reach_id_col="reach_id_v17b",
        calval_file=all_gage_calval_csv,
    )
    n_svs_gages_loaded = len(input_obj.gage_dict)

    integrator = Integrate(
        input_obj.alg_dict,
        input_obj.basin_dict,
        input_obj.sos_dict,
        input_obj.sword_dict,
        getattr(input_obj, "obs_dict", {}),
        params,
        "constrained",
        verbose,
        gage_dict=input_obj.gage_dict,
    )
    n_gages_prepared = len(integrator.gage_dict)
    n_matched_dates = int(
        sum(gage.get("n_matched", 0) for gage in integrator.gage_dict.values())
    )
    integrator.integrate()

    residuals = _standardize_gage_diagnostics(basin_id, integrator)
    if residuals.empty:
        raise RuntimeError(f"Basin {basin_id}: no Mean gage diagnostics were produced")
    solver = _solver_diagnostics(basin_id, integrator)

    basin_dir = result_root / "basins" / basin_id
    basin_dir.mkdir(parents=True, exist_ok=True)
    residual_path = basin_dir / f"basin_{basin_id}_mean_gage_residuals.csv"
    solver_path = basin_dir / f"basin_{basin_id}_solver_diagnostics.csv"
    residuals.to_csv(residual_path, index=False)
    solver.to_csv(solver_path, index=False)

    if write_reach_netcdf:
        netcdf_dir = basin_dir / "moi_reach_netcdf"
        netcdf_dir.mkdir(parents=True, exist_ok=True)
        params["write_fill_only"] = False
        output = Output(
            input_obj.basin_dict,
            netcdf_dir,
            integrator.integ_dict,
            input_obj.alg_dict,
            input_obj.obs_dict,
            run_root / "input" / "sword",
            params,
        )
        output.write_output()

    audit = {
        "basin_id": basin_id,
        "status": "completed",
        "source_json": basin_record.get("source_json", ""),
        "n_basin_reaches": len(input_obj.basin_dict["reach_ids_all"]),
        "n_svs_gages_loaded": n_svs_gages_loaded,
        "n_gages_prepared": n_gages_prepared,
        "n_total_matched_dates": n_matched_dates,
        "n_algorithms_with_residuals": residuals["algorithm"].nunique(),
        "n_residual_rows": len(residuals),
        "residual_csv": str(residual_path),
        "solver_csv": str(solver_path),
    }
    return residuals, solver, audit


def run_selected_basins(
    target_basins: Iterable[str],
    basin_catalog: dict[str, dict[str, Any]],
    *,
    moi_repo: Path,
    run_root: Path,
    svs_file: Path,
    all_gage_calval_csv: Path,
    result_root: Path,
    write_reach_netcdf: bool = False,
    verbose: bool = True,
    parameter_overrides: dict[str, Any] | None = None,
    reuse_existing: bool = True,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run or resume the all-gage experiment for all selected basins."""
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    residual_tables = []
    solver_tables = []
    audits = []

    for basin_value in target_basins:
        basin_id = str(basin_value)
        if basin_id not in basin_catalog:
            raise KeyError(f"Basin {basin_id} is not present in the basin JSON catalog")
        basin_dir = result_root / "basins" / basin_id
        residual_path = basin_dir / f"basin_{basin_id}_mean_gage_residuals.csv"
        solver_path = basin_dir / f"basin_{basin_id}_solver_diagnostics.csv"

        if reuse_existing and residual_path.is_file():
            print(f"[{basin_id}] Reusing existing residual table: {residual_path}")
            residuals = pd.read_csv(residual_path, dtype={"basin_id": str, "reach_id": str})
            solver = (
                pd.read_csv(solver_path, dtype={"basin_id": str})
                if solver_path.is_file()
                else pd.DataFrame()
            )
            residual_tables.append(residuals)
            if not solver.empty:
                solver_tables.append(solver)
            audits.append(
                {
                    "basin_id": basin_id,
                    "status": "reused",
                    "n_gages_prepared": residuals["reach_id"].nunique(),
                    "n_algorithms_with_residuals": residuals["algorithm"].nunique(),
                    "n_residual_rows": len(residuals),
                    "residual_csv": str(residual_path),
                    "solver_csv": str(solver_path),
                }
            )
            continue

        print(f"\n{'=' * 72}\nRunning all-gage constrained MOI for basin {basin_id}\n{'=' * 72}")
        try:
            residuals, solver, audit = run_one_basin(
                basin_id,
                basin_catalog[basin_id],
                moi_repo=moi_repo,
                run_root=run_root,
                svs_file=svs_file,
                all_gage_calval_csv=all_gage_calval_csv,
                result_root=result_root,
                write_reach_netcdf=write_reach_netcdf,
                verbose=verbose,
                parameter_overrides=parameter_overrides,
            )
            residual_tables.append(residuals)
            if not solver.empty:
                solver_tables.append(solver)
            audits.append(audit)
        except Exception as exc:
            basin_dir.mkdir(parents=True, exist_ok=True)
            error_path = basin_dir / f"basin_{basin_id}_error.txt"
            error_path.write_text(traceback.format_exc(), encoding="utf-8")
            audits.append(
                {
                    "basin_id": basin_id,
                    "status": "failed",
                    "error": str(exc),
                    "error_log": str(error_path),
                }
            )
            print(f"[{basin_id}] FAILED: {exc}\nTraceback saved to {error_path}")
            if not continue_on_error:
                raise

        pd.DataFrame(audits).to_csv(result_root / "run_audit.csv", index=False)

    residuals = (
        pd.concat(residual_tables, ignore_index=True)
        if residual_tables
        else pd.DataFrame()
    )
    solver = (
        pd.concat(solver_tables, ignore_index=True)
        if solver_tables
        else pd.DataFrame(
            columns=[
                "basin_id",
                "algorithm",
                "status",
                "converged",
                "outer_iterations",
                "final_reduced_chi_square",
                "estimated_bias_fraction",
                "bias_std_fraction",
                "correlation_rho",
                "n_real_flpe_rows",
            ]
        )
    )
    audit = pd.DataFrame(audits)
    residuals.to_csv(result_root / "all_basins_mean_gage_residuals.csv", index=False)
    solver.to_csv(result_root / "all_basins_solver_diagnostics.csv", index=False)
    audit.to_csv(result_root / "run_audit.csv", index=False)
    return residuals, solver, audit


def summarize_residuals(residuals: pd.DataFrame) -> pd.DataFrame:
    """Calculate fit-quality statistics for each basin and FLPE algorithm."""
    if residuals.empty:
        return pd.DataFrame()
    return (
        residuals.groupby(["basin_id", "algorithm"], as_index=False)
        .agg(
            n_gages=("reach_id", "nunique"),
            median_nbias=("nbias", "median"),
            median_abs_nbias=("abs_nbias", "median"),
            p90_abs_nbias=("abs_nbias", lambda s: float(s.quantile(0.90))),
            p95_abs_nbias=("abs_nbias", lambda s: float(s.quantile(0.95))),
            max_abs_nbias=("abs_nbias", "max"),
            fraction_within_10pct=("abs_nbias", lambda s: float((s <= 0.10).mean())),
            fraction_within_20pct=("abs_nbias", lambda s: float((s <= 0.20).mean())),
            fraction_within_50pct=("abs_nbias", lambda s: float((s <= 0.50).mean())),
        )
        .sort_values(["basin_id", "median_abs_nbias", "algorithm"])
        .reset_index(drop=True)
    )


def _ecdf(values: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return np.array([]), np.array([])
    return values, np.arange(1, values.size + 1) / values.size


def plot_residual_ecdfs(
    residuals: pd.DataFrame,
    basin_order: Iterable[str],
    *,
    signed_xlim: tuple[float, float] = (-1.0, 2.0),
    absolute_xlim: tuple[float, float] = (0.0, 2.0),
):
    """Plot signed and absolute mean-flow nBias ECDFs for every basin."""
    import matplotlib.pyplot as plt

    basin_order = [str(value) for value in basin_order]
    algorithms = sorted(residuals["algorithm"].dropna().unique())
    colors = plt.get_cmap("tab10")(np.linspace(0, 0.9, max(len(algorithms), 1)))
    color_by_algorithm = dict(zip(algorithms, colors))

    figure, axes = plt.subplots(
        len(basin_order),
        2,
        figsize=(14, max(4.0, 3.6 * len(basin_order))),
        squeeze=False,
        sharex="col",
        sharey=True,
    )
    for row, basin_id in enumerate(basin_order):
        basin = residuals.loc[residuals["basin_id"].astype(str) == basin_id]
        for algorithm in algorithms:
            data = basin.loc[basin["algorithm"] == algorithm]
            if data.empty:
                continue
            x_signed, y_signed = _ecdf(data["nbias"])
            x_abs, y_abs = _ecdf(data["abs_nbias"])
            label = f"{algorithm} (n={data['reach_id'].nunique()})"
            axes[row, 0].step(
                x_signed,
                y_signed,
                where="post",
                linewidth=1.8,
                color=color_by_algorithm[algorithm],
                label=label,
            )
            axes[row, 1].step(
                x_abs,
                y_abs,
                where="post",
                linewidth=1.8,
                color=color_by_algorithm[algorithm],
                label=label,
            )

        axes[row, 0].axvline(0.0, color="black", linestyle=":", linewidth=1.0)
        axes[row, 1].axvline(0.10, color="0.35", linestyle=":", linewidth=1.0)
        axes[row, 1].axvline(0.20, color="0.35", linestyle="--", linewidth=1.0)
        axes[row, 0].set_ylabel(f"Basin {basin_id}\nECDF")
        for column in range(2):
            axes[row, column].set_ylim(0.0, 1.01)
            axes[row, column].grid(True, linestyle=":", alpha=0.45)
        axes[row, 0].set_xlim(signed_xlim)
        axes[row, 1].set_xlim(absolute_xlim)

    axes[0, 0].set_title("Signed mean-flow nBias = (fit - gage) / gage")
    axes[0, 1].set_title("Absolute mean-flow |nBias|")
    axes[-1, 0].set_xlabel("Signed nBias")
    axes[-1, 1].set_xlabel("|nBias|")
    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.975),
            ncol=min(6, len(handles)),
            frameon=True,
        )
    figure.suptitle(
        "All-gage calibration: fitted vs observed mean flow",
        y=0.998,
        fontsize=15,
    )
    figure.tight_layout(rect=[0.0, 0.0, 1.0, 0.93])
    return figure

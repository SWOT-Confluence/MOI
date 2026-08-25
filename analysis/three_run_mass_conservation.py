"""Three-run MOI experiment and MetroMan mass-conservation diagnostics.

The module supports the six-basin notebook experiment:

* run1: unconstrained MOI (no gage constraints),
* run2: constrained MOI using the calibration rows in the supplied Cal/Val CSV,
* run3: constrained MOI after assigning every listed gage in the six basins to
  the calibration group.

No reach NetCDF or SWORD output is needed.  By default the experiment solves
only the ``Mean`` flow level and skips final flow-law parameter estimation.
The functions retain the exact in-memory ``qbar`` values from ``Integrate`` in
compact CSV tables, evaluate mass closure at every SWORD junction, and plot the
SWORD river centerlines.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Iterable, Mapping, Sequence

from netCDF4 import Dataset
import numpy as np
import pandas as pd


TARGET_BASINS = ("7429", "7426", "6424", "7521", "7424", "6412")
METROMAN = "metroman"
REACH_TABLE_COLUMNS = [
    "basin_id",
    "run",
    "run_label",
    "branch",
    "reach_id",
    "qbar_reachScale",
    "qbar_basinScale",
    "reach_scale_source",
    "basin_scale_mass_epsilon",
    "is_swot_observed_reach",
    "is_gage_constraint",
    "gage_group",
    "facc",
    "n_rch_up",
    "n_rch_down",
]
JUNCTION_TABLE_COLUMNS = [
    "basin_id",
    "run",
    "run_label",
    "branch",
    "scale",
    "junction_index",
    "originating_reach_id",
    "junction_type",
    "upflow_reach_ids",
    "downflow_reach_ids",
    "n_upflows",
    "n_downflows",
    "sum_upstream_cms",
    "sum_downstream_cms",
    "signed_residual_cms",
    "absolute_residual_cms",
    "relative_residual",
    "allowed_residual_cms",
    "evaluable",
    "conserving",
]


@dataclass(frozen=True)
class RunSpec:
    """Configuration for one experimental run."""

    name: str
    label: str
    branch: str
    calval_csv: Path | None

    @property
    def uses_gages(self) -> bool:
        return self.branch == "constrained"


def normalize_reach_id(value: Any) -> str:
    """Convert integer-like reach identifiers to stable strings."""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>", "--"}:
        return ""
    try:
        return str(int(float(text)))
    except (TypeError, ValueError, OverflowError):
        return text


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
    for path in sorted(input_dir.rglob("*.json")):
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
        raise FileNotFoundError(f"No basin records found below {input_dir}")
    return catalog


def discover_svs_file(svs_dir: Path, pattern: str = "*SVS*.nc") -> Path:
    """Return the single SVS file used by all constrained experiments."""
    candidates = sorted(Path(svs_dir).glob(pattern))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one SVS file matching {pattern!r} in {svs_dir}; "
            f"found {len(candidates)}: {[path.name for path in candidates]}"
        )
    return candidates[0]


def build_run_specs(
    source_calval_csv: Path,
    all_gage_calval_csv: Path,
) -> tuple[RunSpec, ...]:
    """Return the requested three experiment definitions in display order."""
    return (
        RunSpec("run1", "unconstrained MOI", "unconstrained", None),
        RunSpec(
            "run2",
            "Cal/Val calibration gages",
            "constrained",
            Path(source_calval_csv),
        ),
        RunSpec(
            "run3",
            "all gages constrained",
            "constrained",
            Path(all_gage_calval_csv),
        ),
    )


def prepare_all_gage_calval_csv(
    source_csv: Path,
    destination_csv: Path,
    target_basins: Iterable[str] = TARGET_BASINS,
) -> pd.DataFrame:
    """Create run3's CSV without modifying the source Cal/Val assignment."""
    source_csv = Path(source_csv)
    destination_csv = Path(destination_csv)
    targets = {str(value) for value in target_basins}
    table = pd.read_csv(source_csv, dtype=str, keep_default_na=False)
    required = {"reach_id_v17b", "basin_id", "group"}
    missing_columns = required.difference(table.columns)
    if missing_columns:
        raise ValueError(
            f"Cal/Val CSV is missing columns: {sorted(missing_columns)}"
        )
    table["reach_id_v17b"] = table["reach_id_v17b"].map(normalize_reach_id)
    table["basin_id"] = table["basin_id"].str.strip()
    table["group"] = table["group"].str.strip().str.lower()
    selected = table["basin_id"].isin(targets)
    missing_basins = targets.difference(table.loc[selected, "basin_id"].unique())
    if missing_basins:
        raise ValueError(
            f"Target basins missing from Cal/Val CSV: {sorted(missing_basins)}"
        )

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
    audit[["count_before", "count_after"]] = audit[
        ["count_before", "count_after"]
    ].astype(int)
    return audit.sort_values(["basin_id", "group"]).reset_index(drop=True)


def _load_moi_api(moi_repo: Path):
    """Import the checked-out MOI classes selected by the notebook."""
    moi_repo = Path(moi_repo).resolve()
    if not (moi_repo / "run_MOI.py").is_file():
        raise FileNotFoundError(f"run_MOI.py not found in MOI_REPO={moi_repo}")
    if str(moi_repo) not in sys.path:
        sys.path.insert(0, str(moi_repo))

    from moi.Input import Input
    from moi.Integrate import Integrate
    from run_MOI import get_all_sword_reach_in_basin, set_moi_params

    return Input, Integrate, get_all_sword_reach_in_basin, set_moi_params


def _as_float(value: Any) -> float:
    """Return one scalar as float, replacing masked/non-scalar values by NaN."""
    if np.ma.is_masked(value):
        return np.nan
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.nan
    if values.size != 1 or not np.isfinite(values[0]):
        return np.nan
    return float(values[0])


def _as_int(value: Any, default: int = 0) -> int:
    number = _as_float(value)
    return int(number) if np.isfinite(number) else int(default)


def _boolean_series(values: pd.Series) -> pd.Series:
    """Interpret native booleans and CSV round-tripped True/False strings."""
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _sword_index(sword_dict: Mapping[str, Any]) -> dict[str, int]:
    result = {}
    for index, value in enumerate(np.asarray(sword_dict["reach_id"]).reshape(-1)):
        reach_id = normalize_reach_id(value)
        if reach_id:
            result[reach_id] = index
    return result


def build_metroman_reach_table(
    basin_id: str,
    run_spec: RunSpec,
    input_obj,
    integrator,
) -> pd.DataFrame:
    """Extract exact pre-/post-integration MetroMan mean flows from memory."""
    observed_reaches = {
        normalize_reach_id(value)
        for value in input_obj.basin_dict.get("reach_ids", [])
    }
    gage_reaches = {
        normalize_reach_id(value)
        for value in getattr(integrator, "gage_dict", {})
    }
    sword_rows = _sword_index(input_obj.sword_dict)
    epsilons = getattr(integrator, "reach_epsilons", {}).get(METROMAN, {})
    records = []

    for raw_reach in input_obj.basin_dict["reach_ids_all"]:
        reach_id = normalize_reach_id(raw_reach)
        reach_data = integrator.alg_dict[METROMAN].get(reach_id, {})
        integrated = reach_data.get("integrator", {})
        sword_row = sword_rows.get(reach_id)

        def sword_value(name: str, default=np.nan):
            if sword_row is None or name not in input_obj.sword_dict:
                return default
            values = np.asarray(input_obj.sword_dict[name]).reshape(-1)
            if sword_row >= values.size:
                return default
            return values[sword_row]

        gage = getattr(integrator, "gage_dict", {}).get(reach_id, {})
        records.append(
            {
                "basin_id": str(basin_id),
                "run": run_spec.name,
                "run_label": run_spec.label,
                "branch": run_spec.branch,
                "reach_id": reach_id,
                "qbar_reachScale": _as_float(reach_data.get("qbar", np.nan)),
                "qbar_basinScale": _as_float(integrated.get("qbar", np.nan)),
                "reach_scale_source": str(reach_data.get("qbar_source", "")),
                "basin_scale_mass_epsilon": _as_float(
                    epsilons.get(reach_id, np.nan)
                ),
                "is_swot_observed_reach": reach_id in observed_reaches,
                "is_gage_constraint": reach_id in gage_reaches,
                "gage_group": str(gage.get("group", "")),
                "facc": _as_float(sword_value("facc")),
                "n_rch_up": _as_int(sword_value("n_rch_up", 0)),
                "n_rch_down": _as_int(sword_value("n_rch_down", 0)),
            }
        )

    return pd.DataFrame.from_records(records, columns=REACH_TABLE_COLUMNS)


def _junction_type(n_upflows: int, n_downflows: int) -> str:
    if n_upflows > 1 and n_downflows == 1:
        return "confluence"
    if n_upflows == 1 and n_downflows > 1:
        return "bifurcation"
    if n_upflows > 1 and n_downflows > 1:
        return "complex"
    return "one_to_one"


def evaluate_junction_mass(
    reach_table: pd.DataFrame,
    junctions: Sequence[Mapping[str, Any]],
    *,
    relative_tolerance: float = 0.01,
    absolute_tolerance_cms: float = 5.0,
    minimum_reference_flow_cms: float = 5.0,
) -> pd.DataFrame:
    """Evaluate upstream/downstream mean-flow closure for two qbar scales.

    A junction is conserving when

    ``abs(sum(downstream) - sum(upstream)) <= absolute_tolerance_cms +``
    ``relative_tolerance * max(abs(upstream), abs(downstream), minimum_flow)``.

    Only junctions for which every upstream and downstream reach has a finite
    positive qbar enter the reported denominator.
    """
    if relative_tolerance < 0 or absolute_tolerance_cms < 0:
        raise ValueError("Mass-conservation tolerances must be non-negative")
    if minimum_reference_flow_cms <= 0:
        raise ValueError("minimum_reference_flow_cms must be positive")
    if reach_table.empty:
        return pd.DataFrame(columns=JUNCTION_TABLE_COLUMNS)

    context = reach_table.iloc[0]
    scales = ("qbar_reachScale", "qbar_basinScale")
    value_maps = {
        scale: {
            normalize_reach_id(row.reach_id): float(getattr(row, scale))
            for row in reach_table[["reach_id", scale]].itertuples(index=False)
        }
        for scale in scales
    }
    records = []

    for junction_index, junction in enumerate(junctions):
        upflows = [
            normalize_reach_id(value) for value in junction.get("upflows", [])
        ]
        downflows = [
            normalize_reach_id(value) for value in junction.get("downflows", [])
        ]
        upflows = [value for value in upflows if value]
        downflows = [value for value in downflows if value]
        originating = normalize_reach_id(junction.get("originating_reach_id", ""))

        for scale in scales:
            value_map = value_maps[scale]
            upstream = np.asarray(
                [value_map.get(reach_id, np.nan) for reach_id in upflows],
                dtype=float,
            )
            downstream = np.asarray(
                [value_map.get(reach_id, np.nan) for reach_id in downflows],
                dtype=float,
            )
            evaluable = bool(
                upstream.size
                and downstream.size
                and np.all(np.isfinite(upstream) & (upstream > 0))
                and np.all(np.isfinite(downstream) & (downstream > 0))
            )
            sum_up = float(np.sum(upstream)) if evaluable else np.nan
            sum_down = float(np.sum(downstream)) if evaluable else np.nan
            signed = sum_down - sum_up if evaluable else np.nan
            absolute = abs(signed) if evaluable else np.nan
            reference = (
                max(abs(sum_up), abs(sum_down), minimum_reference_flow_cms)
                if evaluable
                else np.nan
            )
            allowed = (
                absolute_tolerance_cms + relative_tolerance * reference
                if evaluable
                else np.nan
            )
            records.append(
                {
                    "basin_id": str(context["basin_id"]),
                    "run": str(context["run"]),
                    "run_label": str(context["run_label"]),
                    "branch": str(context["branch"]),
                    "scale": scale,
                    "junction_index": junction_index,
                    "originating_reach_id": originating,
                    "junction_type": _junction_type(len(upflows), len(downflows)),
                    "upflow_reach_ids": json.dumps(upflows),
                    "downflow_reach_ids": json.dumps(downflows),
                    "n_upflows": len(upflows),
                    "n_downflows": len(downflows),
                    "sum_upstream_cms": sum_up,
                    "sum_downstream_cms": sum_down,
                    "signed_residual_cms": signed,
                    "absolute_residual_cms": absolute,
                    "relative_residual": absolute / reference if evaluable else np.nan,
                    "allowed_residual_cms": allowed,
                    "evaluable": evaluable,
                    "conserving": bool(evaluable and absolute <= allowed),
                }
            )

    return pd.DataFrame.from_records(records, columns=JUNCTION_TABLE_COLUMNS)


def summarize_junction_mass(junction_table: pd.DataFrame) -> pd.DataFrame:
    """Count and calculate the fraction of evaluable junctions that fail."""
    columns = [
        "basin_id",
        "run",
        "run_label",
        "branch",
        "scale",
        "n_junctions_total",
        "n_junctions_evaluable",
        "n_junctions_not_conserving",
        "fraction_junctions_not_conserving",
        "median_relative_residual",
        "p95_relative_residual",
        "max_relative_residual",
    ]
    if junction_table.empty:
        return pd.DataFrame(columns=columns)

    records = []
    group_columns = ["basin_id", "run", "run_label", "branch", "scale"]
    for keys, group in junction_table.groupby(group_columns, sort=False):
        evaluable = group.loc[_boolean_series(group["evaluable"])]
        n_evaluable = len(evaluable)
        n_failed = int((~_boolean_series(evaluable["conserving"])).sum())
        relative = pd.to_numeric(evaluable["relative_residual"], errors="coerce")
        records.append(
            {
                **dict(zip(group_columns, keys)),
                "n_junctions_total": int(len(group)),
                "n_junctions_evaluable": int(n_evaluable),
                "n_junctions_not_conserving": n_failed,
                "fraction_junctions_not_conserving": (
                    n_failed / n_evaluable if n_evaluable else np.nan
                ),
                "median_relative_residual": float(relative.median()),
                "p95_relative_residual": float(relative.quantile(0.95)),
                "max_relative_residual": float(relative.max()),
            }
        )
    return pd.DataFrame.from_records(records, columns=columns)


def print_junction_mass_summary(summary: pd.DataFrame) -> None:
    """Print the requested non-conserving-junction ratios."""
    if summary.empty:
        print("No evaluable junction mass-conservation results are available.")
        return

    ordered = summary.sort_values(["basin_id", "run", "scale"])
    for row in ordered.itertuples(index=False):
        fraction = row.fraction_junctions_not_conserving
        fraction_text = "NA" if not np.isfinite(fraction) else f"{fraction:.2%}"
        print(
            f"basin={row.basin_id} {row.run} {row.scale}: "
            f"{row.n_junctions_not_conserving}/"
            f"{row.n_junctions_evaluable} evaluable junctions do not conserve "
            f"mass ({fraction_text}); total junctions={row.n_junctions_total}"
        )

    print("\nAll six basins combined:")
    for (run, scale), group in ordered.groupby(["run", "scale"], sort=False):
        n_evaluable = int(group["n_junctions_evaluable"].sum())
        n_failed = int(group["n_junctions_not_conserving"].sum())
        fraction = n_failed / n_evaluable if n_evaluable else np.nan
        fraction_text = "NA" if not np.isfinite(fraction) else f"{fraction:.2%}"
        print(
            f"{run} {scale}: {n_failed}/{n_evaluable} evaluable junctions "
            f"do not conserve mass ({fraction_text})"
        )


def run_one_basin_experiment(
    basin_id: str,
    basin_record: Mapping[str, Any],
    run_spec: RunSpec,
    *,
    moi_repo: Path,
    run_root: Path,
    svs_file: Path,
    result_root: Path,
    relative_tolerance: float = 0.01,
    absolute_tolerance_cms: float = 5.0,
    minimum_reference_flow_cms: float = 5.0,
    parameter_overrides: Mapping[str, Any] | None = None,
    mean_only: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run one basin/configuration and immediately persist compact diagnostics."""
    Input, Integrate, get_all_sword_reach_in_basin, set_moi_params = _load_moi_api(
        moi_repo
    )
    basin_id = str(basin_id)
    run_root = Path(run_root)
    result_root = Path(result_root)
    basin_data = {
        "basin_id": basin_id,
        "reach_ids": list(basin_record["reach_ids"]),
        "sos": basin_record["sos"],
        "sword": basin_record["sword"],
    }
    params = set_moi_params()
    params["write_sword_output"] = False
    if parameter_overrides:
        params.update(dict(parameter_overrides))
    if mean_only:
        params["SFOI_Flow_Levels"] = ("Mean",)
        params["SFOI_Compute_FLPs"] = False
    configured_flow_levels = params["SFOI_Flow_Levels"]
    flow_levels_audit = (
        configured_flow_levels
        if isinstance(configured_flow_levels, str)
        else ",".join(configured_flow_levels)
    )

    input_obj = Input(
        run_root / "flpe",
        run_root / "input" / "sos",
        run_root / "input" / "swot",
        run_root / "input" / "sword",
        basin_data,
        run_spec.branch,
        verbose,
    )
    input_obj.extract_sword()
    input_obj = get_all_sword_reach_in_basin(input_obj, verbose)
    input_obj.extract_swot()
    input_obj.extract_sos()
    input_obj.extract_alg()

    gage_dict = {}
    if run_spec.uses_gages:
        if run_spec.calval_csv is None:
            raise ValueError(f"{run_spec.name} requires a Cal/Val CSV")
        input_obj.extract_svs(
            Path(svs_file),
            reach_id_col="reach_id_v17b",
            calval_file=run_spec.calval_csv,
        )
        gage_dict = input_obj.gage_dict
    n_svs_gages_loaded = len(gage_dict)

    # Passing an explicit dictionary prevents the legacy SoS fallback from
    # silently adding gages outside the selected experimental group.
    integrator = Integrate(
        input_obj.alg_dict,
        input_obj.basin_dict,
        input_obj.sos_dict,
        input_obj.sword_dict,
        getattr(input_obj, "obs_dict", {}),
        params,
        run_spec.branch,
        verbose,
        gage_dict=gage_dict,
    )
    n_gages_prepared = len(integrator.gage_dict)
    integrator.integrate()

    reach_table = build_metroman_reach_table(
        basin_id,
        run_spec,
        input_obj,
        integrator,
    )
    junction_table = evaluate_junction_mass(
        reach_table,
        integrator.junctions,
        relative_tolerance=relative_tolerance,
        absolute_tolerance_cms=absolute_tolerance_cms,
        minimum_reference_flow_cms=minimum_reference_flow_cms,
    )

    basin_dir = result_root / "runs" / run_spec.name / "basins" / basin_id
    basin_dir.mkdir(parents=True, exist_ok=True)
    reach_path = basin_dir / f"basin_{basin_id}_metroman_qbar.csv"
    junction_path = basin_dir / f"basin_{basin_id}_junction_mass.csv"
    reach_table.to_csv(reach_path, index=False)
    junction_table.to_csv(junction_path, index=False)

    evaluable_mask = _boolean_series(junction_table["evaluable"])
    conserving_mask = _boolean_series(junction_table["conserving"])
    n_evaluable = int(evaluable_mask.sum())
    n_failed = int(
        (evaluable_mask & ~conserving_mask).sum()
    )
    audit = {
        "basin_id": basin_id,
        "run": run_spec.name,
        "run_label": run_spec.label,
        "branch": run_spec.branch,
        "flow_levels": flow_levels_audit,
        "compute_flps": bool(params["SFOI_Compute_FLPs"]),
        "status": "completed",
        "source_json": basin_record.get("source_json", ""),
        "n_basin_reaches": len(input_obj.basin_dict["reach_ids_all"]),
        "n_svs_gages_loaded": n_svs_gages_loaded,
        "n_gages_prepared": n_gages_prepared,
        "n_junction_scale_rows": len(junction_table),
        "n_evaluable_junction_scale_rows": n_evaluable,
        "n_nonconserving_junction_scale_rows": n_failed,
        "reach_csv": str(reach_path),
        "junction_csv": str(junction_path),
    }
    return reach_table, junction_table, audit


def _existing_basin_results(
    result_root: Path,
    run_spec: RunSpec,
    basin_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    basin_dir = Path(result_root) / "runs" / run_spec.name / "basins" / str(basin_id)
    reach_path = basin_dir / f"basin_{basin_id}_metroman_qbar.csv"
    junction_path = basin_dir / f"basin_{basin_id}_junction_mass.csv"
    if not (reach_path.is_file() and junction_path.is_file()):
        return None
    reach_table = pd.read_csv(
        reach_path,
        dtype={"basin_id": str, "reach_id": str, "run": str},
    )
    junction_table = pd.read_csv(
        junction_path,
        dtype={"basin_id": str, "originating_reach_id": str, "run": str},
    )
    return reach_table, junction_table


def run_three_experiments(
    target_basins: Iterable[str],
    basin_catalog: Mapping[str, Mapping[str, Any]],
    run_specs: Sequence[RunSpec],
    *,
    moi_repo: Path,
    run_root: Path,
    svs_file: Path,
    result_root: Path,
    relative_tolerance: float = 0.01,
    absolute_tolerance_cms: float = 5.0,
    minimum_reference_flow_cms: float = 5.0,
    parameter_overrides: Mapping[str, Any] | None = None,
    mean_only: bool = True,
    verbose: bool = True,
    reuse_existing: bool = True,
    continue_on_error: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run/resume all 18 basin/configuration combinations."""
    result_root = Path(result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    target_basins = [str(value) for value in target_basins]
    reach_tables = []
    junction_tables = []
    audits = []

    if mean_only:
        print("Execution mode: Mean-only; q33 and final FLP estimation are skipped.")
    else:
        print("Execution mode: full Mean + q33 + final FLP estimation.")

    for run_spec in run_specs:
        for basin_id in target_basins:
            if basin_id not in basin_catalog:
                raise KeyError(f"Basin {basin_id} is absent from the basin catalog")
            existing = (
                _existing_basin_results(result_root, run_spec, basin_id)
                if reuse_existing
                else None
            )
            if existing is not None:
                reach_table, junction_table = existing
                print(f"[{run_spec.name} basin {basin_id}] reusing existing CSVs")
                reach_tables.append(reach_table)
                junction_tables.append(junction_table)
                audits.append(
                    {
                        "basin_id": basin_id,
                        "run": run_spec.name,
                        "run_label": run_spec.label,
                        "branch": run_spec.branch,
                        "flow_levels": "existing_output",
                        "compute_flps": np.nan,
                        "status": "reused",
                        "n_basin_reaches": len(reach_table),
                        "n_gages_prepared": int(
                            _boolean_series(reach_table["is_gage_constraint"]).sum()
                        ),
                        "n_junction_scale_rows": len(junction_table),
                    }
                )
                continue

            print("\n" + "=" * 78)
            print(f"Running {run_spec.name} ({run_spec.label}) for basin {basin_id}")
            print("=" * 78)
            try:
                reach_table, junction_table, audit = run_one_basin_experiment(
                    basin_id,
                    basin_catalog[basin_id],
                    run_spec,
                    moi_repo=moi_repo,
                    run_root=run_root,
                    svs_file=svs_file,
                    result_root=result_root,
                    relative_tolerance=relative_tolerance,
                    absolute_tolerance_cms=absolute_tolerance_cms,
                    minimum_reference_flow_cms=minimum_reference_flow_cms,
                    parameter_overrides=parameter_overrides,
                    mean_only=mean_only,
                    verbose=verbose,
                )
                reach_tables.append(reach_table)
                junction_tables.append(junction_table)
                audits.append(audit)
            except Exception as exc:
                error_dir = result_root / "runs" / run_spec.name / "basins" / basin_id
                error_dir.mkdir(parents=True, exist_ok=True)
                error_path = error_dir / f"basin_{basin_id}_error.txt"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                audits.append(
                    {
                        "basin_id": basin_id,
                        "run": run_spec.name,
                        "run_label": run_spec.label,
                        "branch": run_spec.branch,
                        "flow_levels": "Mean" if mean_only else "Mean,q33",
                        "compute_flps": not mean_only,
                        "status": "failed",
                        "error": str(exc),
                        "error_log": str(error_path),
                    }
                )
                print(f"[{run_spec.name} basin {basin_id}] FAILED: {exc}")
                if not continue_on_error:
                    raise

            pd.DataFrame(audits).to_csv(result_root / "run_audit.csv", index=False)

    reach_results = (
        pd.concat(reach_tables, ignore_index=True)
        if reach_tables
        else pd.DataFrame(columns=REACH_TABLE_COLUMNS)
    )
    junction_results = (
        pd.concat(junction_tables, ignore_index=True)
        if junction_tables
        else pd.DataFrame(columns=JUNCTION_TABLE_COLUMNS)
    )
    audit = pd.DataFrame(audits)
    summary = summarize_junction_mass(junction_results)
    reach_results.to_csv(result_root / "all_metroman_qbar.csv", index=False)
    junction_results.to_csv(result_root / "all_junction_mass.csv", index=False)
    summary.to_csv(result_root / "junction_mass_summary.csv", index=False)
    audit.to_csv(result_root / "run_audit.csv", index=False)
    return reach_results, junction_results, summary, audit


def load_experiment_results(
    result_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load aggregate outputs from a previously completed/resumed experiment."""
    result_root = Path(result_root)
    reach_results = pd.read_csv(
        result_root / "all_metroman_qbar.csv",
        dtype={"basin_id": str, "reach_id": str, "run": str},
    )
    junction_results = pd.read_csv(
        result_root / "all_junction_mass.csv",
        dtype={"basin_id": str, "originating_reach_id": str, "run": str},
    )
    summary = pd.read_csv(
        result_root / "junction_mass_summary.csv",
        dtype={"basin_id": str, "run": str},
    )
    audit = pd.read_csv(
        result_root / "run_audit.csv",
        dtype={"basin_id": str, "run": str},
    )
    return reach_results, junction_results, summary, audit


def _netcdf_variable(dataset: Dataset, paths: Sequence[str]):
    for path in paths:
        try:
            return dataset[path]
        except (IndexError, KeyError):
            continue
    return None


def _numeric_array(variable) -> np.ndarray:
    values = variable[:]
    if hasattr(values, "filled"):
        values = values.filled(np.nan)
    return np.asarray(values, dtype=float).reshape(-1)


def load_sword_centerlines(
    sword_path: Path,
    reach_ids: Iterable[str] | None = None,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Load SWORD reach centerlines from a NetCDF or GeoPackage.

    The return value maps each reach ID to one or more ``(longitude, latitude)``
    segments, so MultiLineString GeoPackage geometries and NetCDF node tracks
    share a common plotting interface.
    """
    sword_path = Path(sword_path)
    selected = (
        {normalize_reach_id(value) for value in reach_ids}
        if reach_ids is not None
        else None
    )

    if sword_path.suffix.lower() == ".gpkg":
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                "geopandas is required to plot a SWORD GeoPackage"
            ) from exc
        frame = gpd.read_file(sword_path)
        reach_column = next(
            (name for name in ("reach_id", "reach_id_v17c", "reach_id_v17b") if name in frame),
            None,
        )
        if reach_column is None:
            raise KeyError(f"No reach-ID column found in {sword_path}")
        centerlines: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for row in frame[[reach_column, "geometry"]].itertuples(index=False):
            reach_id = normalize_reach_id(row[0])
            if selected is not None and reach_id not in selected:
                continue
            geometry = row[1]
            pieces = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
            for piece in pieces:
                coordinates = np.asarray(piece.coords, dtype=float)
                if coordinates.ndim == 2 and coordinates.shape[0] >= 2:
                    centerlines.setdefault(reach_id, []).append(
                        (coordinates[:, 0], coordinates[:, 1])
                    )
        return centerlines

    with Dataset(sword_path, mode="r") as dataset:
        node_reach = _netcdf_variable(
            dataset,
            ("nodes/reach_id", "nodes/reach_id_v17c", "nodes/reach_id_v17b"),
        )
        node_lon = _netcdf_variable(
            dataset,
            ("nodes/x", "nodes/lon", "nodes/longitude"),
        )
        node_lat = _netcdf_variable(
            dataset,
            ("nodes/y", "nodes/lat", "nodes/latitude"),
        )
        if node_reach is None or node_lon is None or node_lat is None:
            raise KeyError(
                f"SWORD node centerline variables were not found in {sword_path}; "
                "expected nodes/reach_id plus nodes/x,y (or lon,lat)."
            )
        raw_reaches = node_reach[:]
        if hasattr(raw_reaches, "filled"):
            raw_reaches = raw_reaches.filled(0)
        raw_reaches = np.asarray(raw_reaches).reshape(-1)
        longitude = _numeric_array(node_lon)
        latitude = _numeric_array(node_lat)
        if not (raw_reaches.size == longitude.size == latitude.size):
            raise ValueError(f"SWORD node arrays have incompatible sizes in {sword_path}")

    centerlines = {}
    for raw_reach in np.unique(raw_reaches):
        reach_id = normalize_reach_id(raw_reach)
        if not reach_id or reach_id == "0":
            continue
        if selected is not None and reach_id not in selected:
            continue
        mask = raw_reaches == raw_reach
        valid = mask & np.isfinite(longitude) & np.isfinite(latitude)
        if np.count_nonzero(valid) >= 2:
            centerlines[reach_id] = [(longitude[valid], latitude[valid])]
    return centerlines


def plot_basin_metroman_maps(
    reach_results: pd.DataFrame,
    basin_id: str,
    run_specs: Sequence[RunSpec],
    sword_path: Path,
    *,
    junction_results: pd.DataFrame | None = None,
    centerlines: Mapping[str, list[tuple[np.ndarray, np.ndarray]]] | None = None,
    figure_path: Path | None = None,
    dpi: int = 250,
):
    """Plot a two-scale by three-run qbar map on SWORD centerlines."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LogNorm, Normalize

    basin_id = str(basin_id)
    basin = reach_results.loc[reach_results["basin_id"].astype(str) == basin_id].copy()
    if basin.empty:
        raise ValueError(f"No reach results are available for basin {basin_id}")
    reach_ids = basin["reach_id"].map(normalize_reach_id).unique()
    reach_id_set = set(reach_ids)
    if centerlines is None:
        centerlines = load_sword_centerlines(sword_path, reach_ids)
    else:
        centerlines = {
            reach_id: segments
            for reach_id, segments in centerlines.items()
            if reach_id in reach_id_set
        }
    if not centerlines:
        raise ValueError(f"No centerlines were loaded for basin {basin_id}")

    scales = (
        ("qbar_reachScale", "MetroMan qbar_reachScale (m³/s)"),
        ("qbar_basinScale", "MetroMan qbar_basinScale (m³/s)"),
    )
    figure, axes = plt.subplots(
        nrows=2,
        ncols=len(run_specs),
        figsize=(6.0 * len(run_specs), 11.0),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_index, (scale, scale_label) in enumerate(scales):
        all_values = pd.to_numeric(basin[scale], errors="coerce").to_numpy(dtype=float)
        positive = all_values[np.isfinite(all_values) & (all_values > 0)]
        if positive.size == 0:
            norm = Normalize(vmin=0.0, vmax=1.0)
        else:
            vmin = float(np.min(positive))
            vmax = float(np.max(positive))
            if np.isclose(vmin, vmax):
                vmin = max(vmax / 2.0, np.finfo(float).tiny)
            norm = LogNorm(vmin=vmin, vmax=vmax)

        last_collection = None
        for column_index, run_spec in enumerate(run_specs):
            axis = axes[row_index, column_index]
            run = basin.loc[basin["run"] == run_spec.name]
            values = {
                normalize_reach_id(row.reach_id): float(getattr(row, scale))
                for row in run[["reach_id", scale]].itertuples(index=False)
            }

            # Every SWORD centerline is drawn first; reaches without a valid
            # qbar remain visible in gray rather than disappearing from the map.
            for segments in centerlines.values():
                for x, y in segments:
                    axis.plot(x, y, color="0.78", linewidth=0.45, zorder=1)

            line_segments = []
            line_values = []
            for reach_id, pieces in centerlines.items():
                value = values.get(reach_id, np.nan)
                if not np.isfinite(value) or value <= 0:
                    continue
                for x, y in pieces:
                    line_segments.append(np.column_stack([x, y]))
                    line_values.append(value)
            if line_segments:
                last_collection = LineCollection(
                    line_segments,
                    cmap="viridis",
                    norm=norm,
                    linewidths=1.35,
                    zorder=2,
                )
                last_collection.set_array(np.asarray(line_values, dtype=float))
                axis.add_collection(last_collection)

            junction_title = ""
            if junction_results is not None and not junction_results.empty:
                panel_junctions = junction_results.loc[
                    (junction_results["basin_id"].astype(str) == basin_id)
                    & (junction_results["run"].astype(str) == run_spec.name)
                    & (junction_results["scale"].astype(str) == scale)
                ]
                evaluable_mask = _boolean_series(panel_junctions["evaluable"])
                conserving_mask = _boolean_series(panel_junctions["conserving"])
                failures = panel_junctions.loc[evaluable_mask & ~conserving_mask]
                failure_points = []
                for value in failures["originating_reach_id"]:
                    pieces = centerlines.get(normalize_reach_id(value), [])
                    if not pieces:
                        continue
                    x, y = pieces[0]
                    failure_points.append((float(np.nanmean(x)), float(np.nanmean(y))))
                if failure_points:
                    x_points, y_points = np.asarray(failure_points).T
                    axis.scatter(
                        x_points,
                        y_points,
                        marker="x",
                        s=17,
                        linewidths=0.8,
                        color="red",
                        zorder=3,
                        label="non-conserving junction",
                    )
                    axis.legend(loc="lower left", fontsize=7, frameon=True)
                junction_title = (
                    f"\nnon-conserving junctions="
                    f"{len(failures)}/{int(evaluable_mask.sum())}"
                )

            valid_count = int(
                np.count_nonzero(
                    np.isfinite(pd.to_numeric(run[scale], errors="coerce"))
                    & (pd.to_numeric(run[scale], errors="coerce") > 0)
                )
            )
            axis.set_title(
                f"{run_spec.name}: {run_spec.label}\nvalid reaches={valid_count}"
                f"{junction_title}"
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(True, linestyle=":", linewidth=0.45, alpha=0.45)
            if column_index == 0:
                axis.set_ylabel(f"{scale_label}\nLatitude")
            if row_index == len(scales) - 1:
                axis.set_xlabel("Longitude")

        if last_collection is not None:
            colorbar = figure.colorbar(
                last_collection,
                ax=axes[row_index, :].tolist(),
                fraction=0.018,
                pad=0.015,
            )
            colorbar.set_label(scale_label + " (log color scale)")

    figure.suptitle(
        f"Basin {basin_id}: MetroMan mean flow on SWORD river centerlines",
        fontsize=15,
        y=0.995,
    )
    figure.subplots_adjust(left=0.07, right=0.90, bottom=0.06, top=0.93, wspace=0.10, hspace=0.16)
    if figure_path is not None:
        figure_path = Path(figure_path)
        figure_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(figure_path, dpi=dpi, bbox_inches="tight")
    return figure


def plot_all_basin_maps(
    reach_results: pd.DataFrame,
    target_basins: Iterable[str],
    run_specs: Sequence[RunSpec],
    basin_catalog: Mapping[str, Mapping[str, Any]],
    *,
    junction_results: pd.DataFrame | None = None,
    sword_dir: Path,
    result_root: Path,
    dpi: int = 250,
) -> dict[str, Any]:
    """Create and save one 2x3 centerline map for each target basin."""
    figures = {}
    figure_dir = Path(result_root) / "figures"
    target_basins = [str(value) for value in target_basins]

    # Several basins commonly share one continental SWORD file. Load that
    # large node table once for the union of requested reaches.
    reaches_by_sword: dict[Path, set[str]] = {}
    for basin_id in target_basins:
        sword_path = Path(sword_dir) / basin_catalog[basin_id]["sword"]
        basin_reaches = reach_results.loc[
            reach_results["basin_id"].astype(str) == basin_id, "reach_id"
        ]
        reaches_by_sword.setdefault(sword_path, set()).update(
            basin_reaches.map(normalize_reach_id)
        )
    centerline_cache = {
        sword_path: load_sword_centerlines(sword_path, reach_ids)
        for sword_path, reach_ids in reaches_by_sword.items()
    }

    for basin_id in target_basins:
        sword_path = Path(sword_dir) / basin_catalog[basin_id]["sword"]
        figure_path = figure_dir / f"basin_{basin_id}_metroman_qbar_maps.png"
        figures[basin_id] = plot_basin_metroman_maps(
            reach_results,
            basin_id,
            run_specs,
            sword_path,
            junction_results=junction_results,
            centerlines=centerline_cache[sword_path],
            figure_path=figure_path,
            dpi=dpi,
        )
        print(f"Basin {basin_id} map: {figure_path}")
    return figures


__all__ = [
    "TARGET_BASINS",
    "RunSpec",
    "build_run_specs",
    "discover_basin_catalog",
    "discover_svs_file",
    "evaluate_junction_mass",
    "load_experiment_results",
    "load_sword_centerlines",
    "plot_all_basin_maps",
    "plot_basin_metroman_maps",
    "prepare_all_gage_calval_csv",
    "print_junction_mass_summary",
    "run_one_basin_experiment",
    "run_three_experiments",
    "summarize_junction_mass",
]

"""Generate the portable SOS validation extraction/comparison notebooks.

The generated notebooks are self-contained so they can be copied to OSC,
Unity, or a local computer without importing code from this repository.
"""

from pathlib import Path

import nbformat as nbf


HERE = Path(__file__).resolve().parent


EXTRACTION_IMPORTS = r'''from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import gc
import json
import platform
import re
import sys

import numpy as np
import pandas as pd
from netCDF4 import Dataset, chartostring

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 180)

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)'''


EXTRACTION_HELPERS = r'''def clean_text(value):
    """Decode one bytes/string scalar and remove NetCDF padding."""
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    return str(value).replace("\x00", "").strip()


def decode_text_array(variable):
    """Read fixed-width char arrays and VLEN strings into a text ndarray."""
    raw = np.ma.filled(variable[:], b"")
    values = np.asarray(raw)

    if values.dtype.kind == "S" and values.dtype.itemsize == 1 and values.ndim:
        values = np.asarray(chartostring(values, encoding="utf-8"))

    if values.ndim == 0:
        return np.asarray(clean_text(values.item()), dtype=object)
    return np.vectorize(clean_text, otypes=[object])(values)


def read_numeric_array(variable):
    """Read a numeric variable as float64, converting masks/fill to NaN."""
    masked = np.ma.asarray(variable[:]).astype(np.float64)
    values = np.asarray(np.ma.filled(masked, np.nan), dtype=np.float64)

    fill_value = getattr(variable, "_FillValue", None)
    if fill_value is not None:
        try:
            values[np.isclose(values, float(fill_value), equal_nan=False)] = np.nan
        except (TypeError, ValueError):
            pass

    valid_min = getattr(variable, "valid_min", None)
    valid_max = getattr(variable, "valid_max", None)
    valid_range = getattr(variable, "valid_range", None)
    if valid_range is not None and np.asarray(valid_range).size == 2:
        valid_min, valid_max = np.asarray(valid_range).reshape(-1)[:2]
    if valid_min is not None:
        values[values < float(valid_min)] = np.nan
    if valid_max is not None:
        values[values > float(valid_max)] = np.nan
    return values


def normalize_algorithm(value):
    text = clean_text(value).lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"^(flpe|moi)_+", "", text)
    text = re.sub(r"_+(flpe|moi)$", "", text)
    return text.strip("_")


def normalize_reach_id(value):
    if value is None or np.ma.is_masked(value):
        return ""
    text = clean_text(value)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(number) or number <= 0:
        return ""
    return str(int(round(number)))


def normalize_gage_id(value):
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def partition_key(path):
    """Create a filename-independent partition key, normally a continent."""
    match = re.match(r"([a-z]{2})_sword_", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    prefix = re.split(r"_SOS_results_", path.stem, maxsplit=1, flags=re.IGNORECASE)[0]
    return re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")


def candidate_reach_variables(dataset):
    candidates = []
    if "reaches" in dataset.groups:
        group = dataset.groups["reaches"]
        for name in ("reach_id", "reach_ids", "reachid"):
            if name in group.variables:
                candidates.append((f"/reaches/{name}", group.variables[name]))
    for name in ("reach_id", "reach_ids", "reachid"):
        if name in dataset.variables:
            candidates.append((f"/{name}", dataset.variables[name]))
    return candidates


def find_reach_ids(dataset, expected_size):
    for source, variable in candidate_reach_variables(dataset):
        try:
            if getattr(variable.dtype, "kind", "") in {"S", "U", "O"}:
                values = decode_text_array(variable).reshape(-1)
            else:
                values = np.ma.filled(variable[:], np.nan).reshape(-1)
        except Exception:
            continue
        if values.size == expected_size:
            return np.asarray(values, dtype=object), source
    fallback = np.asarray([f"row:{index}" for index in range(expected_size)], dtype=object)
    return fallback, "fallback_row_index"


def align_algorithm_names(variable, n_reaches, n_algorithms):
    values = np.asarray(decode_text_array(variable), dtype=object)
    target_shape = (n_reaches, n_algorithms)
    if values.shape == target_shape:
        return values
    if values.ndim == 1 and values.size == n_algorithms:
        return np.broadcast_to(values.reshape(1, -1), target_shape).copy()
    if values.size == n_reaches * n_algorithms:
        return values.reshape(target_shape)
    raise ValueError(
        f"algo_names shape {values.shape} cannot align to {target_shape}"
    )


def align_reach_vector(variable, n_reaches, name):
    values = np.asarray(decode_text_array(variable), dtype=object).reshape(-1)
    if values.size != n_reaches:
        raise ValueError(f"{name} length {values.size} != reach rows {n_reaches}")
    return values


def align_numeric_matrix(variable, target_shape):
    values = read_numeric_array(variable)
    if values.shape == target_shape:
        return values
    if target_shape[1] == 1 and values.shape == (target_shape[0],):
        return values.reshape(-1, 1)
    return None


def float64_hex(value):
    return float(np.float64(value)).hex()


def extract_validation_group(dataset, path, group_name, file_metadata):
    validation = dataset.groups["validation"]
    if group_name not in validation.groups:
        raise KeyError(f"/validation/{group_name} is missing")
    group = validation.groups[group_name]

    for required in ("nbias", "algo_names", "has_validation"):
        if required not in group.variables:
            raise KeyError(f"/validation/{group_name}/{required} is missing")

    nbias = read_numeric_array(group.variables["nbias"])
    if nbias.ndim == 1:
        nbias = nbias.reshape(-1, 1)
    if nbias.ndim != 2:
        raise ValueError(f"nbias must be 1-D or 2-D, found {nbias.shape}")
    n_reaches, n_algorithms = nbias.shape
    algorithms = align_algorithm_names(
        group.variables["algo_names"], n_reaches, n_algorithms
    )

    has_validation = read_numeric_array(group.variables["has_validation"]).reshape(-1)
    if has_validation.size != n_reaches:
        raise ValueError(
            f"has_validation length {has_validation.size} != reach rows {n_reaches}"
        )
    has_validation = np.isfinite(has_validation) & (has_validation == 1)

    if "gageid" in group.variables:
        gageids = align_reach_vector(group.variables["gageid"], n_reaches, "gageid")
    else:
        gageids = np.full(n_reaches, "", dtype=object)

    reach_ids_raw, reach_id_source = find_reach_ids(dataset, n_reaches)
    reach_ids = np.asarray([normalize_reach_id(value) for value in reach_ids_raw])

    metrics = {}
    variable_rows = []
    target_shape = (n_reaches, n_algorithms)
    for variable_name, variable in group.variables.items():
        dtype_kind = getattr(variable.dtype, "kind", "")
        aligned = None
        if dtype_kind in {"b", "i", "u", "f"} and variable_name != "has_validation":
            aligned = align_numeric_matrix(variable, target_shape)
            if aligned is not None:
                metrics[variable_name] = aligned
        variable_rows.append(
            {
                **file_metadata,
                "group": group_name,
                "variable": variable_name,
                "dtype": str(variable.dtype),
                "dimensions": "|".join(variable.dimensions),
                "shape": "x".join(str(value) for value in variable.shape),
                "aligned_result_metric": aligned is not None,
            }
        )

    if "nbias" not in metrics:
        raise ValueError("nbias could not be aligned to reach-by-algorithm cells")

    cell_rows = []
    value_rows = []
    for reach_index in range(n_reaches):
        for algorithm_index in range(n_algorithms):
            algorithm_raw = clean_text(algorithms[reach_index, algorithm_index])
            algorithm = normalize_algorithm(algorithm_raw)
            if not algorithm:
                algorithm = f"algorithm_column_{algorithm_index}"
            nbias_value = metrics["nbias"][reach_index, algorithm_index]
            local_cell_id = (
                f"{file_metadata['relative_path']}|{group_name}|"
                f"{reach_index}|{algorithm_index}"
            )
            cell = {
                **file_metadata,
                "local_cell_id": local_cell_id,
                "group": group_name,
                "row_index": reach_index,
                "algorithm_index": algorithm_index,
                "reach_id_raw": clean_text(reach_ids_raw[reach_index]),
                "reach_id": reach_ids[reach_index],
                "reach_id_source": reach_id_source,
                "gageid": clean_text(gageids[reach_index]),
                "gageid_key": normalize_gage_id(gageids[reach_index]),
                "algorithm_raw": algorithm_raw,
                "algorithm": algorithm,
                "has_validation": int(has_validation[reach_index]),
                "nbias_is_finite": bool(np.isfinite(nbias_value)),
                "result_present": bool(
                    has_validation[reach_index] and np.isfinite(nbias_value)
                ),
            }
            cell_rows.append(cell)

            for metric_name, matrix in metrics.items():
                value = float(matrix[reach_index, algorithm_index])
                value_rows.append(
                    {
                        **file_metadata,
                        "local_cell_id": local_cell_id,
                        "group": group_name,
                        "row_index": reach_index,
                        "algorithm_index": algorithm_index,
                        "reach_id": reach_ids[reach_index],
                        "gageid_key": normalize_gage_id(gageids[reach_index]),
                        "algorithm": algorithm,
                        "has_validation": int(has_validation[reach_index]),
                        "metric": metric_name,
                        "metric_dtype": str(group.variables[metric_name].dtype),
                        "value": value,
                        "value_hex64": float64_hex(value),
                        "is_finite": bool(np.isfinite(value)),
                    }
                )
    return cell_rows, value_rows, variable_rows


def attach_stable_keys(cells, values):
    cells = cells.sort_values(
        ["partition_key", "relative_path", "group", "row_index", "algorithm_index"],
        kind="stable",
    ).reset_index(drop=True)
    semantic_columns = ["group", "partition_key", "reach_id", "gageid_key", "algorithm"]
    cells["occurrence"] = cells.groupby(semantic_columns, dropna=False).cumcount()
    cells["record_key"] = cells[semantic_columns + ["occurrence"]].astype(str).agg("|".join, axis=1)

    key_lookup = cells[["local_cell_id", "occurrence", "record_key"]]
    values = values.merge(key_lookup, on="local_cell_id", how="left", validate="many_to_one")
    if values["record_key"].isna().any():
        raise RuntimeError("Some metric rows did not receive a stable record key")
    return cells, values


def summarize_counts(cells):
    def summarize(frame, group_name, algorithm):
        results = frame.loc[frame["result_present"]]
        return {
            "run_label": RUN_LABEL,
            "group": group_name,
            "algorithm": algorithm,
            "total_cells": len(frame),
            "validation_flag_count": int(frame["has_validation"].sum()),
            "finite_nbias_count": int(frame["nbias_is_finite"].sum()),
            "result_count": int(frame["result_present"].sum()),
            "unique_result_reaches": int(results["reach_id"].nunique()),
            "unique_result_gages": int(results.loc[results["gageid_key"] != "", "gageid_key"].nunique()),
        }

    rows = []
    for group_name, group_frame in cells.groupby("group", sort=True):
        rows.append(summarize(group_frame, group_name, "ALL"))
        for algorithm, frame in group_frame.groupby("algorithm", sort=True):
            rows.append(summarize(frame, group_name, algorithm))
    return pd.DataFrame(rows)


def safe_json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value'''


EXTRACTION_RUN = r'''if not SOS_DIR.is_dir():
    raise NotADirectoryError(f"SOS_DIR does not exist: {SOS_DIR}")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
paths = sorted(SOS_DIR.rglob(FILE_GLOB) if RECURSIVE else SOS_DIR.glob(FILE_GLOB))
if not paths:
    raise FileNotFoundError(f"No files matching {FILE_GLOB!r} under {SOS_DIR}")

all_cells = []
all_values = []
all_variables = []
inventory_rows = []
issue_rows = []

print(f"Found {len(paths):,} candidate NetCDF files")
for file_number, path in enumerate(paths, start=1):
    relative_path = str(path.relative_to(SOS_DIR))
    metadata = {
        "run_label": RUN_LABEL,
        "source_file": path.name,
        "relative_path": relative_path,
        "partition_key": partition_key(path),
        "file_size_bytes": path.stat().st_size,
    }
    inventory = {**metadata, "extract_status": "pending", "validation_groups": ""}
    print(f"[{file_number}/{len(paths)}] {relative_path}")
    try:
        with Dataset(path, mode="r") as dataset:
            if "validation" not in dataset.groups:
                inventory["extract_status"] = "skipped_no_validation_group"
                inventory_rows.append(inventory)
                continue
            available = sorted(dataset.groups["validation"].groups)
            inventory["validation_groups"] = "|".join(available)
            extracted_groups = []
            for group_name in VALIDATION_GROUPS:
                try:
                    cells, values, variables = extract_validation_group(
                        dataset, path, group_name, metadata
                    )
                    all_cells.extend(cells)
                    all_values.extend(values)
                    all_variables.extend(variables)
                    extracted_groups.append(group_name)
                except Exception as exc:
                    issue_rows.append(
                        {
                            **metadata,
                            "severity": "error",
                            "group": group_name,
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
            inventory["extract_status"] = (
                "ok:" + "|".join(extracted_groups) if extracted_groups else "error_no_groups_extracted"
            )
    except Exception as exc:
        inventory["extract_status"] = "error_opening_file"
        issue_rows.append(
            {
                **metadata,
                "severity": "error",
                "group": "",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
    inventory_rows.append(inventory)

    inventory_df = pd.DataFrame(inventory_rows)
    issue_columns = [
        "run_label", "source_file", "relative_path", "partition_key",
        "file_size_bytes", "severity", "group", "message",
    ]
    issues_df = pd.DataFrame(issue_rows, columns=issue_columns)
cells_df = pd.DataFrame(all_cells)
values_df = pd.DataFrame(all_values)
variables_df = pd.DataFrame(all_variables)

if cells_df.empty or values_df.empty:
    inventory_df.to_csv(EXPORT_DIR / "file_inventory.csv", index=False)
    issues_df.to_csv(EXPORT_DIR / "extraction_issues.csv", index=False)
    raise RuntimeError("No validation records were extracted; inspect the inventory/issues CSV files")

cells_df, values_df = attach_stable_keys(cells_df, values_df)
counts_df = summarize_counts(cells_df)

duplicate_key_count = int(cells_df.duplicated("record_key", keep=False).sum())
if duplicate_key_count:
    raise RuntimeError(f"Stable record keys are not unique for {duplicate_key_count} rows")

csv_options = {"index": False, "float_format": "%.17g", "na_rep": "NaN"}
inventory_df.to_csv(EXPORT_DIR / "file_inventory.csv", **csv_options)
issues_df.to_csv(EXPORT_DIR / "extraction_issues.csv", **csv_options)
variables_df.to_csv(EXPORT_DIR / "variable_inventory.csv", **csv_options)
cells_df.to_csv(EXPORT_DIR / "validation_cells.csv", **csv_options)
values_df.to_csv(EXPORT_DIR / "validation_values.csv", **csv_options)
counts_df.to_csv(EXPORT_DIR / "count_summary.csv", **csv_options)

file_counts = (
    cells_df.groupby(["relative_path", "partition_key", "group", "algorithm"], dropna=False)
    .agg(
        validation_flag_count=("has_validation", "sum"),
        result_count=("result_present", "sum"),
        total_cells=("record_key", "size"),
    )
    .reset_index()
)
file_counts.to_csv(EXPORT_DIR / "count_by_file.csv", **csv_options)

run_info = {
    "run_label": RUN_LABEL,
    "sos_dir": str(SOS_DIR),
    "export_dir": str(EXPORT_DIR.resolve()),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "file_glob": FILE_GLOB,
    "recursive": RECURSIVE,
    "candidate_file_count": len(paths),
    "extracted_cell_count": len(cells_df),
    "extracted_value_count": len(values_df),
    "issue_count": len(issues_df),
    "count_definition": "result_count = has_validation == 1 AND finite nbias",
}
with (EXPORT_DIR / "run_info.json").open("w", encoding="utf-8") as stream:
    json.dump(run_info, stream, ensure_ascii=False, indent=2, default=safe_json_value)

print(f"\nExport complete: {EXPORT_DIR.resolve()}")
print("Count definition: result_count = has_validation == 1 AND finite nbias")
display(counts_df.sort_values(["group", "algorithm"]).reset_index(drop=True))

fallback_rows = int((cells_df["reach_id_source"] == "fallback_row_index").sum())
print(f"Rows using fallback row index instead of reach ID: {fallback_rows:,}")
print(f"Extraction issues: {len(issues_df):,}")
if not issues_df.empty:
    display(issues_df)
    if STRICT:
        raise RuntimeError(
            "Extraction finished with errors. CSV files were saved; inspect extraction_issues.csv. "
            "Set STRICT=False only if the affected files/groups are intentionally out of scope."
        )'''


EXTRACTION_HELPERS_STREAMING = r'''def clean_text(value):
    """Decode one bytes/string scalar and remove NetCDF padding."""
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8", errors="replace")
    return str(value).replace("\x00", "").strip()


def decode_text_data(raw):
    """Decode a NetCDF char/VLEN selection into a text ndarray."""
    values = np.asarray(np.ma.filled(raw, b""))
    if values.dtype.kind == "S" and values.dtype.itemsize == 1 and values.ndim:
        values = np.asarray(chartostring(values, encoding="utf-8"))
    if values.ndim == 0:
        return np.asarray(clean_text(values.item()), dtype=object)
    return np.vectorize(clean_text, otypes=[object])(values)


def selected_data(variable, row_indices=None):
    if row_indices is None:
        return variable[:]
    return variable[np.asarray(row_indices, dtype=np.int64), ...]


def read_text_selection(variable, row_indices=None):
    return decode_text_data(selected_data(variable, row_indices))


def numeric_data_to_float64(variable, raw):
    masked = np.ma.asarray(raw).astype(np.float64)
    values = np.asarray(np.ma.filled(masked, np.nan), dtype=np.float64)
    fill_value = getattr(variable, "_FillValue", None)
    if fill_value is not None:
        try:
            values[np.isclose(values, float(fill_value), equal_nan=False)] = np.nan
        except (TypeError, ValueError):
            pass
    valid_min = getattr(variable, "valid_min", None)
    valid_max = getattr(variable, "valid_max", None)
    valid_range = getattr(variable, "valid_range", None)
    if valid_range is not None and np.asarray(valid_range).size == 2:
        valid_min, valid_max = np.asarray(valid_range).reshape(-1)[:2]
    if valid_min is not None:
        values[values < float(valid_min)] = np.nan
    if valid_max is not None:
        values[values > float(valid_max)] = np.nan
    return values


def read_numeric_selection(variable, row_indices=None):
    return numeric_data_to_float64(variable, selected_data(variable, row_indices))


def is_numeric_variable(variable):
    try:
        return np.issubdtype(np.dtype(variable.dtype), np.number)
    except TypeError:
        return False


def normalize_algorithm(value):
    text = clean_text(value).lower().replace("-", "_").replace(" ", "_")
    text = re.sub(r"_+", "_", text)
    text = re.sub(r"^(flpe|moi)_+", "", text)
    text = re.sub(r"_+(flpe|moi)$", "", text)
    return text.strip("_")


def normalize_reach_id(value):
    if value is None or np.ma.is_masked(value):
        return ""
    text = clean_text(value)
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    if not np.isfinite(number) or number <= 0:
        return ""
    return str(int(round(number)))


def normalize_gage_id(value):
    return re.sub(r"[^a-z0-9]+", "", clean_text(value).lower())


def partition_key(path):
    match = re.match(r"([a-z]{2})_sword_", path.name, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    prefix = re.split(r"_SOS_results_", path.stem, maxsplit=1, flags=re.IGNORECASE)[0]
    return re.sub(r"[^a-z0-9]+", "_", prefix.lower()).strip("_")


def candidate_reach_variables(dataset):
    candidates = []
    if "reaches" in dataset.groups:
        group = dataset.groups["reaches"]
        for name in ("reach_id", "reach_ids", "reachid"):
            if name in group.variables:
                candidates.append((f"/reaches/{name}", group.variables[name]))
    for name in ("reach_id", "reach_ids", "reachid"):
        if name in dataset.variables:
            candidates.append((f"/{name}", dataset.variables[name]))
    return candidates


def find_selected_reach_ids(dataset, n_reaches, row_indices):
    for source, variable in candidate_reach_variables(dataset):
        if not variable.shape or variable.shape[0] != n_reaches:
            continue
        try:
            if is_numeric_variable(variable):
                values = np.ma.filled(selected_data(variable, row_indices), np.nan)
            else:
                values = read_text_selection(variable, row_indices)
            values = np.asarray(values, dtype=object).reshape(-1)
        except Exception:
            continue
        if values.size == len(row_indices):
            return values, source
    fallback = np.asarray([f"row:{index}" for index in row_indices], dtype=object)
    return fallback, "fallback_row_index"


def align_selected_algorithms(variable, n_reaches, n_algorithms, row_indices):
    if variable.shape and variable.shape[0] == n_reaches:
        values = read_text_selection(variable, row_indices)
    else:
        values = read_text_selection(variable)
    values = np.asarray(values, dtype=object)
    target = (len(row_indices), n_algorithms)
    if values.shape == target:
        return values
    if values.ndim == 1 and values.size == n_algorithms:
        return np.broadcast_to(values.reshape(1, -1), target).copy()
    if values.size == target[0] * target[1]:
        return values.reshape(target)
    raise ValueError(f"algo_names shape {values.shape} cannot align to {target}")


def read_selected_gageids(group, n_reaches, row_indices):
    if "gageid" not in group.variables:
        return np.full(len(row_indices), "", dtype=object)
    variable = group.variables["gageid"]
    if not variable.shape or variable.shape[0] != n_reaches:
        raise ValueError(f"gageid first dimension does not match {n_reaches} reach rows")
    values = np.asarray(read_text_selection(variable, row_indices), dtype=object).reshape(-1)
    if values.size != len(row_indices):
        raise ValueError(f"gageid selection has {values.size} values for {len(row_indices)} rows")
    return values


def aligned_numeric_metric(variable, n_reaches, n_algorithms):
    return (
        variable.shape == (n_reaches, n_algorithms)
        or (n_algorithms == 1 and variable.shape == (n_reaches,))
    )


def inspect_validation_group(dataset, group_name, file_metadata):
    validation = dataset.groups["validation"]
    if group_name not in validation.groups:
        raise KeyError(f"/validation/{group_name} is missing")
    group = validation.groups[group_name]
    for required in ("nbias", "algo_names", "has_validation"):
        if required not in group.variables:
            raise KeyError(f"/validation/{group_name}/{required} is missing")

    nbias_variable = group.variables["nbias"]
    if len(nbias_variable.shape) == 1:
        n_reaches, n_algorithms = nbias_variable.shape[0], 1
    elif len(nbias_variable.shape) == 2:
        n_reaches, n_algorithms = nbias_variable.shape
    else:
        raise ValueError(f"nbias must be 1-D or 2-D, found {nbias_variable.shape}")

    has_validation = read_numeric_selection(group.variables["has_validation"]).reshape(-1)
    if has_validation.size != n_reaches:
        raise ValueError(
            f"has_validation length {has_validation.size} != reach rows {n_reaches}"
        )
    validation_indices = np.flatnonzero(np.isfinite(has_validation) & (has_validation == 1))

    available_metrics = []
    variable_rows = []
    requested = None if METRICS_TO_EXPORT is None else set(METRICS_TO_EXPORT)
    for variable_name, variable in group.variables.items():
        aligned = (
            variable_name != "has_validation"
            and is_numeric_variable(variable)
            and aligned_numeric_metric(variable, n_reaches, n_algorithms)
        )
        should_export = aligned and (
            requested is None or variable_name in requested or variable_name == "nbias"
        )
        if should_export:
            available_metrics.append(variable_name)
        variable_rows.append(
            {
                **file_metadata,
                "group": group_name,
                "variable": variable_name,
                "dtype": str(variable.dtype),
                "dimensions": "|".join(variable.dimensions),
                "shape": "x".join(str(value) for value in variable.shape),
                "aligned_result_metric": aligned,
                "exported_metric": should_export,
            }
        )
    if "nbias" not in available_metrics:
        raise ValueError("nbias could not be aligned/exported")

    return {
        "group": group,
        "n_reaches": n_reaches,
        "n_algorithms": n_algorithms,
        "validation_indices": validation_indices,
        "metrics": available_metrics,
        "variable_rows": variable_rows,
    }


def extract_validation_chunk(dataset, info, row_indices, group_name, file_metadata):
    group = info["group"]
    n_reaches = info["n_reaches"]
    n_algorithms = info["n_algorithms"]
    row_indices = np.asarray(row_indices, dtype=np.int64)
    selected_count = len(row_indices)
    target_shape = (selected_count, n_algorithms)

    algorithms = align_selected_algorithms(
        group.variables["algo_names"], n_reaches, n_algorithms, row_indices
    )
    algorithms_raw = np.asarray(algorithms, dtype=object).reshape(-1)
    algorithms_normalized = np.asarray(
        [normalize_algorithm(value) for value in algorithms_raw], dtype=object
    )
    blank = algorithms_normalized == ""
    if blank.any():
        algorithm_columns = np.tile(np.arange(n_algorithms), selected_count)
        algorithms_normalized[blank] = [
            f"algorithm_column_{index}" for index in algorithm_columns[blank]
        ]

    gageids = read_selected_gageids(group, n_reaches, row_indices)
    reach_ids_raw, reach_id_source = find_selected_reach_ids(
        dataset, n_reaches, row_indices
    )
    reach_ids = np.asarray([normalize_reach_id(value) for value in reach_ids_raw], dtype=object)
    gageid_keys = np.asarray([normalize_gage_id(value) for value in gageids], dtype=object)

    metric_arrays = {}
    for metric_name in info["metrics"]:
        values = read_numeric_selection(group.variables[metric_name], row_indices)
        if n_algorithms == 1 and values.shape == (selected_count,):
            values = values.reshape(-1, 1)
        if values.shape != target_shape:
            raise ValueError(
                f"{metric_name} selection shape {values.shape} != {target_shape}"
            )
        metric_arrays[metric_name] = values

    reach_index_flat = np.repeat(row_indices, n_algorithms)
    algorithm_index_flat = np.tile(np.arange(n_algorithms), selected_count)
    nbias_flat = metric_arrays["nbias"].reshape(-1)
    local_ids = np.asarray(
        [
            f"{file_metadata['relative_path']}|{group_name}|{reach}|{algorithm}"
            for reach, algorithm in zip(reach_index_flat, algorithm_index_flat)
        ],
        dtype=object,
    )
    cells = pd.DataFrame(
        {
            **{name: value for name, value in file_metadata.items()},
            "local_cell_id": local_ids,
            "group": group_name,
            "row_index": reach_index_flat,
            "algorithm_index": algorithm_index_flat,
            "reach_id_raw": np.repeat(
                np.asarray([clean_text(value) for value in reach_ids_raw], dtype=object),
                n_algorithms,
            ),
            "reach_id": np.repeat(reach_ids, n_algorithms),
            "reach_id_source": reach_id_source,
            "gageid": np.repeat(
                np.asarray([clean_text(value) for value in gageids], dtype=object),
                n_algorithms,
            ),
            "gageid_key": np.repeat(gageid_keys, n_algorithms),
            "algorithm_raw": np.asarray([clean_text(value) for value in algorithms_raw]),
            "algorithm": algorithms_normalized,
            "has_validation": 1,
            "nbias_is_finite": np.isfinite(nbias_flat),
            "result_present": np.isfinite(nbias_flat),
        }
    )

    value_frames = []
    for metric_name, matrix in metric_arrays.items():
        flat = matrix.reshape(-1)
        if EXPORT_FLOAT_HEX:
            hex_values = np.asarray([float(np.float64(value)).hex() for value in flat])
        else:
            hex_values = np.full(flat.size, "", dtype=object)
        value_frames.append(
            pd.DataFrame(
                {
                    **{name: value for name, value in file_metadata.items()},
                    "local_cell_id": local_ids,
                    "group": group_name,
                    "row_index": reach_index_flat,
                    "algorithm_index": algorithm_index_flat,
                    "reach_id": np.repeat(reach_ids, n_algorithms),
                    "gageid_key": np.repeat(gageid_keys, n_algorithms),
                    "algorithm": algorithms_normalized,
                    "has_validation": 1,
                    "metric": metric_name,
                    "metric_dtype": str(group.variables[metric_name].dtype),
                    "value": flat,
                    "value_hex64": hex_values,
                    "is_finite": np.isfinite(flat),
                }
            )
        )
    values = pd.concat(value_frames, ignore_index=True)
    return cells, values


def attach_stable_keys_to_chunk(cells, values, occurrence_state):
    occurrences = []
    record_keys = []
    key_columns = ["group", "partition_key", "reach_id", "gageid_key", "algorithm"]
    for row in cells[key_columns].itertuples(index=False, name=None):
        occurrence = occurrence_state[row]
        occurrence_state[row] += 1
        occurrences.append(occurrence)
        record_keys.append("|".join([*(str(value) for value in row), str(occurrence)]))
    cells = cells.copy()
    cells["occurrence"] = occurrences
    cells["record_key"] = record_keys
    lookup = cells[["local_cell_id", "occurrence", "record_key"]]
    values = values.merge(lookup, on="local_cell_id", how="left", validate="many_to_one")
    return cells, values


def update_count_accumulator(accumulator, cells):
    group_name = cells["group"].iloc[0]
    for algorithm, frame in cells.groupby("algorithm", sort=False):
        for key in ((group_name, algorithm), (group_name, "ALL")):
            entry = accumulator.setdefault(
                key,
                {
                    "total_cells": 0,
                    "validation_flag_count": 0,
                    "finite_nbias_count": 0,
                    "result_count": 0,
                    "reaches": set(),
                    "gages": set(),
                },
            )
            result_frame = frame.loc[frame["result_present"]]
            entry["total_cells"] += len(frame)
            entry["validation_flag_count"] += len(frame)
            entry["finite_nbias_count"] += int(frame["nbias_is_finite"].sum())
            entry["result_count"] += int(frame["result_present"].sum())
            entry["reaches"].update(result_frame["reach_id"].astype(str))
            entry["gages"].update(
                value for value in result_frame["gageid_key"].astype(str) if value
            )


def count_accumulator_frame(accumulator):
    rows = []
    for (group_name, algorithm), entry in sorted(accumulator.items()):
        rows.append(
            {
                "run_label": RUN_LABEL,
                "group": group_name,
                "algorithm": algorithm,
                "total_cells": entry["total_cells"],
                "validation_flag_count": entry["validation_flag_count"],
                "finite_nbias_count": entry["finite_nbias_count"],
                "result_count": entry["result_count"],
                "unique_result_reaches": len(entry["reaches"]),
                "unique_result_gages": len(entry["gages"]),
            }
        )
    return pd.DataFrame(rows)


class StreamingCsvWriter:
    """Append chunks to a partial CSV and publish only after extraction."""
    def __init__(self, final_path):
        self.final_path = Path(final_path)
        self.partial_path = self.final_path.with_name(self.final_path.name + ".partial")
        self.started = False
        self.rows = 0

    def append(self, frame):
        if frame.empty:
            return
        frame.to_csv(
            self.partial_path,
            mode="a" if self.started else "w",
            header=not self.started,
            index=False,
            float_format="%.17g",
            na_rep="NaN",
        )
        self.started = True
        self.rows += len(frame)

    def publish(self):
        if not self.started:
            raise RuntimeError(f"No rows were written to {self.final_path.name}")
        self.partial_path.replace(self.final_path)


def atomic_to_csv(frame, final_path):
    final_path = Path(final_path)
    partial_path = final_path.with_name(final_path.name + ".partial")
    frame.to_csv(
        partial_path, index=False, float_format="%.17g", na_rep="NaN"
    )
    partial_path.replace(final_path)


def safe_json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value'''


EXTRACTION_RUN_STREAMING = r'''if not SOS_DIR.is_dir():
    raise NotADirectoryError(f"SOS_DIR does not exist: {SOS_DIR}")

EXPORT_DIR.mkdir(parents=True, exist_ok=True)
paths = sorted(SOS_DIR.rglob(FILE_GLOB) if RECURSIVE else SOS_DIR.glob(FILE_GLOB))
if not paths:
    raise FileNotFoundError(f"No files matching {FILE_GLOB!r} under {SOS_DIR}")

cells_writer = StreamingCsvWriter(EXPORT_DIR / "validation_cells.csv")
values_writer = StreamingCsvWriter(EXPORT_DIR / "validation_values.csv")
occurrence_state = defaultdict(int)
count_accumulator = {}
file_count_accumulator = defaultdict(
    lambda: {"validation_flag_count": 0, "result_count": 0, "total_cells": 0}
)
inventory_rows = []
issue_rows = []
variable_rows = []
fallback_rows = 0

print(f"Found {len(paths):,} candidate NetCDF files")
for file_number, path in enumerate(paths, start=1):
    relative_path = str(path.relative_to(SOS_DIR))
    metadata = {
        "run_label": RUN_LABEL,
        "source_file": path.name,
        "relative_path": relative_path,
        "partition_key": partition_key(path),
        "file_size_bytes": path.stat().st_size,
    }
    inventory = {**metadata, "extract_status": "pending", "validation_groups": ""}
    print(f"\n[{file_number}/{len(paths)}] {relative_path}")
    try:
        with Dataset(path, mode="r") as dataset:
            if "validation" not in dataset.groups:
                inventory["extract_status"] = "skipped_no_validation_group"
                inventory_rows.append(inventory)
                continue
            available = sorted(dataset.groups["validation"].groups)
            inventory["validation_groups"] = "|".join(available)
            extracted_groups = []
            for group_name in VALIDATION_GROUPS:
                try:
                    info = inspect_validation_group(dataset, group_name, metadata)
                    variable_rows.extend(info["variable_rows"])
                    validation_indices = info["validation_indices"]
                    print(
                        f"  {group_name}: {info['n_reaches']:,} reach rows, "
                        f"{info['n_algorithms']} algorithms, "
                        f"{len(validation_indices):,} validation rows, "
                        f"metrics={info['metrics']}"
                    )
                    for chunk_start in range(
                        0, len(validation_indices), VALIDATION_ROW_CHUNK_SIZE
                    ):
                        row_indices = validation_indices[
                            chunk_start : chunk_start + VALIDATION_ROW_CHUNK_SIZE
                        ]
                        cells, values = extract_validation_chunk(
                            dataset, info, row_indices, group_name, metadata
                        )
                        cells, values = attach_stable_keys_to_chunk(
                            cells, values, occurrence_state
                        )
                        update_count_accumulator(count_accumulator, cells)
                        for algorithm, frame in cells.groupby("algorithm", sort=False):
                            key = (
                                relative_path,
                                metadata["partition_key"],
                                group_name,
                                algorithm,
                            )
                            file_entry = file_count_accumulator[key]
                            file_entry["validation_flag_count"] += len(frame)
                            file_entry["result_count"] += int(
                                frame["result_present"].sum()
                            )
                            file_entry["total_cells"] += len(frame)
                        fallback_rows += int(
                            (cells["reach_id_source"] == "fallback_row_index").sum()
                        )
                        cells_writer.append(cells)
                        values_writer.append(values)
                        del cells, values
                    extracted_groups.append(group_name)
                    del info
                    gc.collect()
                except Exception as exc:
                    issue_rows.append(
                        {
                            **metadata,
                            "severity": "error",
                            "group": group_name,
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    print(f"  ERROR {group_name}: {type(exc).__name__}: {exc}")
            inventory["extract_status"] = (
                "ok:" + "|".join(extracted_groups)
                if extracted_groups
                else "error_no_groups_extracted"
            )
    except Exception as exc:
        inventory["extract_status"] = "error_opening_file"
        issue_rows.append(
            {
                **metadata,
                "severity": "error",
                "group": "",
                "message": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"  ERROR opening file: {type(exc).__name__}: {exc}")
    inventory_rows.append(inventory)
    gc.collect()

issue_columns = [
    "run_label", "source_file", "relative_path", "partition_key",
    "file_size_bytes", "severity", "group", "message",
]
inventory_df = pd.DataFrame(inventory_rows)
issues_df = pd.DataFrame(issue_rows, columns=issue_columns)
variables_df = pd.DataFrame(variable_rows)
counts_df = count_accumulator_frame(count_accumulator)

atomic_to_csv(inventory_df, EXPORT_DIR / "file_inventory.csv")
atomic_to_csv(issues_df, EXPORT_DIR / "extraction_issues.csv")
atomic_to_csv(variables_df, EXPORT_DIR / "variable_inventory.csv")

if not cells_writer.started or not values_writer.started:
    raise RuntimeError(
        "No validation records were extracted. Inspect file_inventory.csv and "
        "extraction_issues.csv."
    )

cells_writer.publish()
values_writer.publish()
atomic_to_csv(counts_df, EXPORT_DIR / "count_summary.csv")

file_count_rows = []
for key, entry in sorted(file_count_accumulator.items()):
    relative_path, current_partition, group_name, algorithm = key
    file_count_rows.append(
        {
            "relative_path": relative_path,
            "partition_key": current_partition,
            "group": group_name,
            "algorithm": algorithm,
            **entry,
        }
    )
atomic_to_csv(pd.DataFrame(file_count_rows), EXPORT_DIR / "count_by_file.csv")

run_info = {
    "run_label": RUN_LABEL,
    "sos_dir": str(SOS_DIR),
    "export_dir": str(EXPORT_DIR.resolve()),
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version,
    "platform": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "file_glob": FILE_GLOB,
    "recursive": RECURSIVE,
    "validation_row_chunk_size": VALIDATION_ROW_CHUNK_SIZE,
    "metrics_to_export": METRICS_TO_EXPORT,
    "candidate_file_count": len(paths),
    "extracted_cell_count": cells_writer.rows,
    "extracted_value_count": values_writer.rows,
    "issue_count": len(issues_df),
    "count_definition": "result_count = has_validation == 1 AND finite nbias",
}
run_info_path = EXPORT_DIR / "run_info.json"
run_info_partial = run_info_path.with_name(run_info_path.name + ".partial")
with run_info_partial.open("w", encoding="utf-8") as stream:
    json.dump(
        run_info, stream, ensure_ascii=False, indent=2, default=safe_json_value
    )
run_info_partial.replace(run_info_path)

print(f"\nExport complete: {EXPORT_DIR.resolve()}")
print("Only has_validation == 1 rows were materialized.")
print("Count definition: result_count = has_validation == 1 AND finite nbias")
display(counts_df.sort_values(["group", "algorithm"]).reset_index(drop=True))
print(f"Rows using fallback row index instead of reach ID: {fallback_rows:,}")
print(f"Extraction issues: {len(issues_df):,}")
if not issues_df.empty:
    display(issues_df)
    if STRICT:
        raise RuntimeError(
            "Extraction finished with errors. Completed CSV files were saved; inspect "
            "extraction_issues.csv. Set STRICT=False only if the affected files/groups "
            "are intentionally out of scope."
        )'''


COMPARISON_IMPORTS = r'''from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)

print("Python:", sys.version.split()[0])
print("NumPy:", np.__version__)
print("pandas:", pd.__version__)'''


COMPARISON_HELPERS = r'''KEY_COLUMNS = ["partition_key", "reach_id", "gageid_key", "algorithm", "occurrence"]
VALUE_KEY_COLUMNS = KEY_COLUMNS + ["metric"]
STRING_COLUMNS = [
    "partition_key", "reach_id", "gageid_key", "algorithm", "record_key", "metric",
    "value_hex64", "group", "source_file", "relative_path",
]


def read_export(export_dir):
    export_dir = Path(export_dir)
    required = ["validation_cells.csv", "validation_values.csv", "count_summary.csv"]
    missing = [name for name in required if not (export_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing files in {export_dir}: {missing}")

    # keep_default_na=False preserves the literal float-hex token "nan".
    cells = pd.read_csv(
        export_dir / "validation_cells.csv", low_memory=False, keep_default_na=False
    )
    values = pd.read_csv(
        export_dir / "validation_values.csv", low_memory=False, keep_default_na=False
    )
    counts = pd.read_csv(
        export_dir / "count_summary.csv", low_memory=False, keep_default_na=False
    )
    for frame in (cells, values):
        for column in STRING_COLUMNS:
            if column in frame:
                frame[column] = frame[column].astype("string").fillna("")
    for frame in (cells, values):
        if "occurrence" in frame:
            frame["occurrence"] = pd.to_numeric(frame["occurrence"], errors="raise").astype(int)
        if "has_validation" in frame:
            frame["has_validation"] = pd.to_numeric(frame["has_validation"], errors="coerce").fillna(0).astype(int)
    values["value"] = pd.to_numeric(values["value"], errors="coerce")
    return {"dir": export_dir, "cells": cells, "values": values, "counts": counts}


def assert_unique(frame, keys, label):
    duplicate = frame.duplicated(keys, keep=False)
    if duplicate.any():
        display(frame.loc[duplicate, keys].head(20))
        raise ValueError(f"{label}: {int(duplicate.sum())} rows have duplicate comparison keys")


def compare_counts(osc_counts, unity_counts):
    index_columns = ["group", "algorithm"]
    value_columns = [
        "total_cells", "validation_flag_count", "finite_nbias_count",
        "result_count", "unique_result_reaches", "unique_result_gages",
    ]
    left = osc_counts[index_columns + value_columns].rename(
        columns={name: f"{name}_osc" for name in value_columns}
    )
    right = unity_counts[index_columns + value_columns].rename(
        columns={name: f"{name}_unity" for name in value_columns}
    )
    compared = left.merge(right, on=index_columns, how="outer", indicator=True)
    for name in value_columns:
        compared[f"{name}_osc"] = compared[f"{name}_osc"].fillna(0).astype(int)
        compared[f"{name}_unity"] = compared[f"{name}_unity"].fillna(0).astype(int)
        compared[f"{name}_delta_unity_minus_osc"] = (
            compared[f"{name}_unity"] - compared[f"{name}_osc"]
        )
    compared["result_count_matches"] = (
        compared["result_count_osc"] == compared["result_count_unity"]
    )
    return compared.sort_values(index_columns).reset_index(drop=True)


def compare_record_coverage(osc_cells, unity_cells):
    osc = osc_cells.loc[osc_cells["has_validation"] == 1].copy()
    unity = unity_cells.loc[unity_cells["has_validation"] == 1].copy()
    assert_unique(osc, ["group"] + KEY_COLUMNS, "OSC validation cells")
    assert_unique(unity, ["group"] + KEY_COLUMNS, "Unity validation cells")
    coverage = osc.merge(
        unity,
        on=["group"] + KEY_COLUMNS,
        how="outer",
        suffixes=("_osc", "_unity"),
        indicator="coverage_status",
    )
    coverage["coverage_status"] = coverage["coverage_status"].map(
        {"both": "both", "left_only": "osc_only", "right_only": "unity_only"}
    ).astype("string")
    return coverage


def compare_moi_values(osc_values, unity_values):
    osc = osc_values.loc[
        (osc_values["group"] == "moi") & (osc_values["has_validation"] == 1)
    ].copy()
    unity = unity_values.loc[
        (unity_values["group"] == "moi") & (unity_values["has_validation"] == 1)
    ].copy()
    assert_unique(osc, VALUE_KEY_COLUMNS, "OSC MOI values")
    assert_unique(unity, VALUE_KEY_COLUMNS, "Unity MOI values")

    keep = VALUE_KEY_COLUMNS + ["value", "value_hex64", "metric_dtype", "is_finite"]
    compared = osc[keep].merge(
        unity[keep],
        on=VALUE_KEY_COLUMNS,
        how="outer",
        suffixes=("_osc", "_unity"),
        indicator="row_status",
    )
    compared["row_status"] = compared["row_status"].map(
        {"both": "both", "left_only": "osc_only", "right_only": "unity_only"}
    ).astype("string")
    matched = compared["row_status"] == "both"
    osc_value = pd.to_numeric(compared["value_osc"], errors="coerce")
    unity_value = pd.to_numeric(compared["value_unity"], errors="coerce")
    compared["value_osc"] = osc_value
    compared["value_unity"] = unity_value
    both_nan = osc_value.isna() & unity_value.isna()
    both_finite = np.isfinite(osc_value) & np.isfinite(unity_value)

    compared["both_finite"] = matched & both_finite
    compared["same_numeric_value"] = matched & ((osc_value == unity_value) | both_nan)
    compared["exact_float64"] = matched & (
        compared["value_hex64_osc"].astype("string")
        == compared["value_hex64_unity"].astype("string")
    )
    compared["within_tolerance"] = False
    compared.loc[matched, "within_tolerance"] = np.isclose(
        osc_value[matched], unity_value[matched], rtol=RTOL, atol=ATOL, equal_nan=True
    )
    compared["delta_unity_minus_osc"] = np.where(
        compared["both_finite"], unity_value - osc_value, np.nan
    )
    compared["absolute_difference"] = np.abs(compared["delta_unity_minus_osc"])
    denominator = np.maximum(np.abs(osc_value), np.abs(unity_value))
    compared["relative_difference"] = np.where(
        compared["both_finite"] & (denominator > 0),
        compared["absolute_difference"] / denominator,
        np.where(compared["both_finite"] & (compared["absolute_difference"] == 0), 0.0, np.nan),
    )
    return compared


def precision_summary(compared):
    rows = []

    def one_summary(frame, metric, algorithm):
        matched = frame.loc[frame["row_status"] == "both"]
        finite = matched.loc[matched["both_finite"]]
        differences = finite["absolute_difference"].dropna()
        return {
            "metric": metric,
            "algorithm": algorithm,
            "union_rows": len(frame),
            "matched_rows": len(matched),
            "osc_only_rows": int((frame["row_status"] == "osc_only").sum()),
            "unity_only_rows": int((frame["row_status"] == "unity_only").sum()),
            "both_finite_rows": len(finite),
            "exact_float64_rows": int(matched["exact_float64"].sum()),
            "same_numeric_value_rows": int(matched["same_numeric_value"].sum()),
            "within_tolerance_rows": int(matched["within_tolerance"].sum()),
            "different_float64_rows": int((~matched["exact_float64"]).sum()),
            "max_absolute_difference": differences.max() if not differences.empty else np.nan,
            "median_absolute_difference": differences.median() if not differences.empty else np.nan,
            "p95_absolute_difference": differences.quantile(0.95) if not differences.empty else np.nan,
            "max_relative_difference": finite["relative_difference"].max() if not finite.empty else np.nan,
        }

    for metric, metric_frame in compared.groupby("metric", sort=True):
        rows.append(one_summary(metric_frame, metric, "ALL"))
        for algorithm, frame in metric_frame.groupby("algorithm", sort=True):
            rows.append(one_summary(frame, metric, algorithm))
    return pd.DataFrame(rows)


def nbias_accuracy_tables(compared):
    paired = compared.loc[
        (compared["metric"] == PRIMARY_METRIC)
        & (compared["row_status"] == "both")
        & compared["both_finite"]
    ].copy()
    paired["abs_nbias_osc"] = paired["value_osc"].abs()
    paired["abs_nbias_unity"] = paired["value_unity"].abs()
    paired["abs_nbias_delta_unity_minus_osc"] = (
        paired["abs_nbias_unity"] - paired["abs_nbias_osc"]
    )
    paired["accuracy_outcome"] = np.select(
        [
            paired["abs_nbias_delta_unity_minus_osc"] < -ACCURACY_TIE_ATOL,
            paired["abs_nbias_delta_unity_minus_osc"] > ACCURACY_TIE_ATOL,
        ],
        ["unity_better", "osc_better"],
        default="tie",
    )

    rows = []
    def summarize(frame, algorithm):
        delta = frame["abs_nbias_delta_unity_minus_osc"]
        return {
            "metric": PRIMARY_METRIC,
            "algorithm": algorithm,
            "paired_samples": len(frame),
            "median_abs_nbias_osc": frame["abs_nbias_osc"].median(),
            "median_abs_nbias_unity": frame["abs_nbias_unity"].median(),
            "mean_abs_nbias_osc": frame["abs_nbias_osc"].mean(),
            "mean_abs_nbias_unity": frame["abs_nbias_unity"].mean(),
            "p90_abs_nbias_osc": frame["abs_nbias_osc"].quantile(0.90),
            "p90_abs_nbias_unity": frame["abs_nbias_unity"].quantile(0.90),
            "median_paired_abs_nbias_delta_unity_minus_osc": delta.median(),
            "mean_paired_abs_nbias_delta_unity_minus_osc": delta.mean(),
            "unity_better_count": int((frame["accuracy_outcome"] == "unity_better").sum()),
            "osc_better_count": int((frame["accuracy_outcome"] == "osc_better").sum()),
            "tie_count": int((frame["accuracy_outcome"] == "tie").sum()),
        }

    if not paired.empty:
        rows.append(summarize(paired, "ALL"))
        for algorithm, frame in paired.groupby("algorithm", sort=True):
            rows.append(summarize(frame, algorithm))
    return paired, pd.DataFrame(rows)'''


COMPARISON_RUN = r'''for directory, label in ((OSC_EXPORT_DIR, "OSC"), (UNITY_EXPORT_DIR, "Unity")):
    if not directory.is_dir():
        raise NotADirectoryError(f"{label} export directory not found: {directory.resolve()}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
osc = read_export(OSC_EXPORT_DIR)
unity = read_export(UNITY_EXPORT_DIR)

count_comparison = compare_counts(osc["counts"], unity["counts"])
record_coverage = compare_record_coverage(osc["cells"], unity["cells"])
moi_differences = compare_moi_values(osc["values"], unity["values"])
moi_precision = precision_summary(moi_differences)
moi_nbias_paired, moi_accuracy = nbias_accuracy_tables(moi_differences)

csv_options = {"index": False, "float_format": "%.17g", "na_rep": "NaN"}
count_comparison.to_csv(OUTPUT_DIR / "count_comparison.csv", **csv_options)
record_coverage.to_csv(OUTPUT_DIR / "record_coverage.csv", **csv_options)
moi_differences.to_csv(OUTPUT_DIR / "moi_value_differences.csv", **csv_options)
moi_precision.to_csv(OUTPUT_DIR / "moi_precision_summary.csv", **csv_options)
moi_nbias_paired.to_csv(OUTPUT_DIR / "moi_nbias_pairwise.csv", **csv_options)
moi_accuracy.to_csv(OUTPUT_DIR / "moi_accuracy_summary.csv", **csv_options)

print("\n=== 1) validation MOI / FLPE result counts ===")
display(
    count_comparison.loc[
        :, [
            "group", "algorithm", "result_count_osc", "result_count_unity",
            "result_count_delta_unity_minus_osc", "result_count_matches", "_merge",
        ]
    ]
)

print("\n=== 2) Record-key coverage ===")
coverage_summary = (
    record_coverage.groupby(["group", "coverage_status"], observed=True)
    .size().rename("rows").reset_index()
)
display(coverage_summary)

print("\n=== 3) MOI numeric reproducibility (exact and tolerance-aware) ===")
display(moi_precision.loc[moi_precision["algorithm"] == "ALL"].reset_index(drop=True))

print("\n=== 4) MOI validation accuracy based on paired |nBias| (lower is better) ===")
if moi_accuracy.empty:
    print(f"No common finite {PRIMARY_METRIC!r} samples were available.")
else:
    display(moi_accuracy)

count_mismatches = int((~count_comparison["result_count_matches"]).sum())
coverage_mismatches = int((record_coverage["coverage_status"] != "both").sum())
matched_precision = moi_precision.loc[moi_precision["algorithm"] == "ALL"]
numeric_differences = int(matched_precision["different_float64_rows"].sum())
tolerance_failures = int(
    (matched_precision["matched_rows"] - matched_precision["within_tolerance_rows"]).sum()
)

print("\n=== Automatic conclusion ===")
print("Count comparison:", "MATCH" if count_mismatches == 0 else f"DIFFER ({count_mismatches} rows)")
print("Record coverage:", "MATCH" if coverage_mismatches == 0 else f"DIFFER ({coverage_mismatches} records)")
print("MOI exact float64 values:", "MATCH" if numeric_differences == 0 else f"DIFFER ({numeric_differences} matched values)")
print(
    f"MOI values within atol={ATOL:g}, rtol={RTOL:g}:",
    "YES" if tolerance_failures == 0 else f"NO ({tolerance_failures} matched values outside tolerance)",
)
print(f"\nAll comparison CSV files: {OUTPUT_DIR.resolve()}")'''


COMPARISON_PLOTS = r'''if MAKE_PLOTS:
    import matplotlib.pyplot as plt

    flpe_counts = count_comparison.loc[
        (count_comparison["group"] == "flpe") & (count_comparison["algorithm"] != "ALL")
    ].copy()
    if not flpe_counts.empty:
        x = np.arange(len(flpe_counts))
        width = 0.38
        fig, ax = plt.subplots(figsize=(max(8, len(flpe_counts) * 1.2), 4.8))
        ax.bar(x - width / 2, flpe_counts["result_count_osc"], width, label="OSC")
        ax.bar(x + width / 2, flpe_counts["result_count_unity"], width, label="Unity")
        ax.set_xticks(x, flpe_counts["algorithm"], rotation=35, ha="right")
        ax.set_ylabel("validation FLPE result count")
        ax.set_title("FLPE validation counts by algorithm")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "flpe_count_comparison.png", dpi=180)
        plt.show()

    if not moi_nbias_paired.empty:
        fig, ax = plt.subplots(figsize=(6.2, 6.2))
        for algorithm, frame in moi_nbias_paired.groupby("algorithm", sort=True):
            ax.scatter(frame["value_osc"], frame["value_unity"], s=12, alpha=0.45, label=algorithm)
        finite_values = np.concatenate(
            [moi_nbias_paired["value_osc"].to_numpy(), moi_nbias_paired["value_unity"].to_numpy()]
        )
        lower, upper = np.nanpercentile(finite_values, [1, 99])
        if np.isfinite(lower) and np.isfinite(upper):
            ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=1)
            ax.set_xlim(lower, upper)
            ax.set_ylim(lower, upper)
        ax.set_xlabel("OSC MOI validation nBias")
        ax.set_ylabel("Unity MOI validation nBias")
        ax.set_title("Paired MOI nBias: OSC vs Unity")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "moi_nbias_scatter.png", dpi=180)
        plt.show()'''


def markdown_cell(text):
    return nbf.v4.new_markdown_cell(text)


def code_cell(source):
    return nbf.v4.new_code_cell(source)


def extraction_notebook(run_label, sos_dir, export_name):
    title = f"Extract SOS validation results — {run_label}"
    config = f'''# Only this cell normally needs editing.
RUN_LABEL = {run_label!r}
SOS_DIR = Path({sos_dir!r})
EXPORT_DIR = Path.cwd() / {export_name!r}

# Search all NetCDF files directly in SOS_DIR. Set RECURSIVE=True if needed.
FILE_GLOB = "*.nc"
RECURSIVE = False
VALIDATION_GROUPS = ("moi", "flpe")

# Only validation rows are read. This limit bounds peak memory for unusually
# large validation sets; 20,000 rows is conservative on shared HPC notebooks.
VALIDATION_ROW_CHUNK_SIZE = 20_000

# nBias is sufficient for the requested count/precision comparison. Set this
# to None to export every aligned numeric metric after the nBias run succeeds.
METRICS_TO_EXPORT = ("nbias",)
EXPORT_FLOAT_HEX = True

# With STRICT=True, extraction errors are reported after all readable CSVs are saved.
STRICT = True

print("SOS_DIR:", SOS_DIR)
print("EXPORT_DIR:", EXPORT_DIR.resolve())'''
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        markdown_cell(
            f"# {title}\n\n"
            "This notebook reads every SOS NetCDF file, extracts `/validation/moi` and "
            "`/validation/flpe`, counts valid results, and exports portable CSV files. "
            "A result is counted when `has_validation == 1` and `nbias` is finite.\n\n"
            "Requirements: `numpy`, `pandas`, and `netCDF4`. Run all cells, then download "
            f"the complete `{export_name}` directory. The comparison notebook needs that directory unchanged."
        ),
        code_cell(EXTRACTION_IMPORTS),
        code_cell(config),
        markdown_cell(
            "## Extraction helpers\n\n"
            "The extractor first selects `has_validation == 1`, then reads those rows in "
            "bounded chunks and streams them directly to partial CSV files. By default only "
            "`nbias` is exported. Each value also has a float64 hexadecimal representation."
        ),
        code_cell(EXTRACTION_HELPERS_STREAMING),
        markdown_cell("## Run extraction and write CSV files"),
        code_cell(EXTRACTION_RUN_STREAMING),
        markdown_cell(
            "## Files to download\n\n"
            "Download the entire export directory. Important files are `count_summary.csv`, "
            "`validation_cells.csv`, `validation_values.csv`, `file_inventory.csv`, and "
            "`extraction_issues.csv`."
        ),
    ]
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


def comparison_notebook():
    config = r'''# Put both downloaded export directories beside this notebook, or edit these paths.
BASE_DIR = Path.cwd()
OSC_EXPORT_DIR = BASE_DIR / "osc_validation_export"
UNITY_EXPORT_DIR = BASE_DIR / "unity_validation_export"
OUTPUT_DIR = BASE_DIR / "osc_unity_validation_comparison"

# Numeric reproducibility tolerance. Exact float64 equality is always reported separately.
ATOL = 1e-12
RTOL = 1e-10

# Accuracy means closeness to validation observations; |nBias| is lower-is-better.
PRIMARY_METRIC = "nbias"
ACCURACY_TIE_ATOL = 1e-12
MAKE_PLOTS = True

print("OSC export:", OSC_EXPORT_DIR.resolve())
print("Unity export:", UNITY_EXPORT_DIR.resolve())
print("Comparison output:", OUTPUT_DIR.resolve())'''
    notebook = nbf.v4.new_notebook()
    notebook["cells"] = [
        markdown_cell(
            "# Compare OSC and Unity SOS validation results\n\n"
            "This notebook compares: (1) the number of MOI validation results; "
            "(2) FLPE validation counts by algorithm; and (3) MOI numerical reproducibility "
            "and validation accuracy. It performs an outer join first, so missing/extra records "
            "cannot be hidden by a paired inner join.\n\n"
            "Place the complete `osc_validation_export` and `unity_validation_export` "
            "directories beside this notebook, then run all cells."
        ),
        code_cell(COMPARISON_IMPORTS),
        code_cell(config),
        markdown_cell(
            "## Comparison definitions\n\n"
            "- **Counted result:** `has_validation == 1` and finite `nbias`.\n"
            "- **Numerical reproducibility:** same keyed MOI metric value across runs; both exact "
            "float64 and configurable tolerance checks are reported.\n"
            "- **Validation accuracy:** paired absolute MOI `nBias`; lower is better. This is distinct "
            "from machine-level numerical precision."
        ),
        code_cell(COMPARISON_HELPERS),
        markdown_cell("## Run comparisons and export detailed CSV files"),
        code_cell(COMPARISON_RUN),
        markdown_cell("## Diagnostic plots"),
        code_cell(COMPARISON_PLOTS),
        markdown_cell(
            "## How to interpret\n\n"
            "Start with `count_comparison.csv`. If counts match, inspect `record_coverage.csv` "
            "to ensure the actual reach/gage/algorithm keys also match. Then use "
            "`moi_precision_summary.csv` for numerical equality and `moi_accuracy_summary.csv` "
            "for paired validation performance. `moi_value_differences.csv` contains every metric-level difference."
        ),
    ]
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    }
    return notebook


def write_notebook(path, notebook):
    nbf.write(notebook, path)
    print(f"Wrote {path}")


def main():
    HERE.mkdir(parents=True, exist_ok=True)
    write_notebook(
        HERE / "01_extract_OSC_validation.ipynb",
        extraction_notebook(
            "OSC",
            "/fs/ess/PAS1926/2026confluence/SVSfull/confluence_svs_full_MOI/svs_full_MOI_mnt/output/sos",
            "osc_validation_export",
        ),
    )
    write_notebook(
        HERE / "02_extract_Unity_validation.ipynb",
        extraction_notebook(
            "Unity",
            "/nas/cee-water/cjgleason/Yushan/Confluence_Aug/confluence_global_v17c_gagecorr/global_v17c_gagecorr_mnt/output/sos",
            "unity_validation_export",
        ),
    )
    write_notebook(
        HERE / "03_compare_OSC_Unity_validation.ipynb",
        comparison_notebook(),
    )


if __name__ == "__main__":
    main()

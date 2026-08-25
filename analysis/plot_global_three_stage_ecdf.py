"""Plot global FLPE -> integrated mean -> final MOI validation ECDFs.

This script implements the diagnostic requested for ``compute_FLPs``:

1. FLPE validation nBias from the continental SOS result files;
2. the pre-FLP basin-integrated mean stored as ``qbar_basinScale`` in each
   ``{reach_id}_integrator.nc`` file;
3. final MOI validation nBias from the continental SOS result files.

The three ECDF curves use the same reach/algorithm samples.  For the middle
curve, the SVS reference mean is calculated on dates where the final MOI
discharge and the selected validation gage both have valid values.
"""

from collections import defaultdict
from pathlib import Path
import re
import warnings

from netCDF4 import Dataset, chartostring
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# User configuration
# ============================================================

RUN_ROOT = Path(
    "/nas/cee-water/cjgleason/Yushan/Confluence_Aug/"
    "confluence_global_v17c_gagecorr/"
    "global_v17c_gagecorr_mnt"
)

SOS_DIR = RUN_ROOT / "output" / "sos"
MOI_DIR = RUN_ROOT / "moi"
SVS_DIR = RUN_ROOT / "input" / "svs"

FILE_PATTERN = "*_sword_v17_SOS_results_*.nc"
SVS_PATTERN = "*SVS*.nc"

FLPE_ALGORITHMS = (
    "busboi",
    "metroman",
    "momma",
    "sic4dvar",
)

ALGORITHM_LABELS = {
    "busboi": "BUSBOI",
    "metroman": "MetroMan",
    "momma": "MOMMA",
    "sic4dvar": "SIC4DVar",
}

CONTINENT_ORDER = ["AF", "AS", "EU", "NA", "OC", "SA"]

SAVE_FIGURES = True
SAVE_TABLES = True
SHOW_FIGURE = True
FIGURE_DIR = Path.cwd()


# ============================================================
# Generic NetCDF helpers (adapted from the original notebook)
# ============================================================

def decode_char_variable(variable):
    """Decode a NetCDF character array into cleaned strings."""
    raw = np.ma.filled(variable[:], b"")
    decoded = np.asarray(chartostring(raw, encoding="utf-8")).astype(str)
    return np.vectorize(
        lambda value: value.replace("\x00", "").strip(),
        otypes=[str],
    )(decoded)


def decode_strings_along_dimension(variable, record_dimension):
    """Decode one string per record for either char arrays or VLEN strings."""
    raw = np.ma.filled(variable[:], b"")
    values = np.asarray(raw)

    if record_dimension not in variable.dimensions:
        raise ValueError(
            f"{variable.name} does not have dimension {record_dimension!r}: "
            f"{variable.dimensions}"
        )

    record_axis = variable.dimensions.index(record_dimension)
    values = np.moveaxis(values, record_axis, 0)

    decoded = []
    for record in values:
        pieces = []
        for value in np.asarray(record).reshape(-1):
            if isinstance(value, (bytes, np.bytes_)):
                pieces.append(bytes(value).decode("utf-8", errors="ignore"))
            else:
                pieces.append(str(value))
        decoded.append("".join(pieces).replace("\x00", "").strip())

    return np.asarray(decoded, dtype=object)


def read_numeric_variable(variable):
    """Read a flow/metric variable and convert fill or invalid values to NaN."""
    values = np.asarray(np.ma.filled(variable[:], np.nan), dtype=float)

    fill_value = getattr(variable, "_FillValue", None)
    if fill_value is not None:
        try:
            values[np.isclose(values, float(fill_value))] = np.nan
        except (TypeError, ValueError):
            pass

    valid_min = getattr(variable, "valid_min", None)
    valid_max = getattr(variable, "valid_max", None)

    if valid_min is not None:
        values[values < float(valid_min)] = np.nan
    if valid_max is not None:
        values[values > float(valid_max)] = np.nan

    values[np.abs(values) > 1e8] = np.nan
    values[~np.isfinite(values)] = np.nan
    return values


def read_scalar(variable):
    """Read one finite scalar from a NetCDF variable."""
    values = read_numeric_variable(variable).reshape(-1)
    if values.size != 1:
        raise ValueError(
            f"Expected scalar variable {variable.name}, found shape "
            f"{variable.shape}"
        )
    return float(values[0])


def normalize_algorithm_name(name):
    """Normalize algorithm names for reliable matching."""
    name = str(name).replace("\x00", "").strip().lower()
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"_+", "_", name)
    name = re.sub(r"^(flpe|moi)_+", "", name)
    name = re.sub(r"_+(flpe|moi)$", "", name)
    return name.strip("_")


def normalize_reach_id(value):
    """Return a stable integer reach ID string."""
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


def canonical_station_id(value):
    """Normalize gage IDs while tolerating spaces and punctuation."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).replace("\x00", "").strip().lower(),
    )


def get_continent_from_path(nc_path):
    """Extract AF/AS/EU/NA/OC/SA from a continental SOS filename."""
    match = re.match(
        r"([a-z]{2})_sword_",
        nc_path.name,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"Cannot determine continent from filename: {nc_path.name}"
        )
    return match.group(1).upper()


def find_reach_ids(nc, expected_size):
    """Find a one-dimensional reach-ID variable matching validation rows."""
    candidates = []

    if "reaches" in nc.groups:
        reaches_group = nc.groups["reaches"]
        for variable_name in ("reach_id", "reach_ids"):
            if variable_name in reaches_group.variables:
                candidates.append(reaches_group.variables[variable_name])

    for variable_name in ("reach_id", "reach_ids"):
        if variable_name in nc.variables:
            candidates.append(nc.variables[variable_name])

    for variable in candidates:
        values = np.asarray(np.ma.filled(variable[:], -1)).squeeze()
        if values.ndim == 1 and values.size == expected_size:
            return values

    return None


# ============================================================
# Read existing FLPE and final-MOI validation nBias
# ============================================================

def load_selected_algorithm_validation(
    nc,
    group_name,
    target_algorithms,
    reach_ids=None,
):
    """Read selected algorithms from /validation/flpe or /validation/moi."""
    group = nc.groups["validation"].groups[group_name]

    algo_names = decode_char_variable(group.variables["algo_names"])
    normalized_algorithms = np.vectorize(
        normalize_algorithm_name,
        otypes=[str],
    )(algo_names)

    nbias = read_numeric_variable(group.variables["nbias"])
    gageid = decode_char_variable(group.variables["gageid"]).reshape(-1)
    has_validation = np.asarray(
        np.ma.filled(group.variables["has_validation"][:], 0),
        dtype=int,
    ).reshape(-1)

    if normalized_algorithms.shape != nbias.shape:
        raise ValueError(
            f"/validation/{group_name}: algo_names shape "
            f"{normalized_algorithms.shape} != nbias shape {nbias.shape}"
        )

    n_reaches, n_algorithms = nbias.shape
    if gageid.size != n_reaches or has_validation.size != n_reaches:
        raise ValueError(
            f"/validation/{group_name}: validation metadata does not match "
            f"the {n_reaches} reach rows"
        )

    target_algorithms = {
        normalize_algorithm_name(name) for name in target_algorithms
    }

    row_index = np.repeat(np.arange(n_reaches), n_algorithms)
    algorithms_flat = normalized_algorithms.reshape(-1)
    nbias_flat = nbias.reshape(-1)

    valid = (
        np.repeat(has_validation == 1, n_algorithms)
        & np.isin(algorithms_flat, list(target_algorithms))
        & np.isfinite(nbias_flat)
    )

    if reach_ids is None:
        reach_id_flat = np.full(
            n_reaches * n_algorithms,
            None,
            dtype=object,
        )
    else:
        reach_ids = np.asarray(reach_ids).reshape(-1)
        if reach_ids.size != n_reaches:
            raise ValueError(
                f"reach_ids has {reach_ids.size} rows, but validation has "
                f"{n_reaches} rows"
            )
        reach_id_flat = np.repeat(reach_ids, n_algorithms)

    dataframe = pd.DataFrame(
        {
            "row_index": row_index[valid],
            "reach_id": reach_id_flat[valid],
            "gageid": np.repeat(gageid, n_algorithms)[valid],
            "algorithm": algorithms_flat[valid],
            "nbias": nbias_flat[valid],
            "abs_nbias": np.abs(nbias_flat[valid]),
            "source": group_name.upper(),
        }
    )

    duplicate_mask = dataframe.duplicated(
        subset=["row_index", "algorithm"],
        keep=False,
    )
    if duplicate_mask.any():
        raise ValueError(
            f"Duplicate rows found in /validation/{group_name}:\n"
            f"{dataframe.loc[duplicate_mask].head(10)}"
        )

    return dataframe.reset_index(drop=True)


def load_global_paired_validation():
    """Load paired FLPE/final-MOI validation from all continental files."""
    if not SOS_DIR.is_dir():
        raise NotADirectoryError(f"SOS directory not found: {SOS_DIR}")

    paths = list(SOS_DIR.glob(FILE_PATTERN))
    if not paths:
        raise FileNotFoundError(
            f"No SOS files matching {FILE_PATTERN!r} in {SOS_DIR}"
        )

    continent_rank = {
        continent: index for index, continent in enumerate(CONTINENT_ORDER)
    }
    paths = sorted(
        paths,
        key=lambda path: (
            continent_rank.get(get_continent_from_path(path), 999),
            path.name,
        ),
    )

    tables = []
    for path in paths:
        continent = get_continent_from_path(path)
        print(f"Reading SOS validation {continent}: {path.name}")

        with Dataset(path, mode="r") as nc:
            if "validation" not in nc.groups:
                raise KeyError(f"{path.name}: /validation not found")

            validation_group = nc.groups["validation"]
            for required_group in ("flpe", "moi"):
                if required_group not in validation_group.groups:
                    raise KeyError(
                        f"{path.name}: /validation/{required_group} not found"
                    )

            n_flpe_rows = validation_group.groups["flpe"].variables[
                "nbias"
            ].shape[0]
            n_moi_rows = validation_group.groups["moi"].variables[
                "nbias"
            ].shape[0]
            if n_flpe_rows != n_moi_rows:
                raise ValueError(
                    f"{path.name}: FLPE has {n_flpe_rows} rows, MOI has "
                    f"{n_moi_rows} rows"
                )

            reach_ids = find_reach_ids(nc, expected_size=n_flpe_rows)
            if reach_ids is None:
                raise KeyError(
                    f"{path.name}: could not find validation reach IDs"
                )

            flpe = load_selected_algorithm_validation(
                nc,
                group_name="flpe",
                target_algorithms=FLPE_ALGORITHMS,
                reach_ids=reach_ids,
            )
            moi = load_selected_algorithm_validation(
                nc,
                group_name="moi",
                target_algorithms=FLPE_ALGORITHMS,
                reach_ids=reach_ids,
            )

        paired = flpe.merge(
            moi,
            on=["row_index", "algorithm"],
            how="inner",
            suffixes=("_flpe", "_moi"),
            validate="one_to_one",
        )

        if not paired.empty:
            left_gages = paired["gageid_flpe"].astype("string").fillna("")
            right_gages = paired["gageid_moi"].astype("string").fillna("")
            mismatch = left_gages != right_gages
            if mismatch.any():
                raise ValueError(
                    f"{path.name}: FLPE/MOI gage IDs differ:\n"
                    f"{paired.loc[mismatch].head(10)}"
                )

        paired["reach_id"] = paired["reach_id_flpe"].combine_first(
            paired["reach_id_moi"]
        )
        paired["reach_id_str"] = paired["reach_id"].map(normalize_reach_id)
        paired["gageid"] = paired["gageid_flpe"]
        paired["continent"] = continent
        paired["source_file"] = path.name
        paired["abs_nbias_flpe"] = paired["nbias_flpe"].abs()
        paired["abs_nbias_moi"] = paired["nbias_moi"].abs()
        tables.append(paired)

    combined = pd.concat(tables, ignore_index=True)
    if combined.empty:
        raise ValueError("No paired FLPE/final-MOI validation samples found")
    if (combined["reach_id_str"] == "").any():
        raise ValueError("Some paired validation rows do not have a reach ID")
    return combined


# ============================================================
# Read SVS once and match the SOS validation gage
# ============================================================

def discover_svs_file():
    """Require exactly one SVS file in the run input directory."""
    if not SVS_DIR.is_dir():
        raise NotADirectoryError(f"SVS directory not found: {SVS_DIR}")

    candidates = sorted(SVS_DIR.glob(SVS_PATTERN))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one {SVS_PATTERN!r} file in {SVS_DIR}; "
            f"found {len(candidates)}: {[path.name for path in candidates]}"
        )
    return candidates[0]


def svs_date_keys(dataset):
    """Convert SVS date_ymd to YYYYMMDD integer keys."""
    values = np.asarray(
        np.ma.filled(dataset.variables["date_ymd"][:], np.nan),
        dtype=float,
    )
    if values.ndim != 2:
        raise ValueError("SVS date_ymd must be two-dimensional")
    if values.shape[0] == 3:
        years, months, days = values
    elif values.shape[1] == 3:
        years, months, days = values.T
    else:
        raise ValueError(
            f"SVS date_ymd must have one dimension of length 3: "
            f"{values.shape}"
        )

    valid = (
        np.isfinite(years)
        & np.isfinite(months)
        & np.isfinite(days)
        & (years >= 1900)
        & (months >= 1)
        & (months <= 12)
        & (days >= 1)
        & (days <= 31)
    )
    keys = np.full(years.size, -1, dtype=np.int64)
    keys[valid] = (
        years[valid].astype(np.int64) * 10000
        + months[valid].astype(np.int64) * 100
        + days[valid].astype(np.int64)
    )
    return keys


def station_reach_sets(variable):
    """Return all positive SWORD reach IDs associated with each SVS station."""
    if "station" not in variable.dimensions:
        raise ValueError(
            f"SVS reach variable {variable.name} has no station dimension"
        )

    values = np.asarray(np.ma.filled(variable[:], np.nan), dtype=float)
    station_axis = variable.dimensions.index("station")
    values = np.moveaxis(values, station_axis, 0)

    result = []
    for row in values.reshape(values.shape[0], -1):
        valid = row[np.isfinite(row) & (row > 0)]
        result.append({str(int(round(value))) for value in valid})
    return result


def load_svs():
    """Load SVS metadata and Q into memory for repeated reach matching."""
    path = discover_svs_file()
    print(f"Reading SVS: {path.name}")

    with Dataset(path, mode="r") as dataset:
        q_variable = dataset.variables["Q"]
        if "station" not in q_variable.dimensions:
            raise ValueError("SVS Q has no station dimension")

        q = read_numeric_variable(q_variable)
        station_axis = q_variable.dimensions.index("station")
        q = np.moveaxis(q, station_axis, 0)
        if q.ndim != 2:
            raise ValueError(f"SVS Q must be station-by-time: {q.shape}")

        date_keys = svs_date_keys(dataset)
        if q.shape[1] != date_keys.size:
            raise ValueError(
                f"SVS Q time length {q.shape[1]} != date length "
                f"{date_keys.size}"
            )

        station_ids = decode_strings_along_dimension(
            dataset.variables["station_id"],
            "station",
        )
        if station_ids.size != q.shape[0]:
            raise ValueError(
                f"SVS station_id count {station_ids.size} != Q station count "
                f"{q.shape[0]}"
            )

        # Union all compatible SWORD versions.  The current MOI CLI defaults
        # to v17b, while some global SOS products expose v17c reach IDs.
        # Keeping every valid mapping avoids silently losing cross-version
        # validation reaches.
        reach_sets = [set() for _ in range(q.shape[0])]
        reach_variable_names = []
        for name in (
            "reach_id_v17b",
            "reach_id_v17c",
            "reach_id_v17",
            "reach_id",
        ):
            if name in dataset.variables:
                current_sets = station_reach_sets(dataset.variables[name])
                if len(current_sets) != q.shape[0]:
                    raise ValueError(
                        f"SVS {name} station count {len(current_sets)} != Q "
                        f"station count {q.shape[0]}"
                    )
                for station_index, reaches in enumerate(current_sets):
                    reach_sets[station_index].update(reaches)
                reach_variable_names.append(name)

        if not reach_variable_names:
            raise KeyError("No compatible SVS reach-ID variable found")

    reach_to_stations = defaultdict(list)
    for station_index, reaches in enumerate(reach_sets):
        for reach in reaches:
            reach_to_stations[reach].append(station_index)

    print(
        "SVS reach-ID mappings: "
        + ", ".join(reach_variable_names)
    )

    return {
        "path": path,
        "reach_variable_names": reach_variable_names,
        "q": q,
        "date_keys": date_keys,
        "station_ids": station_ids,
        "canonical_station_ids": np.asarray(
            [canonical_station_id(value) for value in station_ids],
            dtype=object,
        ),
        "reach_to_stations": dict(reach_to_stations),
    }


def select_svs_station(svs, reach_id, gageid):
    """Select the validation station using reach ID and SOS gage ID."""
    candidates = svs["reach_to_stations"].get(reach_id, [])
    if not candidates:
        return None, "reach_not_in_svs"

    target = canonical_station_id(gageid)
    if target:
        exact = [
            index
            for index in candidates
            if svs["canonical_station_ids"][index] == target
        ]
        if exact:
            candidates = exact
            status = "exact_gageid"
        else:
            suffix = [
                index
                for index in candidates
                if min(
                    len(target),
                    len(svs["canonical_station_ids"][index]),
                )
                >= 5
                and (
                    target.endswith(svs["canonical_station_ids"][index])
                    or svs["canonical_station_ids"][index].endswith(target)
                )
            ]
            if suffix:
                candidates = suffix
                status = "suffix_gageid"
            else:
                return None, "gageid_not_found_at_reach"
    else:
        status = "blank_gageid_fallback"

    valid_counts = [
        np.count_nonzero(
            np.isfinite(svs["q"][index]) & (svs["q"][index] > 0)
        )
        for index in candidates
    ]
    chosen = int(candidates[int(np.argmax(valid_counts))])
    return chosen, status


# ============================================================
# Read qbar_basinScale and final q from reach-level MOI files
# ============================================================

def output_time_day_keys(variable):
    """Decode root time_str and convert it to YYYYMMDD integer keys."""
    strings = decode_strings_along_dimension(variable, "nt")
    timestamps = pd.to_datetime(
        pd.Series(strings, dtype="string"),
        errors="coerce",
        utc=True,
    )
    return (
        timestamps.dt.year * 10000
        + timestamps.dt.month * 100
        + timestamps.dt.day
    ).fillna(-1).astype(np.int64).to_numpy()


def read_integrator_algorithm(reach_id, algorithm):
    """Read one algorithm's qbar, final q, and time axis."""
    path = MOI_DIR / f"{reach_id}_integrator.nc"
    if not path.is_file():
        return None, "integrator_file_missing"

    with Dataset(path, mode="r") as dataset:
        if algorithm not in dataset.groups:
            return None, "algorithm_group_missing"

        group = dataset.groups[algorithm]
        if "qbar_basinScale" not in group.variables:
            return None, "qbar_basinScale_missing"
        if "q" not in group.variables:
            return None, "final_q_missing"
        if "time_str" not in dataset.variables:
            return None, "time_str_missing"

        qbar = read_scalar(group.variables["qbar_basinScale"])
        q = read_numeric_variable(group.variables["q"]).reshape(-1)
        day_keys = output_time_day_keys(dataset.variables["time_str"])

    if q.size != day_keys.size:
        return None, "q_time_length_mismatch"

    return {
        "path": path,
        "qbar_basinScale": qbar,
        "final_q": q,
        "day_keys": day_keys,
    }, "ok"


def match_one_validation_sample(row, svs, integrator_cache):
    """Calculate the middle-stage nBias inputs for one paired sample."""
    reach_id = row["reach_id_str"]
    algorithm = row["algorithm"]
    cache_key = (reach_id, algorithm)

    if cache_key not in integrator_cache:
        integrator_cache[cache_key] = read_integrator_algorithm(
            reach_id,
            algorithm,
        )
    integrator, integrator_status = integrator_cache[cache_key]
    if integrator is None:
        return {"middle_stage_status": integrator_status}

    station_index, station_status = select_svs_station(
        svs,
        reach_id,
        row["gageid"],
    )
    if station_index is None:
        return {"middle_stage_status": station_status}

    qbar = integrator["qbar_basinScale"]
    if not np.isfinite(qbar) or qbar <= 0:
        return {"middle_stage_status": "invalid_qbar_basinScale"}

    model_by_day = {}
    for day, value in zip(integrator["day_keys"], integrator["final_q"]):
        if day > 0 and np.isfinite(value) and value > 0:
            model_by_day.setdefault(int(day), float(value))

    observed_by_day = {}
    station_q = svs["q"][station_index]
    for day, value in zip(svs["date_keys"], station_q):
        if day > 0 and np.isfinite(value) and value > 0:
            observed_by_day.setdefault(int(day), float(value))

    common_days = sorted(set(model_by_day).intersection(observed_by_day))
    if not common_days:
        return {"middle_stage_status": "no_common_valid_dates"}

    final_values = np.asarray(
        [model_by_day[day] for day in common_days],
        dtype=float,
    )
    observed_values = np.asarray(
        [observed_by_day[day] for day in common_days],
        dtype=float,
    )

    final_mean = float(np.mean(final_values))
    observed_mean = float(np.mean(observed_values))
    if not np.isfinite(observed_mean) or observed_mean <= 0:
        return {"middle_stage_status": "invalid_svs_mean"}

    return {
        "middle_stage_status": "ok",
        "station_match_status": station_status,
        "svs_station_id": svs["station_ids"][station_index],
        "n_common_dates": len(common_days),
        "qbar_basinScale": qbar,
        "qbar_svs": observed_mean,
        "final_q_mean_on_common_dates": final_mean,
        "nbias_integrated_mean": (qbar - observed_mean) / observed_mean,
        "nbias_moi_recomputed": (final_mean - observed_mean) / observed_mean,
        "flp_mean_closure": (final_mean - qbar) / qbar,
    }


def build_three_stage_table(paired_validation, svs):
    """Add the integrated-mean stage and retain strict three-way samples."""
    integrator_cache = {}
    diagnostic_records = []

    for index, row in paired_validation.iterrows():
        if (index + 1) % 100 == 0 or index == 0:
            print(
                f"Matching integrator/SVS sample {index + 1:,}/"
                f"{len(paired_validation):,}"
            )
        result = match_one_validation_sample(row, svs, integrator_cache)
        result["validation_index"] = index
        diagnostic_records.append(result)

    diagnostics = pd.DataFrame(diagnostic_records).set_index(
        "validation_index"
    )
    all_samples = paired_validation.join(diagnostics, how="left")

    all_samples["abs_nbias_integrated_mean"] = all_samples[
        "nbias_integrated_mean"
    ].abs()
    all_samples["abs_flp_mean_closure"] = all_samples[
        "flp_mean_closure"
    ].abs()
    all_samples["nbias_moi_recompute_difference"] = (
        all_samples["nbias_moi_recomputed"] - all_samples["nbias_moi"]
    )

    finite_three_way = np.logical_and.reduce(
        [
            np.isfinite(all_samples["nbias_flpe"]),
            np.isfinite(all_samples["nbias_integrated_mean"]),
            np.isfinite(all_samples["nbias_moi"]),
        ]
    )
    three_way = all_samples.loc[
        (all_samples["middle_stage_status"] == "ok") & finite_three_way
    ].copy()

    if three_way.empty:
        raise ValueError("No strict FLPE/integrated-mean/MOI samples found")

    three_way["final_minus_integrated_abs_nbias"] = (
        three_way["abs_nbias_moi"]
        - three_way["abs_nbias_integrated_mean"]
    )
    three_way["flp_stage_worsened"] = (
        three_way["final_minus_integrated_abs_nbias"] > 0
    )
    return all_samples, three_way


# ============================================================
# Summaries and ECDF plot
# ============================================================

def calculate_bounded_ecdf(values, display_min, display_max):
    """Calculate an ECDF using every finite sample without clipping."""
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return np.array([]), np.array([])

    probabilities = np.arange(1, values.size + 1) / values.size
    x_values = np.concatenate(
        ([min(display_min, float(values[0]))], values, [max(display_max, float(values[-1]))])
    )
    y_values = np.concatenate(([0.0], probabilities, [1.0]))
    return x_values, y_values


def summarize_three_stage(three_way):
    """Return algorithm-level diagnostics for the FLP stage."""
    records = []
    for algorithm in FLPE_ALGORITHMS:
        data = three_way.loc[three_way["algorithm"] == algorithm]
        if data.empty:
            continue
        records.append(
            {
                "algorithm": ALGORITHM_LABELS[algorithm],
                "three_way_samples": len(data),
                "median_signed_nbias_flpe": data["nbias_flpe"].median(),
                "median_signed_nbias_integrated_mean": data[
                    "nbias_integrated_mean"
                ].median(),
                "median_signed_nbias_final_moi": data["nbias_moi"].median(),
                "median_abs_nbias_flpe": data["abs_nbias_flpe"].median(),
                "median_abs_nbias_integrated_mean": data[
                    "abs_nbias_integrated_mean"
                ].median(),
                "median_abs_nbias_final_moi": data["abs_nbias_moi"].median(),
                "median_abs_flp_mean_closure": data[
                    "abs_flp_mean_closure"
                ].median(),
                "flp_stage_worsened_count": int(
                    data["flp_stage_worsened"].sum()
                ),
                "flp_stage_worsened_percent": 100
                * data["flp_stage_worsened"].mean(),
                "median_moi_nbias_recompute_difference": data[
                    "nbias_moi_recompute_difference"
                ].median(),
            }
        )
    return pd.DataFrame(records)


def plot_three_stage_ecdfs(three_way):
    """Create the requested four-algorithm, three-stage ECDF figure."""
    colors = {
        "FLPE": "#1f77b4",
        "INTEGRATED_MEAN": "#2ca02c",
        "MOI": "#d62728",
    }

    metric_settings = [
        {
            "flpe_column": "nbias_flpe",
            "integrated_column": "nbias_integrated_mean",
            "moi_column": "nbias_moi",
            "label": "Signed nBias",
            "xlim": (-2, 2),
            "xticks": [-2, -1, 0, 1, 2],
            "zero_line": True,
        },
        {
            "flpe_column": "abs_nbias_flpe",
            "integrated_column": "abs_nbias_integrated_mean",
            "moi_column": "abs_nbias_moi",
            "label": "|nBias|",
            "xlim": (0, 2),
            "xticks": [0, 0.5, 1.0, 1.5, 2.0],
            "zero_line": False,
        },
    ]

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(
        nrows=len(FLPE_ALGORITHMS),
        ncols=2,
        figsize=(12, 16),
        dpi=150,
        sharey=True,
    )

    for row_index, algorithm in enumerate(FLPE_ALGORITHMS):
        algorithm_data = three_way.loc[
            three_way["algorithm"] == algorithm
        ].copy()

        if algorithm_data.empty:
            for ax in axes[row_index, :]:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.5,
                    f"{ALGORITHM_LABELS[algorithm]}: no three-way samples",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
            continue

        for column_index, settings in enumerate(metric_settings):
            ax = axes[row_index, column_index]
            stage_specs = [
                (
                    "FLPE",
                    settings["flpe_column"],
                    "--",
                    "FLPE",
                ),
                (
                    "INTEGRATED_MEAN",
                    settings["integrated_column"],
                    "-.",
                    "Integrated mean (pre-FLP)",
                ),
                (
                    "MOI",
                    settings["moi_column"],
                    "-",
                    "Final MOI",
                ),
            ]

            for stage_key, column, linestyle, label in stage_specs:
                values = algorithm_data[column].to_numpy(dtype=float)
                x_values, y_values = calculate_bounded_ecdf(
                    values,
                    display_min=settings["xlim"][0],
                    display_max=settings["xlim"][1],
                )
                ax.step(
                    x_values,
                    y_values,
                    where="post",
                    color=colors[stage_key],
                    linewidth=2.1,
                    linestyle=linestyle,
                    label=(
                        f"{label} "
                        f"(n={len(values):,}, median={np.median(values):.3f})"
                    ),
                )

            if settings["zero_line"]:
                ax.axvline(
                    0,
                    color="black",
                    linewidth=1.0,
                    linestyle=":",
                    alpha=0.75,
                )

            ax.set_xlim(settings["xlim"])
            ax.set_xticks(settings["xticks"])
            ax.set_ylim(0, 1.01)
            ax.set_title(
                f"{ALGORITHM_LABELS[algorithm]}: {settings['label']}"
            )
            ax.set_xlabel(settings["label"])
            if column_index == 0:
                ax.set_ylabel("Empirical cumulative probability")
            ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.5)
            ax.legend(loc="lower right", frameon=True)

    fig.suptitle(
        "Global Validation ECDFs by FLPE Algorithm\n"
        "FLPE vs Integrated Mean (pre-FLP) vs Final MOI\n"
        "Strict Three-Way Paired Samples Across Six Continents",
        fontsize=15,
        y=0.997,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.955])
    return fig


def main():
    """Load the global run, calculate the third curve, and save outputs."""
    if not MOI_DIR.is_dir():
        raise NotADirectoryError(f"MOI directory not found: {MOI_DIR}")

    paired_validation = load_global_paired_validation()
    print(
        f"\nPaired FLPE/final-MOI validation rows: "
        f"{len(paired_validation):,}"
    )

    svs = load_svs()
    all_samples, three_way = build_three_stage_table(
        paired_validation,
        svs,
    )

    print("\n===== MIDDLE-STAGE LOAD STATUS =====")
    print(all_samples["middle_stage_status"].value_counts(dropna=False))

    print("\n===== SVS STATION MATCH STATUS (VALID MIDDLE STAGE) =====")
    print(three_way["station_match_status"].value_counts(dropna=False))

    print("\n===== STRICT THREE-WAY COUNTS =====")
    print(
        three_way["algorithm"].value_counts().reindex(FLPE_ALGORITHMS)
    )

    recompute_abs_difference = three_way[
        "nbias_moi_recompute_difference"
    ].abs()
    print("\n===== VALIDATION REPRODUCTION CHECK =====")
    print(
        "|recomputed MOI nBias - SOS MOI nBias|: "
        f"median={recompute_abs_difference.median():.6g}, "
        f"p95={recompute_abs_difference.quantile(0.95):.6g}, "
        f"max={recompute_abs_difference.max():.6g}"
    )
    if recompute_abs_difference.quantile(0.95) > 0.05:
        warnings.warn(
            "The recomputed final-MOI nBias differs materially from the SOS "
            "nBias for at least 5% of samples. Inspect the detailed CSV before "
            "interpreting the middle curve; the SOS validator may use a "
            "different date/station matching rule."
        )

    summary = summarize_three_stage(three_way)
    print("\n===== THREE-STAGE ALGORITHM SUMMARY =====")
    print(summary.round(4).to_string(index=False))

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    figure = plot_three_stage_ecdfs(three_way)

    figure_path = (
        FIGURE_DIR
        / "global-four-algorithms-flpe-integrated-mean-moi-ecdf.png"
    )
    if SAVE_FIGURES:
        figure.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"\nSaved figure: {figure_path}")

    if SAVE_TABLES:
        summary_path = FIGURE_DIR / "global-three-stage-summary.csv"
        detail_path = FIGURE_DIR / "global-three-stage-paired-samples.csv"
        failure_path = FIGURE_DIR / "global-three-stage-load-audit.csv"
        summary.to_csv(summary_path, index=False)
        three_way.to_csv(detail_path, index=False)
        all_samples.to_csv(failure_path, index=False)
        print(f"Saved summary: {summary_path}")
        print(f"Saved paired samples: {detail_path}")
        print(f"Saved load audit: {failure_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)


if __name__ == "__main__":
    main()

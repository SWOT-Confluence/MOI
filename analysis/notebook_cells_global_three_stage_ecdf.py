# %% [markdown]
# Run the original notebook helper cell and original Cell 5 first.
# The following cells require:
# global_algorithm_paired, FLPE_ALGORITHMS, ALGORITHM_LABELS,
# read_numeric_variable, Dataset, np, pd, plt, Path.

# %% Cell 6A: paths and compact decoding helpers
from collections import defaultdict

RUN_ROOT = Path(
    "/nas/cee-water/cjgleason/Yushan/Confluence_Aug/"
    "confluence_global_v17c_gagecorr/"
    "global_v17c_gagecorr_mnt"
)

MOI_DIR = RUN_ROOT / "moi"
SVS_DIR = RUN_ROOT / "input" / "svs"
THREE_STAGE_FIGURE_DIR = Path.cwd()

svs_candidates = sorted(SVS_DIR.glob("*SVS*.nc"))
if len(svs_candidates) != 1:
    raise RuntimeError(
        f"Expected exactly one SVS file in {SVS_DIR}; "
        f"found {len(svs_candidates)}: {[p.name for p in svs_candidates]}"
    )
SVS_PATH = svs_candidates[0]

if not MOI_DIR.is_dir():
    raise NotADirectoryError(f"MOI directory not found: {MOI_DIR}")


def normalize_reach_id(value):
    """Convert a numeric/float/string reach ID to a stable integer string."""
    if value is None or np.ma.is_masked(value):
        return ""
    text = str(value).replace("\x00", "").strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return ""
    try:
        number = float(text)
    except (TypeError, ValueError):
        return text
    return str(int(round(number))) if np.isfinite(number) and number > 0 else ""


def canonical_station_id(value):
    """Compare station IDs while ignoring case, spaces, and punctuation."""
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).replace("\x00", "").strip().lower(),
    )


def decode_strings_by_record(variable, record_dimension):
    """Decode one string per station or per nt from char/VLEN variables."""
    raw = np.ma.filled(variable[:], b"")
    values = np.asarray(raw)
    axis = variable.dimensions.index(record_dimension)
    values = np.moveaxis(values, axis, 0)

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


def ymd_to_day_keys(values):
    """Convert SVS date_ymd to YYYYMMDD integer keys."""
    values = np.asarray(np.ma.filled(values, np.nan), dtype=float)
    if values.shape[0] == 3:
        years, months, days = values
    elif values.shape[1] == 3:
        years, months, days = values.T
    else:
        raise ValueError(f"Unexpected date_ymd shape: {values.shape}")

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


def time_str_to_day_keys(variable):
    """Convert integrator root time_str to YYYYMMDD integer keys."""
    strings = decode_strings_by_record(variable, "nt")
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


print(f"MOI directory: {MOI_DIR}")
print(f"SVS file:      {SVS_PATH}")


# %% Cell 6B: load SVS once and build reach/station lookup
with Dataset(SVS_PATH, mode="r") as svs_nc:
    q_variable = svs_nc.variables["Q"]
    svs_q = read_numeric_variable(q_variable)
    station_axis = q_variable.dimensions.index("station")
    svs_q = np.moveaxis(svs_q, station_axis, 0)

    if svs_q.ndim != 2:
        raise ValueError(f"SVS Q must be station-by-time: {svs_q.shape}")

    svs_day_keys = ymd_to_day_keys(svs_nc.variables["date_ymd"][:])
    if svs_q.shape[1] != svs_day_keys.size:
        raise ValueError(
            f"SVS Q time length {svs_q.shape[1]} != "
            f"date length {svs_day_keys.size}"
        )

    svs_station_ids = decode_strings_by_record(
        svs_nc.variables["station_id"],
        "station",
    )
    svs_canonical_station_ids = np.asarray(
        [canonical_station_id(value) for value in svs_station_ids],
        dtype=object,
    )

    # Union v17b/v17c mappings so either SOS reach-ID version can match.
    station_reach_sets = [set() for _ in range(svs_q.shape[0])]
    used_reach_variables = []

    for variable_name in (
        "reach_id_v17b",
        "reach_id_v17c",
        "reach_id_v17",
        "reach_id",
    ):
        if variable_name not in svs_nc.variables:
            continue

        variable = svs_nc.variables[variable_name]
        values = np.asarray(
            np.ma.filled(variable[:], np.nan),
            dtype=float,
        )
        axis = variable.dimensions.index("station")
        values = np.moveaxis(values, axis, 0)
        values = values.reshape(values.shape[0], -1)

        if values.shape[0] != svs_q.shape[0]:
            raise ValueError(
                f"{variable_name} station count does not match SVS Q"
            )

        for station_index, row in enumerate(values):
            valid_reaches = row[np.isfinite(row) & (row > 0)]
            station_reach_sets[station_index].update(
                str(int(round(value))) for value in valid_reaches
            )

        used_reach_variables.append(variable_name)


svs_reach_to_stations = defaultdict(list)
for station_index, reach_set in enumerate(station_reach_sets):
    for reach_id in reach_set:
        svs_reach_to_stations[reach_id].append(station_index)


def select_validation_station(reach_id, gageid):
    """Match the SOS validation gage to an SVS station at this reach."""
    candidates = svs_reach_to_stations.get(reach_id, [])
    if not candidates:
        return None, "reach_not_in_svs"

    target = canonical_station_id(gageid)
    if target:
        exact = [
            index
            for index in candidates
            if svs_canonical_station_ids[index] == target
        ]
        if exact:
            candidates = exact
            status = "exact_gageid"
        else:
            suffix = [
                index
                for index in candidates
                if min(len(target), len(svs_canonical_station_ids[index])) >= 5
                and (
                    target.endswith(svs_canonical_station_ids[index])
                    or svs_canonical_station_ids[index].endswith(target)
                )
            ]
            if not suffix:
                return None, "gageid_not_found_at_reach"
            candidates = suffix
            status = "suffix_gageid"
    else:
        status = "blank_gageid_fallback"

    # If multiple rows represent the same station/reach, keep the longest record.
    valid_counts = [
        np.count_nonzero(
            np.isfinite(svs_q[index]) & (svs_q[index] > 0)
        )
        for index in candidates
    ]
    return int(candidates[int(np.argmax(valid_counts))]), status


print(f"Loaded SVS stations: {svs_q.shape[0]:,}")
print(f"SVS reach variables: {used_reach_variables}")
print(f"Reach IDs in SVS lookup: {len(svs_reach_to_stations):,}")


# %% Cell 6C: calculate integrated-mean nBias and build strict three-way table
validation_for_middle = global_algorithm_paired.copy().reset_index(drop=True)
validation_for_middle["reach_id_str"] = validation_for_middle[
    "reach_id"
].map(normalize_reach_id)

if (validation_for_middle["reach_id_str"] == "").any():
    raise ValueError("Some paired FLPE/MOI validation rows have no reach ID")


def read_integrator_record(reach_id, algorithm):
    """Read qbar_basinScale, final q, and time_str for one algorithm/reach."""
    path = MOI_DIR / f"{reach_id}_integrator.nc"
    if not path.is_file():
        return None, "integrator_file_missing"

    with Dataset(path, mode="r") as nc:
        if algorithm not in nc.groups:
            return None, "algorithm_group_missing"

        group = nc.groups[algorithm]
        for variable_name in ("qbar_basinScale", "q"):
            if variable_name not in group.variables:
                return None, f"{variable_name}_missing"
        if "time_str" not in nc.variables:
            return None, "time_str_missing"

        qbar_values = read_numeric_variable(
            group.variables["qbar_basinScale"]
        ).reshape(-1)
        if qbar_values.size != 1:
            return None, "qbar_not_scalar"

        record = {
            "qbar": float(qbar_values[0]),
            "q": read_numeric_variable(group.variables["q"]).reshape(-1),
            "day_keys": time_str_to_day_keys(nc.variables["time_str"]),
        }

    if record["q"].size != record["day_keys"].size:
        return None, "q_time_length_mismatch"
    return record, "ok"


integrator_cache = {}
middle_records = []

for index, row in validation_for_middle.iterrows():
    if index == 0 or (index + 1) % 100 == 0:
        print(
            f"Matching sample {index + 1:,}/"
            f"{len(validation_for_middle):,}"
        )

    reach_id = row["reach_id_str"]
    algorithm = row["algorithm"]
    cache_key = (reach_id, algorithm)

    if cache_key not in integrator_cache:
        integrator_cache[cache_key] = read_integrator_record(
            reach_id,
            algorithm,
        )

    record, status = integrator_cache[cache_key]
    result = {
        "validation_index": index,
        "middle_stage_status": status,
    }

    if record is not None:
        station_index, station_status = select_validation_station(
            reach_id,
            row["gageid"],
        )

        if station_index is None:
            result["middle_stage_status"] = station_status
        elif not np.isfinite(record["qbar"]) or record["qbar"] <= 0:
            result["middle_stage_status"] = "invalid_qbar_basinScale"
        else:
            model_by_day = {}
            for day, value in zip(record["day_keys"], record["q"]):
                if day > 0 and np.isfinite(value) and value > 0:
                    model_by_day.setdefault(int(day), float(value))

            observed_by_day = {}
            for day, value in zip(
                svs_day_keys,
                svs_q[station_index],
            ):
                if day > 0 and np.isfinite(value) and value > 0:
                    observed_by_day.setdefault(int(day), float(value))

            common_days = sorted(
                set(model_by_day).intersection(observed_by_day)
            )

            if not common_days:
                result["middle_stage_status"] = "no_common_valid_dates"
            else:
                final_mean = np.mean(
                    [model_by_day[day] for day in common_days]
                )
                svs_mean = np.mean(
                    [observed_by_day[day] for day in common_days]
                )
                qbar = record["qbar"]

                result.update(
                    {
                        "middle_stage_status": "ok",
                        "station_match_status": station_status,
                        "svs_station_id": svs_station_ids[station_index],
                        "n_common_dates": len(common_days),
                        "qbar_basinScale": qbar,
                        "qbar_svs": svs_mean,
                        "final_q_mean_on_common_dates": final_mean,
                        "nbias_integrated_mean": (
                            qbar - svs_mean
                        ) / svs_mean,
                        "nbias_moi_recomputed": (
                            final_mean - svs_mean
                        ) / svs_mean,
                        "flp_mean_closure": (
                            final_mean - qbar
                        ) / qbar,
                    }
                )

    middle_records.append(result)


middle_diagnostics = pd.DataFrame(middle_records).set_index(
    "validation_index"
)
three_stage_all = validation_for_middle.join(middle_diagnostics)

if "nbias_integrated_mean" not in three_stage_all:
    raise ValueError("No integrated-mean nBias values were calculated")

three_stage_all["abs_nbias_integrated_mean"] = three_stage_all[
    "nbias_integrated_mean"
].abs()
three_stage_all["abs_flp_mean_closure"] = three_stage_all[
    "flp_mean_closure"
].abs()
three_stage_all["nbias_moi_recompute_difference"] = (
    three_stage_all["nbias_moi_recomputed"]
    - three_stage_all["nbias_moi"]
)

three_way_valid = (
    (three_stage_all["middle_stage_status"] == "ok")
    & np.isfinite(three_stage_all["nbias_flpe"])
    & np.isfinite(three_stage_all["nbias_integrated_mean"])
    & np.isfinite(three_stage_all["nbias_moi"])
)

global_three_stage_paired = three_stage_all.loc[
    three_way_valid
].copy()

if global_three_stage_paired.empty:
    raise ValueError("No strict FLPE/integrated-mean/MOI samples found")

global_three_stage_paired["final_minus_integrated_abs_nbias"] = (
    global_three_stage_paired["abs_nbias_moi"]
    - global_three_stage_paired["abs_nbias_integrated_mean"]
)
global_three_stage_paired["flp_stage_worsened"] = (
    global_three_stage_paired["final_minus_integrated_abs_nbias"] > 0
)


print("\n===== MIDDLE-STAGE LOAD STATUS =====")
print(three_stage_all["middle_stage_status"].value_counts(dropna=False))

print("\n===== STRICT THREE-WAY COUNTS =====")
print(
    global_three_stage_paired["algorithm"]
    .value_counts()
    .reindex(FLPE_ALGORITHMS)
)

reproduction_error = global_three_stage_paired[
    "nbias_moi_recompute_difference"
].abs()

print("\n===== VALIDATION REPRODUCTION CHECK =====")
print(
    "|recomputed MOI nBias - SOS MOI nBias|: "
    f"median={reproduction_error.median():.6g}, "
    f"p95={reproduction_error.quantile(0.95):.6g}, "
    f"max={reproduction_error.max():.6g}"
)

if reproduction_error.quantile(0.95) > 0.05:
    warnings.warn(
        "The recomputed MOI nBias differs from the SOS nBias. "
        "Inspect station/date matching before interpreting the green curve."
    )


summary_rows = []
for algorithm in FLPE_ALGORITHMS:
    data = global_three_stage_paired.loc[
        global_three_stage_paired["algorithm"] == algorithm
    ]
    if data.empty:
        continue
    summary_rows.append(
        {
            "algorithm": ALGORITHM_LABELS[algorithm],
            "samples": len(data),
            "median_|nBias|_FLPE": data["abs_nbias_flpe"].median(),
            "median_|nBias|_integrated_mean": data[
                "abs_nbias_integrated_mean"
            ].median(),
            "median_|nBias|_final_MOI": data["abs_nbias_moi"].median(),
            "median_|FLP_closure|": data[
                "abs_flp_mean_closure"
            ].median(),
            "FLP_stage_worsened_percent": 100
            * data["flp_stage_worsened"].mean(),
        }
    )

global_three_stage_summary = pd.DataFrame(summary_rows)
display(global_three_stage_summary.round(4))


# %% Cell 6D: plot FLPE vs integrated mean vs final MOI ECDFs
def calculate_bounded_ecdf(values, display_min, display_max):
    values = np.asarray(values, dtype=float)
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return np.array([]), np.array([])

    probabilities = np.arange(1, values.size + 1) / values.size
    x_values = np.concatenate(
        (
            [min(display_min, float(values[0]))],
            values,
            [max(display_max, float(values[-1]))],
        )
    )
    y_values = np.concatenate(([0.0], probabilities, [1.0]))
    return x_values, y_values


colors = {
    "FLPE": "#1f77b4",
    "INTEGRATED": "#2ca02c",
    "MOI": "#d62728",
}

metric_settings = [
    {
        "flpe": "nbias_flpe",
        "integrated": "nbias_integrated_mean",
        "moi": "nbias_moi",
        "label": "Signed nBias",
        "xlim": (-2, 2),
        "xticks": [-2, -1, 0, 1, 2],
        "zero_line": True,
    },
    {
        "flpe": "abs_nbias_flpe",
        "integrated": "abs_nbias_integrated_mean",
        "moi": "abs_nbias_moi",
        "label": "|nBias|",
        "xlim": (0, 2),
        "xticks": [0, 0.5, 1.0, 1.5, 2.0],
        "zero_line": False,
    },
]

fig, axes = plt.subplots(
    nrows=len(FLPE_ALGORITHMS),
    ncols=2,
    figsize=(12, 16),
    dpi=150,
    sharey=True,
)

for row_index, algorithm in enumerate(FLPE_ALGORITHMS):
    algorithm_data = global_three_stage_paired.loc[
        global_three_stage_paired["algorithm"] == algorithm
    ]

    for column_index, settings in enumerate(metric_settings):
        ax = axes[row_index, column_index]

        stage_specs = [
            ("FLPE", settings["flpe"], "--", "FLPE"),
            (
                "INTEGRATED",
                settings["integrated"],
                "-.",
                "Integrated mean (pre-FLP)",
            ),
            ("MOI", settings["moi"], "-", "Final MOI"),
        ]

        for color_key, column, linestyle, label in stage_specs:
            values = algorithm_data[column].to_numpy(dtype=float)
            x_values, y_values = calculate_bounded_ecdf(
                values,
                settings["xlim"][0],
                settings["xlim"][1],
            )
            ax.step(
                x_values,
                y_values,
                where="post",
                color=colors[color_key],
                linewidth=2.1,
                linestyle=linestyle,
                label=(
                    f"{label} "
                    f"(n={len(values):,}, "
                    f"median={np.median(values):.3f})"
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
        ax.legend(loc="lower right", frameon=True, fontsize=9)


fig.suptitle(
    "Global Validation ECDFs by FLPE Algorithm\n"
    "FLPE vs Integrated Mean (pre-FLP) vs Final MOI\n"
    "Strict Three-Way Paired Samples Across Six Continents",
    fontsize=15,
    y=0.997,
)
plt.tight_layout(rect=[0, 0, 1, 0.955])

figure_path = (
    THREE_STAGE_FIGURE_DIR
    / "global-four-algorithms-flpe-integrated-mean-moi-ecdf.png"
)
fig.savefig(figure_path, dpi=300, bbox_inches="tight")

summary_path = THREE_STAGE_FIGURE_DIR / "global-three-stage-summary.csv"
detail_path = THREE_STAGE_FIGURE_DIR / "global-three-stage-paired.csv"
audit_path = THREE_STAGE_FIGURE_DIR / "global-three-stage-audit.csv"

global_three_stage_summary.to_csv(summary_path, index=False)
global_three_stage_paired.to_csv(detail_path, index=False)
three_stage_all.to_csv(audit_path, index=False)

plt.show()

print(f"Saved figure: {figure_path}")
print(f"Saved summary: {summary_path}")
print(f"Saved paired table: {detail_path}")
print(f"Saved audit table: {audit_path}")

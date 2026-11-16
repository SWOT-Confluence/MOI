# MOI

MOI serves as the integrator module to the Confluence workflow. It extracts reach-level FLPE algorithm results and SoS data to integrate results on the basin-level. This module writes output to a specified output directory.

Observed reach-level integrator NetCDF files include a root-level VLEN string
variable `time_str(nt)`, copied from the corresponding SWOT
`reach/time_str`. The complete pre-QC time axis is retained so each timestamp
stays aligned with the final discharge arrays. The primary `q` preserves the
full-record FLPE hydrograph after basin-scale rescaling when that FLPE record
is available. Geometry-rejected observations remain fill for flow-law fallback
reaches and in geometry-dependent products such as `q_flowlaw`. For older SWOT
files without `time_str`, MOI derives UTC ISO timestamps from `reach/time`.

## Gage-constrained SFOI

For `constrained` runs, MOI can read the Surface-water Validation System (SVS)
NetCDF and append study-period mean discharge and q33 as independent soft
observations. The FLPE value at a gaged reach is retained; it is not replaced
by the gage.

The sparse state remains `x = [Q_all, R_regions]`. Each gage contributes one
selector row `H_g x = Q_gage` with finite relative uncertainty. Gage rows use a
fixed observation-scale weight in the multiplicative outer loop, participate in
the global chi-square diagnostic, and are protected from robust downweighting in
the inner loop. The study-period gage mean constrains the `Mean` run and the
33rd-percentile gage discharge over the same period constrains the `q33` run.

By default, each basin uses a single inclusive UTC calendar-date interval from
its first to its last downloaded SWOT observation, **before geometry QC**.
Files in the basin that did not survive the reach selection also contribute
their raw timestamps; geometry and setfinder cannot silently shorten the window.
Every eligible real calibration gage uses all valid daily observations within
that continuous interval, including days without SWOT overpasses. A gage does
not need usable SWOT geometry or an overpass on its own reach to participate.
Data outside the interval are excluded; the window is never shortened to an
individual station's available record. This estimates the mean over the finite
SWOT study period, not a multi-decade climatological mean.

The default minimum gage coverage is **20% of all calendar days in the study
period**, with at least one valid observation. Coverage counts unique valid
dates; duplicate values on a date are averaged before computing Mean/q33.
Finite zero discharge is retained; negative and missing values are excluded.
The existing solver observation-scale floor keeps zero-valued constraints at
finite uncertainty. Coverage alone does not establish seasonal representativeness.
Gages below the threshold are omitted with an explicit diagnostic, without
falling back to observations outside the period. Calibration/validation and
include/exclude filters still apply.
When multiple SVS stations map to one reach, the station with the most valid
days inside the chosen study period is selected.

Use `--gage-period-start YYYY-MM-DD` and `--gage-period-end YYYY-MM-DD` to
replace either automatically inferred endpoint. Supplying both dates gives
all basins a common study period. Set `--gage-min-period-coverage 0.20` to
change the coverage threshold. A missing automatic endpoint is an error rather
than an implicit use of a station's historical record.

```bash
python run_MOI.py -i 0 -j basins.json -b constrained \
  --svs-file /mnt/data/input/svs/SVS_v1_1_2.nc \
  --svs-reach-id-col reach_id_v17c \
  --gage-period-start 2023-01-01 --gage-period-end 2026-10-16 \
  --gage-min-period-coverage 0.20
```

For comparison experiments, `--gage-statistic-mode swot_days` retains the old
per-reach geometry-filtered SWOT-day target. Legacy library callers can still
set `Gage_Match_SWOT_Days` explicitly when `Gage_Statistic_Mode` is absent.
CORRIDORS pseudo-gages remain matched to SWOT days: they are sparse synthetic
observations, not daily station records, and are not subject to the real-gage
period-coverage gate.

Each run writes `output/logs/gage_constraints_<basin_id>.json` before solving,
including the period, coverage threshold, included/rejected status, valid-day
count, coverage and used Mean/q33 for each available gage. The `gage` group in
each integrator NetCDF carries the same study-period and per-reach diagnostics.
These targets do not change how the FLPE hydrograph is rescaled; Qbar and
same-day q(t) accuracy must still be assessed separately.

When exactly one `*SVS*.nc` file exists in `/mnt/data/input/svs`, constrained
runs discover it automatically. A calibration/validation CSV is required and
defaults to `CalValSeparation_basin_stratified_v2.csv` in the cloned MOI module
directory (for example,
`modules/moi/CalValSeparation_basin_stratified_v2.csv`). It must contain
`reach_id_v17b` and `group` columns; only rows whose group is `calibration` are
eligible for gage constraints. Validation, excluded, and unclassified reaches
are retained for independent evaluation and are not refilled from SoS gage
data.

The SVS input can also be selected explicitly; the module CSV remains the
default unless `--gage-calval-csv` is provided as an override:

```bash
python run_MOI.py \
  -i 0 \
  -j basins.json \
  -b constrained \
  --svs-file /mnt/data/input/svs/SVS_v1_0_1.nc \
  --svs-reach-id-col reach_id_v17b
```

Use `--gage-include-json` to select training/constraint gages and
`--gage-exclude-json` to apply an additional filter within the calibration
group. Both files may be a JSON list of reach IDs or a dictionary whose values
are reach IDs.

## Bias augmentation and correlated FLPE errors

The complete equations, convergence definitions, parameter defaults, and code
map are documented in
[`docs/SFOI_MATHEMATICAL_MODEL.md`](docs/SFOI_MATHEMATICAL_MODEL.md).

Constrained Mean-flow runs estimate one systematic multiplicative FLPE bias for
each algorithm:

```text
Q_FLPE_i = (1 + bias) * Q_physical_i + error_i
```

The augmented Jacobian applies this bias only to reach rows backed by genuine
FLPE data. Missing/invalid FLPE values use a separately labeled SoS Prior, then
a runoff-derived Fill if necessary; neither receives bias or regional
correlation. Runoff, mass-balance, and gage rows also remain unbiased. A
positive estimate means the FLPE estimate is systematically high; for example,
`bias = 0.20` corresponds to a direct debiasing factor of `1 / 1.20`.

Random FLPE errors can be correlated within the runoff regions already built by
MOI. The implementation uses one penalized Gaussian latent effect per region:

```text
error_i = sigma_i * (
    sqrt(rho) * region_effect[region(i)]
    + sqrt(1-rho) * independent_error_i
)
```

Marginalizing the latent effects gives equicorrelation `rho` among FLPE errors
in the same region and zero correlation between regions. This formulation
preserves the sparse basin solver and does not construct a dense reach-by-reach
covariance matrix.

The default settings are deliberately conservative:

- augmentation is enabled for `Mean` only;
- bias augmentation and correlation are enabled only when the basin has at
  least one usable calibration gage; an ungaged basin uses ordinary sparse MOI;
- bias prior standard deviation is `0.50`;
- within-region correlation is `0.20`;
- gage and mass rows are protected from robust downweighting;
- robust downweighting occurs only above the upper chi-square bound;
- when consecutive augmented Gauss-Newton steps point in strongly opposite
  directions, the new step is progressively under-relaxed (default factor
  `0.5`) to collapse period-two oscillations toward their midpoint;
- relaxation recovers by a factor of `1.25` after three stable directions;
- convergence requires both raw and applied physical RMS/p95, bias, regional
  effect, and robust-factor deltas to satisfy their component-specific
  tolerances;
- an augmented solve that reaches `maxiter` without satisfying all gates is
  rejected before NetCDF export;
- algorithms with no genuine FLPE data reuse an identical prior-only ordinary
  solve instead of repeating it under different algorithm labels.

Runtime overrides are available:

```bash
python run_MOI.py \
  -i 0 \
  -j basins.json \
  -b constrained \
  --svs-file /mnt/data/input/svs/SVS_v1_0_1.nc \
  --correlation-rho 0.20 \
  --bias-prior-std 0.50
```

Use `--disable-correlation` or `--disable-bias-augmentation` for ablation runs.
The estimated bias, approximate bias standard deviation, correlation
coefficient, latent region effects, and solver status are stored under the
`moi_bias_correlation` group in every reach-level NetCDF output.

## Rolled-back bifurcation behavior

MOI reads the corrected SWORD v17c `facc` values and retains `facc_quality` as
correction provenance. The mathematical core intentionally matches the
pre-bifurcation `c9b28ac` baseline: downstream width, with equal weights as a
fallback, partitions distributary flow. Corrected `facc` is not interpreted as
a dynamic branch-flow fraction.

Lateral drainage area is always computed with the baseline nonnegative formula
`max(0, facc_current - weighted_upstream_facc)`. Fractional coefficients no
longer force a bifurcation child's lateral area to zero. The `facc_quality` fill
value (`-9999`) means that the reach retained its v17b value; it is preserved as
metadata and does not affect the rolled-back equations.

This baseline still uses aggregate junction construction. Complex or incomplete
bifurcations therefore remain outside the validated scope and should be excluded
upstream until an edge-based formulation is implemented and tested separately.

## installation

## setup

## execution

***Example Run***

```bash
%run /Users/mtd/GitHub/SWOT-confluence/moi/run_MOI.py basin.json -v 'unconstrained' 0 
```

So the command line arguments are the basin file, the verbose flag, and the branch name, where branch name can be either constrained or unconstrained. The final argument is the basin number, to be provided only for offline runs.

## deployment

There is a script to deploy the Docker container image and Terraform AWS infrastructure found in the `deploy` directory.

Script to deploy Terraform and Docker image AWS infrastructure

REQUIRES:

- jq (<https://jqlang.github.io/jq/>)
- docker (<https://docs.docker.com/desktop/>) > version Docker 1.5
- AWS CLI (<https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>)
- Terraform (<https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli>)

Command line arguments:

[1] registry: Registry URI
[2] repository: Name of repository to create
[3] prefix: Prefix to use for AWS resources associated with environment deploying to
[4] s3_state_bucket: Name of the S3 bucket to store Terraform state in (no need for s3:// prefix)
[5] profile: Name of profile used to authenticate AWS CLI commands

Example usage: ``./deploy.sh "account-id.dkr.ecr.region.amazonaws.com" "container-image-name" "prefix-for-environment" "s3-state-bucket-name" "confluence-named-profile"`

Note: Run the script from the deploy directory.

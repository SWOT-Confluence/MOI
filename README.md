# MOI

MOI serves as the integrator module to the Confluence workflow. It extracts reach-level FLPE algorithm results and SoS data to integrate results on the basin-level. This module writes output to a specified output directory.

## Gage-constrained SFOI

For `constrained` runs, MOI can read the Surface-water Validation System (SVS)
NetCDF and append time-matched mean discharge and q33 as independent soft
observations. The FLPE value at a gaged reach is retained; it is not replaced
by the gage.

The sparse state remains `x = [Q_all, R_regions]`. Each gage contributes one
selector row `H_g x = Q_gage` with finite relative uncertainty. Gage rows use a
fixed observation-scale weight in the multiplicative outer loop, participate in
the global chi-square diagnostic, and are protected from robust downweighting in
the inner loop. The matched gage mean constrains the `Mean` run and the matched
33rd-percentile gage discharge constrains the `q33` run.

When exactly one `*SVS*.nc` file exists in `/mnt/data/input/svs`, constrained
runs discover it automatically. A calibration/validation CSV is required and
defaults to `CalValSeparation.csv` in the cloned MOI module directory (for
example, `modules/moi/CalValSeparation.csv`). It must contain `reach_id_v17b`
and `group` columns; only rows whose group is `calibration` are eligible for
gage constraints. Validation and unclassified reaches are retained for
independent evaluation and are not refilled from SoS gage data.

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

Constrained Mean-flow runs estimate one systematic multiplicative FLPE bias for
each algorithm:

```text
Q_FLPE_i = (1 + bias) * Q_physical_i + error_i
```

Only the FLPE discharge rows receive this bias. Runoff priors, mass-balance
rows, and gages remain unbiased. A positive estimate means that the FLPE prior
is systematically high; for example, `bias = 0.20` corresponds to a direct
debiasing factor of `1 / 1.20`.

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
- bias prior standard deviation is `0.50`;
- within-region correlation is `0.20`;
- gage and mass rows are protected from robust downweighting;
- robust downweighting occurs only above the upper chi-square bound.

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

## SWORD v17c bifurcation handling

MOI reads the corrected SWORD v17c `facc` values and retains `facc_quality` as
correction provenance. At bifurcations, discharge is partitioned using the
normalized corrected-`facc` shares of the downstream reaches. If those shares
are unavailable, MOI falls back to downstream width and then to an equal split.
The fractions always sum to one.

Immediate bifurcation children receive zero lateral drainage-area increment.
This keeps the MOI mass equations consistent with the v17c area partition and
prevents cloned pre-v17c parent area from being reintroduced as lateral runoff.
Normal one-to-one links and confluences continue to use the nonnegative
difference between downstream and weighted-upstream `facc` as lateral area.
The `facc_quality` fill value (`-9999`) means that the reach retained its v17b
value; it does not cause that reach to be filtered out.

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

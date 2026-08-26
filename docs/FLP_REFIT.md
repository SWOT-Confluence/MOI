# Flow-law parameter refit: what changed and how to run it

Companion to `SFOI_MATHEMATICAL_MODEL.md`. Nothing here touches the SFOI
integrator solve; this is all downstream of it.

## The problem

`compute_FLPs()` was six near-identical copies of one loop, each handing a
parameter vector to a single `scipy.optimize.minimize` call started from the
FLPE prior. The copies had drifted: `busboi` and `momma` checked `res.success`
and wrapped the solve in `try/except`; `hivdi`, `metroman`, `sad` and
`sic4dvar` used `res.x` unconditionally.

That matters more than it looks. A bounded local solve started far from the
answer converges to a wrong optimum **and reports `success == True` while doing
it** — `tests/test_flp_fit.py::test_grid_beats_a_badly_started_local_solve`
pins that behaviour down. On four of six algorithms, that wrong optimum was
exported as though it were a converged fit, with nothing in the output file able
to tell a consumer otherwise.

## Variable projection

For every algorithm except MOMMA the roughness parameter is a pure *scale*:
the flow law is linear in it. Write `q_t = s · C_t(θ)`. Minimising over `s` for
fixed `θ` has a closed form:

| space | solution |
|---|---|
| linear | `s* = Σ(w·C·y) / Σ(w·C²)` |
| log | `log s* = Σ(w·(log y − log C)) / Σ(w)` |

`n` enters as `q = C/n`, so `s = 1/n`. HiVDI's `alpha` enters directly, so
`s = alpha`. What remains is one dimension (`A0`) for BUSBOI/SAD/SIC4DVar and
two (`A0`, exponent) for MetroMan/HiVDI — small enough for a bounded grid to
cover exhaustively, followed by a local polish from the grid optimum.

MOMMA has no scale parameter: `n_b = 0.11·Save^0.18` is pinned by slope. It gets
a plain 2-D grid over `(B, H−B)`. That asymmetry is itself informative — it is
part of why MOMMA's residual behaves unlike the dA-driven laws.

The result is **deterministic and restart-free**. Same inputs, same parameters,
no dependence on where the optimiser happened to start. That is the property a
200k-reach global run needs.

The `A0` grid is bounded by geometry rather than by a numeric guess: `A0` is a
cross-sectional area, so the grid spans roughly 0.05 m to 100 m of mean depth
(`width × depth`), log-spaced. The FLPE prior is appended as an extra candidate,
so the grid can never do worse than the old starting point.

## Diagnostics

Every reach × algorithm now carries, in the output netCDF group:

| variable | meaning |
|---|---|
| `flp_fit_nrmse` | linear-space nRMSE against the integrator hydrograph (unchanged name and meaning) |
| `flp_fit_nrmse_log` | the same in log space |
| `flp_fit_status_code` | 0 good, 1 unidentifiable, 2 fallback_prior, 3 no_series, 4 failed |
| `flp_param_at_bound` | bitmask; bit *i* set means parameter *i* is pinned to a bound |
| `flp_n_valid_obs` | usable timesteps the fit actually saw |
| `dA_valid_frac` | fraction of retained timesteps with finite `d_x_area` |
| `rescale_factor` | `qbar_integrator / qbar_FLPE` |
| `rescale_exponent` | `b` in the power-law rescale; 1.0 for the mean rescale |

and, as group attributes: `flp_fit_status`, `flp_fit_method`, `rescale_method`,
`hydrograph_source`, `qbar_source`, `q33_source`.

The last two close the provenance gap: they were already tracked in
`Integrate.py` and already drove prior-vs-FLPE uncertainty inside the solve, but
were never written out, so a consumer could not tell a real FLPE estimate from a
prior reconstruction wearing the same variable names.

**These gate nothing.** Thresholds chosen before the global distribution is
known are guesses. Run once, record the distribution, set thresholds from it.

## Flags

| flag | default | what it does |
|---|---|---|
| `--flp-optimizer {varpro,legacy}` | `varpro` | analytic scale + bounded grid, or the old single-start L-BFGS-B |
| `--flp-fit {series,series_log,moments}` | `series` | objective; `series_log` fits in log space |
| `--flp-loss {l2,soft_l1,huber}` | `l2` | robust loss down-weights bad `dA` timesteps without discarding the reach |
| `--flp-weighting {none,obs_quality}` | `none` | per-timestep weights from `reach_q` / `xovr_cal_q` |
| `--no-flp-refine` | off | grid optimum only, skip the local polish |
| `--rescale-transform {mean,powerlaw}` | `mean` | `powerlaw` solves `q' = a·q^b` matching both `qbar` and `q33` |
| `--sfoi-uncertainty-model {step,continuous}` | `step` | FLPE uncertainty vs drainage area |
| `--flpe-outlier-nstdev FLOAT` | `10.0` | how far an FLPE estimate may sit from SoS climatology before it is replaced |
| `--require-plain-convergence` | off | refuse output from a plain solver that exhausted its iterations |

Defaults are chosen so that only the optimiser changes. `--flp-fit`,
`--rescale-transform` and `--sfoi-uncertainty-model` all default to existing
behaviour, so the canary matrix's zero-drift gate on `qbar_basinScale` still
holds and each new arm can be measured on its own.

`legacy` differs from the pre-change code in one deliberate way: a solve that
fails or raises now falls back to the FLPE prior and is labelled, instead of
exporting whatever the optimiser last held. That was a bug, not a behaviour
worth reproducing faithfully.

## Robustness

- Per-reach `try/except`: one bad reach can no longer take down a basin.
- `_sword_index()` replaces `np.argwhere(...)[0, 0]`, which raised `IndexError`
  on a reach missing from SWORD and lost every reach in the basin — every
  algorithm, both flow levels — over one bad link.
- Topology pre-check after `CreateJunctionList()`: counts bifurcating, dangling,
  isolated and SWORD-missing reaches.
- Plain-solver non-convergence is now recorded (the augmented solver already
  refused it). Gate on it with `--require-plain-convergence` once you know how
  many basins it costs.
- `Integrate.run_report()` / `print_run_report()` collect all of the above into
  one summary per run.

## Open question this does not answer

Whether the residual `flp_fit_nrmse` of 0.17–0.32 is the optimiser or the flow
law family. `analysis/flow_law_family_bound.py` answers it: for every reach it
refits with a search far denser than production could afford and reports the
best achievable nRMSE — the ceiling on all future optimiser work.

```
python analysis/flow_law_family_bound.py \
    --output-dir /path/to/moi/output \
    --swot-dir   /path/to/input/swot \
    --csv        family_bound.csv
```

If the floor comes back near 0.15, the residual is mostly irreducible and the
remaining effort belongs in `d_x_area` quality and the rescale transform, not in
the optimiser. Run this before spending more on either.

## Tests

- `tests/test_flp_fit.py` — closed-form scale vs numerical optimum, determinism,
  parameter recovery for all five separable laws, vectorised MOMMA vs the looped
  one, identifiability refusals, power-law rescale.
- `tests/test_compute_flps.py` — end-to-end `compute_FLPs()` on a synthetic
  basin: all six algorithms, both optimisers, both objective spaces, both
  rescale transforms, and that a broken reach does not take down its algorithm.
- `tests/test_flp_diagnostics_output.py` — the diagnostics survive the trip to
  disk.

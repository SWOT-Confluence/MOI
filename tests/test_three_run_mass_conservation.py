from pathlib import Path

from netCDF4 import Dataset
import numpy as np
import pandas as pd

from analysis.three_run_mass_conservation import (
    RunSpec,
    evaluate_junction_mass,
    load_sword_centerlines,
    summarize_junction_mass,
)
from moi.Integrate import Integrate
from moi.Output import Output


def _reach_table():
    return pd.DataFrame(
        {
            "basin_id": ["7429"] * 4,
            "run": ["run1"] * 4,
            "run_label": ["unconstrained MOI"] * 4,
            "branch": ["unconstrained"] * 4,
            "reach_id": ["1", "2", "3", "4"],
            "qbar_reachScale": [40.0, 60.0, 100.0, np.nan],
            "qbar_basinScale": [40.0, 60.0, 110.0, np.nan],
        }
    )


def test_junction_mass_counts_only_fully_evaluable_junctions():
    junctions = [
        {
            "originating_reach_id": 3,
            "upflows": [1, 2],
            "downflows": [3],
        },
        {
            "originating_reach_id": 4,
            "upflows": [3],
            "downflows": [4],
        },
    ]
    diagnostics = evaluate_junction_mass(
        _reach_table(),
        junctions,
        relative_tolerance=0.0,
        absolute_tolerance_cms=5.0,
    )
    summary = summarize_junction_mass(diagnostics).set_index("scale")

    assert summary.loc["qbar_reachScale", "n_junctions_total"] == 2
    assert summary.loc["qbar_reachScale", "n_junctions_evaluable"] == 1
    assert summary.loc["qbar_reachScale", "n_junctions_not_conserving"] == 0
    assert summary.loc["qbar_basinScale", "n_junctions_not_conserving"] == 1
    assert summary.loc["qbar_basinScale", "fraction_junctions_not_conserving"] == 1.0


def test_load_sword_netcdf_centerlines_filters_reaches(tmp_path):
    sword_path = tmp_path / "sword.nc"
    with Dataset(sword_path, "w") as dataset:
        nodes = dataset.createGroup("nodes")
        nodes.createDimension("num_nodes", 5)
        nodes.createVariable("reach_id", "i8", ("num_nodes",))[:] = [10, 10, 20, 20, 20]
        nodes.createVariable("x", "f8", ("num_nodes",))[:] = [-72, -71, -70, -69, -68]
        nodes.createVariable("y", "f8", ("num_nodes",))[:] = [40, 41, 42, 43, 44]

    centerlines = load_sword_centerlines(sword_path, reach_ids=["20"])

    assert list(centerlines) == ["20"]
    longitude, latitude = centerlines["20"][0]
    np.testing.assert_allclose(longitude, [-70, -69, -68])
    np.testing.assert_allclose(latitude, [42, 43, 44])


def test_sword_output_is_disabled_by_default(tmp_path, capsys):
    writer = Output(
        basin_dict={"sword": "missing.nc"},
        out_dir=Path(tmp_path),
        integ_dict={},
        alg_dict={},
        obs_dict={},
        sword_dir=Path(tmp_path),
        params_dict={},
    )

    assert writer.write_sword_output("constrained") is None
    assert "Skipping SWORD output" in capsys.readouterr().out
    assert list(tmp_path.iterdir()) == []


def test_run_spec_marks_only_constrained_runs_as_gaged():
    assert not RunSpec("run1", "u", "unconstrained", None).uses_gages
    assert RunSpec("run2", "c", "constrained", Path("calval.csv")).uses_gages


def _integration_call_log(params):
    integrator = object.__new__(Integrate)
    integrator.params_dict = params
    integrator.VerboseFlag = False
    integrator.junctions = []
    integrator.basin_dict = {"reach_ids_all": ["1"]}
    integrator.alg_dict = {"metroman": {}}
    integrator.CreateJunctionList = lambda: None
    integrator.build_mass_diagnostics = lambda n, conversion: {}
    calls = []

    def solve(m, n, flow_level, residuals):
        calls.append(flow_level)
        return residuals

    integrator.integrator_optimization_calcs = solve
    integrator.compute_FLPs = lambda: calls.append("FLPs")
    integrator.integrate()
    return calls


def test_integrate_retains_full_production_workflow_by_default():
    assert _integration_call_log({}) == ["Mean", "q33", "FLPs"]


def test_integrate_mean_only_skips_q33_and_final_flps():
    params = {
        "SFOI_Flow_Levels": ("Mean",),
        "SFOI_Compute_FLPs": False,
    }

    assert _integration_call_log(params) == ["Mean"]

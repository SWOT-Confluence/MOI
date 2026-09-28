from pathlib import Path

from netCDF4 import Dataset
import numpy as np

from moi.Input import Input


def _input(alg_dir):
    obj = Input(alg_dir, None, None, None, {"reach_ids": ["1"], "reach_ids_all": ["1"]},
                "constrained", False)
    obj.sos_dict = {"1": {"Qbar": 50.0, "q33": 20.0}}
    return obj


def _write_momma(path):
    with Dataset(path, "w") as ds:
        ds.createDimension("nt", 3)
        ds.createVariable("Q", "f8", ("nt",))[:] = [1.0, 2.0, 3.0]
        ds.createVariable("zero_flow_stage", "f8")[...] = 1.0
        ds.createVariable("bankfull_stage", "f8")[...] = 5.0
        ds.createVariable("slope", "f8", ("nt",))[:] = [1e-4, 1e-4, 1e-4]


def _write_hivdi_failure(path):
    # H2iVDI error_code=1 output: global attributes and dimensions only.
    with Dataset(path, "w") as ds:
        ds.error_code = 1
        ds.createDimension("nt", 0)


def test_unreadable_flpe_file_only_drops_that_algorithm(tmp_path):
    for alg in ("hivdi", "momma"):
        (tmp_path / alg).mkdir()
    _write_hivdi_failure(tmp_path / "hivdi" / "1_hivdi.nc")
    _write_momma(tmp_path / "momma" / "1_momma.nc")

    obj = _input(Path(tmp_path))
    obj.extract_alg()

    hivdi = obj.alg_dict["hivdi"]["1"]
    assert hivdi["s1-flpe-exists"] is False
    assert hivdi["qbar"] == 50.0 and hivdi["q33"] == 20.0
    assert set(hivdi) == {"s1-flpe-exists", "q", "alpha", "beta", "qbar", "q33"}

    momma = obj.alg_dict["momma"]["1"]
    assert momma["s1-flpe-exists"] is True
    np.testing.assert_allclose(momma["q"], [1.0, 2.0, 3.0])

    assert [f["algorithm"] for f in obj.flpe_read_failures] == ["hivdi"]
    # Absent files are the normal case and are not reported as failures.
    assert obj.alg_dict["sad"]["1"]["s1-flpe-exists"] is False


def test_busboi_without_discharge_is_missing_not_scalar(tmp_path):
    (tmp_path / "busboi").mkdir()
    with Dataset(tmp_path / "busboi" / "1_busboi.nc", "w") as ds:
        ds.createGroup("r")

    obj = _input(Path(tmp_path))
    obj.extract_alg()

    busboi = obj.alg_dict["busboi"]["1"]
    assert busboi["s1-flpe-exists"] is False
    assert busboi["qbar"] == 50.0

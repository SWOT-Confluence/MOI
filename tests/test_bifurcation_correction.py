import numpy as np

from moi.sfoi_math_core import (
    _build_upstream_contribs,
    build_sparse_sfoi_problem,
)


def _bifurcation_sword(*, child_facc=(600.0, 400.0), widths=(90.0, 10.0)):
    """Build parent -> two children -> outlet SWORD attributes."""
    return {
        'num_reaches': 4,
        'reach_id': np.array([100, 110, 120, 130], dtype=np.int64),
        'facc': np.array([1000.0, child_facc[0], child_facc[1], 1100.0]),
        # facc_quality remains available as provenance, but the rolled-back
        # mathematical core must not use it to partition discharge.
        'facc_quality': np.array([-9999, 1, 1, 1], dtype=np.int32),
        'width': np.array([100.0, widths[0], widths[1], 100.0]),
        'n_rch_up': np.array([0, 1, 1, 2], dtype=np.int32),
        'n_rch_down': np.array([2, 1, 1, 0], dtype=np.int32),
        'rch_id_up': np.array(
            [
                [0, 100, 100, 110],
                [0, 0, 0, 120],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
        'rch_id_dn': np.array(
            [
                [110, 130, 130, 0],
                [120, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        ),
    }


def _junctions():
    return [
        {'upflows': [100], 'downflows': [110, 120]},
        {'upflows': [110, 120], 'downflows': [130]},
    ]


def test_rolled_back_bifurcation_uses_width_not_corrected_facc():
    sword = _bifurcation_sword(
        child_facc=(600.0, 400.0),
        widths=(90.0, 10.0),
    )

    contribs = _build_upstream_contribs(
        ['100', '110', '120', '130'],
        sword,
        _junctions(),
    )

    assert contribs['110'] == [('100', 0.9)]
    assert contribs['120'] == [('100', 0.1)]


def test_rolled_back_bifurcation_falls_back_to_equal_invalid_widths():
    sword = _bifurcation_sword(widths=(np.nan, np.nan))

    contribs = _build_upstream_contribs(
        ['100', '110', '120', '130'],
        sword,
        _junctions(),
    )

    assert contribs['110'] == [('100', 0.5)]
    assert contribs['120'] == [('100', 0.5)]


def test_fractional_coefficients_do_not_force_delta_area_to_zero():
    sword = _bifurcation_sword(
        child_facc=(600.0, 400.0),
        widths=(90.0, 10.0),
    )

    problem = build_sparse_sfoi_problem(
        reach_ids=['100', '110', '120', '130'],
        sword_dict=sword,
        junctions=_junctions(),
        Qbar=np.array([100.0, 90.0, 10.0, 110.0]),
        sigQ=np.full(4, 10.0),
        facc=sword['facc'],
        u_conversion=0.1,
        prefix_len=1,
    )

    # Width shares imply weighted upstream areas [900, 100]. The natural
    # max(0, child_facc - weighted_upstream_facc) formula therefore retains
    # 300 km2 of lateral area for child 120 instead of forcing both to zero.
    np.testing.assert_allclose(problem['delta_A'][1:3], [0.0, 300.0])


def test_cloned_child_area_is_not_special_cased():
    sword = _bifurcation_sword(
        child_facc=(1000.0, 1000.0),
        widths=(60.0, 40.0),
    )

    problem = build_sparse_sfoi_problem(
        reach_ids=['100', '110', '120', '130'],
        sword_dict=sword,
        junctions=_junctions(),
        Qbar=np.array([100.0, 60.0, 40.0, 110.0]),
        sigQ=np.full(4, 10.0),
        facc=sword['facc'],
        u_conversion=0.1,
        prefix_len=1,
    )

    np.testing.assert_allclose(problem['delta_A'][1:3], [400.0, 600.0])

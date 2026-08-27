import numpy as np
import pytest
import scipy.sparse as sp
from types import SimpleNamespace

import moi.sfoi_math_core as sfoi_math_core
from moi.Integrate import Integrate
from moi.sfoi_math_core import adjust_lsq_bias_correlated_sparse


def _identity_prior_with_gages(n_reaches, n_regions, gage_indices):
    n_physical = n_reaches + n_regions
    prior = sp.eye(n_physical, format='csr')
    gages = sp.csr_matrix(
        (
            np.ones(len(gage_indices)),
            (np.arange(len(gage_indices)), np.asarray(gage_indices)),
        ),
        shape=(len(gage_indices), n_physical),
    )
    return sp.vstack([prior, gages], format='csr')


@pytest.mark.parametrize('planted_bias', [0.0, 0.2, 0.4])
def test_multiplicative_augmentation_recovers_planted_flow_bias(planted_bias):
    n_reaches = 6
    n_regions = 2
    true_q = np.array([80.0, 100.0, 120.0, 180.0, 220.0, 260.0])
    true_r = np.array([0.7, 1.1])
    true_x = np.r_[true_q, true_r]
    gage_indices = np.array([0, 2, 3, 5])

    A = _identity_prior_with_gages(n_reaches, n_regions, gage_indices)
    observations = np.r_[
        (1.0 + planted_bias) * true_q,
        true_r,
        true_q[gage_indices],
    ]
    cov = np.r_[
        np.full(n_reaches, 0.20),
        np.full(n_regions, 0.25),
        np.full(gage_indices.size, 0.01),
    ]
    fixed = np.r_[
        np.zeros(n_reaches + n_regions, dtype=bool),
        np.ones(gage_indices.size, dtype=bool),
    ]

    result = adjust_lsq_bias_correlated_sparse(
        A_obs=A,
        L=observations,
        cov=cov,
        n_reaches=n_reaches,
        n_regions=n_regions,
        correlation_groups=np.array([0, 0, 0, 1, 1, 1]),
        correlation_rho=0.0,
        bias_enabled=True,
        bias_prior_std=10.0,
        x0=np.r_[(1.0 + planted_bias) * true_q, true_r],
        lb=np.zeros(n_reaches + n_regions),
        maxiter=40,
        change_thresh=1.0e-7,
        fixed_weight_mask=fixed,
        robust_eligible_mask=~fixed,
    )

    assert result.bias == pytest.approx(planted_bias, abs=1.0e-3)
    assert result.x == pytest.approx(true_x, rel=2.0e-4, abs=2.0e-2)
    assert np.all(result.index)
    # The unchanged runoff rows must not receive the multiplicative flow bias.
    assert result.x[-n_regions:] == pytest.approx(true_r, abs=1.0e-6)


def test_region_correlation_effects_do_not_become_global_bias():
    n_reaches = 6
    n_regions = 2
    rho = 0.36
    flow_cov = 0.10
    true_q = np.array([100.0, 120.0, 140.0, 200.0, 240.0, 280.0])
    true_r = np.array([0.8, 1.2])
    true_x = np.r_[true_q, true_r]
    groups = np.array([0, 0, 0, 1, 1, 1])
    planted_effects = np.array([0.8, -0.8])
    gage_indices = np.arange(n_reaches)

    correlated_error = (
        flow_cov
        * true_q
        * np.sqrt(rho)
        * planted_effects[groups]
    )
    A = _identity_prior_with_gages(n_reaches, n_regions, gage_indices)
    observations = np.r_[true_q + correlated_error, true_r, true_q]
    cov = np.r_[
        np.full(n_reaches, flow_cov),
        np.full(n_regions, 0.25),
        np.full(n_reaches, 0.005),
    ]
    fixed = np.r_[
        np.zeros(n_reaches + n_regions, dtype=bool),
        np.ones(n_reaches, dtype=bool),
    ]

    result = adjust_lsq_bias_correlated_sparse(
        A_obs=A,
        L=observations,
        cov=cov,
        n_reaches=n_reaches,
        n_regions=n_regions,
        correlation_groups=groups,
        correlation_rho=rho,
        bias_enabled=True,
        bias_prior_std=0.5,
        x0=np.r_[true_q + correlated_error, true_r],
        lb=np.zeros(n_reaches + n_regions),
        maxiter=40,
        change_thresh=1.0e-7,
        fixed_weight_mask=fixed,
        robust_eligible_mask=~fixed,
    )

    assert result.bias == pytest.approx(0.0, abs=1.0e-3)
    assert result.correlation_effects[0] > 0
    assert result.correlation_effects[1] < 0
    assert abs(result.correlation_effects[0] + result.correlation_effects[1]) < 0.01
    assert result.x == pytest.approx(true_x, rel=2.0e-4, abs=3.0e-2)
    assert np.all(result.index)


def test_bias_recovery_preserves_soft_mass_balance():
    true_q = np.array([100.0, 110.0, 130.0])
    true_x = np.r_[true_q, 1.0]
    planted_bias = 0.40

    prior = sp.eye(4, format='csr')
    mass = sp.csr_matrix(
        [
            [-1.0, 1.0, 0.0, -10.0],
            [0.0, -1.0, 1.0, -20.0],
        ]
    )
    downstream_gage = sp.csr_matrix([[0.0, 0.0, 1.0, 0.0]])
    A = sp.vstack([prior, mass, downstream_gage], format='csr')
    observations = np.r_[
        (1.0 + planted_bias) * true_q,
        1.0,
        0.0,
        0.0,
        true_q[-1],
    ]
    cov = np.r_[np.full(3, 0.20), 0.25, 0.10, 0.10, 0.01]
    fixed = np.r_[np.zeros(4, dtype=bool), np.ones(3, dtype=bool)]

    result = adjust_lsq_bias_correlated_sparse(
        A_obs=A,
        L=observations,
        cov=cov,
        n_reaches=3,
        n_regions=1,
        correlation_groups=[0, 0, 0],
        correlation_rho=0.20,
        bias_enabled=True,
        bias_prior_std=10.0,
        x0=np.r_[(1.0 + planted_bias) * true_q, 1.0],
        lb=np.zeros(4),
        maxiter=50,
        change_thresh=1.0e-7,
        fixed_weight_mask=fixed,
        robust_eligible_mask=~fixed,
    )

    assert result.bias == pytest.approx(planted_bias, abs=1.0e-3)
    assert result.x == pytest.approx(true_x, rel=2.0e-4, abs=2.0e-2)
    assert np.asarray(mass @ result.x).ravel() == pytest.approx(
        np.zeros(2), abs=1.0e-4
    )


def test_correlation_rho_is_validated():
    with pytest.raises(ValueError, match='0 <= rho < 1'):
        adjust_lsq_bias_correlated_sparse(
            A_obs=sp.eye(2, format='csr'),
            L=np.ones(2),
            cov=np.ones(2),
            n_reaches=1,
            n_regions=1,
            correlation_groups=[0],
            correlation_rho=1.0,
        )


def test_augmented_result_damps_period_two_oscillation_to_convergence(monkeypatch):
    """A reversing fixed-point direction is relaxed to the cycle midpoint."""

    def period_two_candidate(
        A_obs,
        L,
        W,
        A_eq=None,
        b_eq=None,
        lb=None,
        ub=None,
        x0=None,
        verbose=False,
    ):
        # Fixed-point map F(z) = 2 - z has the undamped cycle 0 -> 2 -> 0.
        return 2.0 - np.asarray(x0, dtype=float), 'success_mock_optimal'

    monkeypatch.setattr(
        sfoi_math_core,
        '_solve_wls_sparse',
        period_two_candidate,
    )

    common = dict(
        A_obs=sp.eye(1, format='csr'),
        L=np.array([1.0]),
        cov=np.array([0.20]),
        n_reaches=1,
        n_regions=0,
        correlation_rho=0.0,
        bias_enabled=False,
        x0=np.array([0.0]),
        lb=np.array([0.0]),
        maxiter=6,
        change_thresh=1.0e-9,
        fixed_weight_mask=np.array([True]),
        robust_eligible_mask=np.array([False]),
    )

    result = adjust_lsq_bias_correlated_sparse(**common)

    assert result.converged
    assert result.status.startswith(
        'success_bias_correlated_converged_after_relaxation_'
    )
    assert result.outer_iterations == 3
    assert result.x == pytest.approx([1.0], abs=1.0e-12)
    assert result.delta[-1] <= common['change_thresh']
    assert result.oscillation_events == 1
    assert result.step_relaxation == pytest.approx(0.5)
    assert result.diagnostics[1]['oscillation_detected']
    assert result.diagnostics[1]['direction_cosine'] == pytest.approx(-1.0)
    assert result.diagnostics[-1]['robust_delta'] == 0.0


def test_bias_and_correlation_apply_only_to_genuine_flpe_rows():
    true_q = np.array([100.0, 200.0])
    planted_bias = 0.20
    A = _identity_prior_with_gages(2, 0, [0, 1])
    observations = np.array(
        [
            (1.0 + planted_bias) * true_q[0],
            true_q[1],  # SoS prior: deliberately not bias augmented.
            true_q[0],
            true_q[1],
        ]
    )
    cov = np.array([0.20, 0.60, 0.005, 0.005])
    fixed = np.array([False, False, True, True])

    result = adjust_lsq_bias_correlated_sparse(
        A_obs=A,
        L=observations,
        cov=cov,
        n_reaches=2,
        n_regions=0,
        flpe_eligible_mask=np.array([True, False]),
        correlation_groups=np.array([0, 0]),
        correlation_rho=0.20,
        bias_enabled=True,
        bias_prior_std=10.0,
        x0=np.array([120.0, 200.0]),
        lb=np.zeros(2),
        maxiter=40,
        change_thresh=1.0e-7,
        fixed_weight_mask=fixed,
        robust_eligible_mask=~fixed,
    )

    assert result.converged
    assert result.bias == pytest.approx(planted_bias, abs=1.0e-3)
    assert result.x == pytest.approx(true_q, rel=2.0e-4, abs=2.0e-2)
    np.testing.assert_array_equal(result.flpe_eligible_mask, [True, False])
    np.testing.assert_array_equal(result.correlation_groups, [0, -1])
    assert result.flpe_prediction[1] == pytest.approx(result.x[1])


def test_physical_p95_threshold_prevents_hidden_local_nonconvergence(monkeypatch):
    def one_local_change(
        A_obs,
        L,
        W,
        A_eq=None,
        b_eq=None,
        lb=None,
        ub=None,
        x0=None,
        verbose=False,
    ):
        candidate = np.asarray(x0, dtype=float).copy()
        candidate[0] += 5.0
        return candidate, 'success_mock_optimal'

    monkeypatch.setattr(sfoi_math_core, '_solve_wls_sparse', one_local_change)
    n_reaches = 10
    result = adjust_lsq_bias_correlated_sparse(
        A_obs=sp.eye(n_reaches, format='csr'),
        L=np.full(n_reaches, 100.0),
        cov=np.full(n_reaches, 0.20),
        n_reaches=n_reaches,
        n_regions=0,
        correlation_rho=0.0,
        bias_enabled=False,
        x0=np.full(n_reaches, 100.0),
        maxiter=1,
        physical_rms_thresh=0.02,
        physical_p95_thresh=0.01,
        bias_thresh=0.01,
        effect_thresh=0.01,
        robust_thresh=0.01,
        fixed_weight_mask=np.ones(n_reaches, dtype=bool),
        robust_eligible_mask=np.zeros(n_reaches, dtype=bool),
    )

    assert result.diagnostics[-1]['physical_delta'] < 0.02
    assert result.diagnostics[-1]['physical_p95_delta'] > 0.01
    assert not result.converged


def test_relaxation_recovers_after_stable_directions(monkeypatch):
    call = {'count': 0}

    def reversing_then_stable(
        A_obs,
        L,
        W,
        A_eq=None,
        b_eq=None,
        lb=None,
        ub=None,
        x0=None,
        verbose=False,
    ):
        steps = [1.0, -1.0, -0.1, -0.1, -0.1]
        step = steps[call['count']]
        call['count'] += 1
        return np.asarray(x0, dtype=float) + step, 'success_mock_optimal'

    monkeypatch.setattr(
        sfoi_math_core, '_solve_wls_sparse', reversing_then_stable
    )
    result = adjust_lsq_bias_correlated_sparse(
        A_obs=sp.eye(1, format='csr'),
        L=np.array([1.0]),
        cov=np.array([0.20]),
        n_reaches=1,
        n_regions=0,
        correlation_rho=0.0,
        bias_enabled=False,
        x0=np.array([1.0]),
        lb=np.array([0.0]),
        maxiter=5,
        change_thresh=1.0e-12,
        relaxation_recovery=1.25,
        relaxation_recovery_patience=3,
        fixed_weight_mask=np.array([True]),
        robust_eligible_mask=np.array([False]),
    )

    assert not result.converged
    assert result.oscillation_events == 1
    assert result.relaxation_recoveries == 1
    assert result.step_relaxation == pytest.approx(0.625)
    assert result.diagnostics[-1]['relaxation_recovered']


def test_persistent_bounded_oscillation_converges_via_physical_hold(monkeypatch):
    """A non-decaying raw 2-cycle must not block acceptance once the
    *applied* state has been small for several iterations running.

    This reproduces the exact signature diagnosed in production basins
    7823/7240/6230: relaxation gets pinned low by an oscillation that
    re-triggers every single iteration (so the raw candidate never shrinks),
    while the actually-exported physical state has been oscillating inside a
    tiny, bounded band the whole time. The old single-iteration,
    raw-and-applied gate never terminates on this pattern; physical-hold
    accepts it once the applied step has stayed small for
    convergence_hold_iters iterations in a row -- without requiring the raw
    candidate, which never gets small here by construction, to agree.
    """
    call = {'count': 0}

    def persistent_two_cycle(
        A_obs, L, W, A_eq=None, b_eq=None, lb=None, ub=None, x0=None, verbose=False
    ):
        call['count'] += 1
        sign = 1.0 if call['count'] % 2 else -1.0
        # Jump by a fixed absolute amount from wherever the state currently
        # sits (not a fixed-point map) -- so the raw step never decays,
        # unlike test_augmented_result_damps_period_two_oscillation_to_convergence's
        # F(z) = C - z, which converges to a fixed point and lets raw shrink
        # along with everything else.
        return np.asarray(x0, dtype=float) + sign * 2.0, 'success_mock_optimal'

    monkeypatch.setattr(sfoi_math_core, '_solve_wls_sparse', persistent_two_cycle)

    result = adjust_lsq_bias_correlated_sparse(
        A_obs=sp.eye(1, format='csr'),
        L=np.array([100.0]),
        cov=np.array([0.20]),
        n_reaches=1,
        n_regions=0,
        correlation_rho=0.0,
        bias_enabled=False,
        x0=np.array([100.0]),
        lb=np.array([0.0]),
        maxiter=15,
        physical_rms_thresh=0.01,
        physical_p95_thresh=0.01,
        convergence_hold_iters=3,
        fixed_weight_mask=np.array([True]),
        robust_eligible_mask=np.array([False]),
    )

    assert result.converged
    assert result.convergence_mode == 'physical_hold'
    assert result.physical_stable_streak >= 3
    # The raw candidate's relative jump (~2/100 = 0.02) never drops under
    # physical_rms_thresh=0.01 by construction, so the strict, all-components
    # criterion this basin used to be stuck on must never have fired.
    assert not result.fully_converged
    assert not result.diagnostics[-1]['physical_converged']


def test_pipeline_falls_back_instead_of_rejecting_nonconverged_result():
    """A non-converged augmented fit is discarded, not raised.

    The pipeline used to crash the whole basin on this outcome. It now asks
    whether the result is usable and, if not, the caller in Integrate.py
    retries with the plain multiplicative solver so a basin never loses its
    qbar_basinscale output over one algorithm's unresolved bias/correlation
    fit.
    """
    unconverged = SimpleNamespace(
        converged=False,
        status='failed_bias_correlated_maxiter_success_mock_optimal',
        outer_iterations=30,
        delta=[0.083],
    )
    converged = SimpleNamespace(
        converged=True,
        status='success_bias_correlated_converged_naturally_success_mock_optimal',
        outer_iterations=12,
        delta=[0.0004],
    )

    assert not Integrate._augmented_result_is_usable(unconverged)
    assert Integrate._augmented_result_is_usable(converged)

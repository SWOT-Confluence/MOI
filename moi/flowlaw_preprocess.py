"""SWOT observation preprocessing shared by MOI and offline.

MOI fits the flow-law parameters (FLPs) that offline later evaluates to make
L2 discharge.  Both must evaluate the flow laws on the same observations and
the same width and slope, or the exported parameters do not describe the L2
series.  This file is therefore kept byte-identical in

    MOI:     moi/flowlaw_preprocess.py
    offline: offline/flowlaw_preprocess.py

and MOI records the options it used in every *_integrator.nc (see
CONFIG_ATTRIBUTES), which offline reads back.

Order of operations (prepare_flowlaw_inputs):

    raw SWOT reach series
      -> observation filter      (NRT quality checks + outlier limits)
      -> width                   (hypsometric fit if it passes QC, else the
                                  median QC-passing observed width)
      -> slope policy            (slope2; invalid or floored per policy)
      -> flowlaw_valid mask      (observation kept, slope valid, width and
                                  WSE finite)
      -> flowlaw_valid_da mask   (flowlaw_valid and d_x_area finite; used by
                                  the five dA-driven laws, not MOMMA)

Nothing here modifies its inputs.
"""

import numpy as np

PREPROCESS_VERSION = '2026-09-28'

# ---------------------------------------------------------------------------
# options and the integrator attributes that carry them from MOI to offline
# ---------------------------------------------------------------------------
FILTER_LEGACY = 'legacy'
FILTER_ROBUST_V2 = 'robust_v2'
OBSERVATION_FILTERS = (FILTER_LEGACY, FILTER_ROBUST_V2)
DEFAULT_OBSERVATION_FILTER = FILTER_LEGACY

SLOPE_LEGACY_SAFE = 'legacy_safe'   # slope2 <= 0 invalid; 0 < slope2 < 1.7e-5 -> 1.7e-5
SLOPE_L2_3P4 = 'l2_3p4'             # slope2 < 3.4e-5 invalid
SLOPE_POLICIES = (SLOPE_LEGACY_SAFE, SLOPE_L2_3P4)
DEFAULT_SLOPE_POLICY = SLOPE_LEGACY_SAFE
SLOPE_FLOOR = 1.7e-5
SLOPE_L2_MINIMUM = 3.4e-5

CONFIG_ATTRIBUTES = {
    'preprocess_version': 'flowlaw_preprocess_version',
    'observation_filter': 'flowlaw_observation_filter',
    'slope_policy': 'flowlaw_slope_policy',
}

# ---------------------------------------------------------------------------
# observation filter
# ---------------------------------------------------------------------------
QUALITY_VARIABLES = (
    'time', 'xtrk_dist', 'ice_clim_f', 'dark_frac', 'obs_frac_n',
    'xovr_cal_q', 'n_good_nod', 'p_width', 'p_length', 'reach_q_b',
)
TUKEY_NUMBER = 1.5
# robust_v2 skips the outlier test for a variable with fewer NRT-passing
# finite values than this, instead of pretending to filter.
MIN_OUTLIER_SAMPLE = 8

# ---------------------------------------------------------------------------
# width
# ---------------------------------------------------------------------------
# A hypsometric width is rejected when it leaves [0.5, 2] x the range of the
# QC-passing observed widths, or when fewer than MIN_RAW_WIDTHS observed
# widths are available to check it against.
MIN_WIDTH_RATIO = 0.5
MAX_WIDTH_RATIO = 2.0
MIN_RAW_WIDTHS = 3

WIDTH_SOURCE_HYPSOMETRIC = 'hypsometric'
WIDTH_SOURCE_MEDIAN = 'median_observed'
WIDTH_SOURCE_NONE = 'none'

# Per-timestep slope status codes.
SLOPE_OK = 0
SLOPE_FLOORED = 1
SLOPE_NONPOSITIVE = 2
SLOPE_BELOW_MINIMUM = 3
SLOPE_MISSING = 4


def _float_array(values):
    """1-D float copy with masked values as NaN."""
    return np.ma.filled(np.ma.asarray(values, dtype=float), np.nan).ravel()


def _nrt_flags(q):
    """Per-timestep NRT failures (a NaN quality value never trips a check)."""
    with np.errstate(invalid='ignore'):
        return (
            (np.abs(q['xtrk_dist']) > 60e3)
            | (np.abs(q['xtrk_dist']) < 10e3)
            | (q['ice_clim_f'] > 1)
            | (q['dark_frac'] > .6)
            | (q['obs_frac_n'] < .4)
            | (q['xovr_cal_q'] > 1)
            | (q['n_good_nod'] < 10)
            | (q['p_width'] < 60)
            | (q['p_length'] < 5000)
            | (q['reach_q_b'] > 507510784)
        )


def _tukey_limits(values):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    iqr = q75 - q25
    return q25 - TUKEY_NUMBER * iqr, q75 + TUKEY_NUMBER * iqr


def _outside(values, limits):
    lower, upper = limits
    with np.errstate(invalid='ignore'):
        return (values < lower) | (values > upper)


def _legacy_filter(h, w, s, q):
    """Offline NRTfilter behaviour (with its indexing/argument bugs fixed).

    Tukey limits come from the NRT-passing observations only when the series
    has no NaN WSE/width/slope at a valid time; otherwise the limits are the
    series min/max, i.e. no outlier filtering.  NaN quality values pass.
    """
    nrt_bad = np.isnan(q['time']) | _nrt_flags(q)
    has_time = ~np.isnan(q['time'])
    complete = not any(np.any(np.isnan(x[has_time])) for x in (h, w, s))
    outlier = np.zeros(h.size, dtype=bool)
    for values in (h, w, s):
        if complete:
            limits = _tukey_limits(np.where(nrt_bad, np.nan, values))
        else:
            finite = values[np.isfinite(values)]
            limits = ((finite.min(), finite.max()) if finite.size
                      else (np.nan, np.nan))
        outlier |= _outside(values, limits)
    info = {'n_nrt_bad': int(nrt_bad.sum()), 'n_quality_unknown': 0,
            'n_outlier': int((outlier & ~nrt_bad).sum()),
            'outlier_test_skipped': ''}
    return ~(nrt_bad | outlier), info


def _robust_v2_filter(h, w, s, q):
    """Outlier limits per variable from its NRT-passing finite values.

    A NaN quality value makes the observation unknown and it is dropped.  A
    variable with too few NRT-passing values skips its own outlier test only.
    """
    unknown = np.zeros(h.size, dtype=bool)
    for name in QUALITY_VARIABLES:
        unknown |= np.isnan(q[name])
    nrt_bad = unknown | _nrt_flags(q)
    outlier = np.zeros(h.size, dtype=bool)
    skipped = []
    for name, values in (('wse', h), ('width', w), ('slope2', s)):
        sample = np.where(nrt_bad, np.nan, values)
        if np.sum(np.isfinite(sample)) < MIN_OUTLIER_SAMPLE:
            skipped.append(name)
            continue
        outlier |= _outside(values, _tukey_limits(sample))
    info = {'n_nrt_bad': int(nrt_bad.sum()),
            'n_quality_unknown': int(unknown.sum()),
            'n_outlier': int((outlier & ~nrt_bad).sum()),
            'outlier_test_skipped': ','.join(skipped)}
    return ~(nrt_bad | outlier), info


def observation_keep_mask(h, w, s, quality, version=DEFAULT_OBSERVATION_FILTER):
    """``(keep, info)``: True where the observation passes the filter.

    ``h``, ``w`` and ``s`` are the observed WSE, width and slope2; ``quality``
    maps every name in QUALITY_VARIABLES to its series.  When ``quality`` is
    None (the SWOT file lacks the variables) every observation is kept.
    """
    h, w, s = (_float_array(x) for x in (h, w, s))
    if quality is None:
        return np.ones(h.size, dtype=bool), {
            'n_nrt_bad': 0, 'n_quality_unknown': 0, 'n_outlier': 0,
            'outlier_test_skipped': 'no_quality_variables'}
    q = {name: _float_array(quality[name]) for name in QUALITY_VARIABLES}
    if version == FILTER_LEGACY:
        return _legacy_filter(h, w, s, q)
    if version == FILTER_ROBUST_V2:
        return _robust_v2_filter(h, w, s, q)
    raise ValueError('unknown observation filter %r' % (version,))


# ---------------------------------------------------------------------------
# width
# ---------------------------------------------------------------------------
def _hypsometric_width(h, area_fit):
    """Width from the three-segment height-width fit, or None if unusable."""
    try:
        h_break = np.asarray(area_fit['h_break'], dtype=float).ravel()
        coeffs = np.asarray(area_fit['fit_coeffs'], dtype=float)
    except Exception:
        return None
    if h_break.size < 4 or not np.all(np.isfinite(h_break)):
        return None
    # The four h_break values are the lower boundary, two internal breakpoints
    # and the upper boundary, so the segments switch at h_break[1] and [2].
    segment = np.where(h < h_break[1], 0, np.where(h < h_break[2], 1, 2))
    intercept = coeffs[1, segment, 0]
    slope = coeffs[0, segment, 0]
    return intercept + slope * h


def constrain_width(h, w, area_fit, keep=None):
    """``(width, source)`` for every timestep.

    The hypsometric width is used when, at the kept observations, it is
    finite, positive and within [MIN_WIDTH_RATIO, MAX_WIDTH_RATIO] x the range
    of the kept observed widths (at least MIN_RAW_WIDTHS of them).  Otherwise
    every width is the median kept observed width; with no kept observed
    width it is NaN (no discharge).
    """
    h, w = _float_array(h), _float_array(w)
    keep = np.ones(h.size, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
    observed = w[keep & np.isfinite(w) & (w > 0)]

    predicted = _hypsometric_width(h, area_fit)
    if predicted is not None and observed.size >= MIN_RAW_WIDTHS:
        check = predicted[keep & np.isfinite(h)]
        if (check.size and np.all(np.isfinite(check)) and np.all(check > 0)
                and check.min() >= MIN_WIDTH_RATIO * observed.min()
                and check.max() <= MAX_WIDTH_RATIO * observed.max()):
            return predicted, WIDTH_SOURCE_HYPSOMETRIC

    if observed.size:
        return np.full(h.size, float(np.median(observed))), WIDTH_SOURCE_MEDIAN
    return np.full(h.size, np.nan), WIDTH_SOURCE_NONE


# ---------------------------------------------------------------------------
# slope
# ---------------------------------------------------------------------------
def prepare_slope(slope2, policy=DEFAULT_SLOPE_POLICY):
    """``(slope_used, slope_valid, slope_status)`` for a slope2 series.

    legacy_safe: slope2 <= 0 is invalid; 0 < slope2 < 1.7e-5 is raised to
                 1.7e-5.
    l2_3p4:      slope2 < 3.4e-5 is invalid.
    Invalid and missing slopes are NaN in ``slope_used``.
    """
    s = _float_array(slope2)
    status = np.full(s.size, SLOPE_OK, dtype=np.int8)
    status[~np.isfinite(s)] = SLOPE_MISSING
    finite = np.isfinite(s)
    if policy == SLOPE_LEGACY_SAFE:
        status[finite & (s <= 0)] = SLOPE_NONPOSITIVE
        status[finite & (s > 0) & (s < SLOPE_FLOOR)] = SLOPE_FLOORED
    elif policy == SLOPE_L2_3P4:
        status[finite & (s < SLOPE_L2_MINIMUM)] = SLOPE_BELOW_MINIMUM
    else:
        raise ValueError('unknown slope policy %r' % (policy,))

    valid = (status == SLOPE_OK) | (status == SLOPE_FLOORED)
    used = np.where(valid, s, np.nan)
    used[status == SLOPE_FLOORED] = SLOPE_FLOOR
    return used, valid, status


# ---------------------------------------------------------------------------
# everything together
# ---------------------------------------------------------------------------
def prepare_flowlaw_inputs(h, w, s, area_fit, quality,
                           observation_filter=DEFAULT_OBSERVATION_FILTER,
                           slope_policy=DEFAULT_SLOPE_POLICY, dA=None):
    """Width, slope and validity mask at which to evaluate the flow laws.

    Returns a dict with per-timestep arrays ``keep`` (observation filter),
    ``width_used``, ``slope_used``, ``slope_status`` and ``flowlaw_valid``
    (common validity for every law), plus ``flowlaw_valid_da`` for laws that
    require d_x_area, ``width_source`` and the filter counts.  WSE and
    d_x_area are used as observed.
    """
    h = _float_array(h)
    keep, info = observation_keep_mask(h, w, s, quality, observation_filter)
    width_used, width_source = constrain_width(h, w, area_fit, keep)
    slope_used, slope_valid, slope_status = prepare_slope(s, slope_policy)
    with np.errstate(invalid='ignore'):
        flowlaw_valid = (keep & slope_valid & np.isfinite(h)
                         & np.isfinite(width_used) & (width_used > 0))
    if dA is None:
        flowlaw_valid_da = flowlaw_valid.copy()
    else:
        dA = _float_array(dA)
        if dA.size != h.size:
            raise ValueError('d_x_area must have the same length as WSE')
        flowlaw_valid_da = flowlaw_valid & np.isfinite(dA)
    result = {
        'keep': keep,
        'width_used': width_used,
        'width_source': width_source,
        'slope_used': slope_used,
        'slope_status': slope_status,
        'flowlaw_valid': flowlaw_valid,
        'flowlaw_valid_da': flowlaw_valid_da,
    }
    result.update(info)
    return result


def read_config(dataset, require_version=True):
    """Preprocessing options recorded on an open *_integrator.nc dataset.

    ``require_version`` is True on production MOI/offline handoffs: silently
    interpreting an old file with today's defaults would apply a different
    slope rule from the one its parameters were fitted with.  Tests and
    explicit legacy tools may opt into the old fallback with False.
    """
    def attribute(key, default):
        name = CONFIG_ATTRIBUTES[key]
        return str(dataset.getncattr(name)) if name in dataset.ncattrs() else default
    version = attribute('preprocess_version', None)
    if require_version and version is None:
        raise ValueError('flow-law preprocessing version is missing')
    if version is not None and version != PREPROCESS_VERSION:
        raise ValueError('unsupported flow-law preprocessing version %r; expected %r'
                         % (version, PREPROCESS_VERSION))
    observation_filter = attribute('observation_filter',
                                   DEFAULT_OBSERVATION_FILTER)
    slope_policy = attribute('slope_policy', DEFAULT_SLOPE_POLICY)
    if observation_filter not in OBSERVATION_FILTERS:
        raise ValueError('unknown observation filter %r' % observation_filter)
    if slope_policy not in SLOPE_POLICIES:
        raise ValueError('unknown slope policy %r' % slope_policy)
    return {'observation_filter': observation_filter,
            'slope_policy': slope_policy}

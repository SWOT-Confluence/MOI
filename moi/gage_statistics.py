"""UTC study periods and daily gage statistics, independent of geometry QC."""

import datetime as dt

import numpy as np


def swot_ordinal_days(seconds):
    """Convert valid SWOT seconds since 2000-01-01 to UTC calendar days."""
    values = np.ma.asarray(seconds, dtype=float).filled(np.nan).ravel()
    epoch = dt.datetime(2000, 1, 1)
    days = []
    for value in values:
        if not np.isfinite(value) or value < 0:
            continue
        try:
            days.append((epoch + dt.timedelta(seconds=float(value))).toordinal())
        except (OverflowError, ValueError):
            continue
    return np.asarray(days, dtype=np.int64)


def observation_days(obs):
    """Use the complete record, including dates rejected for missing geometry."""
    if 't_full' in obs:
        return swot_ordinal_days(obs['t_full'])
    # Older callers retain full time_str but only geometry-filtered t.
    if 'time_str' in obs:
        days = []
        for value in np.asarray(obs['time_str']).ravel():
            if isinstance(value, bytes):
                value = value.decode('utf-8', errors='replace')
            try:
                stamp = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
                if stamp.tzinfo is not None:
                    stamp = stamp.astimezone(dt.timezone.utc)
                days.append(stamp.date().toordinal())
            except (ValueError, TypeError, OverflowError):
                continue
        return np.asarray(days, dtype=np.int64)
    return swot_ordinal_days(obs.get('t', []))


def study_period(obs_dict, start=None, end=None):
    """Inclusive basin-wide period; explicit ISO dates may override either end."""
    bounds = []
    for obs in obs_dict.values():
        days = observation_days(obs)
        if days.size:
            bounds.extend((int(days.min()), int(days.max())))
    first = dt.date.fromisoformat(start).toordinal() if start else min(bounds, default=None)
    last = dt.date.fromisoformat(end).toordinal() if end else max(bounds, default=None)
    if first is None or last is None:
        raise ValueError(
            'Cannot determine the gage study period without raw SWOT dates; '
            'provide both Gage_Period_Start and Gage_Period_End '
            '(CLI: --gage-period-start and --gage-period-end).'
        )
    if first > last:
        raise ValueError('Gage study-period start must not be after its end')
    return first, last


def daily_gage_values(times, discharge, include_zero=True):
    """One value per valid ordinal day, averaging duplicate daily records."""
    times = np.ma.asarray(times, dtype=float).filled(np.nan).ravel()
    discharge = np.ma.asarray(discharge, dtype=float).filled(np.nan).ravel()
    count = min(times.size, discharge.size)
    times, discharge = times[:count], discharge[:count]
    valid = (np.isfinite(times) & (times >= 1)
             & (times <= dt.date.max.toordinal()) & (times == np.floor(times))
             & np.isfinite(discharge))
    valid &= discharge >= 0 if include_zero else discharge > 0
    days, inverse = np.unique(times[valid].astype(np.int64), return_inverse=True)
    sums = np.bincount(inverse, weights=discharge[valid])
    counts = np.bincount(inverse)
    return days, sums / counts

"""Tests for reading CORRIDORS resources and building pseudo-gages."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moi.Corridors import Corridors, TIME_COLUMN, DISCHARGE_COLUMN
from moi.Input import Input


REPO_ROOT = Path(__file__).resolve().parents[1]
TRANSLATION_SOURCE = REPO_ROOT / 'corridors' / 'SWORD_v16_v17_translation_reach.csv'
NOATAK_SOURCE = REPO_ROOT / 'corridors' / 'SWOT_hydroshare_Noatak_v2_SWORD16.csv'

# 81340100031 (v16) -> 81340100011 (v17), the Noatak reach with the most
# CORRIDORS measurements.
NOATAK_V16 = 81340100031
NOATAK_V17 = '81340100011'
NOATAK_BASIN = '8134'


def write_translation(directory, pairs):
    path = directory / 'SWORD_v16_v17_translation_reach.csv'
    pd.DataFrame(
        {
            'v17_reach_id': [v17 for _, v17 in pairs],
            'v16_reach_id': [v16 for v16, _ in pairs],
            'reach_id': [v17 for _, v17 in pairs],
        }
    ).to_csv(path, index=False)
    return path


def corridors_rows(reach_id, dates, discharges):
    """A minimal but realistically-named CORRIDORS resource."""
    return pd.DataFrame(
        {
            'Reach_ID': [reach_id] * len(dates),
            'Node_ID': [-9999] * len(dates),
            'SWORD_Version': [16] * len(dates),
            'X': [-162.36] * len(dates),
            'Y': [67.11] * len(dates),
            TIME_COLUMN: [f"{d}'" for d in dates],
            DISCHARGE_COLUMN: discharges,
            'Qu_(m^3/s_daily)': [-9999] * len(dates),
        }
    )


def swot_obs(dates, n_delete=0):
    """An obs_dict entry shaped the way Input.extract_swot leaves it.

    h/w/S/dA are trimmed by iDelete while time_str is kept at full length,
    which is the asymmetry Corridors has to undo.
    """
    nt_full = len(dates)
    nt_valid = nt_full - n_delete
    rng = np.random.default_rng(0)
    return {
        'h': np.linspace(10.0, 12.0, nt_valid),
        'w': np.linspace(100.0, 140.0, nt_valid),
        'S': np.full(nt_valid, 1.0e-4) + rng.normal(0, 1e-6, nt_valid),
        'dA': np.linspace(0.0, 200.0, nt_valid),
        'time_str': np.array([f'{d}T22:00:00Z' for d in dates]),
        # np.where-style: a tuple of index arrays, as Input stores it.
        'iDelete': (np.arange(nt_delete_start(nt_full, n_delete), nt_full),),
        'nt': nt_valid,
    }


def nt_delete_start(nt_full, n_delete):
    return nt_full - n_delete


# ---------------------------------------------------------------------------
# Graceful degradation: none of these may raise
# ---------------------------------------------------------------------------

def test_missing_directory_returns_none(tmp_path):
    corridors = Corridors(tmp_path / 'absent', {'basin_id': NOATAK_BASIN}, {})
    with pytest.warns(UserWarning, match='not found'):
        assert corridors.integrate_corridors_data() is None


def test_empty_directory_returns_none(tmp_path):
    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    with pytest.warns(UserWarning, match='No CORRIDORS CSV files'):
        assert corridors.integrate_corridors_data() is None


def test_missing_translation_file_returns_none(tmp_path):
    corridors_rows(NOATAK_V16, ['02-07-2024'], [646.58]).to_csv(
        tmp_path / 'resource.csv', index=False
    )
    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    with pytest.warns(UserWarning, match='translation file'):
        assert corridors.integrate_corridors_data() is None


def test_reach_outside_basin_returns_none(tmp_path):
    corridors_rows(NOATAK_V16, ['02-07-2024'], [646.58]).to_csv(
        tmp_path / 'resource.csv', index=False
    )
    write_translation(tmp_path, [(NOATAK_V16, int(NOATAK_V17))])

    # A basin that shares no prefix with the reach.
    corridors = Corridors(tmp_path, {'basin_id': '2222'}, {})
    assert corridors.integrate_corridors_data() is None


def test_reach_without_swot_observations_returns_none(tmp_path):
    corridors_rows(NOATAK_V16, ['02-07-2024'], [646.58]).to_csv(
        tmp_path / 'resource.csv', index=False
    )
    write_translation(tmp_path, [(NOATAK_V16, int(NOATAK_V17))])

    # Right basin, but obs_dict has nothing for the reach.
    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    assert corridors.integrate_corridors_data() is None


def test_unreadable_resource_is_skipped_not_fatal(tmp_path):
    (tmp_path / 'broken.csv').write_text('this,is\nnot,a,corridors,resource\n')
    write_translation(tmp_path, [(NOATAK_V16, int(NOATAK_V17))])

    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    with pytest.warns(UserWarning):
        assert corridors.integrate_corridors_data() is None


def test_merge_corridors_and_gages_accepts_none():
    """integrate_corridors_data returns None for most basins."""
    input_obj = Input.__new__(Input)
    input_obj.gage_dict = {'1001': {'source': 'SVS'}}
    input_obj.corridors_reaches = set()
    input_obj.VerboseFlag = False

    assert input_obj.merge_corridors_and_gages(None) == {'1001': {'source': 'SVS'}}
    assert input_obj.corridors_reaches == set()


def fresh_input(gage_dict):
    input_obj = Input.__new__(Input)
    input_obj.gage_dict = dict(gage_dict)
    input_obj.corridors_reaches = set()
    input_obj.VerboseFlag = False
    return input_obj


def test_merge_corridors_and_gages_keeps_the_real_gage():
    """Default precedence: the rated station beats the fitted pseudo-gage.

    This is a deliberate change from the original CORRIDORS branch, which let
    the pseudo-gage overwrite the station -- see
    test_corridors_can_override_the_gage for the flag that restores it.
    """
    input_obj = fresh_input({'1001': {'source': 'SVS'}})

    input_obj.merge_corridors_and_gages({
        '1001': {'source': 'corridors'},
        '2002': {'source': 'corridors'},
    })

    assert input_obj.gage_dict['1001']['source'] == 'SVS'
    assert input_obj.gage_dict['2002']['source'] == 'corridors'
    # Availability is recorded for both, including the reach that lost.
    assert input_obj.corridors_reaches == {'1001', '2002'}


def test_corridors_can_override_the_gage():
    """override_gage=True reproduces the original CORRIDORS branch."""
    input_obj = fresh_input({'1001': {'source': 'SVS'}})

    input_obj.merge_corridors_and_gages(
        {'1001': {'source': 'corridors'}}, override_gage=True
    )

    assert input_obj.gage_dict['1001']['source'] == 'corridors'
    assert input_obj.corridors_reaches == {'1001'}


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def test_quoted_resource_is_read_like_a_well_formed_one(tmp_path):
    """Conaway's USGS Alaska resource wraps every line in stray quotes.

    pd.read_csv cannot unpick them because they are unbalanced, so the reader
    has to fall back to QUOTE_NONE and strip them off by hand.
    """
    header = (
        'Reach_ID,Node_ID,SWORD_Version,X,Y,'
        "Time_('dd-mm-yyyy'),Q_(m^3/s_daily),Qu_(m^3/s_daily)"
    )
    rows = [
        f'{NOATAK_V16},-9999,16,-162.36,67.11,\'02-07-2024\',646.58,-9999',
        f'{NOATAK_V16},-9999,16,-162.36,67.11,\'03-07-2024\',676.71,-9999',
    ]
    path = tmp_path / 'USGS_Alaska_SWOT_ADCP_dataC.csv'
    path.write_text(
        '"' + header + '"\n' + ''.join(f'"{row}"\n' for row in rows)
    )

    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    df = corridors.read_corridors_csv(path)

    assert list(df.columns[:5]) == [
        'Reach_ID', 'Node_ID', 'SWORD_Version', 'X', 'Y'
    ]
    # Reach ids must come back numeric, or concat with a well-formed resource
    # produces an object column and the == comparisons stop matching.
    assert pd.api.types.is_integer_dtype(df['Reach_ID'])
    assert df['Reach_ID'].tolist() == [NOATAK_V16, NOATAK_V16]
    assert df[TIME_COLUMN].tolist() == ['02-07-2024', '03-07-2024']
    assert df[DISCHARGE_COLUMN].tolist() == [646.58, 676.71]
    # Stripping the trailing quote must not leave the last column as text.
    assert pd.api.types.is_numeric_dtype(df['Qu_(m^3/s_daily)'])


def test_quoted_and_plain_resources_concatenate_cleanly(tmp_path):
    """The whole point of normalizing: both readers must agree on dtypes."""
    plain = corridors_rows(NOATAK_V16, ['04-07-2024'], [624.41])
    plain.to_csv(tmp_path / 'plain.csv', index=False)

    header = "Reach_ID,Time_('dd-mm-yyyy'),Q_(m^3/s_daily)"
    quoted = tmp_path / 'USGS_Alaska_SWOT_ADCP_dataC.csv'
    quoted.write_text(
        f'"{header}"\n"{NOATAK_V16},\'02-07-2024\',646.58"\n'
    )

    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    frames = corridors.read_corridors_files([tmp_path / 'plain.csv', quoted])
    assert len(frames) == 2

    combined = pd.concat(frames, ignore_index=True)
    assert pd.api.types.is_integer_dtype(combined['Reach_ID'])
    assert (combined['Reach_ID'] == NOATAK_V16).all()


def test_fill_value_discharge_becomes_nan(tmp_path):
    df = corridors_rows(NOATAK_V16, ['02-07-2024', '03-07-2024'], [646.58, -9999])
    df.to_csv(tmp_path / 'resource.csv', index=False)

    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    out = corridors.read_corridors_csv(tmp_path / 'resource.csv')

    assert out[DISCHARGE_COLUMN].tolist()[0] == 646.58
    assert np.isnan(out[DISCHARGE_COLUMN].tolist()[1])


def test_non_positive_discharge_never_reaches_the_fit(tmp_path):
    """A zero or negative discharge is not a measurement.

    The flow law cannot reproduce it and rRMSE divides by the observation, so
    one zero makes the fit residual infinite.  Screening on q > 0 rather than
    on the -9999 fill also catches values that were never flagged.
    """
    df = corridors_rows(
        NOATAK_V16,
        ['02-07-2024', '03-07-2024', '04-07-2024', '05-07-2024', '06-07-2024'],
        [646.58, 0.0, -5.0, -9999, 624.41],
    )
    df.to_csv(tmp_path / 'resource.csv', index=False)

    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    out = corridors.read_corridors_csv(tmp_path / 'resource.csv')

    kept = out[DISCHARGE_COLUMN].dropna().tolist()
    assert kept == [646.58, 624.41]


def test_non_positive_discharge_does_not_pad_the_observation_count(tmp_path):
    """The min_observations threshold must count fittable measurements only."""
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024', '04-07-2024', '05-07-2024'],
        # Only two usable measurements, dressed up as four.
        discharges=[646.58, 0.0, -5.0, 624.41],
        swot_dates=['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05'],
    )
    # Two usable measurements is below the default threshold of three.
    assert corridors.integrate_corridors_data() is None


def test_missing_basin_id_does_not_claim_every_reach(tmp_path):
    """Every reach id starts with the empty string."""
    corridors_rows(NOATAK_V16, ['02-07-2024'], [646.58]).to_csv(
        tmp_path / 'resource.csv', index=False
    )
    write_translation(tmp_path, [(NOATAK_V16, int(NOATAK_V17))])

    corridors = Corridors(tmp_path, {}, {NOATAK_V17: swot_obs(['2024-07-02'])})
    with pytest.warns(UserWarning, match='No basin_id'):
        assert corridors.integrate_corridors_data() is None


def test_untranslatable_v16_reach_is_dropped_not_fatal(tmp_path):
    df = pd.concat([
        corridors_rows(NOATAK_V16, ['02-07-2024'], [646.58]),
        corridors_rows(99999999999, ['02-07-2024'], [12.3]),
    ], ignore_index=True)
    df.to_csv(tmp_path / 'resource.csv', index=False)
    trans = write_translation(tmp_path, [(NOATAK_V16, int(NOATAK_V17))])

    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {})
    corridors.corridors_df = corridors.read_corridors_csv(tmp_path / 'resource.csv')
    assert corridors.add_sword_17_ids(trans) is True

    translated = corridors.corridors_df['reach_id_17']
    assert translated.notna().sum() == 1
    assert int(translated.dropna().iloc[0]) == int(NOATAK_V17)


# ---------------------------------------------------------------------------
# Matching and the pseudo-gage
# ---------------------------------------------------------------------------

def build_corridors(tmp_path, dates, discharges, swot_dates, n_delete=0):
    corridors_rows(NOATAK_V16, dates, discharges).to_csv(
        tmp_path / 'resource.csv', index=False
    )
    write_translation(tmp_path, [(NOATAK_V16, int(NOATAK_V17))])
    obs_dict = {NOATAK_V17: swot_obs(swot_dates, n_delete=n_delete)}
    return Corridors(
        tmp_path, {'basin_id': NOATAK_BASIN}, obs_dict, verbose=True
    )


def test_measurements_beyond_one_day_are_not_matched(tmp_path):
    """merge_asof without a tolerance pairs measurements months apart."""
    corridors = build_corridors(
        tmp_path,
        # Only the first measurement has an overpass within a day; the others
        # are a week and a fortnight away from the single overpass.
        dates=['02-07-2024', '10-07-2024', '20-07-2024'],
        discharges=[646.58, 676.71, 624.41],
        swot_dates=['2024-07-02'],
    )
    corridors.corridors_df = corridors.read_corridors_csv(tmp_path / 'resource.csv')
    corridors.add_sword_17_ids(tmp_path / 'SWORD_v16_v17_translation_reach.csv')
    corridors.find_corridors_in_basin()
    assert corridors.prepare_corridors_time() is True

    _, reachdf = corridors.create_reach_df(int(NOATAK_V17))
    assert len(reachdf) == 1
    assert reachdf[DISCHARGE_COLUMN].iloc[0] == 646.58


def test_too_few_matches_yields_no_pseudo_gage(tmp_path):
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024'],
        discharges=[646.58, 676.71],
        swot_dates=['2024-07-02', '2024-07-03', '2024-07-04'],
    )
    # Two matches, below MIN_FIT_OBSERVATIONS.
    assert corridors.integrate_corridors_data() is None


def test_pseudo_gage_is_indexed_on_utc_ordinal_days(tmp_path):
    """Integrate.prepare_gage_constraints matches on UTC ordinal days.

    The overpass timestamps are held in local time to pair them with the
    CORRIDORS calendar dates, so indexing the pseudo-gage on the local date
    would silently drop every overpass that falls on a different UTC day --
    which in Alaska is most evening overpasses.
    """
    swot_dates = ['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05']
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024', '04-07-2024'],
        discharges=[646.58, 676.71, 624.41],
        swot_dates=swot_dates,
    )
    result = corridors.integrate_corridors_data()
    assert result is not None

    entry = result[NOATAK_V17]
    assert entry['source'] == 'corridors'
    assert entry['station_id'] is None
    assert entry['n_corridors_measurements'] == 3

    # swot_obs stamps every overpass at 22:00Z, so the UTC ordinal is the
    # nominal date; the Anchorage local date would be the day before.
    expected = {
        datetime.date.fromisoformat(d).toordinal() for d in swot_dates
    }
    assert set(int(t) for t in entry['t']).issubset(expected)
    assert np.all(entry['Q'] > 0)
    assert entry['Q'].size == entry['t'].size


def test_time_str_is_trimmed_by_idelete(tmp_path):
    """Input keeps time_str at full length while trimming h/w/S/dA."""
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024', '04-07-2024'],
        discharges=[646.58, 676.71, 624.41],
        swot_dates=['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05'],
        n_delete=1,
    )
    swotdf = corridors.swot_reach_frame(NOATAK_V17)

    # One overpass deleted, so three remain and the dropped one is the last.
    assert len(swotdf) == 3
    assert swotdf['time_str'].tolist() == [
        '2024-07-02T22:00:00Z', '2024-07-03T22:00:00Z', '2024-07-04T22:00:00Z'
    ]


def test_ambiguous_v16_reach_is_skipped_with_a_warning(tmp_path):
    """SWORD splits some v16 reaches into several v17 reaches.

    125 of the 16525 reaches in the shipped translation table do, up to a
    fanout of four.  Picking one is arbitrary and copying the discharge to all
    of them invents constraints that break mass conservation at the
    confluence, so the reach has to be dropped.
    """
    corridors_rows(NOATAK_V16, ['02-07-2024', '03-07-2024', '04-07-2024'],
                   [646.58, 676.71, 624.41]).to_csv(
        tmp_path / 'resource.csv', index=False
    )
    # One v16 reach, two v17 successors.
    write_translation(tmp_path, [
        (NOATAK_V16, int(NOATAK_V17)),
        (NOATAK_V16, int(NOATAK_V17) + 10),
    ])

    swot_dates = ['2024-07-02', '2024-07-03', '2024-07-04']
    obs_dict = {
        NOATAK_V17: swot_obs(swot_dates),
        str(int(NOATAK_V17) + 10): swot_obs(swot_dates),
    }
    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, obs_dict)

    with pytest.warns(UserWarning, match='mass conservation'):
        result = corridors.integrate_corridors_data()

    # Neither successor may receive the discharge.
    assert result is None
    assert corridors.ambiguous_v16_reaches == [
        (NOATAK_V16, [int(NOATAK_V17), int(NOATAK_V17) + 10])
    ]


def test_unambiguous_reach_still_translates_alongside_an_ambiguous_one(tmp_path):
    """Dropping the split reach must not cost us the well-defined ones."""
    other_v16 = 81340100041
    other_v17 = 81340100021
    pd.concat([
        corridors_rows(NOATAK_V16, ['02-07-2024'], [646.58]),
        corridors_rows(other_v16, ['02-07-2024', '03-07-2024', '04-07-2024'],
                       [624.41, 610.0, 600.0]),
    ], ignore_index=True).to_csv(tmp_path / 'resource.csv', index=False)
    write_translation(tmp_path, [
        (NOATAK_V16, int(NOATAK_V17)),
        (NOATAK_V16, int(NOATAK_V17) + 10),
        (other_v16, other_v17),
    ])

    swot_dates = ['2024-07-02', '2024-07-03', '2024-07-04']
    obs_dict = {
        NOATAK_V17: swot_obs(swot_dates),
        str(other_v17): swot_obs(swot_dates),
    }
    corridors = Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, obs_dict)

    with pytest.warns(UserWarning, match='mass conservation'):
        result = corridors.integrate_corridors_data()

    assert result is not None
    assert set(result) == {str(other_v17)}


# ---------------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------------

def test_pseudo_gage_carries_its_own_uncertainty(tmp_path):
    """Integrate reads relative_uncertainty straight off the gage entry.

    Without it the pseudo-gage inherits Gage_Uncertainty, i.e. a flow law
    fitted to three field measurements would constrain the integrator as
    tightly as a rated station.
    """
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024', '04-07-2024'],
        discharges=[646.58, 676.71, 624.41],
        swot_dates=['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05'],
    )
    entry = corridors.integrate_corridors_data()[NOATAK_V17]

    assert 'relative_uncertainty' in entry
    assert entry['relative_uncertainty'] >= 0.10
    assert entry['n_corridors_measurements'] == 3
    assert 'corridors_fit_relative_rmse' in entry
    # The floor may only be raised by the fit residual, never lowered.
    rrmse = entry['corridors_fit_relative_rmse']
    if np.isfinite(rrmse):
        assert entry['relative_uncertainty'] == pytest.approx(max(0.10, rrmse))


def test_a_poor_fit_is_downweighted_beyond_the_floor(tmp_path):
    """A badly fitting flow law must not constrain as tightly as a good one."""
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024', '04-07-2024', '05-07-2024'],
        # Discharge that no monotonic flow law can follow.
        discharges=[10.0, 5000.0, 20.0, 4000.0],
        swot_dates=['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05'],
    )
    entry = corridors.integrate_corridors_data()[NOATAK_V17]
    assert entry['relative_uncertainty'] > 0.10


def test_min_uncertainty_is_configurable(tmp_path):
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024', '04-07-2024'],
        discharges=[646.58, 676.71, 624.41],
        swot_dates=['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05'],
    )
    corridors.min_uncertainty = 0.5
    entry = corridors.integrate_corridors_data()[NOATAK_V17]
    assert entry['relative_uncertainty'] >= 0.5


def test_unknown_timezone_is_rejected_up_front(tmp_path):
    with pytest.raises(ValueError, match='Unknown CORRIDORS timezone'):
        Corridors(tmp_path, {'basin_id': NOATAK_BASIN}, {},
                  timezone='Mars/Olympus_Mons')


# ---------------------------------------------------------------------------
# Integration: the pseudo-gage has to survive all the way into the solver
# ---------------------------------------------------------------------------

def fake_integrator(gage_dict, obs):
    """An Integrate carrying only what prepare_gage_constraints touches."""
    from moi.Integrate import Integrate

    integrator = Integrate.__new__(Integrate)
    integrator.gage_dict = gage_dict
    integrator.obs_dict = {NOATAK_V17: obs}
    integrator.sos_dict = {}
    # gage_dict was passed explicitly, so SoS gages must not refill it.
    integrator._use_sos_gage_fallback = False
    integrator.basin_dict = {
        'reach_ids': [NOATAK_V17], 'reach_ids_all': [NOATAK_V17]
    }
    integrator.params_dict = {}
    integrator.VerboseFlag = False
    return integrator


def test_pseudo_gage_survives_into_prepared_gage_constraints(tmp_path):
    """Corridors -> Input.merge -> Integrate.prepare_gage_constraints.

    prepare_gage_constraints keeps a gage only where its ordinal days match
    the SWOT overpass days it derives from obs_dict['t'] (seconds since 2000,
    UTC).  Asserting on the Corridors return value alone would not catch the
    pseudo-gage being indexed on local dates, because it looks perfectly well
    formed right up until the day matching silently drops every sample.
    """
    swot_dates = ['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05']
    corridors = build_corridors(
        tmp_path,
        dates=['02-07-2024', '03-07-2024', '04-07-2024'],
        discharges=[646.58, 676.71, 624.41],
        swot_dates=swot_dates,
    )
    corridors_dict = corridors.integrate_corridors_data()
    assert corridors_dict is not None

    input_obj = Input.__new__(Input)
    input_obj.gage_dict = {}
    input_obj.corridors_reaches = set()
    input_obj.VerboseFlag = False
    input_obj.merge_corridors_and_gages(corridors_dict)
    assert input_obj.corridors_reaches == {NOATAK_V17}

    # obs_dict['t'] is seconds since 2000-01-01 UTC, the form Integrate reads.
    epoch = datetime.datetime(2000, 1, 1)
    obs = dict(corridors.obs_dict[NOATAK_V17])
    obs['t'] = np.array([
        (datetime.datetime.fromisoformat(f'{d} 22:00:00') - epoch).total_seconds()
        for d in swot_dates
    ])

    integrator = fake_integrator(dict(input_obj.gage_dict), obs)
    integrator.prepare_gage_constraints()

    assert NOATAK_V17 in integrator.gage_dict, (
        'the pseudo-gage was dropped by SWOT-day matching, which is what an '
        'ordinal built from local rather than UTC time causes'
    )
    prepared = integrator.gage_dict[NOATAK_V17]
    assert prepared['n_matched'] > 0
    assert prepared['Qbar'] > 0
    # The uncertainty has to reach the solver, not just the dict.
    assert prepared['relative_uncertainty'] >= 0.10


def test_local_date_indexing_corrupts_the_day_match(tmp_path):
    """Guards the guard: local-time ordinals really are matched differently.

    Anchorage is UTC-8, so a 06:00Z overpass falls on the previous local day.
    Indexing the pseudo-gage on that local date shifts its whole day series by
    one, so prepare_gage_constraints matches a different subset of overpasses
    and Qbar is computed from the wrong samples.  The loss is partial rather
    than total, which is what makes it easy to miss.
    """
    swot_dates = ['2024-07-02', '2024-07-03', '2024-07-04', '2024-07-05']
    epoch = datetime.datetime(2000, 1, 1)
    obs = swot_obs(swot_dates)
    obs['t'] = np.array([
        (datetime.datetime.fromisoformat(f'{d} 06:00:00') - epoch).total_seconds()
        for d in swot_dates
    ])

    Q = np.array([600.0, 620.0, 640.0, 660.0])
    utc_ordinals = np.array(
        [datetime.date.fromisoformat(d).toordinal() for d in swot_dates]
    )
    local_ordinals = utc_ordinals - 1

    def prepared_with(ordinals):
        integrator = fake_integrator(
            {NOATAK_V17: {'source': 'corridors', 't': ordinals, 'Q': Q}}, obs
        )
        integrator.prepare_gage_constraints()
        return integrator.gage_dict.get(NOATAK_V17)

    correct = prepared_with(utc_ordinals)
    shifted = prepared_with(local_ordinals)

    assert correct is not None
    assert correct['n_matched'] == len(swot_dates)
    assert correct['Qbar'] == pytest.approx(Q.mean())

    # The shifted series still produces a gage row, so nothing errors -- it
    # just quietly constrains the integrator with the wrong mean.
    assert shifted is not None
    assert shifted['n_matched'] < correct['n_matched']
    assert shifted['Qbar'] != pytest.approx(correct['Qbar'])


def test_real_noatak_resource_produces_pseudo_gages(tmp_path):
    """End to end against the resource actually committed to the repo."""
    import shutil

    shutil.copy(NOATAK_SOURCE, tmp_path / NOATAK_SOURCE.name)
    shutil.copy(TRANSLATION_SOURCE, tmp_path / TRANSLATION_SOURCE.name)

    # The Noatak measurements fall on 2-4 Jul 2024 and 7-9 Sep 2025.
    swot_dates = [
        '2024-07-02', '2024-07-03', '2024-07-04',
        '2025-09-07', '2025-09-08', '2025-09-09',
    ]
    obs_dict = {
        '81340100011': swot_obs(swot_dates),
        '81340100021': swot_obs(swot_dates),
    }

    corridors = Corridors(
        tmp_path, {'basin_id': NOATAK_BASIN}, obs_dict, verbose=True
    )
    result = corridors.integrate_corridors_data()

    assert result is not None
    assert '81340100011' in result
    entry = result['81340100011']
    assert entry['source'] == 'corridors'
    assert entry['Q'].size > 0
    assert np.all(np.isfinite(entry['Q']))

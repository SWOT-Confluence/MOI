"""CORRIDORS field discharge -> integrator pseudo-gages.

CORRIDORS (COmmunity Repository of RIver Discharge Observations foR SWOT,
formerly SWOT SHCQ) publishes in-situ discharge as one CSV "resource" per
contributing PI.  This module reads those CSVs, pairs each field measurement
with the SWOT overpass nearest in time, fits a one-parameter flow law to the
pair, and evaluates that flow law over the whole SWOT record.  The result is
handed to the integrator as an extra gage constraint -- a "CORRIDORS
pseudo-gage" -- through Input.merge_corridors_and_gages().

The CSVs are contributed by different groups and are not uniformly formatted,
so every read is defensive: a resource that cannot be parsed is skipped with a
warning rather than taking down the basin.
"""

import csv
from pathlib import Path
import warnings
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np

from moi.FlowLaws import MWAPN, MWACN, MWHFN
from moi.Domain import Domain
from moi.FlowLawCalibration import FlowLawCalibration


# The SWORD v16 -> v17 lookup lives alongside the resources but is not one.
# CORRIDORS lists reaches by their SWORD v16 id; MOI works in v17.
TRANSLATION_FILE = 'SWORD_v16_v17_translation_reach.csv'

REACH_COLUMN = 'Reach_ID'
TIME_COLUMN = "Time_('dd-mm-yyyy')"
DISCHARGE_COLUMN = 'Q_(m^3/s_daily)'

# A resource missing any of these cannot be used at all.
REQUIRED_COLUMNS = (REACH_COLUMN, TIME_COLUMN, DISCHARGE_COLUMN)

# Resources whose values and headers carry stray double quotes that
# pd.read_csv cannot unpick on its own.  Matched case-insensitively on the
# file name and read by _read_quoted_csv instead.
QUOTED_CSV_FILES = frozenset({
    'usgs_alaska_swot_adcp_datac.csv',   # Conaway, USGS Alaska
})

# CORRIDORS uses -9999 rather than an empty field for "not measured".  Only
# the discharge column is screened, and it is screened on q > 0 rather than on
# this value, so an unflagged zero or negative is caught too.
FILL_VALUE = -9999

# Field measurements carry a calendar date only, so a match is allowed at most
# one day either side of the overpass.  Without a limit merge_asof happily
# pairs a measurement with an overpass months away.
MATCH_TOLERANCE = pd.Timedelta(days=1)

# Below this many matched pairs the one-parameter fit is not worth trusting.
# Three is a floor, not a recommendation: a one-parameter law fitted to three
# points has two degrees of freedom and its in-sample residual says very little
# about how it extrapolates.  Configurable through Corridors_Min_Observations.
MIN_FIT_OBSERVATIONS = 3

# Relative uncertainty floor for a pseudo-gage, matching the default for a real
# station (Integrate.Gage_Uncertainty).  A pseudo-gage that fits its field
# measurements well is therefore weighted like a gage, reproducing the
# behaviour CORRIDORS was first tested with; a poorly fitting one is
# automatically downweighted by its own residual.  See build_pseudo_gage.
MIN_RELATIVE_UNCERTAINTY = 0.10

# CORRIDORS dates are local calendar dates, so the overpass timestamps have to
# be compared in local time.  TODO: pick this from the reach lat/lon instead of
# assuming Alaska -- every resource released so far is Alaskan, but that will
# stop being true.
DEFAULT_TIMEZONE = 'America/Anchorage'


def validate_timezone(name):
    """Reject an unknown IANA zone here rather than per reach, later.

    zoneinfo is standard library and the tzdata package is already a
    requirement, so this needs nothing new.
    """
    try:
        ZoneInfo(str(name))
    except Exception as e:
        raise ValueError(f'Unknown CORRIDORS timezone {name!r}: {e}') from e
    return str(name)


class Corridors:
    """Extracts and formats CORRIDORS data from CSV files."""

    def __init__(self, corridors_dir, basin_dict, obs_dict, verbose=False,
                 timezone=DEFAULT_TIMEZONE,
                 min_observations=MIN_FIT_OBSERVATIONS,
                 min_uncertainty=MIN_RELATIVE_UNCERTAINTY):
        self.corridors_dir = Path(corridors_dir)
        self.basin_dict = basin_dict
        self.obs_dict = obs_dict
        self.verbose = verbose
        self.timezone = validate_timezone(timezone)
        self.min_observations = int(min_observations)
        self.min_uncertainty = float(min_uncertainty)
        self.corridors_dict = {}
        self.corridors_df = None
        self.rids_in_basin = []
        # v16 reaches that SWORD splits into several v17 reaches; recorded so
        # integrate_corridors_data can report them once.
        self.ambiguous_v16_reaches = []

    def integrate_corridors_data(self):
        """Build the CORRIDORS pseudo-gage for every reach in the basin.

        Returns
        -------
        dict or None
            A dictionary keyed by SWORD v17 reach id, shaped like gage_dict.
            None when this basin has no usable CORRIDORS data at all, which is
            the common case and not an error.
        """
        if self.verbose:
            print(f"  -> Scanning for CORRIDORS CSV files in: {self.corridors_dir}")

        if not self.corridors_dir.is_dir():
            warnings.warn(f'CORRIDORS directory not found: {self.corridors_dir}.')
            return None

        # 1. separate the resources from the SWORD translation table
        csv_files = sorted(self.corridors_dir.glob('*.csv'))
        trans_file = next(
            (f for f in csv_files if f.name == TRANSLATION_FILE), None
        )
        csv_files = [f for f in csv_files if f.name != TRANSLATION_FILE]

        if not csv_files:
            warnings.warn(f'No CORRIDORS CSV files found in {self.corridors_dir}.')
            return None
        if trans_file is None:
            warnings.warn(
                f'SWORD v16-v17 translation file {TRANSLATION_FILE} not found in '
                f'{self.corridors_dir}; CORRIDORS reaches cannot be matched to '
                'SWORD v17 ids.'
            )
            return None

        # 2. read every resource we can and stack them into one frame
        corridors_dfs = self.read_corridors_files(csv_files)
        if not corridors_dfs:
            warnings.warn('No CORRIDORS resource could be read.')
            return None

        try:
            self.corridors_df = pd.concat(corridors_dfs, ignore_index=True)
        except Exception as e:
            # Mismatched columns or dtypes across resources land here.
            warnings.warn(f'Could not combine CORRIDORS resources: {e}')
            return None

        # 3. add sword17 rids via translation
        if not self.add_sword_17_ids(trans_file):
            return None

        # 4. check whether there are any corridors data in this basin
        self.find_corridors_in_basin()
        if not self.rids_in_basin:
            if self.verbose:
                print('  -> No CORRIDORS reaches fall in this basin')
            return None

        # 5. corridors timestamps are shared by every reach, so parse them once
        if not self.prepare_corridors_time():
            return None

        # 6. for each reach, fit flow law and evaluate Q over the SWOT record
        for rid in self.rids_in_basin:
            try:
                entry = self.build_pseudo_gage(rid)
            except Exception as e:
                # One malformed reach must not cost us the other reaches.
                warnings.warn(f'CORRIDORS reach {rid} skipped: {e}')
                continue
            if entry is not None:
                self.corridors_dict[str(rid)] = entry

        if not self.corridors_dict:
            if self.verbose:
                print('  -> No CORRIDORS reach yielded a usable pseudo-gage')
            return None

        if self.verbose:
            print(f'  -> Built {len(self.corridors_dict)} CORRIDORS pseudo-gages: '
                  f'{", ".join(sorted(self.corridors_dict))}')

        return self.corridors_dict

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def read_corridors_files(self, csv_files):
        """Read each resource, skipping any that cannot be parsed."""
        corridors_dfs = []
        for csv_file in csv_files:
            if self.verbose:
                print(f"  -> Processing {csv_file.name}...")
            try:
                df = self.read_corridors_csv(csv_file)
            except Exception as e:
                warnings.warn(f'Error reading {csv_file.name}: {e}')
                continue

            missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
            if missing:
                warnings.warn(
                    f'{csv_file.name} skipped; missing required columns: '
                    f'{", ".join(missing)}'
                )
                continue

            corridors_dfs.append(df)
        return corridors_dfs

    def read_corridors_csv(self, csv_file):
        """Read one CORRIDORS resource into the canonical frame layout.

        Resources on the known-quirky list go straight to the quoted reader.
        Everything else is read normally, and falls back to the quoted reader
        only if the plain read fails to produce the required columns -- that is
        the signature of stray quoting confusing the header.
        """
        if csv_file.name.lower() in QUOTED_CSV_FILES:
            return self.read_quoted_csv(csv_file)

        df = self.normalize_frame(pd.read_csv(csv_file))
        if all(column in df.columns for column in REQUIRED_COLUMNS):
            return df

        if self.verbose:
            print(f'  -> {csv_file.name} has unexpected columns; '
                  're-reading as a quoted CSV')
        return self.read_quoted_csv(csv_file)

    def read_quoted_csv(self, csv_file):
        """Read a resource whose headers and values carry stray double quotes.

        Conaway's USGS Alaska ADCP resource is written with a leading quote on
        the first field of every line and a trailing quote on the last, which
        pd.read_csv cannot unpick because the quotes are unbalanced.  Reading
        with QUOTE_NONE keeps them as literal text so they can be stripped off.
        """
        df = pd.read_csv(csv_file, quoting=csv.QUOTE_NONE)
        if df.empty or not len(df.columns):
            return df

        df.columns = df.columns.str.strip().str.strip('"')
        # Only the outermost fields on each line pick up the stray quotes.
        for position in (0, -1):
            column = df.columns[position]
            stripped = df[column].astype(str).str.strip('"')
            # Stripping forced the column to text; put a numeric column back
            # the way it was, or concat with a plain resource makes it object.
            numeric = pd.to_numeric(stripped, errors='coerce')
            df[column] = stripped if numeric.isna().any() else numeric

        return self.normalize_frame(df)

    def normalize_frame(self, df):
        """Give one resource the column names and dtypes the rest expects.

        Both readers funnel through here so a quoted resource ends up
        indistinguishable from a well-formed one, which is what lets
        pd.concat() stack them without producing object columns.
        """
        df = df.copy()
        df.columns = [str(c).strip().strip('"').strip() for c in df.columns]

        # Reach ids arrive as text from the quoted reader and as int64 from the
        # plain one; concat would make the column object and break the ==
        # comparisons in add_sword_17_ids.
        if REACH_COLUMN in df.columns:
            df[REACH_COLUMN] = pd.to_numeric(
                df[REACH_COLUMN].astype(str).str.strip().str.strip('"\''),
                errors='coerce',
            ).astype('Int64')

        if TIME_COLUMN in df.columns:
            df[TIME_COLUMN] = (
                df[TIME_COLUMN].astype(str).str.strip().str.strip('"\'').str.strip()
            )

        if DISCHARGE_COLUMN in df.columns:
            q = pd.to_numeric(df[DISCHARGE_COLUMN], errors='coerce')
            # Drop every non-positive discharge, not just the -9999 fill.  A
            # zero or a small negative is not a dischargeable value: the flow
            # law cannot reproduce it, and rRMSE divides by the observation, so
            # a single zero makes the fit residual infinite.  Doing it here
            # means create_reach_df drops the rows and the min_observations
            # threshold counts only measurements that can actually be fitted.
            df[DISCHARGE_COLUMN] = q.mask(~(q > 0))

        return df

    def add_sword_17_ids(self, trans_file):
        """Attach the SWORD v17 reach id to every CORRIDORS row.

        Returns False when the translation table is unusable, which makes the
        whole CORRIDORS pass a no-op.
        """
        self.corridors_df['reach_id_17'] = pd.NA
        try:
            transdf = pd.read_csv(trans_file)
        except Exception as e:
            warnings.warn(
                f'Error reading SWORD16-17 translation file {trans_file.name}: {e}'
            )
            return False

        if not {'v16_reach_id', 'v17_reach_id'}.issubset(transdf.columns):
            warnings.warn(
                f'{trans_file.name} does not contain v16_reach_id and '
                'v17_reach_id columns.'
            )
            return False

        pairs = transdf.dropna(subset=['v16_reach_id', 'v17_reach_id'])
        candidates = (
            pairs.groupby('v16_reach_id')['v17_reach_id']
            .apply(lambda s: sorted({int(v) for v in s}))
        )

        rids_16 = list(self.corridors_df[REACH_COLUMN].dropna().unique())
        untranslated = []
        for rid in rids_16:
            try:
                rid17s = candidates.loc[rid]
            except (KeyError, TypeError, ValueError):
                # A v16 reach absent from the table is expected as SWORD
                # evolves; note it and move on.
                untranslated.append(rid)
                continue

            if len(rid17s) > 1:
                # SWORD split this reach.  Picking one is arbitrary and
                # copying the discharge to all of them would invent constraints
                # that violate mass conservation at the confluence, so the
                # reach is dropped until there is a rule for dividing it.
                self.ambiguous_v16_reaches.append((int(rid), rid17s))
                continue

            self.corridors_df.loc[
                self.corridors_df[REACH_COLUMN] == rid, 'reach_id_17'
            ] = rid17s[0]

        if self.ambiguous_v16_reaches:
            detail = '; '.join(
                f'{rid} -> {v17s}' for rid, v17s in self.ambiguous_v16_reaches
            )
            warnings.warn(
                f'{len(self.ambiguous_v16_reaches)} CORRIDORS v16 reach(es) map '
                f'to several SWORD v17 reaches and were skipped, because '
                f'assigning the discharge to any one of them is arbitrary and '
                f'assigning it to all of them would break mass conservation: '
                f'{detail}'
            )

        if untranslated and self.verbose:
            print(f'  -> {len(untranslated)} CORRIDORS v16 reach(es) absent from '
                  f'{TRANSLATION_FILE}: {untranslated}')

        return True

    def find_corridors_in_basin(self):
        """Reaches with CORRIDORS data that also have SWOT observations here."""
        basin_id = str(self.basin_dict.get('basin_id', '')).strip()
        if not basin_id:
            # Every reach id starts with the empty string, so an absent basin
            # id would quietly claim the whole global CORRIDORS record.
            warnings.warn(
                'No basin_id available; cannot select CORRIDORS reaches for '
                'this basin.'
            )
            self.rids_in_basin = []
            return

        rids = self.corridors_df['reach_id_17'].dropna().unique()

        in_basin = sorted(
            {int(rid) for rid in rids if str(int(rid)).startswith(basin_id)}
        )

        # The flow law is fitted against SWOT observations, so a reach without
        # them cannot produce a pseudo-gage however much field data it has.
        self.rids_in_basin = [
            rid for rid in in_basin if str(rid) in self.obs_dict
        ]

        dropped = [rid for rid in in_basin if str(rid) not in self.obs_dict]
        if dropped and self.verbose:
            print(f'  -> {len(dropped)} CORRIDORS reach(es) in basin have no SWOT '
                  f'observations: {dropped}')

    # ------------------------------------------------------------------
    # Time handling
    # ------------------------------------------------------------------

    def prepare_corridors_time(self):
        """Parse the CORRIDORS calendar dates into localized timestamps.

        Shared by every reach, so this runs once rather than per reach as it
        used to.  Returns False when no date could be parsed at all.
        """
        raw_dates = self.corridors_df[TIME_COLUMN].astype(str).str.strip("'\" ")

        parsed = pd.to_datetime(raw_dates, format='%d-%m-%Y', errors='coerce')
        if parsed.isna().all():
            # A resource using another convention: let pandas infer, still
            # reading an ambiguous date as day-first.
            parsed = pd.to_datetime(raw_dates, errors='coerce', dayfirst=True)

        if parsed.isna().all():
            warnings.warn(
                'No CORRIDORS measurement date could be parsed; expected '
                'dd-mm-yyyy.'
            )
            return False

        n_unparsed = int(parsed.isna().sum())
        if n_unparsed:
            warnings.warn(
                f'{n_unparsed} CORRIDORS measurement date(s) could not be '
                'parsed and will be ignored.'
            )

        # Midday local time: the date is all we know, and centring it keeps the
        # one-day match tolerance symmetric about the measurement.
        parsed = parsed + pd.Timedelta(hours=12)
        self.corridors_df['t'] = parsed.dt.tz_localize(
            self.timezone, ambiguous='NaT', nonexistent='NaT'
        )
        return True

    def swot_reach_frame(self, rid):
        """SWOT observations for one reach, with local and UTC timestamps."""
        obs = self.obs_dict[str(rid)]

        fields_to_keep = ['h', 'w', 'S', 'dA']
        swotdf = pd.DataFrame(data={k: obs[k] for k in fields_to_keep})

        # h/w/S/dA are already trimmed by iDelete in Input.extract_swot, but
        # time_str is deliberately kept at full length there, so it has to be
        # trimmed the same way here before the columns will line up.
        time_str = np.asarray(obs['time_str'])
        i_delete = obs.get('iDelete')
        if i_delete is not None and time_str.size != len(swotdf):
            time_str = np.delete(time_str, i_delete, 0)
        if time_str.size != len(swotdf):
            raise ValueError(
                f'SWOT time_str length {time_str.size} does not match '
                f'{len(swotdf)} valid observations'
            )

        swotdf['time_str'] = time_str
        t_utc = pd.to_datetime(swotdf['time_str'], utc=True, errors='coerce')
        # UTC ordinal day is what Integrate.prepare_gage_constraints matches
        # the pseudo-gage against; local time is only for pairing with the
        # CORRIDORS calendar dates below.
        swotdf['t_utc'] = t_utc
        swotdf['t'] = t_utc.dt.tz_convert(self.timezone)
        swotdf['time_str_local'] = swotdf['t'].dt.strftime('%Y-%m-%d %H:%M')

        swotdf = swotdf.dropna(subset=['t'])
        if swotdf.empty:
            raise ValueError('no SWOT observation carries a usable timestamp')

        return swotdf.sort_values('t').reset_index(drop=True)

    def create_reach_df(self, rid):
        """Pair this reach's field measurements with the nearest overpass."""
        swotdf = self.swot_reach_frame(rid)

        reach_rows = self.corridors_df[
            self.corridors_df['reach_id_17'] == int(rid)
        ].dropna(subset=['t', DISCHARGE_COLUMN])

        if reach_rows.empty:
            return swotdf, reach_rows

        # merge_asof needs both sides sorted on the key, the right side
        # included -- swotdf comes back sorted from swot_reach_frame.
        reachdf = pd.merge_asof(
            reach_rows.sort_values('t'),
            swotdf,
            on='t',
            direction='nearest',
            tolerance=MATCH_TOLERANCE,
            suffixes=('_corridors', '_swot'),
        )

        # Rows outside the tolerance come back with the SWOT columns unfilled;
        # they carry no information for the fit.
        reachdf = reachdf.dropna(subset=['h', 'w', 'S', 'dA'])

        # 4 drop unwanted columns
        cols_to_drop = [
            'Node_ID', 'SWORD_Version', REACH_COLUMN, 'X', 'Y',
            'Qu_(m^3/s_daily)', 'WSE_(m)', 'WSEu_(m)', 'W_(m)', 'Wu_(m)',
            'Cross-sectionalArea_(m^2)', 'Cross-sectionalAreau_(m^2)',
            'MaxV_(m/s)', 'MaxVu_(m/s)', 'MeanV_(m/s)', 'MeanVu_(m/s)',
            'MaxD_(m)', 'MaxDu_(m)', 'MeanD_(m)', 'MeanDu_(m)',
        ]
        # errors='ignore': resources do not all carry the same optional columns.
        reachdf = reachdf.drop(columns=cols_to_drop, errors='ignore')

        return swotdf, reachdf

    # ------------------------------------------------------------------
    # Flow law
    # ------------------------------------------------------------------

    def build_pseudo_gage(self, rid):
        """Fit and evaluate the flow law for one reach, or None if unusable."""
        swotdf, reachdf = self.create_reach_df(rid)

        if len(reachdf) < self.min_observations:
            if self.verbose:
                print(f'  -> CORRIDORS reach {rid} has {len(reachdf)} matched '
                      f'measurement(s), fewer than {self.min_observations}; skipped')
            return None

        flow_law_cal = self.fit_flow_law(reachdf)
        Qhat = np.asarray(self.evaluate_flow_law(swotdf, flow_law_cal), dtype=float)
        fit_rrmse = self.fit_relative_rmse(flow_law_cal)

        t_ordinal = swotdf['t_utc'].map(pd.Timestamp.toordinal).to_numpy()
        count = min(Qhat.size, t_ordinal.size)
        Qhat = Qhat[:count]
        t_ordinal = t_ordinal[:count]

        valid = np.isfinite(Qhat) & (Qhat > 0)
        if not np.any(valid):
            if self.verbose:
                print(f'  -> CORRIDORS reach {rid} flow law produced no positive '
                      'discharge; skipped')
            return None

        # A pseudo-gage is a flow law fitted to a handful of field
        # measurements, not a measurement.  Integrate.build_gage_system reads
        # relative_uncertainty straight off this dict, so setting it here is
        # what stops the integrator weighting it like a real station.
        relative_uncertainty = self.min_uncertainty
        if np.isfinite(fit_rrmse):
            relative_uncertainty = max(relative_uncertainty, float(fit_rrmse))

        return {
            'source': 'corridors',
            'station_id': None,
            'station_index': None,
            'reach_id_variable': 'sword_17c',
            't': t_ordinal[valid],
            'Q': Qhat[valid],
            'relative_uncertainty': float(relative_uncertainty),
            'n_corridors_measurements': int(len(reachdf)),
            'corridors_fit_relative_rmse': float(fit_rrmse),
        }

    @staticmethod
    def fit_relative_rmse(flow_law_cal):
        """In-sample relative RMSE of the calibrated flow law, or NaN.

        This is measured on the same few measurements the law was fitted to,
        so it is optimistic and says nothing about extrapolating across the
        SWOT record -- it can only raise the uncertainty above the floor, never
        lower it.
        """
        performance = getattr(flow_law_cal, 'Performance', None)
        value = getattr(performance, 'rRMSE', None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return np.nan
        return value if np.isfinite(value) and value >= 0.0 else np.nan

    def fit_flow_law(self, reachdf):
        # initialize flow law TODO: switch flow laws depending how many observations are available
        #flow_law=MWAPN(
        #flow_law=MWACN(
        flow_law = MWHFN(
            np.array(reachdf['dA']),
            np.array(reachdf['w']),
            np.array(reachdf['S']),
            np.array(reachdf['h'])
        )

        D = Domain({
            'nR': 1,
            'xkm': np.nan,
            'L': np.nan,
            'nt': len(reachdf),
            't': reachdf['t'],
            'dt': np.nan,
        })

        flow_law_cal = FlowLawCalibration(
            D, np.array(reachdf[DISCHARGE_COLUMN]), flow_law
        )
        flow_law_cal.CalibrateReach(verbose=False, suppress_warnings=True)

        return flow_law_cal

    def evaluate_flow_law(self, swotdf, flow_law_cal):
        # initialize flow law TODO: switch flow laws depending how many observations are available
        #flow_law=MWAPN(
        #flow_law=MWACN(
        flow_law = MWHFN(
            np.array(swotdf['dA']),
            np.array(swotdf['w']),
            np.array(swotdf['S']),
            np.array(swotdf['h'])
        )

        return flow_law.CalcQ(flow_law_cal.param_est)

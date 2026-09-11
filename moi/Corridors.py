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

Two input layouts are accepted:

* the raw per-PI resources, in the layout CORRIDORS publishes, together with
  the SWORD v16->v17 translation table.  Every reach id has to be translated
  here, and a v16 reach that SWORD split into several v17 reaches is dropped
  because this module has no way to divide the discharge between them.
* one merged dataset (``corridors_measurements.csv``), where the reach ids are
  already resolved to SWORD v17 -- including the split reaches, which are
  resolved against the measurement coordinates -- and every timestamp is an
  absolute UTC instant.  This is the preferred input: it carries information
  the raw layout cannot express, and it needs no translation table.

The merged layout is detected from its columns.  When one is present the raw
resources in the same directory are ignored, because the merged dataset is
built from them and reading both would count every measurement twice.
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

# The merged dataset's conventional file name.  Detection is by columns, not
# by name, so a renamed file still reads; this is what the repo ships and what
# the documentation points at.  A .gz alongside it reads too.
MERGED_DATASET_FILE = 'corridors_measurements.csv'

# The merged dataset's own column names.  Its reach ids are already SWORD v17
# and its timestamps are already UTC, which is what lets the merged path skip
# both the translation table and the local-time reconstruction.
MERGED_REACH_COLUMN = 'reach_id_v17'
MERGED_TIME_COLUMN = 'time_utc'
MERGED_DISCHARGE_COLUMN = 'q_cms'
MERGED_DISTANCE_COLUMN = 'reach_match_km'

# A file carrying all three is a merged dataset, not a raw resource.
MERGED_REQUIRED_COLUMNS = (
    MERGED_REACH_COLUMN, MERGED_TIME_COLUMN, MERGED_DISCHARGE_COLUMN,
)

# Columns worth carrying through the merge.  Absent ones are skipped, so an
# older merged file still reads.  measurement_id is the stable per-measurement
# key, used to drop a measurement that arrives twice; lon/lat identify the
# contributing station, which matters when one reach carries several.
MERGED_PROVENANCE_COLUMNS = (
    'measurement_id', 'resource', 'data_type', 'time_precision', 'lon', 'lat',
)

# The stable per-measurement key, and the columns that identify one station's
# series within a reach.
MEASUREMENT_ID_COLUMN = 'measurement_id'
SITE_COLUMNS = ('lon', 'lat')

# A resource whose rows are a continuous daily series rather than field
# campaigns.  It is paired with the overpasses differently -- see
# create_reach_df.
DAILY_SERIES_TYPE = 'daily_series'
DATA_TYPE_COLUMN = 'data_type'

# How far a measurement may sit from the reach it was assigned to before the
# record is treated as broken rather than merely imprecise.  The merged
# dataset's 99.9th percentile is 1.8 km, so 2 km excludes only the records
# whose stated reach and coordinates disagree outright -- one of them puts a
# Chilean measurement 2822 km from its reach.  This is error exclusion, not an
# uncertainty penalty: a row past the cap carries no usable position at all.
MAX_REACH_MATCH_KM = 2.0

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

# Matched field measurements below this leave the reach without a pseudo-gage.
# One, by decision: a single measurement scales the flow law to an observed
# discharge, which is a one-point rating and worth having.  Note what it is
# not -- with one point a one-parameter law has no degrees of freedom left, so
# it reproduces that measurement exactly and its in-sample residual is zero by
# construction, saying nothing about the fit.  The residual and both sample
# counts are written to the output as diagnostics for exactly this reason.
# Configurable through Corridors_Min_Observations.
MIN_FIT_OBSERVATIONS = 1

# Relative uncertainty of a pseudo-gage, matching the default for a real
# station (Integrate.Gage_Uncertainty).  Applied as a fixed value: every
# pseudo-gage enters the integrator with this weight whatever its fit residual
# or sample size.  See build_pseudo_gage for why the diagnostics do not feed
# back into it.
MIN_RELATIVE_UNCERTAINTY = 0.10

# What a pseudo-gage records about its own fit.  None of it is acted on: the
# weight is fixed at MIN_RELATIVE_UNCERTAINTY, and these travel to the output
# so the distribution can be measured on a global run before anything is gated
# on it.  Input.merge_corridors_and_gages copies them out of the entry.
CORRIDORS_DIAGNOSTIC_KEYS = (
    'n_corridors_measurements',
    'n_corridors_overpasses',
    'corridors_fit_relative_rmse',
    'relative_uncertainty',
)

# Raw CORRIDORS dates are local calendar dates, so the overpass timestamps have
# to be compared in local time and one zone has to be assumed for every
# resource.  This applies to the raw path only: the merged dataset carries an
# absolute UTC instant per measurement, resolved against that measurement's own
# time zone when it was built, so nothing is assumed there.
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
                 min_uncertainty=MIN_RELATIVE_UNCERTAINTY,
                 max_match_km=MAX_REACH_MATCH_KM):
        self.corridors_dir = Path(corridors_dir)
        self.basin_dict = basin_dict
        self.obs_dict = obs_dict
        self.verbose = verbose
        self.timezone = validate_timezone(timezone)
        self.min_observations = int(min_observations)
        self.min_uncertainty = float(min_uncertainty)
        self.max_match_km = float(max_match_km)
        self.corridors_dict = {}
        self.corridors_df = None
        self.rids_in_basin = []
        # True once a merged dataset has been recognised.  It changes two
        # things: the translation table is not needed, and the measurement
        # timestamps are absolute UTC rather than local calendar dates, so the
        # pairing with the overpasses happens in UTC.
        self.merged_mode = False
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

        # 1. separate the resources from the SWORD translation table.  .csv.gz
        # is accepted so the merged dataset can travel compressed; pandas reads
        # it transparently and a 13 MB table becomes 1.3 MB in the repo.
        csv_files = sorted(
            list(self.corridors_dir.glob('*.csv'))
            + list(self.corridors_dir.glob('*.csv.gz'))
        )
        trans_file = next(
            (f for f in csv_files if f.name == TRANSLATION_FILE), None
        )
        csv_files = [f for f in csv_files if f.name != TRANSLATION_FILE]

        if not csv_files:
            warnings.warn(f'No CORRIDORS CSV files found in {self.corridors_dir}.')
            return None

        # 2. read the measurements into one frame, either merged or raw
        merged_files = [f for f in csv_files if self.is_merged_dataset(f)]
        if merged_files:
            if not self.read_merged_dataset(merged_files, csv_files):
                return None
        else:
            if trans_file is None:
                warnings.warn(
                    f'SWORD v16-v17 translation file {TRANSLATION_FILE} not '
                    f'found in {self.corridors_dir}; CORRIDORS reaches cannot '
                    'be matched to SWORD v17 ids.'
                )
                return None
            if not self.read_raw_resources(csv_files, trans_file):
                return None

        # 3. check whether there are any corridors data in this basin
        self.find_corridors_in_basin()
        if not self.rids_in_basin:
            if self.verbose:
                print('  -> No CORRIDORS reaches fall in this basin')
            return None

        # 4. for each reach, fit flow law and evaluate Q over the SWOT record
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

    def is_merged_dataset(self, csv_file):
        """True when a file carries the merged layout rather than a resource.

        Only the header is read.  A file that cannot be opened at all is not
        called merged; the raw reader reports it, so it is not silently lost.
        """
        try:
            header = pd.read_csv(csv_file, nrows=0)
        except Exception:
            return False
        return all(c in header.columns for c in MERGED_REQUIRED_COLUMNS)

    def read_merged_dataset(self, merged_files, csv_files):
        """Read the merged dataset(s) into corridors_df.

        The raw resources sitting in the same directory are deliberately
        ignored: the merged dataset is built from them, so reading both would
        enter every measurement twice and double the apparent sample size.
        """
        self.merged_mode = True

        ignored = [f for f in csv_files if f not in merged_files]
        if ignored:
            if self.verbose:
                print(f'  -> Merged CORRIDORS dataset found; ignoring '
                      f'{len(ignored)} raw resource(s) it already contains: '
                      f'{", ".join(f.name for f in ignored)}')

        frames = []
        for csv_file in merged_files:
            if self.verbose:
                print(f"  -> Processing merged dataset {csv_file.name}...")
            try:
                frames.append(self.read_merged_csv(csv_file))
            except Exception as e:
                warnings.warn(f'Error reading merged dataset {csv_file.name}: {e}')

        if not frames:
            warnings.warn('No merged CORRIDORS dataset could be read.')
            return False

        if len(merged_files) > 1:
            # Almost always a mistake -- a .csv left beside its own .csv.gz,
            # or two builds of the same dataset.  Reading both would enter
            # every measurement twice, so say so rather than quietly doubling
            # the sample size.
            warnings.warn(
                f'{len(merged_files)} merged CORRIDORS datasets found in '
                f'{self.corridors_dir} and all were read: '
                f'{", ".join(f.name for f in merged_files)}. Keep one.'
            )

        self.corridors_df = pd.concat(frames, ignore_index=True)

        # Belt and braces for the case above: a measurement that arrives twice
        # under the same id is one measurement.
        if MEASUREMENT_ID_COLUMN in self.corridors_df.columns:
            duplicated = self.corridors_df[MEASUREMENT_ID_COLUMN].duplicated()
            if duplicated.any():
                warnings.warn(
                    f'{int(duplicated.sum())} merged CORRIDORS measurement(s) '
                    'appeared more than once and the repeats were dropped.'
                )
                self.corridors_df = self.corridors_df.loc[~duplicated]
            self.corridors_df = self.corridors_df.reset_index(drop=True)

        if self.verbose:
            n_reaches = self.corridors_df['reach_id_17'].dropna().nunique()
            print(f'  -> {len(self.corridors_df)} merged measurement(s) on '
                  f'{n_reaches} SWORD v17 reach(es)')
        return True

    def read_merged_csv(self, csv_file):
        """One merged dataset, reduced to the columns the fit needs.

        The output is deliberately the same shape the raw path produces after
        translation and time parsing -- ``reach_id_17``, ``t`` and the
        discharge column -- so everything downstream is shared.
        """
        df = pd.read_csv(csv_file, low_memory=False)

        out = pd.DataFrame(index=df.index)
        out['reach_id_17'] = pd.to_numeric(
            df[MERGED_REACH_COLUMN], errors='coerce'
        ).astype('Int64')

        # Already absolute instants: a day-precision row was centred on local
        # midday and converted to UTC when the dataset was built, which is the
        # same convention prepare_corridors_time applies to a raw resource.
        out['t'] = pd.to_datetime(df[MERGED_TIME_COLUMN], errors='coerce', utc=True)

        # Screened on q > 0 rather than on the fill value, as normalize_frame
        # does, so an unflagged zero or negative is caught too.
        q = pd.to_numeric(df[MERGED_DISCHARGE_COLUMN], errors='coerce')
        out[DISCHARGE_COLUMN] = q.mask(~(q > 0))

        for column in MERGED_PROVENANCE_COLUMNS:
            if column in df.columns:
                out[column] = df[column]

        # Error exclusion, not an uncertainty penalty: past the cap the stated
        # reach and the coordinates disagree outright, so the row does not say
        # which reach it measured.
        if MERGED_DISTANCE_COLUMN in df.columns:
            km = pd.to_numeric(df[MERGED_DISTANCE_COLUMN], errors='coerce')
            too_far = (km > self.max_match_km).fillna(False)
            if too_far.any():
                warnings.warn(
                    f'{int(too_far.sum())} merged CORRIDORS measurement(s) in '
                    f'{csv_file.name} sit more than {self.max_match_km} km from '
                    'the reach they were assigned to and were dropped as broken '
                    f'records (furthest {km[too_far].max():.1f} km).'
                )
                out = out.loc[~too_far]

        n_unresolved = int(out['reach_id_17'].isna().sum())
        if n_unresolved and self.verbose:
            print(f'  -> {n_unresolved} merged measurement(s) in '
                  f'{csv_file.name} carry no SWORD v17 reach')

        n_undated = int(out['t'].isna().sum())
        if n_undated:
            warnings.warn(
                f'{n_undated} merged CORRIDORS measurement(s) in '
                f'{csv_file.name} carry no usable timestamp and were ignored.'
            )

        return out.reset_index(drop=True)

    def read_raw_resources(self, csv_files, trans_file):
        """Read the raw per-PI resources, translate their ids, parse their time."""
        corridors_dfs = self.read_corridors_files(csv_files)
        if not corridors_dfs:
            warnings.warn('No CORRIDORS resource could be read.')
            return False

        try:
            self.corridors_df = pd.concat(corridors_dfs, ignore_index=True)
        except Exception as e:
            # Mismatched columns or dtypes across resources land here.
            warnings.warn(f'Could not combine CORRIDORS resources: {e}')
            return False

        if not self.add_sword_17_ids(trans_file):
            return False

        # Raw timestamps are shared by every reach, so parse them once.
        return self.prepare_corridors_time()

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
        if self.merged_mode:
            # The merged dataset carries an absolute UTC instant per
            # measurement, so the pairing happens in UTC and the per-resource
            # local time zone -- which varies by contributor and is only
            # guessed at by self.timezone -- never has to be reconstructed.
            swotdf['t'] = t_utc
        else:
            swotdf['t'] = t_utc.dt.tz_convert(self.timezone)
        swotdf['time_str_local'] = swotdf['t'].dt.strftime('%Y-%m-%d %H:%M')

        swotdf = swotdf.dropna(subset=['t'])
        if swotdf.empty:
            raise ValueError('no SWOT observation carries a usable timestamp')

        return swotdf.sort_values('t').reset_index(drop=True)

    def create_reach_df(self, rid):
        """Pair this reach's measurements with the overpasses, by resource type.

        The two resource types need the pairing run in opposite directions.

        A field campaign is a handful of instantaneous measurements, so each
        one asks which overpass it belongs to: several campaign measurements
        around one overpass are several genuine observations of it, and all of
        them belong in the fit.

        A daily series has a value for every day of the record, so asking the
        same question pairs both the day before and the day after an overpass
        with it -- and the same SWOT geometry then enters the flow-law fit
        two or three times over, which really does move the fitted parameter
        (measured at about half a percent on the Rhine).  The question is
        turned round for a daily series: each overpass asks which daily value
        is nearest, so it contributes one.

        "One per overpass" is applied per station, not per reach.  A v17 reach
        can carry more than one contributing station -- reach 23267000091
        carries two Rhine gauges 3.8 km apart -- and those are independent
        measurements of the overpass, not repeats of one.  Collapsing to a
        single value per reach would silently discard one station's whole
        record.
        """
        swotdf = self.swot_reach_frame(rid)

        reach_rows = self.corridors_df[
            self.corridors_df['reach_id_17'] == int(rid)
        ].dropna(subset=['t', DISCHARGE_COLUMN])

        if reach_rows.empty:
            return swotdf, reach_rows

        if DATA_TYPE_COLUMN in reach_rows.columns:
            is_daily = reach_rows[DATA_TYPE_COLUMN] == DAILY_SERIES_TYPE
        else:
            # The raw layout does not say, and every raw resource released so
            # far is a field campaign.
            is_daily = pd.Series(False, index=reach_rows.index)

        field_pairs = self.pair_measurements_to_overpasses(
            reach_rows[~is_daily], swotdf
        )
        daily_pairs = self.pair_overpasses_to_daily_series(
            reach_rows[is_daily], swotdf
        )

        # An overpass measured by a field campaign does not also need the day's
        # mean: the instantaneous measurement is nearer to what SWOT saw.  No
        # reach in the current dataset carries both, so this decides nothing
        # today; it is here so that one which does cannot double-count.
        if not field_pairs.empty and not daily_pairs.empty:
            daily_pairs = daily_pairs[
                ~daily_pairs['t_utc'].isin(field_pairs['t_utc'])
            ]

        reachdf = pd.concat(
            [field_pairs, daily_pairs], ignore_index=True, sort=False
        )
        if reachdf.empty:
            return swotdf, reachdf

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

        return swotdf, reachdf.sort_values('t').reset_index(drop=True)

    def pair_measurements_to_overpasses(self, measurements, swotdf):
        """Each measurement takes the overpass nearest to it.

        Several measurements may land on one overpass, which for a field
        campaign is what should happen.
        """
        if measurements.empty:
            return measurements

        # merge_asof needs both sides sorted on the key, the right side
        # included -- swotdf comes back sorted from swot_reach_frame.
        paired = pd.merge_asof(
            measurements.sort_values('t'),
            swotdf,
            on='t',
            direction='nearest',
            tolerance=MATCH_TOLERANCE,
            suffixes=('_corridors', '_swot'),
        )

        # Measurements outside the tolerance come back with the SWOT columns
        # unfilled; they carry no information for the fit.
        return paired.dropna(subset=['h', 'w', 'S', 'dA'])

    def pair_overpasses_to_daily_series(self, measurements, swotdf):
        """Each overpass takes the daily value nearest to it, per station.

        The merge runs with the overpasses on the left, so an overpass can
        claim at most one value from each station's series rather than a
        station's series claiming an overpass several times over.
        """
        if measurements.empty:
            return measurements

        site_columns = [c for c in SITE_COLUMNS if c in measurements.columns]
        if site_columns:
            series = [rows for _, rows in measurements.groupby(site_columns, dropna=False)]
        else:
            series = [measurements]

        overpasses = swotdf.sort_values('t')
        paired = []
        for station_rows in series:
            # The measurement time moves aside so the merge key can stay 't'
            # on both sides of the concat: 't' is the measurement time in the
            # frame this returns, matching the field-campaign branch.
            matched = pd.merge_asof(
                overpasses,
                station_rows.sort_values('t').rename(columns={'t': 't_measurement'}),
                left_on='t',
                right_on='t_measurement',
                direction='nearest',
                tolerance=MATCH_TOLERANCE,
                suffixes=('_swot', '_corridors'),
            )
            # An overpass with no value within the tolerance comes back with
            # the measurement columns unfilled.
            matched = matched.dropna(subset=['t_measurement', DISCHARGE_COLUMN])
            if not matched.empty:
                matched['t'] = matched['t_measurement']
                paired.append(matched.drop(columns=['t_measurement']))

        if not paired:
            return measurements.iloc[0:0]
        return pd.concat(paired, ignore_index=True, sort=False)

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

        # Matched pairs and distinct overpasses are not the same number.  A
        # daily series contributes both the day before and the day after an
        # overpass within MATCH_TOLERANCE, so one overpass can be paired with
        # several measurements and the same SWOT geometry then enters the fit
        # more than once.  Both counts are recorded so a global run can see
        # which reaches that happened on.
        n_overpasses = (
            int(reachdf['t_utc'].nunique()) if 't_utc' in reachdf else 0
        )

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

        # Fixed, by decision: every pseudo-gage enters the integrator with the
        # same relative uncertainty as a real station, whatever its fit
        # residual or sample size.  The residual is not used to downweight,
        # because it is not comparable across sample sizes -- a one-parameter
        # law fitted to one point has a zero residual by construction, so
        # feeding it back would weight the least-supported pseudo-gages the
        # highest.  It is written to the output as a diagnostic instead, next
        # to both sample counts, so the first global run can measure what those
        # numbers are worth before any of them gates anything.
        relative_uncertainty = self.min_uncertainty

        return {
            'source': 'corridors',
            'station_id': None,
            'station_index': None,
            'reach_id_variable': 'sword_17c',
            't': t_ordinal[valid],
            'Q': Qhat[valid],
            'relative_uncertainty': float(relative_uncertainty),
            'n_corridors_measurements': int(len(reachdf)),
            'n_corridors_overpasses': n_overpasses,
            'corridors_fit_relative_rmse': float(fit_rrmse),
        }

    @staticmethod
    def fit_relative_rmse(flow_law_cal):
        """In-sample relative RMSE of the calibrated flow law, or NaN.

        Recorded, not acted on.  It is measured on the same measurements the
        law was fitted to, so it is optimistic, says nothing about
        extrapolating across the SWOT record, and is not comparable between a
        reach fitted to one measurement and one fitted to twenty.  It reaches
        the output as a diagnostic and never touches the weight.
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

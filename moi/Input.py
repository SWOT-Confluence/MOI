# Standard imports
from pathlib import Path
import csv
import datetime
import json
import warnings
from moi import flowlaw_preprocess

try:
    import geopandas as gpd
except ImportError:  # Optional unless a GeoPackage SWORD input is used.
    gpd = None

# Third-party imports
from netCDF4 import Dataset,chartostring
import numpy as np

# Local imports
from moi.Corridors import CORRIDORS_DIAGNOSTIC_KEYS

class Input:
    """Extracts and stores reach-level FLPE algorithm data.
    
    Attributes
    ----------
    alg_dict: dict
        dictionary of algorithm data stored by algorithm name as numpy arrays
    alg_dir: Path
        path to reach-level FLPE algorithm data
    basin_dict: dict
        dict of reach_ids and SoS file needed to process entire basin of data
    sos_dict: dict
        dictionary of SoS data
    sos_dir: Path
        path to SoS data    
    Methods
    -------
    extract_alg()
        extracts and stores reach-level FLPE algorithm data
    extract_sos()
        extracts and stores SoS data
    __get_ids(self, basin_json):
        Extract reach identifiers and store in basin_dict
    """

    def __init__(self, alg_dir, sos_dir, swot_dir, sword_dir,basin_data,branch,verbose):
        """
        Parameters
        ----------
        alg_dir: Path
            path to reach-level FLPE algorithm data
        sos_dir: Path
            path to SoS data
        swot_dir: Path
            path to SWOT data
        basin_data: dict
            dict of reach_ids and SoS file needed to process entire basin of data
        Branch: str
            either constrained or unconstrained
        """

        self.alg_dict = {
            "busboi": {},
            "hivdi": {},
            "metroman": {},
            "momma": {},
            "sad": {},
            "sic4dvar": {}
        }
        self.basin_dict = basin_data
        self.alg_dir = alg_dir
        self.sos_dict = {}
        self.gage_dict = {}
        self.calval_groups = {}
        # Fit diagnostics for every CORRIDORS pseudo-gage that was built,
        # keyed by reach id: the in-sample residual and the two sample counts.
        # Recorded whether or not the pseudo-gage won, so the output can report
        # them for a reach that also has a real gage.
        self.corridors_diagnostics = {}
        # Reaches whose gage_dict entry is a CORRIDORS pseudo-gage rather than
        # a real station.  Kept separately so downstream consumers can tell the
        # two apart once merge_corridors_and_gages has folded them together.
        self.corridors_reaches = set()
        self.sos_dir = sos_dir
        # FLPE files that existed but could not be read, one entry per
        # algorithm/reach; each was treated as missing for that algorithm.
        self.flpe_read_failures = []
        # How extract_swot prepares observations for the flow laws; must
        # match what offline uses (both are written to the integrator files).
        self.observation_filter = flowlaw_preprocess.DEFAULT_OBSERVATION_FILTER
        self.slope_policy = flowlaw_preprocess.DEFAULT_SLOPE_POLICY
        self.sword_dir = sword_dir
        self.swot_dir = swot_dir
        self.branch = branch
        self.VerboseFlag = verbose

    @staticmethod
    def default_calval_file():
        """Return the basin-stratified Cal/Val CSV shipped with MOI."""
        return (
            Path(__file__).resolve().parents[1]
            / 'CalValSeparation_basin_stratified_v2.csv'
        )

    @staticmethod
    def _filled_array(values, fill=np.nan):
        """Return a normal ndarray from a NetCDF variable or masked array."""
        if hasattr(values, 'filled'):
            if np.issubdtype(np.asarray(values).dtype, np.integer):
                try:
                    if np.isnan(fill):
                        fill = 0
                except TypeError:
                    pass
            values = values.filled(fill)
        return np.asarray(values)

    @staticmethod
    def _decode_time_strings(values, expected_size=None):
        """Decode a SWOT ``time_str`` variable into one string per time step."""
        raw = np.asarray(values)
        if hasattr(values, 'filled'):
            fill = b' ' if raw.dtype.kind == 'S' else ' '
            raw = np.asarray(values.filled(fill))

        # SWOT time strings are normally a two-dimensional S1 character array.
        # netCDF4.chartostring preserves the leading time dimension and joins
        # the trailing character dimension. VLEN strings arrive as 1-D values.
        if raw.ndim >= 2 and raw.dtype.kind in ('S', 'U'):
            try:
                raw = np.asarray(chartostring(raw))
            except (AttributeError, TypeError, ValueError):
                pass
        elif (
            raw.ndim == 1
            and expected_size == 1
            and raw.size != 1
            and raw.dtype.kind in ('S', 'U')
        ):
            raw = raw.reshape(1, -1)

        def decode_one(value):
            pieces = np.asarray(value).ravel() if isinstance(value, np.ndarray) else [value]
            decoded = []
            for piece in pieces:
                if isinstance(piece, (bytes, np.bytes_)):
                    decoded.append(bytes(piece).decode('utf-8', errors='replace'))
                else:
                    decoded.append(str(piece))
            return ''.join(decoded).rstrip('\x00 ')

        if raw.ndim == 0:
            decoded = np.asarray([decode_one(raw.item())], dtype=str)
        elif raw.ndim == 1:
            decoded = np.asarray([decode_one(value) for value in raw], dtype=str)
        else:
            decoded = np.asarray([decode_one(row) for row in raw], dtype=str)

        if expected_size is not None and decoded.size != int(expected_size):
            raise ValueError(
                'SWOT time_str length does not match nt: '
                f'{decoded.size} != {int(expected_size)}'
            )
        return decoded

    @staticmethod
    def _time_strings_from_seconds(times):
        """Build UTC ISO strings when an older SWOT file lacks ``time_str``."""
        epoch = datetime.datetime(2000, 1, 1)
        decoded = []
        for seconds in np.asarray(times, dtype=float).ravel():
            if not np.isfinite(seconds):
                decoded.append('')
                continue
            try:
                value = epoch + datetime.timedelta(seconds=float(seconds))
                decoded.append(value.strftime('%Y-%m-%dT%H:%M:%SZ'))
            except (OverflowError, ValueError):
                decoded.append('')
        return np.asarray(decoded, dtype=str)

    @staticmethod
    def _read_reach_id_json(path):
        """Read a JSON list/dict of reach IDs into a normalized string set."""
        if path is None:
            return set()
        with open(path, 'r') as stream:
            data = json.load(stream)
        values = list(data.values()) if isinstance(data, dict) else data
        if not isinstance(values, (list, tuple, set)):
            raise ValueError(f'Reach-ID JSON must contain a list or dictionary: {path}')
        return {str(int(value)) for value in values}

    @staticmethod
    def _read_calval_groups(
        path,
        reach_id_col='reach_id_v17b',
        group_col='group',
    ):
        """Read reach-level calibration/validation assignments from CSV."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f'Calibration/validation CSV not found: {path}')

        with path.open('r', newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f'Calibration/validation CSV has no header: {path}')

            field_lookup = {
                str(field).strip().lower(): field
                for field in reader.fieldnames
                if field is not None
            }
            requested_reach_col = str(reach_id_col).strip().lower()
            reach_candidates = [
                requested_reach_col,
                'reach_id_v17c',
                'reach_id_v17b',
                'reach_id_v17',
                'reach_id',
            ]
            reach_field = next(
                (field_lookup[name] for name in reach_candidates if name in field_lookup),
                None,
            )
            group_field = field_lookup.get(str(group_col).strip().lower())
            if reach_field is None or group_field is None:
                raise ValueError(
                    f'Calibration/validation CSV must contain {reach_id_col!r} '
                    f'(or a compatible reach-ID column) and {group_col!r}: {path}'
                )

            groups = {}
            for line_number, row in enumerate(reader, start=2):
                raw_reach = row.get(reach_field)
                raw_group = row.get(group_field)
                if raw_reach is None or not str(raw_reach).strip():
                    continue
                try:
                    reach = str(int(str(raw_reach).strip()))
                except ValueError as exc:
                    raise ValueError(
                        f'Invalid reach ID {raw_reach!r} on line {line_number} of {path}'
                    ) from exc

                group = '' if raw_group is None else str(raw_group).strip().lower()
                if not group:
                    raise ValueError(
                        f'Missing calibration/validation group for reach {reach} '
                        f'on line {line_number} of {path}'
                    )
                if reach in groups and groups[reach] != group:
                    raise ValueError(
                        f'Conflicting groups for reach {reach} in {path}: '
                        f'{groups[reach]!r} and {group!r}'
                    )
                groups[reach] = group

        if not groups:
            raise ValueError(f'Calibration/validation CSV contains no reach assignments: {path}')
        return groups

    @staticmethod
    def _station_id(svs_dataset, station_index):
        """Decode the SVS station identifier across known string layouts."""
        try:
            station_id = svs_dataset['station_id']
            if station_id.ndim == 3:
                chars = station_id[:, station_index, 0]
            elif station_id.ndim == 2:
                chars = station_id[:, station_index]
            else:
                chars = station_id[station_index]
            if hasattr(chars, 'filled'):
                chars = chars.filled(b' ')
            pieces = []
            for value in np.asarray(chars).ravel():
                if isinstance(value, bytes):
                    pieces.append(value)
                else:
                    pieces.append(str(value).encode('utf-8'))
            return b''.join(pieces).decode('utf-8', errors='ignore').strip()
        except Exception:
            try:
                return str(svs_dataset['station'][station_index])
            except Exception:
                return str(station_index)

    @staticmethod
    def _svs_ordinal_dates(svs_dataset):
        """Convert the shared SVS date_ymd coordinate to ordinal days."""
        date_ymd = Input._filled_array(svs_dataset['date_ymd'][:])
        if date_ymd.ndim != 2:
            raise ValueError('SVS date_ymd must be a two-dimensional array')
        if date_ymd.shape[0] == 3:
            years, months, days = date_ymd
        elif date_ymd.shape[1] == 3:
            years, months, days = date_ymd.T
        else:
            raise ValueError('SVS date_ymd must have one dimension of length 3')

        ordinals = np.full(years.size, -9999, dtype=np.int64)
        for i, (year, month, day) in enumerate(zip(years, months, days)):
            try:
                ordinals[i] = datetime.date(int(year), int(month), int(day)).toordinal()
            except (TypeError, ValueError, OverflowError):
                pass
        return ordinals

    @staticmethod
    def _resolve_svs_reach_id_variable(svs_dataset, requested):
        candidates = [requested, 'reach_id_v17c', 'reach_id_v17b', 'reach_id_v17', 'reach_id']
        for name in candidates:
            if name and name in svs_dataset.variables:
                return name
        available = ', '.join(sorted(name for name in svs_dataset.variables if 'reach_id' in name))
        raise KeyError(
            f'No compatible SVS reach-ID variable found. Requested {requested!r}; '
            f'available reach-ID variables: {available or "none"}'
        )

    def extract_svs(
        self,
        svs_file,
        reach_id_col='reach_id_v17b',
        include_reach_ids=None,
        exclude_reach_ids=None,
        calval_file=None,
        calval_reach_id_col='reach_id_v17b',
        calibration_group='calibration',
    ):
        """Extract SVS gage time series for reaches in the current basin.

        SVS gages are stored separately from FLPE/SoS observations so that the
        integrator can append them as independent soft constraints rather than
        replacing the FLPE observation at a gaged reach. When a Cal/Val CSV is
        supplied, only reaches assigned to the calibration group are eligible.
        """
        svs_file = Path(svs_file)
        if not svs_file.is_file():
            raise FileNotFoundError(f'SVS file not found: {svs_file}')

        include = self._read_reach_id_json(include_reach_ids) if include_reach_ids else set()
        exclude = self._read_reach_id_json(exclude_reach_ids) if exclude_reach_ids else set()
        basin_reaches = {str(int(reach)) for reach in self.basin_dict['reach_ids_all']}
        self.calval_groups = {}
        if calval_file is not None:
            self.calval_groups = self._read_calval_groups(
                calval_file,
                reach_id_col=calval_reach_id_col,
            )
            calibration_group = str(calibration_group).strip().lower()
            calibration_reaches = {
                reach
                for reach, group in self.calval_groups.items()
                if group == calibration_group
            }
            if not calibration_reaches:
                raise ValueError(
                    f'No reaches assigned to group {calibration_group!r} in {calval_file}'
                )
            basin_reaches &= calibration_reaches
        if include:
            basin_reaches &= include
        basin_reaches -= exclude

        self.gage_dict = {}
        with Dataset(svs_file, 'r') as svs_dataset:
            reach_var_name = self._resolve_svs_reach_id_variable(svs_dataset, reach_id_col)
            svs_reaches = self._filled_array(svs_dataset[reach_var_name][:])
            if svs_reaches.ndim == 1:
                svs_reaches = svs_reaches[:, None]
            if svs_reaches.ndim != 2:
                raise ValueError(f'SVS {reach_var_name} must be one- or two-dimensional')

            q_all = self._filled_array(svs_dataset['Q'][:])
            if q_all.ndim != 2:
                raise ValueError('SVS Q must be a station-by-time array')
            if q_all.shape[0] != svs_reaches.shape[0] and q_all.shape[1] == svs_reaches.shape[0]:
                q_all = q_all.T
            if q_all.shape[0] != svs_reaches.shape[0]:
                raise ValueError('SVS Q station dimension does not match reach-ID station dimension')

            ordinal_dates = self._svs_ordinal_dates(svs_dataset)
            if q_all.shape[1] != ordinal_dates.size:
                raise ValueError('SVS Q time dimension does not match date_ymd')

            for reach in sorted(basin_reaches):
                reach_value = int(reach)
                station_indices = np.where(np.any(svs_reaches == reach_value, axis=1))[0]
                if station_indices.size == 0:
                    continue

                valid_counts = []
                for station_index in station_indices:
                    q = np.asarray(q_all[station_index], dtype=float)
                    valid_counts.append(np.count_nonzero(np.isfinite(q) & (q > 0)))
                station_index = int(station_indices[int(np.argmax(valid_counts))])

                q = np.asarray(q_all[station_index], dtype=float)
                valid = np.isfinite(q) & (q > 0) & (ordinal_dates > 0)
                if not np.any(valid):
                    continue

                self.gage_dict[reach] = {
                    'source': 'SVS',
                    'station_id': self._station_id(svs_dataset, station_index),
                    'station_index': station_index,
                    'reach_id_variable': reach_var_name,
                    't': ordinal_dates[valid].copy(),
                    'Q': q[valid].copy(),
                }
                if self.calval_groups:
                    self.gage_dict[reach]['group'] = self.calval_groups[reach]

        if self.VerboseFlag:
            group_message = ''
            if self.calval_groups:
                n_calibration = sum(
                    group == str(calibration_group).strip().lower()
                    for group in self.calval_groups.values()
                )
                group_message = (
                    f'; Cal/Val CSV contains {n_calibration} calibration reaches '
                    f'out of {len(self.calval_groups)} classified reaches'
                )
            print(
                f'Loaded {len(self.gage_dict)} SVS gages for the basin '
                f'using {reach_var_name} from {svs_file}{group_message}'
            )
        return self.gage_dict

    def merge_corridors_and_gages(self, corridors_dict, override_gage=False):
        """Fold the CORRIDORS pseudo-gages into gage_dict.

        The integrator takes all of its constraints from gage_dict, so a
        CORRIDORS reach has to arrive there to be used.

        ``override_gage`` decides what happens when a reach has both.  The
        default keeps the real station, because it is a rated measurement with
        a continuous record whereas the pseudo-gage is a one-parameter flow law
        fitted to a handful of field measurements and extrapolated across the
        SWOT record.  Set it True to reproduce the original CORRIDORS branch,
        which let the pseudo-gage replace the station.

        ``corridors_reaches`` records every reach a pseudo-gage was built for,
        whether or not it won, so the output can report that CORRIDORS data
        exists for the reach separately from what actually constrained it.

        corridors_dict may be None -- Corridors.integrate_corridors_data()
        returns None for every basin without CORRIDORS data, which is the
        common case.
        """
        if not corridors_dict:
            return self.gage_dict

        n_used = 0
        n_deferred = 0
        for rid, entry in corridors_dict.items():
            rid = str(rid)
            # Availability and fit quality are recorded regardless of
            # precedence, so a reach that keeps its real gage still reports
            # what its pseudo-gage would have been.
            self.corridors_reaches.add(rid)
            self.corridors_diagnostics[rid] = {
                key: entry[key] for key in CORRIDORS_DIAGNOSTIC_KEYS
                if key in entry
            }

            if rid in self.gage_dict and not override_gage:
                n_deferred += 1
                if self.VerboseFlag:
                    print(f'Reach {rid} already has a gage; '
                          'keeping it over the CORRIDORS pseudo-gage')
                continue

            self.gage_dict[rid] = entry
            n_used += 1

        if self.VerboseFlag:
            message = (
                f'Added {n_used} CORRIDORS pseudo-gage(s) to {len(self.gage_dict)} '
                'total gage constraints'
            )
            if n_deferred:
                message += f'; {n_deferred} deferred to an existing gage'
            print(message)

        return self.gage_dict

    def extract_sos(self):
        """Extracts and stores SoS data in sos_dict.
        
        Parameters
        ----------
        """

        sosfile=self.sos_dir.joinpath(self.sos_dir, self.basin_dict['sos'])

        sos_dataset=Dataset(sosfile)
    
        sosreachids=sos_dataset["reaches/reach_id"][:]
        sosQbars=sos_dataset["model/mean_q"][:]
        sosfdc=sos_dataset["model/flow_duration_q"][:]
        if self.branch == 'constrained':
            overwritten_indices=sos_dataset["model/overwritten_indexes"][:]
            overwritten_source=sos_dataset["model/overwritten_source"][:]

        #initialize empty dictionary
        self.sos_dict={}

        #get list of all agencies
        try:
            agencystr=sos_dataset.Gage_Agency
        except:
            agencystr=''

        gage_agencies=agencystr.split(';')

        n_not_found=0
        for reach in self.basin_dict['reach_ids_all']:
            self.sos_dict[reach] = {
                'Qbar': np.nan,
                'q33': np.nan,
                'cal_status': -1,
                'overwritten_indices': np.nan,
                'overwritten_source': ''
            }
            try:
                # find index in the sos data array
                k=np.argwhere(sosreachids==np.int64(reach))
                
                if len(k) == 0:
                    print(f'Reach {reach} not found in SOS. Filled with NaN.')
                    n_not_found += 1
                    continue
                    
                k=k[0,0]
                # assign key data elements
                self.sos_dict[reach]['Qbar']=sosQbars[k]
                self.sos_dict[reach]['q33']=sosfdc[k,13] #probability = .66

                # assign data elements for constrained data
                if self.branch == 'constrained':
                    self.sos_dict[reach]['overwritten_indices']=overwritten_indices[k]
                    raw_chars = overwritten_source[k, :]
                    if hasattr(raw_chars, 'compressed'):
                        raw_chars = raw_chars.compressed()  
                    
                    clean_chars = []
                    for c in raw_chars:
                        if hasattr(c, 'decode'):
                            clean_chars.append(c.decode('utf-8', errors='ignore'))
                        else:
                            clean_chars.append(str(c))
                            
                    source_str = "".join(clean_chars)
                    source_str = source_str.replace("'", "").replace("[", "").replace("]", "").replace(" ", "")
                    self.sos_dict[reach]['overwritten_source'] = source_str.strip('x').strip()
                    # ------------------------------------------------------------------

                    #copy the gage data to this dictionary if it's a constrained reach
                    if (self.sos_dict[reach]['overwritten_indices']==1 and 
                      self.sos_dict[reach]['overwritten_source'] != 'grdc'):

                         # extract agency gage data for each reach in the domain
                         agency=self.sos_dict[reach]['overwritten_source']
                         num_name='num_'+ agency   +'_reaches'
                         num_reaches=sos_dataset[agency].dimensions[num_name].size

                         # determine which index in the sos corresponds to this gage
                         igage=np.nan
                         for i in range(num_reaches):
                             gage_reach=str(sos_dataset[agency][agency + '_reach_id'][i])
                             if gage_reach==reach:
                                 igage=i

                         if not np.isnan(igage):
                             self.sos_dict[reach]['cal_status']=sos_dataset[agency]['CAL'][igage]

                         if not np.isnan(igage) and self.sos_dict[reach]['cal_status']==1:
                             self.sos_dict[reach]['gage']={}
                             self.sos_dict[reach]['gage']['source']=agency
                             self.sos_dict[reach]['gage']['t']=[]
                             self.sos_dict[reach]['gage']['Q']=[]

                             self.sos_dict[reach]['gage']['t']=sos_dataset[agency][agency+'_qt'][igage,:]
                             self.sos_dict[reach]['gage']['Q']=sos_dataset[agency][agency+'_q'][igage,:]
                             
            except Exception as e:
                print(f'Error processing reach {reach}: {str(e)}')
                n_not_found+=1

        sos_dataset.close()


    def extract_sword(self):
        """Extracts and stores SWORD data in sword_dict. (v15/v16/v17 Compatible)"""
        swordfile = self.sword_dir / self.basin_dict['sword']
        sword_dataset = Dataset(swordfile)

        self.sword_dict = {}

        dimfields = ['num_domains', 'num_reaches']
        for field in dimfields:
            self.sword_dict[field] = sword_dataset['reaches'].dimensions[field].size    

        if 'num_orbits' in sword_dataset['reaches'].dimensions:
            self.sword_dict['orbits'] = sword_dataset['reaches'].dimensions['num_orbits'].size
        elif 'orbits' in sword_dataset['reaches'].dimensions:
            self.sword_dict['orbits'] = sword_dataset['reaches'].dimensions['orbits'].size
        else:
            self.sword_dict['orbits'] = 1

        reachfields = ['reach_id', 'facc', 'n_rch_up', 'n_rch_down', 'rch_id_up', 'rch_id_dn', 'swot_obs']
        for field in reachfields:
            self.sword_dict[field] = sword_dataset['reaches/' + field][:]

        # SWORD v17c marks reaches whose flow accumulation was changed by the
        # denoise_v3 correction.  The fill value (-9999) means "unchanged from
        # v17b", not invalid, so retain it as provenance rather than filtering
        # those reaches out.  Older SWORD versions do not have this variable.
        reaches_group = sword_dataset['reaches']
        if 'facc_quality' in reaches_group.variables:
            self.sword_dict['facc_quality'] = self._filled_array(
                reaches_group['facc_quality'][:],
                fill=-9999,
            ).astype(np.int32)

        self.sword_dict['facc'] = self._filled_array(
            self.sword_dict['facc'],
            fill=np.nan,
        ).astype(float)

        try:
            self.sword_dict['swot_orbits'] = sword_dataset['reaches/swot_orbits'][:]
        except Exception:
            self.sword_dict['swot_orbits'] = np.zeros((1, self.sword_dict['num_reaches']))

        try:
            self.sword_dict['width'] = sword_dataset['reaches/width'][:]
        except Exception:
            self.sword_dict['width'] = np.zeros(self.sword_dict['num_reaches'])

        sword_dataset.close()
        
    
    def extract_swot(self):
        self.obs_dict = {}

        for reach in self.basin_dict['reach_ids']:
            reach = str(reach)
            swotfile = self.swot_dir.joinpath(reach + '_SWOT.nc')
            
            # --- NEW DIAGNOSTIC BLOCK ---
            try:
                swot_dataset = Dataset(swotfile)
            except Exception as e:
                # Force print to log so you can see the REAL error
                print(f"FAILED to open {swotfile}. Reason: {e}")
                continue
            # -----------------------------

            nt = swot_dataset.dimensions['nt'].size
            reach_group = swot_dataset['reach']
            obs = self.obs_dict[reach] = {'nt': nt}
            h = reach_group['wse'][0:nt].filled(np.nan)
            w_raw = reach_group['width'][0:nt].filled(np.nan)
            s_raw = reach_group['slope2'][0:nt].filled(np.nan)
            obs['dA'] = reach_group['d_x_area'][0:nt].filled(np.nan)
            obs['t'] = reach_group['time'][0:nt].filled(np.nan)

            # A file without a usable reach/hwfit gets the median observed width.
            area_fit = {'h_break': np.full(4, np.nan), 'fit_coeffs': None}
            if 'hwfit' in reach_group.groups:
                area_fit['h_break'] = reach_group['hwfit']['h_break'][:].filled(np.nan)
                # fit_coeffs[0] holds the slopes, fit_coeffs[1] the intercepts
                area_fit['fit_coeffs'] = reach_group['hwfit']['fit_coeffs'][:].filled(np.nan)

            # Same preprocessing offline applies before evaluating the FLPs.
            prep = flowlaw_preprocess.prepare_flowlaw_inputs(
                h, w_raw, s_raw, area_fit,
                self._quality_variables(reach_group, nt),
                self.observation_filter, self.slope_policy)
            obs['h'] = h
            obs['w'] = prep['width_used']
            obs['S'] = prep['slope_used']
            obs['w_raw'] = w_raw
            obs['S_raw'] = s_raw
            obs['width_source'] = prep['width_source']
            obs['slope_status'] = prep['slope_status']
            obs['flowlaw_valid'] = prep['flowlaw_valid']
            obs['observation_filter_counts'] = {
                key: prep[key] for key in (
                    'n_nrt_bad', 'n_quality_unknown', 'n_outlier',
                    'outlier_test_skipped')}

            if 'time_str' in reach_group.variables:
                obs['time_str'] = self._decode_time_strings(
                    reach_group['time_str'][0:nt],
                    expected_size=nt,
                )
                obs['time_str_source'] = 'SWOT reach/time_str'
            else:
                obs['time_str'] = self._time_strings_from_seconds(obs['t'])
                obs['time_str_source'] = 'derived from SWOT reach/time'

            obs['reach_q'] = reach_group['reach_q'][0:nt].filled(np.nan)
            obs['xovr_cal_q'] = reach_group['xovr_cal_q'][0:nt].filled(np.nan)

            swot_dataset.close()

            # Timesteps with no WSE, width, slope or area at all are dropped
            # from every MOI product (and re-inserted as fill on output).
            iDelete = np.where(
                np.isnan(h) | np.isnan(obs['w']) | np.isnan(s_raw)
                | np.isnan(obs['dA'])
            )
            for key in ('h', 'w', 'S', 'dA', 't', 'w_raw', 'S_raw',
                        'slope_status', 'flowlaw_valid', 'reach_q',
                        'xovr_cal_q'):
                obs[key] = np.delete(obs[key], iDelete, 0)
            obs['iDelete'] = iDelete
            obs['nt'] -= np.shape(iDelete)[1]

            # The flow-law geometry is defined only where offline will also
            # evaluate the flow laws; elsewhere it is NaN, so FLP fitting,
            # parameter bounds and the geometry check see exactly offline's
            # observations and q_flowlaw is NaN where L2 is missing.
            valid = obs['flowlaw_valid']
            for key in ('h', 'w', 'S', 'dA'):
                obs[key] = np.where(valid, obs[key], np.nan)

            corr, status = self.geometry_status(obs['h'], obs['dA'])
            obs['corr_wse_dA'] = corr
            obs['geometry_status'] = status

        # RESTORED CHECK: Triggers if ALL files failed to open or process
        if self.obs_dict == {}:
            raise LookupError('No reaches in basin processed')




    @staticmethod
    def _quality_variables(reach_group, nt):
        """Quality series offline filters on, or None if any is missing."""
        try:
            return {name: np.ma.filled(
                        np.ma.asarray(reach_group[name][0:nt], dtype=float), np.nan)
                    for name in flowlaw_preprocess.QUALITY_VARIABLES}
        except (IndexError, KeyError):
            return None

    GEOMETRY_OK = 'ok'
    GEOMETRY_DA_NOT_INCREASING = 'dA_not_increasing_with_wse'
    GEOMETRY_TOO_FEW_OBS = 'too_few_obs'
    MIN_GEOMETRY_OBS = 3

    @classmethod
    def geometry_status(cls, h, dA):
        """``(corr(wse, dA), status)`` for one reach's valid observations.

        d_x_area is an integral of width over stage, so it has to increase with
        WSE.  A non-positive correlation means the area anomaly is sign-reversed
        or degenerate (for example the Input HWS fallback with a negative
        width), and no flow law driven by dA can reproduce the FLPE dynamics.
        """
        h = np.asarray(h, dtype=float).ravel()
        dA = np.asarray(dA, dtype=float).ravel()
        good = np.isfinite(h) & np.isfinite(dA)
        if good.sum() < cls.MIN_GEOMETRY_OBS:
            return np.nan, cls.GEOMETRY_TOO_FEW_OBS
        if np.std(h[good]) == 0 or np.std(dA[good]) == 0:
            return np.nan, cls.GEOMETRY_DA_NOT_INCREASING
        corr = float(np.corrcoef(h[good], dA[good])[0, 1])
        if not corr > 0:
            return corr, cls.GEOMETRY_DA_NOT_INCREASING
        return corr, cls.GEOMETRY_OK

    def extract_alg(self):
        """Extracts and stores reach-level FLPE algorithm data in alg_dict."""

        reach_ids = self.basin_dict["reach_ids"]
        reach_ids_all = self.basin_dict["reach_ids_all"]
 
        #for r_id in reach_ids:
        for r_id in reach_ids_all:
            if r_id in reach_ids:
                # for observed reaches in the domain
                bb_file = self.alg_dir / "busboi" / f"{r_id}_busboi.nc"
                hv_file = self.alg_dir / "hivdi" / f"{r_id}_hivdi.nc"
                mo_file = self.alg_dir / "momma" / f"{r_id}_momma.nc"
                sd_file = self.alg_dir / "sad" / f"{r_id}_sad.nc"
                sv_file = self.alg_dir / "sic4dvar" / f"{r_id}_sic4dvar.nc"
                #more robust os agnostic approach to finding files
                mm_file = self.alg_dir / "metroman" / f"{r_id}_metroman.nc"

                if not mm_file:
                    mm_file=Path('dir/that/does/not/exist')  #this sets mm_file.exists() to false
                else: 
                    mm_file = Path(mm_file) 

                self.__extract_valid(r_id, bb_file, hv_file, mo_file, sd_file, mm_file, sv_file)

            else:
                #for unobserved reaches
                algs=['busboi','hivdi','metroman','momma','sad','sic4dvar']
                for alg in algs:
                    self.alg_dict[alg][r_id] = {
                        "s1-flpe-exists": False,
                        "qbar": np.nan
                        }

    # Keys each algorithm carries when no usable FLPE file exists.  Identical
    # to what the per-algorithm branches used to build inline.
    FLPE_MISSING_KEYS = {
        "busboi": ("q", "r", "bed", "chainage", "prior_q", "n", "a0"),
        "hivdi": ("q", "alpha", "beta"),
        "momma": ("q", "B", "H", "Save"),
        "sad": ("q", "n", "a0"),
        "metroman": ("q", "na", "x1", "a0"),
        "sic4dvar": ("q_mm", "q", "n", "a0"),
    }

    @staticmethod
    def _optional(read):
        """Value of ``read()``, or NaN when the variable is absent/unreadable."""
        try:
            return read()
        except Exception:
            return np.nan

    @classmethod
    def _read_busboi(cls, ds):
        return {
            "q": np.array(ds["q"]["q"][:].filled(np.nan)),
            "r": cls._optional(lambda: np.array(ds["r"]["mean"][:].filled(np.nan))),
            "bed": cls._optional(lambda: np.array(ds["bed"]["elevation"][:].filled(np.nan))),
            "chainage": cls._optional(lambda: np.array(ds["bed"]["chainage"][:].filled(np.nan))),
            "prior_q": cls._optional(lambda: np.array(ds["prior_q"]["q"][:].filled(np.nan))),
            # placeholders for MOI surrogate BAM-compatible refit
            "n": np.nan,
            "a0": np.nan,
        }

    @staticmethod
    def _read_hivdi(ds):
        return {
            "q": ds["reach"]["Q"][:].filled(np.nan),
            "alpha": ds["reach"]["alpha"][:].filled(np.nan),
            "beta": ds["reach"]["beta"][:].filled(np.nan),
            "a0": ds["reach"]["A0"][:].filled(np.nan),
        }

    @staticmethod
    def _read_momma(ds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                "q": ds["Q"][:].filled(np.nan),
                "B": ds["zero_flow_stage"][:].filled(np.nan),
                "H": ds["bankfull_stage"][:].filled(np.nan),
                "Save": np.nanmean(ds["slope"][:].filled(np.nan)),
            }

    @staticmethod
    def _read_sad(ds):
        return {
            "q": ds["Qa"][:].filled(np.nan),
            "n": ds["n"][:].filled(np.nan),
            "a0": ds["A0"][:].filled(np.nan),
        }

    @staticmethod
    def _read_metroman(ds):
        return {
            "q": ds["average"]["allq"][:].filled(np.nan),
            "na": ds["average"]["nahat"][:].filled(np.nan),
            "x1": ds["average"]["x1hat"][:].filled(np.nan),
            "a0": ds["average"]["A0hat"][:].filled(np.nan),
        }

    @staticmethod
    def _read_sic4dvar(ds):
        return {
            "q_mm": ds["Q_mm"][:].filled(np.nan),
            "q": ds["Q_da"][:].filled(np.nan),
            "n": ds["n"][:].filled(np.nan),
            "a0": ds["A0"][:].filled(np.nan),
        }

    def _missing_flpe(self, alg, r_id):
        entry = {"s1-flpe-exists": False}
        entry.update({key: np.nan for key in self.FLPE_MISSING_KEYS[alg]})
        entry["qbar"] = self.sos_dict[str(r_id)]['Qbar']
        entry["q33"] = self.sos_dict[str(r_id)]['q33']
        return entry

    def _extract_one_flpe(self, alg, r_id, path, reader):
        """One algorithm's FLPE record for one reach.

        A file that exists but cannot be read (for example an H2iVDI failure
        file with no ``reach`` group) is treated exactly like a missing file
        for that algorithm only.  Previously the exception escaped
        extract_alg() and run_MOI skipped the whole basin, so enabling one
        algorithm could remove every other algorithm's MOI output.
        """
        if not path.exists():
            return self._missing_flpe(alg, r_id)
        try:
            with Dataset(path, 'r', format="NETCDF4") as ds:
                entry = reader(ds)
        except Exception as exc:
            self.flpe_read_failures.append(
                {"algorithm": alg, "reach_id": str(r_id), "reason": repr(exc)})
            warnings.warn(f"Unreadable {alg} FLPE file {path}: {exc!r}; "
                          f"treating {alg} as missing for reach {r_id}")
            return self._missing_flpe(alg, r_id)
        return {"s1-flpe-exists": True, **entry}

    def __extract_valid(self, r_id, bb_file, hv_file, mo_file, sd_file, mm_file, sv_file):
        """Extract valid data from the output of each reach-level FLPE alg."""
        for alg, path, reader in (
            ("busboi", bb_file, self._read_busboi),
            ("hivdi", hv_file, self._read_hivdi),
            ("momma", mo_file, self._read_momma),
            ("sad", sd_file, self._read_sad),
            ("metroman", mm_file, self._read_metroman),
            ("sic4dvar", sv_file, self._read_sic4dvar),
        ):
            self.alg_dict[alg][r_id] = self._extract_one_flpe(alg, r_id, path, reader)


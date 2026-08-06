# Standard imports
from glob import glob
from pathlib import Path
import csv
import datetime
import json
import warnings
import os
import sys

try:
    import geopandas as gpd
except ImportError:  # Optional unless a GeoPackage SWORD input is used.
    gpd = None

# Third-party imports
from netCDF4 import Dataset,chartostring
import numpy as np

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
        self.sos_dir = sos_dir
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
        
    
    def extract_sword_gpkg(self):
        swordfile = self.sword_dir.joinpath(self.basin_dict['sword']) 

        if gpd is None:
            raise RuntimeError('geopandas is required to read a SWORD GeoPackage')
        try:
            gdf = gpd.read_file(swordfile)
        except Exception as e:
            raise RuntimeError(f"Failed to read SWORD gpkg file: {swordfile}. Error: {e}")

        self.sword_dict = {}

        self.sword_dict['num_reaches'] = len(gdf)

        self.sword_dict['reach_id'] = gdf['reach_id'].values
        self.sword_dict['facc'] = gdf['facc'].to_numpy(dtype=float, na_value=np.nan)
        self.sword_dict['n_rch_up'] = gdf['n_rch_up'].values
        self.sword_dict['n_rch_down'] = gdf['n_rch_down'].values
        self.sword_dict['swot_obs'] = gdf['swot_obs'].values

        if 'facc_quality' in gdf.columns:
            self.sword_dict['facc_quality'] = (
                gdf['facc_quality'].fillna(-9999).astype(np.int32).values
            )

        self.sword_dict['width'] = gdf['width'].fillna(0).values 

        rch_id_up = np.zeros((4, len(gdf)), dtype=np.int64)
        rch_id_dn = np.zeros((4, len(gdf)), dtype=np.int64)

        for i in range(1, 5):
            up_col = f'rch_id_up_{i}'
            dn_col = f'rch_id_dn_{i}'

            if up_col in gdf.columns:
                rch_id_up[i-1, :] = gdf[up_col].fillna(0).astype(np.int64).values
            if dn_col in gdf.columns:
                rch_id_dn[i-1, :] = gdf[dn_col].fillna(0).astype(np.int64).values

        self.sword_dict['rch_id_up'] = rch_id_up
        self.sword_dict['rch_id_dn'] = rch_id_dn

        self.sword_dict['swot_orbits'] = np.zeros((1, len(gdf)))
        
        
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

            self.obs_dict[reach] = {}
            nt = swot_dataset.dimensions['nt'].size
            self.obs_dict[reach]['nt'] = nt
            self.obs_dict[reach]['h'] = swot_dataset["reach/wse"][0:nt].filled(np.nan)
            self.obs_dict[reach]['w'] = swot_dataset["reach/width"][0:nt].filled(np.nan)
            self.obs_dict[reach]['S'] = swot_dataset["reach/slope2"][0:nt].filled(np.nan)
            self.obs_dict[reach]['dA'] = swot_dataset["reach/d_x_area"][0:nt].filled(np.nan)
            self.obs_dict[reach]['t'] = swot_dataset["reach/time"][0:nt].filled(np.nan)

            self.obs_dict[reach]['reach_q'] = swot_dataset["reach/reach_q"][0:nt].filled(np.nan)
            self.obs_dict[reach]['xovr_cal_q'] = swot_dataset["reach/xovr_cal_q"][0:nt].filled(np.nan)

            swot_dataset.close()

            # select observations that are NOT equal to the fill value
            iDelete = np.where(np.isnan(self.obs_dict[reach]['h']) | \
                               np.isnan(self.obs_dict[reach]['w']) | \
                               np.isnan(self.obs_dict[reach]['S']) | \
                               np.isnan(self.obs_dict[reach]['dA'])| \
                               (self.obs_dict[reach]['reach_q'] > 1) | \
                               (self.obs_dict[reach]['xovr_cal_q'] > 1) )

            self.obs_dict[reach]['h'] = np.delete(self.obs_dict[reach]['h'], iDelete, 0)
            self.obs_dict[reach]['w'] = np.delete(self.obs_dict[reach]['w'], iDelete, 0)
            self.obs_dict[reach]['S'] = np.delete(self.obs_dict[reach]['S'], iDelete, 0)
            self.obs_dict[reach]['dA'] = np.delete(self.obs_dict[reach]['dA'], iDelete, 0)
            self.obs_dict[reach]['t'] = np.delete(self.obs_dict[reach]['t'], iDelete, 0)

            self.obs_dict[reach]['iDelete'] = iDelete

            Smin = 1.7e-5
            np.putmask(
                self.obs_dict[reach]['S'],
                self.obs_dict[reach]['S'] < Smin,
                Smin,
            )

            shape_iDelete = np.shape(iDelete)
            nDelete = shape_iDelete[1]
            self.obs_dict[reach]['nt'] -= nDelete
            
        # RESTORED CHECK: Triggers if ALL files failed to open or process
        if self.obs_dict == {}:
            raise LookupError('No reaches in basin processed')




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

    def __extract_valid(self, r_id, bb_file, hv_file, mo_file, sd_file, mm_file, sv_file):
        """ Extract valid data from the output of each reach-level FLPE alg.
        Parameters
        ----------
        r_id: str
            Unique reach identifier
        bb_file: Path
            Path to BUSBOI results file
        hv_file: Path
            Path to HiVDI results file
        mo_file: Path
            Path to MOMMA results file
        sd_file: Path
            Path to SAD results file
        mm_file: Path
            Path to MetroMan results file
        sv_file: Path
            Path to SIC4DVar results file
        """

        # busboi
        if bb_file.exists():
            bb = Dataset(bb_file, 'r', format="NETCDF4")
            try:
                q = np.array(bb["q"]["q"][:].filled(np.nan))
            except Exception:
                q = np.nan

            try:
                r = np.array(bb["r"]["mean"][:].filled(np.nan))
            except Exception:
                try:
                    r = np.array(bb["r"]["mean"].getValue())
                except Exception:
                    r = np.nan

            try:
                bed = np.array(bb["bed"]["elevation"][:].filled(np.nan))
            except Exception:
                bed = np.nan

            try:
                chainage = np.array(bb["bed"]["chainage"][:].filled(np.nan))
            except Exception:
                chainage = np.nan

            try:
                prior_q = np.array(bb["prior_q"]["q"][:].filled(np.nan))
            except Exception:
                prior_q = np.nan

            self.alg_dict["busboi"][r_id] = {
                "s1-flpe-exists": True,
                "q": q,
                "r": r,
                "bed": bed,
                "chainage": chainage,
                "prior_q": prior_q,
                # placeholders for MOI surrogate BAM-compatible refit
                "n": np.nan,
                "a0": np.nan
            }
            bb.close()

        else:
            self.alg_dict["busboi"][r_id] = { 
                "s1-flpe-exists" : False ,
                "q" : np.nan,
                "r" : np.nan,
                "bed" : np.nan,
                "chainage" : np.nan,
                "prior_q" : np.nan,
                "n" : np.nan,
                "a0" : np.nan,
                "qbar" : self.sos_dict[str(r_id)]['Qbar'],
                "q33" : self.sos_dict[str(r_id)]['q33']
            }

        # hivdi
        if hv_file.exists():
            hv = Dataset(hv_file, 'r', format="NETCDF4")
            self.alg_dict["hivdi"][r_id] = {
                "s1-flpe-exists": True,
                "q" : hv["reach"]["Q"][:].filled(np.nan),
                "alpha" : hv["reach"]["alpha"][:].filled(np.nan),  
                "beta" : hv["reach"]["beta"][:].filled(np.nan),  
                "a0" : hv["reach"]["A0"][:].filled(np.nan)
            }
            hv.close()
        else:
            self.alg_dict["hivdi"][r_id] = { 
                "s1-flpe-exists" : False ,
                "q" : np.nan,
                "alpha" : np.nan,
                "beta" : np.nan,
                "qbar" : self.sos_dict[str(r_id)]['Qbar'],
                "q33" : self.sos_dict[str(r_id)]['q33']
            }

        # momma
        if mo_file.exists():
            mo = Dataset(mo_file, 'r', format="NETCDF4")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                self.alg_dict["momma"][r_id] = {
                    "s1-flpe-exists": True,
                    "q" : mo["Q"][:].filled(np.nan),
                    "B" : mo["zero_flow_stage"][:].filled(np.nan),
                    "H" : mo["bankfull_stage"][:].filled(np.nan),                                  
                    "Save" : np.nanmean(mo["slope"][:].filled(np.nan))
                }
            mo.close()
        else:
            self.alg_dict["momma"][r_id] = { 
                "s1-flpe-exists" : False ,
                "q" : np.nan,
                "B" : np.nan,
                "H" : np.nan,
                "Save" : np.nan,
                "qbar" : self.sos_dict[str(r_id)]['Qbar'],
                "q33" : self.sos_dict[str(r_id)]['q33']
            }

        # sad
        if sd_file.exists():
            sd = Dataset(sd_file, 'r', format="NETCDF4")
            self.alg_dict["sad"][r_id] = {
                "s1-flpe-exists": True,
                "q" : sd["Qa"][:].filled(np.nan),
                "n" : sd["n"][:].filled(np.nan),
                "a0" : sd["A0"][:].filled(np.nan)
            }
            sd.close()
        else:
            self.alg_dict["sad"][r_id] = { 
                "s1-flpe-exists" : False ,
                "q" : np.nan,
                "n" : np.nan,
                "a0" : np.nan,
                "qbar" : self.sos_dict[str(r_id)]['Qbar'],
                "q33" : self.sos_dict[str(r_id)]['q33']
            }

        # metroman    
        if mm_file.exists():
            mm = Dataset(mm_file, 'r', format="NETCDF4")
            # index = np.where(mm["reach_id"][:] == int(r_id))
            self.alg_dict["metroman"][r_id] = {
                 "s1-flpe-exists": True,
                 "q" : mm["average"]["allq"][:].filled(np.nan),
                 "na" : mm["average"]["nahat"][:].filled(np.nan),
                 "x1" : mm["average"]["x1hat"][:].filled(np.nan),
                 "a0" : mm["average"]["A0hat"][:].filled(np.nan)
            }
            mm.close()
            #print('MetroMan file found. ')
        else:
            self.alg_dict["metroman"][r_id] = { 
                "s1-flpe-exists" : False ,
                "q" : np.nan,
                "na" : np.nan,
                "x1" : np.nan,
                "a0" : np.nan,
                "qbar" : self.sos_dict[str(r_id)]['Qbar'],
                "q33" : self.sos_dict[str(r_id)]['q33']
            }
            #print('MetroMan file not found. Using prior')

        # sic4dvar
        if sv_file.exists():
            sv = Dataset(sv_file, 'r', format="NETCDF4")
            self.alg_dict["sic4dvar"][r_id] = {
                #"q31": sv["Qalgo31"][:].filled(np.nan),#unclear which of these to use
                "s1-flpe-exists": True,
                "q_mm": sv["Q_mm"][:].filled(np.nan),
                "q": sv["Q_da"][:].filled(np.nan),
                # "q5": sv["Qalgo5"][:].filled(np.nan),
                "n": sv["n"][:].filled(np.nan),
                "a0": sv["A0"][:].filled(np.nan)
            }
            sv.close()
        else:
            self.alg_dict["sic4dvar"][r_id] = { 
                "s1-flpe-exists" : False ,
                "q_mm": np.nan,
                "q": np.nan,
                "n" : np.nan,
                "a0" : np.nan,
                "qbar" : self.sos_dict[str(r_id)]['Qbar'],
                "q33" : self.sos_dict[str(r_id)]['q33']
            }

    def __indicate_no_data(self, r_id):
        """Indicate no data is available for the reach.
        TODO: Metroman results
        Parameters
        ----------
        r_id: str
            Unique reach identifier
        """

        self.alg_dict["busboi"][r_id] = {
            "q": np.nan,
            "r": np.nan,
            "bed": np.nan,
            "chainage": np.nan,
            "prior_q": np.nan,
            "n": np.nan,
            "a0": np.nan
        }

        # hivdi
        self.alg_dict["hivdi"][r_id] = {
            "q" : np.nan,
            "alpha" : np.nan,  
            "beta" : np.nan,  
            "a0" : np.nan
        }

        # momma
        self.alg_dict["momma"][r_id] = {
            "q" : np.nan,
            "B" : np.nan,
            "H" : np.nan,
            "Save" : np.nan
        }

        # sad
        self.alg_dict["sad"][r_id] = {
            "q" : np.nan,
            "n" : np.nan,
            "a0" : np.nan
        }

        # metroman    
        self.alg_dict["metroman"][r_id] = {
             "q" : np.nan,
             "na" : np.nan,
             "x1" : np.nan,
             "a0" : np.nan
        }

        # sic4dvar
        self.alg_dict["sic4dvar"][r_id] = {
            "q_mm": np.nan,
            "q": np.nan,
            "n": np.nan,
            "a0": np.nan
        }

    def __get_gb_data(self, gb,group, pre, logged):
        """Return legacy neoBAM data as a numpy array.
        
        Parameters
        ----------
        gb: netCDF4.Dataset
            NetCDF file dataset to extract discharge time series
        group: str
            string name of group to access chains
        pre: str
            string prefix of variable name
        logged: bool
            boolean indicating if result is logged
        """

        variables = gb[group].variables
        if pre in variables:
            q = variables[pre][:]
            q = q.filled(np.nan) if hasattr(q, 'filled') else np.asarray(q)
        else:
            chain_names = [
                f'{pre}{chain_number}'
                for chain_number in range(1, 4)
                if f'{pre}{chain_number}' in variables
            ]
            if not chain_names:
                raise KeyError(
                    f'No variable {pre!r} or chain variables {pre!r}1..3 '
                    f'in NetCDF group {group!r}'
                )
            chains = []
            for name in chain_names:
                values = variables[name][:]
                if hasattr(values, 'filled'):
                    values = values.filled(np.nan)
                chains.append(np.asarray(values, dtype=float))
            q = np.vstack(chains)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_q = np.nanmean(q, axis=0) if np.ndim(q) > 1 else np.asarray(q)
            if logged:
                return np.exp(mean_q)
            else:
                return mean_q

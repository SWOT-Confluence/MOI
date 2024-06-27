# Standard imports
from glob import glob
from pathlib import Path
import warnings
import os
import sys

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
            "neobam": {},
            "hivdi": {},
            "metroman": {},
            "momma": {},
            "sad": {},
            "sic4dvar": {}
        }
        self.basin_dict = basin_data
        self.alg_dir = alg_dir
        self.sos_dict = {}
        self.sos_dir = sos_dir
        self.sword_dir = sword_dir
        self.swot_dir = swot_dir
        self.branch = branch
        self.VerboseFlag = verbose

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
            try:
                # initialize reach dictionary
                self.sos_dict[reach]={}
                # find index in the sos data array
                k=np.argwhere(sosreachids==np.int64(reach))
                k=k[0,0]
                # assign key data elements
                self.sos_dict[reach]['Qbar']=sosQbars[k]
                self.sos_dict[reach]['q33']=sosfdc[k,13] #probability = .66
                self.sos_dict[reach]['cal_status']=-1 

                # assign data elements for constrained data
                if self.branch == 'constrained':
                    self.sos_dict[reach]['overwritten_indices']=overwritten_indices[k]
                    source_str=str(chartostring(overwritten_source[k,:]))
                    self.sos_dict[reach]['overwritten_source']=source_str.strip('x')


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
                             #cal_status:
                             #  ungaged: -1
                             #  validation: 0
                             #  calibration: 1
                             #  historical: 2
                             self.sos_dict[reach]['cal_status']=sos_dataset[agency]['CAL'][igage]

                         """
                         if reach=='73120000131':
                             print('Input - got to reach 73120000131')
                             print(self.sos_dict[reach]['overwritten_source'])
                             print('cal_status=',cal_status)
                             sys.exit('stopping at dev point')
                         """

                         if not np.isnan(igage) and self.sos_dict[reach]['cal_status']==1:
                             self.sos_dict[reach]['gage']={}
                             self.sos_dict[reach]['gage']['source']=agency
                             self.sos_dict[reach]['gage']['t']=[]
                             self.sos_dict[reach]['gage']['Q']=[]

                             self.sos_dict[reach]['gage']['t']=sos_dataset[agency][agency+'_qt'][igage,:]
                             self.sos_dict[reach]['gage']['Q']=sos_dataset[agency][agency+'_q'][igage,:]
                             #print('for reach',reach,'c/v=',sos_dataset[agency]['CAL'][igage])
                             #sys.exit('stopping at dev point')
                         
                else:
                    self.sos_dict[reach]['overwritten_indices']=np.nan
            except Exception as e:
                #print(e)
                print(f'reach data not found for {reach}')
                n_not_found+=1


        sos_dataset.close()

        #print('A total of ',n_not_found,' data not found')


    def extract_sword(self):
        """Extracts and stores SWORD data in sword_dict.
        
        Parameters
        ----------
        """
        swordfile=self.sword_dir.joinpath(self.sword_dir, self.basin_dict['sword'])
        sword_dataset=Dataset(swordfile)

        self.sword_dict={} #organized by field rather than by reaches

        # grab sizes of the data
        dimfields=['orbits','num_domains','num_reaches']
        for field in dimfields:
            self.sword_dict[field]=sword_dataset['reaches'].dimensions[field].size    

        # grab data    
        reachfields=['reach_id','facc','n_rch_up','n_rch_down','rch_id_up','rch_id_dn','swot_obs','swot_orbits']
        for field in reachfields:
            self.sword_dict[field]=sword_dataset['reaches/' + field][:]
 
        sword_dataset.close()

    def extract_swot(self):
 
        self.obs_dict={}

        for reach in self.basin_dict['reach_ids']:
             swotfile=self.swot_dir.joinpath(reach+'_SWOT.nc')
             try:
                swot_dataset = Dataset(swotfile)
                # if self.VerboseFlag:
                #    print(f'swot file found for {reach}')
             except:
                if self.VerboseFlag:
                    print(f'swot file not found for {reach}')
                continue

             self.obs_dict[reach]={}
             nt = swot_dataset.dimensions['nt'].size
             self.obs_dict[reach]['nt']=nt
             self.obs_dict[reach]['h']=swot_dataset["reach/wse"][0:nt].filled(np.nan)
             self.obs_dict[reach]['w']=swot_dataset["reach/width"][0:nt].filled(np.nan)
             self.obs_dict[reach]['S']=swot_dataset["reach/slope2"][0:nt].filled(np.nan)
             self.obs_dict[reach]['dA']=swot_dataset["reach/d_x_area"][0:nt].filled(np.nan)
             self.obs_dict[reach]['t']=swot_dataset["reach/time"][0:nt].filled(np.nan)


             self.obs_dict[reach]['reach_q']=swot_dataset["reach/reach_q"][0:nt].filled(np.nan)
             self.obs_dict[reach]['xovr_cal_q']=swot_dataset["reach/xovr_cal_q"][0:nt].filled(np.nan)

             #print(self.obs_dict[reach]['xovr_cal_q']>0)
             #print(self.obs_dict[reach]['reach_q']>0)
             #print(np.isnan(self.obs_dict[reach]['dA']))

             swot_dataset.close()

             #select observations that are NOT equal to the fill value
             iDelete=np.where(np.isnan(self.obs_dict[reach]['h']) | \
                              np.isnan(self.obs_dict[reach]['w']) | \
                              np.isnan(self.obs_dict[reach]['S']) | \
                              np.isnan(self.obs_dict[reach]['dA'])| \
                              (self.obs_dict[reach]['reach_q'] > 1) | \
                              (self.obs_dict[reach]['xovr_cal_q'] > 1) )

             self.obs_dict[reach]['h']=np.delete(self.obs_dict[reach]['h'],iDelete,0)
             self.obs_dict[reach]['w']=np.delete(self.obs_dict[reach]['w'],iDelete,0)
             self.obs_dict[reach]['S']=np.delete(self.obs_dict[reach]['S'],iDelete,0)
             self.obs_dict[reach]['dA']=np.delete(self.obs_dict[reach]['dA'],iDelete,0)
             self.obs_dict[reach]['t']=np.delete(self.obs_dict[reach]['t'],iDelete,0)


             self.obs_dict[reach]['iDelete']=iDelete

             Smin=1.7e-5
             self.obs_dict[reach]['S'][self.obs_dict[reach]['S']<Smin]=\
                np.putmask(self.obs_dict[reach]['S'],self.obs_dict[reach]['S']<Smin,Smin)

             #Obs.S[Obs.S<Smin]=putmask(Obs.S,Obs.S<Smin,Smin) #limit slopes to a minimum value

             #sys.exit('stopping at dev point')

             shape_iDelete=np.shape(iDelete)
             nDelete=shape_iDelete[1]
             self.obs_dict[reach]['nt'] -= nDelete
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
                gb_file = self.alg_dir / "geobam" / f"{r_id}_geobam.nc"
                hv_file = self.alg_dir / "hivdi" / f"{r_id}_hivdi.nc"
                mo_file = self.alg_dir / "momma" / f"{r_id}_momma.nc"
                sd_file = self.alg_dir / "sad" / f"{r_id}_sad.nc"
                sv_file = self.alg_dir / "sic4dvar" / f"{r_id}_sic4dvar.nc"
                #more robust os agnostic approach to finding files
                mm_file = glob(os.path.join(self.alg_dir,"metroman", f"*{r_id}*_metroman.nc"))

                if not mm_file:
                    mm_file=Path('dir/that/does/not/exist')  #this sets mm_file.exists() to false
                else: 
                    mm_file = Path(mm_file[0]) 

                self.__extract_valid(r_id, gb_file, hv_file, mo_file, sd_file, mm_file, sv_file)

            else:
                #for unobserved reaches
                algs=['neobam','hivdi','metroman','momma','sad','sic4dvar']
                for alg in algs:
                    self.alg_dict[alg][r_id] = {
                        "s1-flpe-exists": False,
                        "qbar": np.nan
                        }

    def __extract_valid(self, r_id, gb_file, hv_file, mo_file, sd_file, mm_file, sv_file):
        """ Extract valid data from the output of each reach-level FLPE alg.
        Parameters
        ----------
        r_id: str
            Unique reach identifier
        gb_file: Path
            Path to neoBAM results file
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

        # neobam
        if gb_file.exists():
            #print('reading',gb_file)
            gb = Dataset(gb_file, 'r', format="NETCDF4")
            self.alg_dict["neobam"][r_id] = {
                "s1-flpe-exists": True,
                "q": np.array(self.__get_gb_data(gb,"q", "q", False)),
                "n": np.array(self.__get_gb_data(gb,"logn", "mean", True)),
                "a0": 1.0    # TODO temp value until work out neoBAM A0
            }
            gb.close()

        else:
            self.alg_dict["neobam"][r_id] = { 
                "s1-flpe-exists" : False ,
                "q" : np.nan,
                "n" : np.nan,
                "a0" : np.nan,
                "qbar" : self.sos_dict[r_id]['Qbar'],
                "q33" : self.sos_dict[r_id]['q33']
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
                "qbar" : self.sos_dict[r_id]['Qbar'],
                "q33" : self.sos_dict[r_id]['q33']
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
                "qbar" : self.sos_dict[r_id]['Qbar'],
                "q33" : self.sos_dict[r_id]['q33']
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
                "qbar" : self.sos_dict[r_id]['Qbar'],
                "q33" : self.sos_dict[r_id]['q33']
            }

        # metroman    
        if mm_file.exists():
            mm = Dataset(mm_file, 'r', format="NETCDF4")
            index = np.where(mm["reach_id"][:] == int(r_id))
            self.alg_dict["metroman"][r_id] = {
                 "s1-flpe-exists": True,
                 "q" : mm["allq"][index].filled(np.nan),
                 "na" : mm["nahat"][index].filled(np.nan),
                 "x1" : mm["x1hat"][index].filled(np.nan),
                 "a0" : mm["A0hat"][index].filled(np.nan)
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
                "qbar" : self.sos_dict[r_id]['Qbar'],
                "q33" : self.sos_dict[r_id]['q33']
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
                "qbar" : self.sos_dict[r_id]['Qbar'],
                "q33" : self.sos_dict[r_id]['q33']
            }

    def __indicate_no_data(self, r_id):
        """Indicate no data is available for the reach.
        TODO: Metroman results
        Parameters
        ----------
        r_id: str
            Unique reach identifier
        """

        self.alg_dict["neobam"][r_id] = {
            "q": np.nan,
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
        """Return neoBAM data as a numpy array.
        
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

        q = gb[group][pre][:].filled(np.nan)
        # chain2 = gb[group][f"{pre}2"][:].filled(np.nan)
        # chain3 = gb[group][f"{pre}3"][:].filled(np.nan)

        # chains = np.vstack((chain1, chain2, chain3))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if logged:
                return np.exp(np.nanmean(q, axis=0))
            else:
                return np.nanmean(q, axis=0)

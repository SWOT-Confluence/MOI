# Standard imports
from datetime import datetime
from pathlib import Path
import time
import random
import os

# Third-party imports
from netCDF4 import Dataset
import numpy as np
import shutil

def wait_random(min_seconds=1, max_seconds=10):
    """Wait for a random amount of time between min_seconds and max_seconds."""
    random_wait_time = random.uniform(min_seconds, max_seconds)
    print(f"Waiting for {random_wait_time:.2f} seconds...")
    time.sleep(random_wait_time)
    print("Done waiting!")

class Output:
    """Writes integration results stored in integ_dict to NetCDF file.
    
    Attributes
    ----------
    basin_dict: dict
        dict of reach_ids and SoS file needed to process entire basin of data
    FILL_VALUE: float
        Float fill value for missing data
    out_dir: Path
        path to output dir
    stage_estimate: dict
        dict of integrator estimate data

    Methods
    -------
    write_output()
        Write data stored to NetCDF file labelled with basin id
    """

    def __init__(self, basin_dict, out_dir, integ_dict, alg_dict, obs_dict, sword_dir):
        """
        Parameters
        ----------
        basin_dict: dict
            dict of reach_ids and SoS file needed to process entire basin of data
        out_dir: Path
            path to output dir
        integ_dict: dict
            dict of integrator estimate data
        """

        self.basin_dict = basin_dict
        self.out_dir = out_dir
        self.stage_estimate = integ_dict
        self.alg_dict = alg_dict
        self.obs_dict = obs_dict
        self.sword_dir = sword_dir
        
    def write_output(self):
        """Write data stored to NetCDF files for each reach
        
        TODO: 
        - Add optional attribute metadata like valid
        - Should variable output be (nr, nt)?
        - Storage of FLPE algorithms
        - Storage of actual Integrator processing
        """

        fillvalue = -999999999999

        if self.out_dir == Path('/mnt/data/output'):
            # normal confluence runs in AWS, just write out reaches we have swot data for
            reaches_to_write=self.basin_dict['reach_ids']
        else:
            # offline runs,  it's nice to have the integrator values for reaches we do not have swot data for
            print('debug mode: writing out all reach ids')
            reaches_to_write=self.basin_dict['reach_ids_all']

        for reach in reaches_to_write:
             not_obs = False
             # just write out the steady flow discharge values if this was an unobserved reach
             try:
                #print(self.obs_dict[reach])
                tmpdata=self.obs_dict[reach]
             except:
                print('reach not in obs_dict... filling with default')
                not_obs = True
             if reach not in self.basin_dict['reach_ids']:
                not_obs = True
             if not_obs:
                # NetCDF file creation
                out_file = self.out_dir / f"{reach}_integrator.nc"
                out = Dataset(out_file, 'w', format="NETCDF4")
                out.production_date = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

                #1 neobam
                gb = out.createGroup("neobam")
                gb_qbar_stage2  = out.createVariable("neobam/qbar_basinScale", "f8", fill_value=fillvalue)
                gb_qbar_stage2[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                gb_sbQ_rel = out.createVariable("neobam/sbQ_rel", "f8", fill_value=fillvalue)
                gb_sbQ_rel[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                #2 hivdi
                hv = out.createGroup("hivdi")
                hv_qbar_stage2  = out.createVariable("hivdi/qbar_basinScale", "f8", fill_value=fillvalue)
                hv_qbar_stage2[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                hv_sbQ_rel = out.createVariable("hivdi/sbQ_rel", "f8", fill_value=fillvalue)
                hv_sbQ_rel[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                #3 metroman
                mm = out.createGroup("metroman")
                mm_qbar_stage2  = out.createVariable("metroman/qbar_basinScale", "f8", fill_value=fillvalue)
                mm_qbar_stage2[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                mm_sbQ_rel = out.createVariable("metroman/sbQ_rel", "f8", fill_value=fillvalue)
                mm_sbQ_rel[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                #4 momma
                mo = out.createGroup("momma")
                mo_qbar_stage2  = out.createVariable("momma/qbar_basinScale", "f8", fill_value=fillvalue)
                mo_qbar_stage2[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                mo_sbQ_rel = out.createVariable("momma/sbQ_rel", "f8", fill_value=fillvalue)
                mo_sbQ_rel[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                #5 sad
                sad = out.createGroup("sad")
                sad_qbar_stage2  = out.createVariable("sad/qbar_basinScale", "f8", fill_value=fillvalue)
                sad_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                sad_sbQ_rel = out.createVariable("sad/sbQ_rel", "f8", fill_value=fillvalue)
                sad_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                #6 sic
                sic = out.createGroup("sic4dvar")
                sic_qbar_stage2  = out.createVariable("sic4dvar/qbar_basinScale", "f8", fill_value=fillvalue)
                sic_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                sic_sbQ_rel = out.createVariable("sic4dvar/sbQ_rel", "f8", fill_value=fillvalue)
                sic_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)
                out.close()
                continue

             iDelete=self.obs_dict[reach]['iDelete']
             shape_iDelete=np.shape(iDelete)
             nDelete=shape_iDelete[1]
             iInsert=iDelete-np.arange(nDelete)
             iInsert=np.reshape(iInsert,[nDelete,]) 

             self.obs_dict[reach]['nt'] += nDelete

             self.alg_dict['neobam'][reach]['integrator']['q']=np.insert( \
                   self.alg_dict['neobam'][reach]['integrator']['q'],iInsert,fillvalue,1)

             self.alg_dict['hivdi'][reach]['integrator']['q']=np.insert( \
                   self.alg_dict['hivdi'][reach]['integrator']['q'],iInsert,fillvalue,1)

             self.alg_dict['metroman'][reach]['integrator']['q']=np.insert( \
                   self.alg_dict['metroman'][reach]['integrator']['q'],iInsert,fillvalue,1)

             self.alg_dict['momma'][reach]['integrator']['q']=np.insert( \
                   self.alg_dict['momma'][reach]['integrator']['q'],iInsert,fillvalue,1)

             self.alg_dict['sad'][reach]['integrator']['q']=np.insert( \
                   self.alg_dict['sad'][reach]['integrator']['q'],iInsert,fillvalue,1)

             self.alg_dict['sic4dvar'][reach]['integrator']['q']=np.insert( \
                   self.alg_dict['sic4dvar'][reach]['integrator']['q'],iInsert,fillvalue,1)

             # NetCDF file creation
             out_file = self.out_dir / f"{reach}_integrator.nc"
             out = Dataset(out_file, 'w', format="NETCDF4")
             out.production_date = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

             # Dimensions and coordinate variables
             out.createDimension("nt", self.obs_dict[reach]['nt'] )
             nt = out.createVariable("nt", "i4", ("nt",))
             nt.units = "time steps"
             nt[:] = range(self.obs_dict[reach]['nt'])

             # neobam
             gb = out.createGroup("neobam")
             gbq  = out.createVariable("neobam/q", "f8", ("nt",), fill_value=fillvalue)
             gbq[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['integrator']['q'], copy=True, nan=fillvalue)
             
             gb_a0  = out.createVariable("neobam/a0", "f8", fill_value=fillvalue)
             gb_a0[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['integrator']['a0'], copy=True, nan=fillvalue)
             
             gb_n  = out.createVariable("neobam/n", "f8", fill_value=fillvalue)
             gb_n[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['integrator']['n'], copy=True, nan=fillvalue)
             
             gb_qbar_stage1  = out.createVariable("neobam/qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 gb_qbar_stage1[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 gb_qbar_stage1[:]=np.nan
             
             gb_qbar_stage2  = out.createVariable("neobam/qbar_basinScale", "f8", fill_value=fillvalue)
             gb_qbar_stage2[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)

             gb_sbQ_rel = out.createVariable("neobam/sbQ_rel", "f8", fill_value=fillvalue)
             gb_sbQ_rel[:] = np.nan_to_num(self.alg_dict['neobam'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # hivdi
             hv = out.createGroup("hivdi")
             hvq  = out.createVariable("hivdi/q", "f8", ("nt",), fill_value=fillvalue)
             hvq[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['q'], copy=True, nan=fillvalue)

             hv_Abar = out.createVariable("hivdi/Abar", "f8", fill_value=fillvalue)
             hv_Abar[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['Abar'], copy=True, nan=fillvalue)

             hv_alpha = out.createVariable("hivdi/alpha", "f8", fill_value=fillvalue)
             hv_alpha[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['alpha'], copy=True, nan=fillvalue)

             hv_beta = out.createVariable("hivdi/beta", "f8", fill_value=fillvalue)
             hv_beta[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['beta'], copy=True, nan=fillvalue)

             hv_qbar_stage1  = out.createVariable("hivdi/qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 hv_qbar_stage1[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 hv_qbar_stage1 = np.nan
             
             hv_qbar_stage2  = out.createVariable("hivdi/qbar_basinScale", "f8", fill_value=fillvalue)
             hv_qbar_stage2[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)

             hv_sbQ_rel = out.createVariable("hivdi/sbQ_rel", "f8", fill_value=fillvalue)
             hv_sbQ_rel[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # metroman
             mm = out.createGroup("metroman")
             mmq  = out.createVariable("metroman/q", "f8", ("nt",), fill_value=fillvalue)
             mmq[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['q'], copy=True, nan=fillvalue)

             mm_Abar = out.createVariable("metroman/Abar", "f8", fill_value=fillvalue)
             mm_Abar[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['a0'], copy=True, nan=fillvalue)

             mm_na = out.createVariable("metroman/na", "f8", fill_value=fillvalue)
             mm_na[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['na'], copy=True, nan=fillvalue)

             mm_x1 = out.createVariable("metroman/x1", "f8", fill_value=fillvalue)
             mm_x1[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['x1'], copy=True, nan=fillvalue)

             mm_qbar_stage1  = out.createVariable("metroman/qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 mm_qbar_stage1[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 mm_qbar_stage1[:]=np.nan
             
             mm_qbar_stage2  = out.createVariable("metroman/qbar_basinScale", "f8", fill_value=fillvalue)
             mm_qbar_stage2[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)

             mm_q33_stage2  = out.createVariable("metroman/q33_basinScale", "f8", fill_value=fillvalue)
             mm_q33_stage2[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['q33'], copy=True, nan=fillvalue)

             mm_sbQ_rel = out.createVariable("metroman/sbQ_rel", "f8", fill_value=fillvalue)
             mm_sbQ_rel[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # momma
             mo = out.createGroup("momma")
             moq  = out.createVariable("momma/q", "f8", ("nt",), fill_value=fillvalue)
             moq[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['q'], copy=True, nan=fillvalue)

             mo_B = out.createVariable("momma/B", "f8", fill_value=fillvalue)
             mo_B[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['B'], copy=True, nan=fillvalue)

             mo_H = out.createVariable("momma/H", "f8", fill_value=fillvalue)
             mo_H[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['H'], copy=True, nan=fillvalue)

             mo_Save = out.createVariable("momma/Save", "f8", fill_value=fillvalue)
             mo_Save[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['Save'], copy=True, nan=fillvalue)

             mo_qbar_stage1  = out.createVariable("momma/qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 mo_qbar_stage1[:] = np.nan_to_num(self.alg_dict['momma'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 mo_qbar_stage1[:] = np.nan
             
             mo_qbar_stage2  = out.createVariable("momma/qbar_basinScale", "f8", fill_value=fillvalue)
             mo_qbar_stage2[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)

             mo_sbQ_rel = out.createVariable("momma/sbQ_rel", "f8", fill_value=fillvalue)
             mo_sbQ_rel[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             #sad
             sad=out.createGroup("sad")
             sadq = out.createVariable("sad/q", "f8", ("nt",), fill_value=fillvalue)
             sadq[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['q'], copy=True, nan=fillvalue)

             sad_n = out.createVariable("sad/n", "f8", fill_value=fillvalue)
             sad_n[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['n'], copy=True, nan=fillvalue)

             sad_a0 = out.createVariable("sad/a0", "f8", fill_value=fillvalue)
             sad_a0[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['a0'], copy=True, nan=fillvalue)

             sad_qbar_stage1  = out.createVariable("sad/qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 sad_qbar_stage1[:] = np.nan_to_num(self.alg_dict['sad'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 sad_qbar_stage1[:] = np.nan
             
             sad_qbar_stage2  = out.createVariable("sad/qbar_basinScale", "f8", fill_value=fillvalue)
             sad_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)

             sad_sbQ_rel = out.createVariable("sad/sbQ_rel", "f8", fill_value=fillvalue)
             sad_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             #sic4dvar
             sic4dvar=out.createGroup("sic4dvar")
             sic4dvarq = out.createVariable("sic4dvar/q", "f8", ("nt",), fill_value=fillvalue)
             sic4dvarq[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['q'], copy=True, nan=fillvalue)

             sic4dvar_n = out.createVariable("sic4dvar/n", "f8", fill_value=fillvalue)
             sic4dvar_n[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['n'], copy=True, nan=fillvalue)

             sic4dvar_a0 = out.createVariable("sic4dvar/a0", "f8", fill_value=fillvalue)
             sic4dvar_a0[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['a0'], copy=True, nan=fillvalue)

             sic4dvar_qbar_stage1  = out.createVariable("sic4dvar/qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 sic4dvar_qbar_stage1[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 sic4dvar_qbar_stage1[:] = np.nan
             
             sic4dvar_qbar_stage2  = out.createVariable("sic4dvar/qbar_basinScale", "f8", fill_value=fillvalue)
             sic4dvar_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)

             sic4dvar_sbQ_rel = out.createVariable("sic4dvar/sbQ_rel", "f8", fill_value=fillvalue)
             sic4dvar_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             out.close()

    def write_sword_output(self,branch):
        """Make a new copy of the SWORD file, and write the Confluence estimates of the FLPs into the file.
           by Mike, September 2022
           """

        sword_src_file=self.sword_dir.joinpath(self.basin_dict['sword'])

        if self.out_dir == Path('/mnt/data/output'):
            #during normal operations, write this out to the sword directory
            sword_dest_file=self.sword_dir.joinpath(self.basin_dict['sword'].replace('.nc', '_moi.nc'))
        else:
            print('for debugging purposes, write FLPs to the output directory rather than sword directory')
            sword_dest_file=self.out_dir.joinpath(self.basin_dict['sword'].replace('.nc', '_moi.nc'))


        if not os.path.exists(sword_dest_file):
            shutil.copy(sword_src_file,sword_dest_file)
        try_cnt = 0
        while try_cnt < 20:
            try:
                sword_dataset = Dataset(sword_dest_file,'a')
            

                reaches = sword_dataset['reaches']['reach_id'][:]
                
                for reach in self.basin_dict['reach_ids']:
                    reach_ind = np.where(reaches == int(reach))
                    #print(reach)
                    #print(reach_ind)
                    
                    try:
        
                        #1 bam 
                        sword_dataset['reaches']['discharge_models'][branch]['BAM']['Abar'][reach_ind]= \
                            self.alg_dict['neobam'][reach]['integrator']['a0']
                        sword_dataset['reaches']['discharge_models'][branch]['BAM']['n'][reach_ind]= \
                            self.alg_dict['neobam'][reach]['integrator']['n']
                        sword_dataset['reaches']['discharge_models'][branch]['BAM']['sbQ_rel'][reach_ind]= \
                            self.alg_dict['neobam'][reach]['integrator']['sbQ_rel']
                        #2 hivdi
                        sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['Abar'][reach_ind]=\
                            self.alg_dict['hivdi'][reach]['integrator']['Abar']
                        sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['alpha'][reach_ind]=\
                            self.alg_dict['hivdi'][reach]['integrator']['alpha']
                        sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['beta'][reach_ind]=\
                            self.alg_dict['hivdi'][reach]['integrator']['beta']
                        sword_dataset['reaches']['discharge_models'][branch]['HiVDI']['sbQ_rel'][reach_ind]= \
                            self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel']
                        #3 metroman
                        sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['Abar'][reach_ind]= \
                            self.alg_dict['metroman'][reach]['integrator']['a0']
                        sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['ninf'][reach_ind]= \
                            self.alg_dict['metroman'][reach]['integrator']['na']
                        sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['p'][reach_ind]= \
                            self.alg_dict['metroman'][reach]['integrator']['x1']
                        sword_dataset['reaches']['discharge_models'][branch]['MetroMan']['sbQ_rel'][reach_ind]= \
                            self.alg_dict['metroman'][reach]['integrator']['sbQ_rel']
                        #4 momma
                        sword_dataset['reaches']['discharge_models'][branch]['MOMMA']['B'][reach_ind]= \
                            self.alg_dict['momma'][reach]['integrator']['B']
                        sword_dataset['reaches']['discharge_models'][branch]['MOMMA']['H'][reach_ind]= \
                            self.alg_dict['momma'][reach]['integrator']['H']
                        sword_dataset['reaches']['discharge_models'][branch]['MOMMA']['Save'][reach_ind]= \
                            self.alg_dict['momma'][reach]['integrator']['Save']
                        #sword_dataset['reaches']['discharge_models'][branch]['MOMMA']['sbQ_rel'][reach_ind]= \
                        #    self.alg_dict['momma'][reach]['integrator']['sbQ_rel']
                        #5 sads
                        sword_dataset['reaches']['discharge_models'][branch]['SADS']['Abar'][reach_ind]= \
                            self.alg_dict['sad'][reach]['integrator']['a0']
                        sword_dataset['reaches']['discharge_models'][branch]['SADS']['n'][reach_ind]= \
                            self.alg_dict['sad'][reach]['integrator']['n']
                        sword_dataset['reaches']['discharge_models'][branch]['SADS']['sbQ_rel'][reach_ind]= \
                            self.alg_dict['sad'][reach]['integrator']['sbQ_rel']
                        #6 sicvdvar  
                        sword_dataset['reaches']['discharge_models'][branch]['SIC4DVar']['Abar'][reach_ind]= \
                            self.alg_dict['sic4dvar'][reach]['integrator']['a0']
                        sword_dataset['reaches']['discharge_models'][branch]['SIC4DVar']['n'][reach_ind]= \
                            self.alg_dict['sic4dvar'][reach]['integrator']['n']
                        sword_dataset['reaches']['discharge_models'][branch]['SIC4DVar']['sbQ_rel'][reach_ind]= \
                            self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel']
                    except:
                        print(reach, 'data not found for sword...')

                sword_dataset.close()
                try_cnt = 999
            except:
                wait_random(2,30)
                try_cnt += 1

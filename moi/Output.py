# Standard imports
from datetime import datetime
from pathlib import Path
import time
import random
import os,sys

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

    def __init__(self, basin_dict, out_dir, integ_dict, alg_dict, obs_dict, sword_dir,params_dict):
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
        self.params_dict=params_dict
        
    def write_output(self):
        """Write data stored to NetCDF files for each reach"""
        fillvalue = -999999999999

        if self.out_dir == Path('/mnt/data/output'):
            reaches_to_write = self.basin_dict['reach_ids']
        else:
            reaches_to_write = self.basin_dict['reach_ids_all']

        for reach in reaches_to_write:
             if self.params_dict['write_fill_only']:
                 for algo in self.alg_dict.keys():
                     if 'q' in self.alg_dict[algo][reach]['integrator'].keys():
                         self.alg_dict[algo][reach]['integrator']['q'][:] = np.nan
                     self.alg_dict[algo][reach]['integrator']['qbar'] = np.nan
                     self.alg_dict[algo][reach]['integrator']['q33'] = np.nan
                     self.alg_dict[algo][reach]['integrator']['sbQ_rel'] = np.nan
                     if 'qbar' in self.alg_dict[algo][reach].keys():
                         self.alg_dict[algo][reach]['qbar'] = np.nan

             not_obs = False
             try:
                tmpdata = self.obs_dict[reach]
             except:
                not_obs = True
             if reach not in self.basin_dict['reach_ids']: 
                not_obs = True
             
             if not_obs:
                # NetCDF file creation for unobserved reaches
                out_file = self.out_dir / f"{reach}_integrator.nc"
                out = Dataset(out_file, 'w', format="NETCDF4")
                out.production_date = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

                # 1 busboi (Fixed neobam -> busboi and path creation)
                gb = out.createGroup("busboi")
                gb_qbar_stage2 = gb.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                gb_qbar_stage2[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                gb_sbQ_rel = gb.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                gb_sbQ_rel[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                # 2 hivdi
                hv = out.createGroup("hivdi")
                hv_qbar_stage2 = hv.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                hv_qbar_stage2[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                hv_sbQ_rel = hv.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                hv_sbQ_rel[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                # 3 metroman
                mm = out.createGroup("metroman")
                mm_qbar_stage2 = mm.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                mm_qbar_stage2[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                mm_sbQ_rel = mm.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                mm_sbQ_rel[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                # 4 momma
                mo = out.createGroup("momma")
                mo_qbar_stage2 = mo.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                mo_qbar_stage2[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                mo_sbQ_rel = mo.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                mo_sbQ_rel[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                # 5 sad
                sad = out.createGroup("sad")
                sad_qbar_stage2 = sad.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                sad_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                sad_sbQ_rel = sad.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                sad_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

                # 6 sic4dvar
                sic = out.createGroup("sic4dvar")
                sic_qbar_stage2 = sic.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
                sic_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
                sic_sbQ_rel = sic.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
                sic_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)
                
                out.close()
                continue
             
             iDelete = self.obs_dict[reach]['iDelete']
             shape_iDelete = np.shape(iDelete)
             nDelete = shape_iDelete[1]
             iInsert = iDelete - np.arange(nDelete)
             iInsert = np.reshape(iInsert,[nDelete,]) 
             self.obs_dict[reach]['nt'] += nDelete

             self.alg_dict['busboi'][reach]['integrator']['q']=np.insert( \
                   self.alg_dict['busboi'][reach]['integrator']['q'],iInsert,fillvalue,1)
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

             # NetCDF file creation for observed reaches
             out_file = self.out_dir / f"{reach}_integrator.nc"
             out = Dataset(out_file, 'w', format="NETCDF4")
             out.production_date = datetime.now().strftime('%d-%b-%Y %H:%M:%S')

             # Dimensions and coordinate variables
             out.createDimension("nt", self.obs_dict[reach]['nt'] )
             nt = out.createVariable("nt", "i4", ("nt",))
             nt.units = "time steps"
             nt[:] = range(self.obs_dict[reach]['nt'])

             # busboi (Fixed paths)
             gb = out.createGroup("busboi")
             gbq = gb.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             gbq[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['integrator']['q'], copy=True, nan=fillvalue)
             gb_a0 = gb.createVariable("a0", "f8", fill_value=fillvalue)
             gb_a0[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['integrator']['a0'], copy=True, nan=fillvalue)
             gb_n = gb.createVariable("n", "f8", fill_value=fillvalue)
             gb_n[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['integrator']['n'], copy=True, nan=fillvalue)
             
             gb_qbar_stage1 = gb.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 gb_qbar_stage1[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 gb_qbar_stage1[:] = np.nan
             
             gb_qbar_stage2 = gb.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             gb_qbar_stage2[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
             gb_sbQ_rel = gb.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             gb_sbQ_rel[:] = np.nan_to_num(self.alg_dict['busboi'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # hivdi
             hv = out.createGroup("hivdi")
             hvq = hv.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             hvq[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['q'], copy=True, nan=fillvalue)
             hv_Abar = hv.createVariable("Abar", "f8", fill_value=fillvalue)
             hv_Abar[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['Abar'], copy=True, nan=fillvalue)
             hv_alpha = hv.createVariable("alpha", "f8", fill_value=fillvalue)
             hv_alpha[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['alpha'], copy=True, nan=fillvalue)
             hv_beta = hv.createVariable("beta", "f8", fill_value=fillvalue)
             hv_beta[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['beta'], copy=True, nan=fillvalue)
             
             hv_qbar_stage1 = hv.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 hv_qbar_stage1[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 hv_qbar_stage1[:] = np.nan
             
             hv_qbar_stage2 = hv.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             hv_qbar_stage2[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
             hv_sbQ_rel = hv.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             hv_sbQ_rel[:] = np.nan_to_num(self.alg_dict['hivdi'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # metroman
             mm = out.createGroup("metroman")
             mmq = mm.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             mmq[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['q'], copy=True, nan=fillvalue)
             mm_Abar = mm.createVariable("Abar", "f8", fill_value=fillvalue)
             mm_Abar[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['a0'], copy=True, nan=fillvalue)
             mm_na = mm.createVariable("na", "f8", fill_value=fillvalue)
             mm_na[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['na'], copy=True, nan=fillvalue)
             mm_x1 = mm.createVariable("x1", "f8", fill_value=fillvalue)
             mm_x1[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['x1'], copy=True, nan=fillvalue)
             
             mm_qbar_stage1 = mm.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 mm_qbar_stage1[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 mm_qbar_stage1[:] = np.nan
             
             mm_qbar_stage2 = mm.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             mm_qbar_stage2[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
             mm_q33_stage2 = mm.createVariable("q33_basinScale", "f8", fill_value=fillvalue)
             mm_q33_stage2[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['q33'], copy=True, nan=fillvalue)
             mm_sbQ_rel = mm.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             mm_sbQ_rel[:] = np.nan_to_num(self.alg_dict['metroman'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # momma
             mo = out.createGroup("momma")
             moq = mo.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             moq[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['q'], copy=True, nan=fillvalue)
             mo_B = mo.createVariable("B", "f8", fill_value=fillvalue)
             mo_B[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['B'], copy=True, nan=fillvalue)
             mo_H = mo.createVariable("H", "f8", fill_value=fillvalue)
             mo_H[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['H'], copy=True, nan=fillvalue)
             mo_Save = mo.createVariable("Save", "f8", fill_value=fillvalue)
             mo_Save[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['Save'], copy=True, nan=fillvalue)
             
             mo_qbar_stage1 = mo.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 mo_qbar_stage1[:] = np.nan_to_num(self.alg_dict['momma'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 mo_qbar_stage1[:] = np.nan
             
             mo_qbar_stage2 = mo.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             mo_qbar_stage2[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
             mo_sbQ_rel = mo.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             mo_sbQ_rel[:] = np.nan_to_num(self.alg_dict['momma'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # sad
             sad = out.createGroup("sad")
             sadq = sad.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             sadq[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['q'], copy=True, nan=fillvalue)
             sad_n = sad.createVariable("n", "f8", fill_value=fillvalue)
             sad_n[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['n'], copy=True, nan=fillvalue)
             sad_a0 = sad.createVariable("a0", "f8", fill_value=fillvalue)
             sad_a0[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['a0'], copy=True, nan=fillvalue)
             
             sad_qbar_stage1 = sad.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 sad_qbar_stage1[:] = np.nan_to_num(self.alg_dict['sad'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 sad_qbar_stage1[:] = np.nan
             
             sad_qbar_stage2 = sad.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             sad_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
             sad_sbQ_rel = sad.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             sad_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sad'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             # sic4dvar
             sic4dvar = out.createGroup("sic4dvar")
             sic4dvarq = sic4dvar.createVariable("q", "f8", ("nt",), fill_value=fillvalue)
             sic4dvarq[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['q'], copy=True, nan=fillvalue)
             sic4dvar_n = sic4dvar.createVariable("n", "f8", fill_value=fillvalue)
             sic4dvar_n[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['n'], copy=True, nan=fillvalue)
             sic4dvar_a0 = sic4dvar.createVariable("a0", "f8", fill_value=fillvalue)
             sic4dvar_a0[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['a0'], copy=True, nan=fillvalue)
             
             sic4dvar_qbar_stage1 = sic4dvar.createVariable("qbar_reachScale", "f8", fill_value=fillvalue)
             try:
                 sic4dvar_qbar_stage1[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['qbar'], copy=True, nan=fillvalue)
             except:
                 sic4dvar_qbar_stage1[:] = np.nan
             
             sic4dvar_qbar_stage2 = sic4dvar.createVariable("qbar_basinScale", "f8", fill_value=fillvalue)
             sic4dvar_qbar_stage2[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['qbar'], copy=True, nan=fillvalue)
             sic4dvar_sbQ_rel = sic4dvar.createVariable("sbQ_rel", "f8", fill_value=fillvalue)
             sic4dvar_sbQ_rel[:] = np.nan_to_num(self.alg_dict['sic4dvar'][reach]['integrator']['sbQ_rel'], copy=True, nan=fillvalue)

             out.close()
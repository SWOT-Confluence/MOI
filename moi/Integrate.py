# Standard imports
import warnings
import sys
import datetime
import csv

# Third-party imports
import numpy as np
import pandas as pd
from scipy import optimize
import scipy.sparse as sp

from moi.sfoi_math_core import getP_1D, adjust_lsq_sparse, adjust_lsq_sparse_strict_mass

class Integrate:
    """Integrates reach-level FLPE algorithm data using Fast Sparse SFOI core.
    Maintains 100% downstream compatibility with Output.py expectations.
    """

    def __init__(self, alg_dict, basin_dict, sos_dict, sword_dict, obs_dict, params_dict, Branch, VerboseFlag):
        self.alg_dict = alg_dict
        self.basin_dict = basin_dict
        self.obs_dict = obs_dict
        self.sword_dict = sword_dict
        self.params_dict = params_dict
        self.sos_dict = sos_dict
        self.Branch = Branch
        self.VerboseFlag = VerboseFlag
        self.junctions = []
        self.GoodFLPE = {}
        self.mass_diag_static = None
        self.reach_epsilons = {}
        
        self.integ_dict = {
            "pre_q_mean": np.array([]),
            "q_mean": np.array([]),
            "flpe": {
                "busboi" : np.array([]),
                "hivdi" : np.array([]),
                "metroman" : np.array([]),
                "momma" : np.array([]),
                "sad" : np.array([]),
                "sic4dvar" : np.array([])
            }
        }

        self.patch_input_deficiencies()

        # if self.VerboseFlag: print('getting pre mean q')
        self.get_pre_mean_q()

        if self.Branch == 'constrained':
            self.get_gage_mean_q()

    def patch_input_deficiencies(self):
        """
        [CRITICAL] Silently fixes dict structures so Output.py won't throw KeyErrors.
        Output.py HARD requires the 'integrator' dict and specific parameter keys 
        for ALL reaches (even unobserved ones), which Input.py fails to create.
        """
        for alg in self.alg_dict:
            for reach in self.basin_dict['reach_ids_all']:
                if reach not in self.alg_dict[alg]:
                    self.alg_dict[alg][reach] = {'s1-flpe-exists': False, 'qbar': np.nan, 'q33': np.nan}
                
                # Pre-build the integrator dict with NaNs to satisfy Output.py
                if 'integrator' not in self.alg_dict[alg][reach]:
                    # Determine a safe 'nt' for the dummy q array
                    nt = self.obs_dict.get(reach, {}).get('nt', 1)
                    if not isinstance(nt, int) or nt < 1: nt = 1
                    
                    self.alg_dict[alg][reach]['integrator'] = {
                        'qbar': np.nan,
                        'q33': np.nan,
                        'sbQ_rel': np.nan,
                        'q': np.full((1, nt), np.nan)
                    }
                    
                    # Fill algorithm-specific parameter requirements for Output.py
                    if alg == 'busboi':
                        self.alg_dict[alg][reach]['integrator'].update({'n': np.nan, 'a0': np.nan})
                    elif alg == 'hivdi':
                        self.alg_dict[alg][reach]['integrator'].update({'Abar': np.nan, 'alpha': np.nan, 'beta': np.nan})
                    elif alg == 'metroman':
                        self.alg_dict[alg][reach]['integrator'].update({'a0': np.nan, 'na': np.nan, 'x1': np.nan})
                    elif alg == 'momma':
                        self.alg_dict[alg][reach]['integrator'].update({'B': np.nan, 'H': np.nan, 'Save': np.nan})
                    elif alg == 'sad':
                        self.alg_dict[alg][reach]['integrator'].update({'n': np.nan, 'a0': np.nan})
                    elif alg == 'sic4dvar':
                        self.alg_dict[alg][reach]['integrator'].update({'n': np.nan, 'a0': np.nan})

    def get_pre_mean_q(self):
        for alg in self.alg_dict:
            for reach in self.alg_dict[alg]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if self.alg_dict[alg][reach].get('s1-flpe-exists', False):
                        q_val = self.alg_dict[alg][reach].get('q', np.nan)
                        self.alg_dict[alg][reach]['qbar'] = np.nanmean(q_val)
                        self.alg_dict[alg][reach]['q33'] = np.nanquantile(q_val, .33)
                    
                    if np.isnan(self.alg_dict[alg][reach].get('qbar', np.nan)):
                        self.alg_dict[alg][reach]['qbar'] = self.sos_dict[str(reach)].get('Qbar', np.nan)
                        self.alg_dict[alg][reach]['q33'] = self.sos_dict[str(reach)].get('q33', np.nan)

    def get_gage_mean_q(self):
         # Kept exact Mike logic for constrained runs
         for reach in self.sos_dict.keys():
             if reach in self.obs_dict.keys():
                 try:
                     agency = self.sos_dict[str(reach)]['gage']['source']
                     gaged_reach = True
                 except Exception:
                     gaged_reach = False

                 if gaged_reach:
                     epoch = datetime.datetime(2000,1,1,0,0,0)
                     gagedQs = []
                     for time_val in self.obs_dict[reach]['t']:
                         try:
                             ordinal_time = (epoch + datetime.timedelta(seconds=time_val)).toordinal()
                         except Exception:
                             ordinal_time = np.nan
                             warnings.warn('problem with time conversion to ordinal')

                         try:
                             idx = np.argwhere(self.sos_dict[str(reach)]['gage']['t'] == ordinal_time)[0,0]
                             gagedQs.append(self.sos_dict[str(reach)]['gage']['Q'][idx])
                         except Exception:
                             pass

                     self.sos_dict[str(reach)]['gage']['Qbar'] = np.nan
                     self.sos_dict[str(reach)]['gage']['q33'] = np.nan
                     if gagedQs:
                         try:
                             self.sos_dict[str(reach)]['gage']['Qbar'] = np.nanmean(gagedQs)
                             self.sos_dict[str(reach)]['gage']['q33'] = np.nanquantile(gagedQs, .33)
                         except Exception:
                             pass

    def pull_sword_attributes_for_reach(self, k):
         sword_data_reach = {}
         for key in self.sword_dict:
             if np.shape(self.sword_dict[key]) == (self.sword_dict['num_reaches'],):
                 sword_data_reach[key] = self.sword_dict[key][k]
         for key in self.sword_dict:            
             if key == 'rch_id_up':
                 sword_data_reach[key] = self.sword_dict[key][0:sword_data_reach['n_rch_up'], k]
             elif key == 'rch_id_dn':
                 sword_data_reach[key] = self.sword_dict[key][0:sword_data_reach['n_rch_down'], k]
             elif key == 'swot_orbits':
                 sword_data_reach[key] = self.sword_dict[key][0:sword_data_reach['swot_obs'], k]
         return sword_data_reach

    def ChecksPriorToAddingJunction(self, junction_to_check):
         AlreadyExists = False
         for junction in self.junctions:
             if junction['upflows'] == junction_to_check['upflows'] and junction['downflows'] == junction_to_check['downflows']: 
                 AlreadyExists = True    
         AllReachesInReachFile = True
         for r in junction_to_check['upflows']:
             if str(r) not in self.basin_dict['reach_ids_all']:
                 AllReachesInReachFile = False
         for r in junction_to_check['downflows']:
             if str(r) not in self.basin_dict['reach_ids_all']:
                 AllReachesInReachFile = False            
         return AlreadyExists, AllReachesInReachFile

    def CreateJunctionList(self):
         self.junctions = list()
         self.junctions_valid = True

         for reach in self.basin_dict['reach_ids_all']:
             reach = np.int64(reach)
             k = np.argwhere(self.sword_dict['reach_id'] == reach)[0, 0]
             sword_data_reach = self.pull_sword_attributes_for_reach(k) 

             # Upstream
             junction_up = {'originating_reach_id': reach, 'upflows': list()}
             for i in range(sword_data_reach['n_rch_up']):
                 junction_up['upflows'].append(sword_data_reach['rch_id_up'][i])
       
             if len(junction_up['upflows']) > 0:
                 junction_up['downflows'] = list()
                 if not any(junction_up['upflows']):
                    self.junctions_valid = False
                    continue
                 kup = np.argwhere(self.sword_dict['reach_id'] == junction_up['upflows'][0])[0, 0]
                 sword_data_reach_up = self.pull_sword_attributes_for_reach(kup)
                 for j in range(sword_data_reach_up['n_rch_down']):
                     junction_up['downflows'].append(sword_data_reach_up['rch_id_dn'][j])
                 AlreadyExists, AllReachesInReachFile = self.ChecksPriorToAddingJunction(junction_up)
                 if not AlreadyExists and AllReachesInReachFile:
                     self.junctions.append(junction_up)

             # Downstream
             junction_dn = {'originating_reach_id': reach, 'downflows': list()}
             for i in range(sword_data_reach['n_rch_down']):
                 junction_dn['downflows'].append(sword_data_reach['rch_id_dn'][i])            

             if len(junction_dn['downflows']) > 0:
                 junction_dn['upflows'] = list()
                 if not any(junction_dn['downflows']):
                    self.junctions_valid = False
                    continue
                 kdn = np.argwhere(self.sword_dict['reach_id'] == junction_dn['downflows'][0])[0, 0]
                 sword_data_reach_dn = self.pull_sword_attributes_for_reach(kdn)
                 for j in range(sword_data_reach_dn['n_rch_up']):
                     junction_dn['upflows'].append(sword_data_reach_dn['rch_id_up'][j])
                 AlreadyExists, AllReachesInReachFile = self.ChecksPriorToAddingJunction(junction_dn)
                 if not AlreadyExists and AllReachesInReachFile:
                     self.junctions.append(junction_dn) 


    def calcG(self, m, n):
        G = np.zeros((m, n))
        for junction in self.junctions:
            row = junction['row_num']
            for upflow in junction['upflows']:
                try:
                     kup = self.basin_dict['reach_ids_all'].index(str(upflow))
                     G[row, kup] = 1
                except ValueError: pass
            for downflow in junction['downflows']:
                try:
                     kdn = self.basin_dict['reach_ids_all'].index(str(downflow))
                     G[row, kdn] = -1
                except ValueError: pass
        return G



    def initialize_integration_vars(self, alg, FlowLevel, PreviousResiduals, n):
         self.GoodFLPE[alg] = True
         Qbar = np.empty([n,])
         sigQ = np.empty([n,])
         facc = np.empty([n,])
         runoff = np.empty([n,])
         datasource = []

         i = 0
         for reach in self.basin_dict['reach_ids_all']:
            k = np.argwhere(self.sword_dict['reach_id'] == np.int64(reach))[0,0]
            facc[i] = self.pull_sword_attributes_for_reach(k)['facc']

            if reach in self.alg_dict[alg].keys():
                nrt_gaged_reach = ((self.sos_dict[str(reach)]['overwritten_indices']==1) and 
                                  (self.sos_dict[str(reach)]['overwritten_source']!='grdc') and 
                                  (self.sos_dict[str(reach)]['cal_status']==1 ) and 
                                  ('Qbar' in self.sos_dict[str(reach)]['gage'].keys()) and 
                                  ('q33' in  self.sos_dict[str(reach)]['gage'].keys()))

                if (self.Branch == 'constrained') and nrt_gaged_reach:
                    if FlowLevel == 'Mean':
                        Qbar[i] = self.sos_dict[str(reach)]['gage']['Qbar']
                    elif FlowLevel == 'q33':
                        Qbar[i] = self.sos_dict[str(reach)]['gage']['q33']

                    sigQ[i] = Qbar[i] * self.params_dict['Gage_Uncertainty']
                    datasource.append('Gage')
                else:
                    if FlowLevel == 'Mean':
                        val = self.alg_dict[alg][reach].get('qbar', np.nan)
                        if np.ma.is_masked(val):
                            Qbar[i] = np.nan 
                        else:
                            nstdev = 10.
                            sos_qbar = self.sos_dict[str(reach)].get('Qbar', np.nan)
                            if not np.isnan(val) and not np.isnan(sos_qbar) and abs(val - sos_qbar) > sos_qbar * self.params_dict['FLPE_Uncertainty'] * nstdev:
                                Qbar[i] = np.nan
                            else:
                                Qbar[i] = val
                    elif FlowLevel == 'q33':
                        try:
                            val = self.alg_dict[alg][reach].get('q33', np.nan)
                            if np.ma.is_masked(val):
                                Qbar[i] = np.nan
                            else:
                                nstdev = 10.
                                sos_qbar = self.sos_dict[str(reach)].get('Qbar', np.nan)
                                if not np.isnan(val) and not np.isnan(sos_qbar) and abs(val - sos_qbar) > sos_qbar * self.params_dict['FLPE_Uncertainty'] * nstdev:
                                    Qbar[i] = np.nan
                                else:
                                    Qbar[i] = val
                        except Exception:
                            Qbar[i] = np.nan

                    if facc[i] > 5000:
                        dynamic_unc = 0.20 # 25%
                    elif facc[i] > 500:
                        dynamic_unc = 0.35  # 40%
                    else:
                        dynamic_unc = 0.75  # 75%
                    
                    if np.isnan(PreviousResiduals[alg][i]):
                        sigQ[i] = Qbar[i] * dynamic_unc
                    else:
                        raw_sig = max(abs(PreviousResiduals[alg][i]), Qbar[i]*.01)**(-(self.params_dict.get('norm', 2.0)-2.0))
                        sigQ[i] = min(raw_sig, Qbar[i] * dynamic_unc * 2.0)

                    datasource.append('FLPE')
            else:
                 Qbar[i] = np.nan
                 sigQ[i] = np.nan
                 datasource.append('None')
            i += 1
 
         for i in range(n):
             if not np.isnan(Qbar[i]) and facc[i] > 0:
                 runoff[i] = Qbar[i] / facc[i] / 1000**2 * 86400 * 365
             else:
                 runoff[i] = np.nan

         with warnings.catch_warnings():
             warnings.simplefilter("ignore", category=RuntimeWarning)
             runoff_avg = np.nanmean(runoff)
         
         if np.isnan(runoff_avg) or np.isinf(runoff_avg):
             runoff_avg = 315.36 # fallback

         for i in range(n):
             if np.isnan(Qbar[i]) or np.isinf(Qbar[i]):
                 Qbar[i] = runoff_avg * facc[i] * 1000**2 / 86400 / 365
                 sigQ[i] = Qbar[i] * self.params_dict.get('Fill_Uncertainty', 0.5)

         bignumber = 1e9
         sigQmin = 10.
         for i in range(n):
             if Qbar[i] == 0. and np.isnan(PreviousResiduals[alg][i]):
                 sigQ[i] = bignumber
             if sigQ[i] < sigQmin and datasource[i] != 'Gage':
                sigQ[i] = sigQmin
             if np.isinf(PreviousResiduals[alg][i]):
                 Qbar[i] = 5.
                 sigQ[i] = 0.1

         iFLPE = np.where(np.array(datasource) == 'FLPE')
         FLPE_Data_OK = not np.all(Qbar[iFLPE] == 0)
         self.GoodFLPE[alg] = FLPE_Data_OK

         return Qbar, sigQ, FLPE_Data_OK, facc

    def integrator_optimization_calcs(self, m, n, FlowLevel, PreviousResiduals):
        residuals = {}
        for alg in self.alg_dict:
            if self.VerboseFlag:
                print(f'    RUNNING SPARSE MOI for {alg} ({FlowLevel})')

            Qbar, sigQ, FLPE_Data_OK, facc = self.initialize_integration_vars(alg, FlowLevel, PreviousResiduals, n)
            u_conversion = 1000.0 / (365.25 * 24 * 3600)

            A_sparse, L_vector, W_1d, K_regions = self.build_soft_sfoi_system(n, Qbar, sigQ, facc, u_conversion)

            Success = False
            Qintegrator = np.full((n,), np.nan) #
            x_hat_saved = None

            if FLPE_Data_OK and self.junctions_valid:
                try:
                    x_hat, status = adjust_lsq_sparse_strict_mass(A_sparse=A_sparse, W_1d=W_1d, L=L_vector, n_mass_rows=n, bound=True)
                    
                    if status.startswith('success') or status == 'success_scs':
                        Qintegrator = np.clip(x_hat[:n], 0.1, np.inf) 
                        x_hat_saved = x_hat # 
                        Success = True
                except Exception as e:
                    print(f"      SFOI Solver failed: {e}")

            if Success and getattr(self, 'mass_diag_static', None) is not None and FlowLevel == 'Mean':
                if not hasattr(self, 'reach_epsilons'): 
                    self.reach_epsilons = {}
                self.reach_epsilons[alg] = self.compute_mass_conservation_metrics(x_hat_saved, self.mass_diag_static)

            # compute residuals
            residuals[alg] = Qbar - Qintegrator if Success else np.full((n,), np.nan)
            if Success:
                residuals[alg][Qintegrator < 0.] = np.inf

            # Fast uncertainty proxy (Bypassing O(N^3) dense matrix inversion)
            stdQc_rel_proxy = np.abs(sigQ / Qbar)
            stdQc_rel_proxy[np.isnan(stdQc_rel_proxy) | np.isinf(stdQc_rel_proxy)] = self.params_dict['FLPE_Uncertainty']

            # Save Data into dictionaries for Output.py
            for i, reach in enumerate(self.basin_dict['reach_ids_all']):
                if reach in self.alg_dict[alg]:
                    if FlowLevel == 'Mean':
                        self.alg_dict[alg][reach]['integrator']['qbar'] = float(Qintegrator[i])
                        self.alg_dict[alg][reach]['integrator']['sbQ_rel'] = float(stdQc_rel_proxy[i]) if Success else self.params_dict['FLPE_Uncertainty']
                    elif FlowLevel == 'q33':
                        self.alg_dict[alg][reach]['integrator']['q33'] = float(Qintegrator[i])

        return residuals


    def bam_objfun(self,params,obs,qbar_target,q33_target): 
        qbam=self.bam_flowlaw(params,obs)
        qbam_bar=np.nanmean(qbam)
        y=(qbam_bar-qbar_target)**2
        if not np.isnan(q33_target):
            qbam_33=np.nanquantile(qbam,.33)
            y+=(qbam_33-q33_target)**2 
        return y

    def bam_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        bam_n=params[0]
        bam_Abar=params[1]
        qbam = ((d_x_area+bam_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / bam_n
        qbam=np.reshape(qbam,(1,len(d_x_area)))
        return qbam

    def hivdi_objfun(self,params,obs,qbar_target,q33_target): 
        q=self.hivdi_flowlaw(params,obs)
        qbar=np.nanmean(q)
        y=(qbar-qbar_target)**2
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(q,.33)
            y+=(q33_alg-q33_target)**2 
        return y

    def hivdi_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        hivdi_alpha=params[0]
        hivdi_beta=params[1]
        hivdi_Abar=params[2]
        hivdi_n_inv = hivdi_alpha * ((d_x_area+hivdi_Abar)/reach_width)**hivdi_beta
        qhivdi = ((d_x_area+hivdi_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) * hivdi_n_inv
        qhivdi=np.reshape(qhivdi,(1,len(d_x_area)))
        return qhivdi

    def metroman_objfun(self,params,obs,qbar_target,q33_target): 
        q=self.metroman_flowlaw(params,obs)
        qbar=np.nanmean(q)
        y=abs(qbar-qbar_target)
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(q,.33)
            y+=abs(q33_alg-q33_target)
        return y

    def metroman_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        metro_ninf=params[0]
        metro_p=params[1]
        metro_Abar=params[2]
        metro_n = metro_ninf * ((d_x_area+metro_Abar) / reach_width)**metro_p
        metro_q = ((d_x_area+metro_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / metro_n
        metro_q=np.reshape(metro_q,(1,len(d_x_area)))
        return metro_q

    def momma_objfun(self,params,obs,qbar_target,q33_target,aux_var): 
        q=self.momma_flowlaw(params,obs,aux_var)
        if np.all(np.isnan(q)): return 1e9
        qbar=np.nanmean(q)
        y=(qbar-qbar_target)**2
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(q,.33)
            y+=(q33_alg-q33_target)**2 
        B=params[0]
        H=params[1]
        Db=H-B
        if Db<0.2 and Db >= 0.1: yfac=2.
        elif Db < 0.1: yfac=10.
        else: yfac=1.
        y*=yfac
        return y

    def momma_flowlaw(self,params,obs,aux_var):
        reach_height=obs['h']
        reach_width=obs['w']
        reach_slope=obs['S']
        momma_B=params[0]
        momma_H=params[1]
        momma_Save=aux_var
        momma_r = 2
        momma_nb = 0.11 * momma_Save**0.18
        momma_q=np.empty( (obs['nt'],)) 

        if momma_H <= momma_B+0.1:
             momma_q=np.inf
        else:
             for t in range(obs['nt']):
                  log_factor = np.log10((momma_H-momma_B)/(reach_height[t]-momma_B))
                  if reach_height[t] <= momma_H:
                       momma_n = momma_nb*(1+log_factor)
                  else:
                       momma_n = momma_nb*(1-log_factor)
                  momma_q[t] = (((reach_height[t] - momma_B)*(momma_r/(1+momma_r)))**(5/3) *
                       reach_width[t] * reach_slope[t]**(1/2)) / momma_n
             momma_q=np.reshape(momma_q,(1,len(reach_height)))
        return momma_q

    def sad_objfun(self,params,obs,qbar_target,q33_target): 
        qsad=self.sad_flowlaw(params,obs)
        if np.all(np.isnan(qsad)): return 1e9
        qsad_bar=np.nanmean(qsad)
        y=(qsad_bar-qbar_target)**2
        if not np.isnan(q33_target):
            q33_alg=np.nanquantile(qsad,.33)
            y+=(q33_alg-q33_target)**2 
        return y

    def sad_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        sad_n=params[0]
        sad_Abar=params[1]
        qsad = ((d_x_area+sad_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / sad_n
        qsad=np.reshape(qsad,(1,len(d_x_area)))
        return qsad

    def sic4dvar_objfun(self,params,obs,qbar_target): 
        qsic4dvar=self.sic4dvar_flowlaw(params,obs)
        qsic4dvar_bar=np.nanmean(qsic4dvar)
        y=abs(qsic4dvar_bar-qbar_target)
        return y

    def sic4dvar_flowlaw(self,params,obs):
        d_x_area=obs['dA']
        reach_width=obs['w']
        reach_slope=obs['S']
        sic4dvar_n=params[0]
        sic4dvar_Abar=params[1]
        qsic4dvar = ((d_x_area+sic4dvar_Abar)**(5/3) * reach_width**(-2/3) * (reach_slope)**(1/2)) / sic4dvar_n
        qsic4dvar=np.reshape(qsic4dvar,(1,len(d_x_area)))
        return qsic4dvar

    def compute_FLPs(self):         
        if self.VerboseFlag: print('CALCULATING BUSBOI surrogate FLPs')
        for reach in self.alg_dict['busboi']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nhat = np.nanmean(self.alg_dict['busboi'][reach].get('n', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (nhat, np.nanmean(self.alg_dict['busboi'][reach].get('a0', np.nan))) if not np.isnan(nhat) else (0.03, Abar_min+10.)
                
            param_bounds = ((0.001, np.inf), (Abar_min, np.inf))
            qbar = self.alg_dict['busboi'][reach]['integrator']['qbar'] 
            q33 = self.alg_dict['busboi'][reach]['integrator'].get('q33', np.nan) 

            try:
                res = optimize.minimize(fun=self.bam_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
                if res.success:
                    param_est = res.x
                else:
                    param_est = init_params
            except Exception:
                param_est = init_params
            
            self.alg_dict['busboi'][reach]['integrator']['n'] = param_est[0]
            self.alg_dict['busboi'][reach]['integrator']['a0'] = param_est[1]
            self.alg_dict['busboi'][reach]['integrator']['q'] = self.bam_flowlaw(param_est, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING HiVDI FLPs')
        for reach in self.alg_dict['hivdi']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                alphaflpe = np.nanmean(self.alg_dict['hivdi'][reach].get('alpha', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['hivdi'][reach].get('alpha', np.nan)), np.nanmean(self.alg_dict['hivdi'][reach].get('beta', np.nan)), np.nanmean(self.alg_dict['hivdi'][reach].get('a0', np.nan))) if not np.isnan(alphaflpe) else (33.3, 1.0, Abar_min+10.)
                
            param_bounds = ((0.001, np.inf), (-1e1, 1.e1), (Abar_min, np.inf))
            qbar = self.alg_dict['hivdi'][reach]['integrator']['qbar']
            q33 = self.alg_dict['hivdi'][reach]['integrator'].get('q33', np.nan) 
            res = optimize.minimize(fun=self.hivdi_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
            
            self.alg_dict['hivdi'][reach]['integrator']['alpha'] = res.x[0]
            self.alg_dict['hivdi'][reach]['integrator']['beta'] = res.x[1]
            self.alg_dict['hivdi'][reach]['integrator']['Abar'] = res.x[2]
            self.alg_dict['hivdi'][reach]['integrator']['q'] = self.hivdi_flowlaw(res.x, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING MetroMan FLPs')
        for reach in self.alg_dict['metroman']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                naflpe = np.nanmean(self.alg_dict['metroman'][reach].get('na', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['metroman'][reach].get('na', np.nan)), np.nanmean(self.alg_dict['metroman'][reach].get('x1', np.nan)), np.nanmean(self.alg_dict['metroman'][reach].get('a0', np.nan))) if not np.isnan(naflpe) else (0.03, -1., Abar_min+10.)

            param_bounds = ((0.001, np.inf), (-1e1, 1e1), (Abar_min, np.inf))
            qbar = self.alg_dict['metroman'][reach]['integrator']['qbar']
            q33 = self.alg_dict['metroman'][reach]['integrator'].get('q33', np.nan)
            res = optimize.minimize(fun=self.metroman_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
            
            self.alg_dict['metroman'][reach]['integrator']['na'] = res.x[0]
            self.alg_dict['metroman'][reach]['integrator']['x1'] = res.x[1]
            self.alg_dict['metroman'][reach]['integrator']['a0'] = res.x[2]
            self.alg_dict['metroman'][reach]['integrator']['q'] = self.metroman_flowlaw(res.x, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING MOMMA FLPs')
        for reach in self.alg_dict['momma']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Bflpe = np.nanmean(self.alg_dict['momma'][reach].get('B', np.nan))
                Bmax = np.min(self.obs_dict[reach]['h']) - 0.1
                init_params = (np.nanmean(self.alg_dict['momma'][reach].get('B', np.nan)), np.nanmean(self.alg_dict['momma'][reach].get('H', np.nan))) if not np.isnan(Bflpe) else (Bmax-1.0, Bmax+1.0)
                
            min_H_obs = np.min(self.obs_dict[reach]['h'])
            if min_H_obs - init_params[0] > 10.: init_params = (min_H_obs - 10., min_H_obs)
            param_bounds = ((0.1, Bmax), (Bmax+0.1, np.inf))
            aux_var = self.alg_dict['momma'][reach].get('Save', np.nan)
            if np.isnan(aux_var): aux_var = 20e-5
            qbar = self.alg_dict['momma'][reach]['integrator']['qbar']
            q33 = self.alg_dict['momma'][reach]['integrator'].get('q33', np.nan) 

            try: res = optimize.minimize(fun=self.momma_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33, aux_var), bounds=param_bounds)
            except Exception: res = lambda: None; res.success = False

            if not res.success: param_est = (self.alg_dict['momma'][reach].get('B', np.nan), self.alg_dict['momma'][reach].get('H', np.nan))
            else: param_est = res.x
            
            self.alg_dict['momma'][reach]['integrator']['B'] = param_est[0]
            self.alg_dict['momma'][reach]['integrator']['H'] = param_est[1]
            self.alg_dict['momma'][reach]['integrator']['Save'] = aux_var
            self.alg_dict['momma'][reach]['integrator']['q'] = self.momma_flowlaw(param_est, self.obs_dict[reach], aux_var)

        if self.VerboseFlag: print('CALCULATING SAD FLPs')
        for reach in self.alg_dict['sad']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nflpe = np.nanmean(self.alg_dict['sad'][reach].get('n', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['sad'][reach].get('n', np.nan)), np.nanmean(self.alg_dict['sad'][reach].get('a0', np.nan))) if not np.isnan(nflpe) else (0.03, Abar_min+10.)

            param_bounds = ((0.001, np.inf), (Abar_min, np.inf))
            qbar = self.alg_dict['sad'][reach]['integrator']['qbar']
            q33 = self.alg_dict['sad'][reach]['integrator'].get('q33', np.nan) 
            res = optimize.minimize(fun=self.sad_objfun, x0=init_params, args=(self.obs_dict[reach], qbar, q33), bounds=param_bounds)
            
            self.alg_dict['sad'][reach]['integrator']['n'] = res.x[0]
            self.alg_dict['sad'][reach]['integrator']['a0'] = res.x[1]
            self.alg_dict['sad'][reach]['integrator']['q'] = self.sad_flowlaw(res.x, self.obs_dict[reach])

        if self.VerboseFlag: print('CALCULATING SIC4DVar FLPs')
        for reach in self.alg_dict['sic4dvar']:
            try: datagood = (self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0)
            except Exception: datagood = False
            if reach not in self.basin_dict['reach_ids'] or not datagood: continue
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nflpe = np.nanmean(self.alg_dict['sic4dvar'][reach].get('n', np.nan))
                Abar_min = -min(self.obs_dict[reach]['dA']) + 1
                init_params = (np.nanmean(self.alg_dict['sic4dvar'][reach].get('n', np.nan)), np.nanmean(self.alg_dict['sic4dvar'][reach].get('a0', np.nan))) if not np.isnan(nflpe) else (0.03, Abar_min+10.)

            param_bounds = ((0.001, 10.), (Abar_min, np.inf))
            qbar = self.alg_dict['sic4dvar'][reach]['integrator']['qbar']
            res = optimize.minimize(fun=self.sic4dvar_objfun, x0=init_params, args=(self.obs_dict[reach], qbar), bounds=param_bounds)
            
            self.alg_dict['sic4dvar'][reach]['integrator']['n'] = res.x[0]
            self.alg_dict['sic4dvar'][reach]['integrator']['a0'] = res.x[1]
            self.alg_dict['sic4dvar'][reach]['integrator']['q'] = self.sic4dvar_flowlaw(res.x, self.obs_dict[reach])
    
        # if self.VerboseFlag: print('Enforcing strict array shapes for Output.py compatibility...')
        for alg in self.alg_dict:
            for reach in self.basin_dict['reach_ids_all']:
                if reach in self.obs_dict and 'integrator' in self.alg_dict[alg].get(reach, {}):
                    expected_nt = self.obs_dict[reach].get('nt', 1)
                    if expected_nt < 0: expected_nt = 1
                    
                    q_arr = self.alg_dict[alg][reach]['integrator'].get('q')

                    if q_arr is not None:
                        q_arr = np.atleast_2d(q_arr)
                        if q_arr.shape != (1, expected_nt):
                            new_q = np.full((1, expected_nt), np.nan)
                            
                            copy_len = min(expected_nt, q_arr.shape[1])
                            new_q[0, :copy_len] = q_arr[0, :copy_len]
                                
                            self.alg_dict[alg][reach]['integrator']['q'] = new_q
    

    def build_soft_sfoi_system(self, n, Qbar, sigQ, facc, u_conversion):
        M_data, M_row, M_col = [], [], []
        up_dict = {}

        for junc in self.junctions:
            U = [str(u) for u in junc['upflows'] if str(u) in self.basin_dict['reach_ids_all']]
            D = [str(d) for d in junc['downflows'] if str(d) in self.basin_dict['reach_ids_all']]
            if not D or not U: continue
            
            widths = []
            for d in D:
                try:
                    k = np.argwhere(self.sword_dict['reach_id'] == int(d))[0, 0]
                    w = self.sword_dict['width'][k]
                except Exception: w = 1.0
                widths.append(w)
            total_w = sum(widths)
            
            for idx, d in enumerate(D):
                p = widths[idx] / total_w if total_w > 0 else 1.0 / len(D)
                if d not in up_dict:
                    up_dict[d] = []
                up_dict[d].append((U, p))

        reach_regions = []
        unique_regions = []
        for reach in self.basin_dict['reach_ids_all']:
            sub_basin_id = str(reach)[:6] 
            if sub_basin_id not in unique_regions:
                unique_regions.append(sub_basin_id)
            reach_regions.append(unique_regions.index(sub_basin_id))
            
        K = len(unique_regions)
        if self.VerboseFlag:
            current_basin = str(self.basin_dict['reach_ids_all'][0])[:4] if self.basin_dict['reach_ids_all'] else "the"
            print(f"      [Topology] Divided {current_basin} basin into {K} heterogeneous runoff regions.")

        runoff_prior_vals = np.full(K, 10.0)
        for k in range(K):
            idx_in_region = [i for i, r in enumerate(reach_regions) if r == k]

            facc_region = [facc[i] for i in idx_in_region if facc[i] > 0]
            facc_median = np.median(facc_region) if facc_region else np.inf

            valid_idx = [i for i in idx_in_region if not np.isnan(Qbar[i]) and facc[i] > 0 and facc[i] <= facc_median]

            if not valid_idx:
                valid_idx = [i for i in idx_in_region if not np.isnan(Qbar[i]) and facc[i] > 0]
                
            if valid_idx:
                r_vals = Qbar[valid_idx] / facc[valid_idx] / u_conversion
                runoff_prior_vals[k] = np.nanmean(r_vals)

        for i in range(n):
            reach_id = str(self.basin_dict['reach_ids_all'][i])
            M_row.append(i); M_col.append(i); M_data.append(1.0) # Q_out

            try:
                r_id = int(reach_id)
                k_idx = np.argwhere(self.sword_dict['reach_id'] == r_id)[0, 0]
                facc_current = self.sword_dict['facc'][k_idx]
                facc_up_sum = 0
                n_up = self.sword_dict['n_rch_up'][k_idx]
                if n_up > 0:
                    for up_idx in range(n_up):
                        up_r_id = self.sword_dict['rch_id_up'][up_idx, k_idx]
                        if up_r_id != 0:
                            try:
                                k_up = np.argwhere(self.sword_dict['reach_id'] == up_r_id)[0, 0]
                                facc_up_sum += self.sword_dict['facc'][k_up]
                            except Exception: pass
                delta_A = max(0, facc_current - facc_up_sum)
            except Exception:
                delta_A = 0.0

            region_idx = reach_regions[i]
            M_row.append(i); M_col.append(n + region_idx); M_data.append(-delta_A * u_conversion)

            if reach_id in up_dict:
                for U_list, p in up_dict[reach_id]:
                    for u in U_list:
                        if u in self.basin_dict['reach_ids_all']:
                            u_idx = self.basin_dict['reach_ids_all'].index(u)
                            M_row.append(i); M_col.append(u_idx); M_data.append(-p)

        M_sparse = sp.csr_matrix((M_data, (M_row, M_col)), shape=(n, n + K))
        I_n = sp.eye(n, n + K, format='csr')

        row_R_data = np.ones(K)
        row_R_row = np.arange(K)
        row_R_col = np.arange(n, n + K)
        row_R = sp.csr_matrix((row_R_data, (row_R_row, row_R_col)), shape=(K, n + K))
        
        A_sparse = sp.vstack([I_n, row_R, M_sparse])

        L_mass = np.zeros(n)
        L_vector = np.concatenate([Qbar, runoff_prior_vals, L_mass])

        W_prior = 1.0 / np.clip(sigQ, 1e-6, np.inf)

        sigR = np.clip(runoff_prior_vals * 10.0, 1e-6, np.inf) 
        W_R = 1.0 / sigR

        # covQ_mass = 0.05
        covQ_mass = 0.15
        sigM = np.clip(Qbar * covQ_mass, 10.0, np.inf) 
        W_mass = 1.0 / sigM
        
        W_1d = np.concatenate([W_prior, W_R, W_mass])
        
        invalid = np.isnan(L_vector) | np.isnan(W_1d) | np.isinf(W_1d)
        L_vector[invalid] = 0.0
        W_1d[invalid] = 0.0

        return A_sparse, L_vector, W_1d, K

  
    
    
    
    
    def build_mass_diagnostics(self, n, u_conversion):

        reach_list = [str(r) for r in self.basin_dict['reach_ids_all']]
        reach_set = set(reach_list)
        reach_index = {r: i for i, r in enumerate(reach_list)}
    
        # Rebuild the same partition logic used in build_soft_sfoi_system()
        up_dict = {}
        for junc in self.junctions:
            U = [str(u) for u in junc['upflows'] if str(u) in reach_set]
            D = [str(d) for d in junc['downflows'] if str(d) in reach_set]
            if not U or not D:
                continue
    
            widths = []
            for d in D:
                try:
                    k = np.argwhere(self.sword_dict['reach_id'] == int(d))[0, 0]
                    w = float(self.sword_dict['width'][k])
                except Exception:
                    w = 1.0
                widths.append(w)
    
            total_w = np.sum(widths)
            for idx, d in enumerate(D):
                p = widths[idx] / total_w if total_w > 0 else 1.0 / len(D)
                if d not in up_dict:
                    up_dict[d] = []
                up_dict[d].append((U, p))
    
        # Rebuild the same runoff-region assignment used in build_soft_sfoi_system()
        reach_regions = []
        unique_regions = []
        for reach in reach_list:
            sub_basin_id = str(reach)[:6]
            if sub_basin_id not in unique_regions:
                unique_regions.append(sub_basin_id)
            reach_regions.append(unique_regions.index(sub_basin_id))
        reach_regions = np.array(reach_regions, dtype=int)
        K_regions = len(unique_regions)
    
        # Build diagnostic mass matrix M_sparse and supporting arrays
        M_data, M_row, M_col = [], [], []
        delta_A = np.zeros(n, dtype=float)
        outlet_mask = np.zeros(n, dtype=bool)
    
        for i, reach_id in enumerate(reach_list):
            # Current reach discharge coefficient: +1
            M_row.append(i)
            M_col.append(i)
            M_data.append(1.0)
    
            try:
                r_id = int(reach_id)
                k_idx = np.argwhere(self.sword_dict['reach_id'] == r_id)[0, 0]
    
                facc_current = float(self.sword_dict['facc'][k_idx])
    
                facc_up_sum = 0.0
                n_up = int(self.sword_dict['n_rch_up'][k_idx])
                if n_up > 0:
                    for up_idx in range(n_up):
                        up_r_id = self.sword_dict['rch_id_up'][up_idx, k_idx]
                        if up_r_id != 0:
                            try:
                                k_up = np.argwhere(self.sword_dict['reach_id'] == up_r_id)[0, 0]
                                facc_up_sum += float(self.sword_dict['facc'][k_up])
                            except Exception:
                                pass
    
                delta_A[i] = max(0.0, facc_current - facc_up_sum)
    
                # Determine whether this reach is a basin outlet
                downstream_in_basin = False
                n_dn = int(self.sword_dict['n_rch_down'][k_idx])
                if n_dn > 0:
                    for dn_idx in range(n_dn):
                        dn_r_id = self.sword_dict['rch_id_dn'][dn_idx, k_idx]
                        if dn_r_id != 0 and str(dn_r_id) in reach_set:
                            downstream_in_basin = True
                            break
                outlet_mask[i] = not downstream_in_basin
    
            except Exception:
                delta_A[i] = 0.0
                outlet_mask[i] = False
    
            # Lateral runoff term: -(delta_A * u_conversion) * R_region
            region_idx = reach_regions[i]
            M_row.append(i)
            M_col.append(n + region_idx)
            M_data.append(-delta_A[i] * u_conversion)
    
            # Routed upstream inflow: -p * Q_u
            if reach_id in up_dict:
                for U_list, p in up_dict[reach_id]:
                    for u in U_list:
                        if u in reach_index:
                            u_idx = reach_index[u]
                            M_row.append(i)
                            M_col.append(u_idx)
                            M_data.append(-p)
    
        M_sparse = sp.csr_matrix((M_data, (M_row, M_col)), shape=(n, n + K_regions))
    
        return {
            "n": n,
            "K_regions": K_regions,
            "M_sparse": M_sparse,
            "delta_A": delta_A,
            "reach_regions": reach_regions,
            "outlet_mask": outlet_mask,
            "u_conversion": u_conversion,
        }

    def compute_mass_conservation_metrics(self, x_hat, diag, qmin_local=5.0):
        n = diag["n"]
        x_hat = np.asarray(x_hat).ravel()
        Qintegrator = x_hat[:n]

        mass_resid = np.asarray(diag["M_sparse"] @ x_hat).ravel()

        denom_local = np.maximum(np.abs(Qintegrator), qmin_local)

        eps_local = np.abs(mass_resid / denom_local)

        reach_list = [str(r) for r in self.basin_dict['reach_ids_all']]
        reach_eps_dict = {reach_list[i]: float(eps_local[i]) for i in range(n)}
    
        return reach_eps_dict



    def integrate(self):
          """Integrate reach-level FLPE data. (Main Runner)"""
          # if self.VerboseFlag: print('creating junction list')
          self.CreateJunctionList()       

          FlowLevels = ['Mean', 'q33'] 
          m = len(self.junctions)
          for i, junction in enumerate(self.junctions): junction['row_num'] = i    
          n = len(self.basin_dict['reach_ids_all'])
          
          u_conversion = 1000.0 / (365.25 * 24 * 3600)
          self.mass_diag_static = self.build_mass_diagnostics(n, u_conversion)

          if self.VerboseFlag:
              print(f'Number of junctions = {m}, Number of reaches = {n}')

          for FlowLevel in FlowLevels:
              if self.VerboseFlag:
                  print(f'\nRunning flow level {FlowLevel}')
              residuals = {} 
              for alg in self.alg_dict: residuals[alg] = np.full((n,), np.nan)
              
              for i in range(self.params_dict['niter']):
                  if self.VerboseFlag:
                      print(f'  Running iteration {i+1} / {self.params_dict["niter"]}')
                  residuals = self.integrator_optimization_calcs(m, n, FlowLevel, residuals)

          if self.params_dict.get('quit_before_flpe', False):
              sys.exit('done with integration... exiting')

          if self.VerboseFlag:
              print('Computing all FLPs (Final Parameter Estimation)')
          self.compute_FLPs()
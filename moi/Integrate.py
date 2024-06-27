#Standard imports
import warnings
import sys
import datetime

# Third-party imports
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.linalg import solve
from numpy import random

class Integrate:
     """Integrates reach-level FLPE algorithm data.
     Attributes
     ----------
     alg_dict: dict
         dictionary of algorithm data stored by algorithm name as numpy arrays
     basin_dict: dict
         dict of reach_ids and SoS file needed to process entire basin of data
     integ_dict: dict
         dict of integrator estimate data
     params_dict: dict
         dict of integrator parameters 
     sos_dict: dict
         dictionary of SoS data
     Methods
     -------
     get_pre_mean_q()
         calculate the mean discharge for each reach-level FLPE algorithm
     integrate()
         integrate and store reach-level data
     """

     def __init__(self, alg_dict, basin_dict, sos_dict, sword_dict, obs_dict,params_dict,Branch,VerboseFlag):
          """
          Parameters
          ----------
          alg_dict: dict
               dictionary of algorithm data stored by algorithm name as numpy arrays
          basin_dict: dict
               dict of reach_ids and SoS file needed to process entire basin of data
          sos_dict: dict
               dictionary of SoS data
          sword_dict: dict
               dictionary of SWORD data
          params_dict: dict
               dictionary of MOI parameters
          obs_dict: dict
               dictionary of SWOT observation data
          Branch: string
               constrained or unconstrained
          VerboseFlag: logical
          """

          self.alg_dict = alg_dict
          self.basin_dict = basin_dict
          self.obs_dict = obs_dict
          self.sword_dict = sword_dict
          self.integ_dict = {
               "pre_q_mean": np.array([]),
               "q_mean": np.array([]),
               "flpe": {
                    "neobam" : np.array([]),
                    "hivdi" : np.array([]),
                    "metroman" : np.array([]),
                    "momma" : np.array([]),
                    "sad" : np.array([]),
                    "sic4dvar" : np.array([])
               }
          }
          self.params_dict = params_dict
          self.sos_dict = sos_dict
          self.Branch=Branch
          self.VerboseFlag = VerboseFlag
          print('getting pre mean q')
          self.get_pre_mean_q()

          if self.Branch == 'constrained':
              self.get_gage_mean_q()

     def get_pre_mean_q(self):
        """Calculate the mean discharge for each reach-level FLPE algorithm.
        This represents the mean in time for each algorithm and each reach.
        This should be done prior to integration operations.
        """
        for alg in self.alg_dict:
            for reach in self.alg_dict[alg]:
                with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            
                            if self.alg_dict[alg][reach]['s1-flpe-exists']:
                                self.alg_dict[alg][reach]['qbar']=np.nanmean(self.alg_dict[alg][reach]['q'])
                                self.alg_dict[alg][reach]['q33']=np.nanquantile(self.alg_dict[alg][reach]['q'],.33)
                            
                            if np.isnan(self.alg_dict[alg][reach]['qbar']):
                                self.alg_dict[alg][reach]['qbar']=self.sos_dict[reach]['Qbar']
                                self.alg_dict[alg][reach]['q33']=self.sos_dict[reach]['q33']
     
     def get_gage_mean_q(self):
         """Calculate the gaged mean discharge for each reach in the domain that has a gage
         for the times in the SWOT files, for that reach. This should be done prior to 
         integration operations. This should only be called during a constrained run
         """
         for reach in self.sos_dict.keys():
             if reach in self.obs_dict.keys():

                 # check whether reach is gaged. if not, do nothing
                 try:
                     agency=self.sos_dict[reach]['gage']['source']
                     gaged_reach=True
                 except:
                     #print('no gage found for reach ',reach)
                     gaged_reach=False

                 if gaged_reach:
                     epoch = datetime.datetime(2000,1,1,0,0,0)
                     gagedQs=[]
                     for time in self.obs_dict[reach]['t']:
                         try:
                             ordinal_time=(epoch + datetime.timedelta(seconds=time)).toordinal()
                         except:
                             ordinal_time=np.nan
                             warnings.warn('problem with time conversion to ordinal, most likely nan value')

                         # find index in gaged timeseries that matches this swot observation and add to list
                         try:
                             idx = np.argwhere(self.sos_dict[reach]['gage']['t']==ordinal_time)
                             idx = idx[0,0]
                             gagedQs.append(self.sos_dict[reach]['gage']['Q'][idx])
                         except:
                             idx=np.nan
                             #print('gaged time not found')

                     # use the list to compute stats
                     self.sos_dict[reach]['gage']['Qbar']=np.nan
                     self.sos_dict[reach]['gage']['q33']=np.nan
                     if gagedQs:
                         try:
                             Qbar=np.nanmean(gagedQs)
                             Q33=np.nanquantile(gagedQs,.33)
                             self.sos_dict[reach]['gage']['Qbar']=Qbar
                             self.sos_dict[reach]['gage']['q33']=Q33
                         except:
                             print('problem extracting gage flow stats over swot period for reach',reach)


     def pull_sword_attributes_for_reach(self,k):
         """
         Pull out needed SWORD data from the continent dataset arrays for a particular reach    
         """
    
         sword_data_reach={}
         # extract all single-dimension variables, including number of orbits and reach ids needed for multi-dim vars
         for key in self.sword_dict:
             if np.shape(self.sword_dict[key]) == (self.sword_dict['num_reaches'],):
                 sword_data_reach[key]=self.sword_dict[key][k]

         # extract multi-dim vars
         for key in self.sword_dict:            
             if key == 'rch_id_up':
                 sword_data_reach[key]=self.sword_dict[key][0:sword_data_reach['n_rch_up'],k]
             elif key == 'rch_id_dn':
                 sword_data_reach[key]=self.sword_dict[key][0:sword_data_reach['n_rch_down'],k]
             elif key == 'swot_orbits':
                 sword_data_reach[key]=self.sword_dict[key][0:sword_data_reach['swot_obs'],k]
    
         return sword_data_reach

     def ChecksPriorToAddingJunction(self,junction_to_check):
        #check to see if this one already exists
         AlreadyExists=False
         for junction in self.junctions:
             if junction['upflows']==junction_to_check['upflows'] and junction['downflows']==junction_to_check['downflows']: 
                 AlreadyExists=True    
    
         #check to see if all reaches we've identified area in the basin 
         AllReachesInReachFile=True
         for r in junction_to_check['upflows']:
             if str(r) not in self.basin_dict['reach_ids_all']:
                 AllReachesInReachFile=False
         for r in junction_to_check['downflows']:
             if str(r) not in self.basin_dict['reach_ids_all']:
                 AllReachesInReachFile=False            
            
         return AlreadyExists,AllReachesInReachFile

     def CreateJunctionList(self):
         # create list of junctions
         self.junctions=list()

         self.junctions_valid=True

         for reach in self.basin_dict['reach_ids_all']:
             reach=np.int64(reach)
             k=np.argwhere(self.sword_dict['reach_id'] == reach)
             k=k[0,0]
    
             # extract reach dictionary for reach k
             sword_data_reach=self.pull_sword_attributes_for_reach(k) 

             #1 try adding the upstream junction
             junction_up=dict()    
             junction_up['originating_reach_id']=reach
    
             #1.1 add the reaches upstream of this junction
             junction_up['upflows']=list()
             for i in range(sword_data_reach['n_rch_up']):
                 junction_up['upflows'].append(sword_data_reach['rch_id_up'][i] )
       
             #1.2 for one of the reaches upstream of this junction, add all their downstream reaches
             if len(junction_up['upflows'])>0:
                 junction_up['downflows']=list()

                 # sometimes sword says there are upstream reaches and there actually isnt
                 # in these cases skip the reach and raise a warning
                 if not any(junction_up['upflows']):
                    warnings.warn(f'Upstream reaches not found for reach {reach}')
                    self.junctions_valid=False
                    continue

                 kup=np.argwhere(self.sword_dict['reach_id'] == junction_up['upflows'][0])
                 kup=kup[0,0]
                 sword_data_reach_up=self.pull_sword_attributes_for_reach(kup)
                 for j in range(sword_data_reach_up['n_rch_down']):
                     junction_up['downflows'].append(sword_data_reach_up['rch_id_dn'][j] )

                 AlreadyExists,AllReachesInReachFile=self.ChecksPriorToAddingJunction(junction_up)

                 if not AlreadyExists and AllReachesInReachFile:
                 #if not AlreadyExists:
                     self.junctions.append(junction_up)

             #2 try adding the downstream junction
             junction_dn=dict()
             junction_dn['originating_reach_id']=reach #just adding this for bookkeeping/debugging purposes
    
             #2.1 add the reaches downstream of this junction
             junction_dn['downflows']=list()
             for i in range(sword_data_reach['n_rch_down']):
                 junction_dn['downflows'].append(sword_data_reach['rch_id_dn'][i] )            

             #2.2 for one of the reaches downstream of the junction, add all their upstream reaches
             if len(junction_dn['downflows'])>0:
                 junction_dn['upflows']=list()

                 # sometimes sword says there are downstream reaches and there actually isnt
                 # in these cases skip the reach and raise a warning
                 if not any(junction_dn['downflows']):
                    warnings.warn(f'Downstream reaches not found for reach {reach}')
                    self.junctions_valid=False
                    continue

                 kdn=np.argwhere(self.sword_dict['reach_id'] == junction_dn['downflows'][0])
                 kdn=kdn[0,0]
                 sword_data_reach_dn=self.pull_sword_attributes_for_reach(kdn)
                 for j in range(sword_data_reach_dn['n_rch_up']):
                     junction_dn['upflows'].append(sword_data_reach_dn['rch_id_up'][j] )

                 AlreadyExists,AllReachesInReachFile=self.ChecksPriorToAddingJunction(junction_dn)

                 if junction_dn['upflows']==[0]:
                     print('gotcha!')
                     print('junction=',junction_dn)
                     print('kdn=',kdn)
                     print('junction down=',junction_dn['downflows'][0])
                     print(self.sword_dict['reach_id'][kdn])

                 if not AlreadyExists and AllReachesInReachFile:
                 #if not AlreadyExists:
                     self.junctions.append(junction_dn) 

     def RemoveDamReaches(self):
         for reachid in self.basin_dict['reach_ids']:
             k=np.argwhere(self.sword_dict['reach_id']==int(reachid))
             k=k[0,0]
    
             if self.sword_dict['n_rch_down'][k] == 1:
                 # 0. find the reach downstream of the target reach
                 rid_down=str(self.sword_dict['rch_id_dn'][0,k])
        
                 # if the reach downstream of the target reach is a dam...
                 if rid_down[-1] == '4':
            
                     k_down=np.argwhere(self.sword_dict['reach_id']==int(rid_down))
                     k_down=k_down[0,0]
            
                     # 1. find the reach downstream of the reach downstream of the target reach
                     if self.sword_dict['n_rch_down'][k_down] == 1:
                         rid_down_down=str(self.sword_dict['rch_id_dn'][0,k_down])
                         k_down_down=np.argwhere(self.sword_dict['reach_id']==int(rid_down_down))
                         k_down_down=k_down_down[0,0]             
                                
                         if rid_down_down[-1] == '1':
#                             if self.VerboseFlag:
#                                 print('Removing reach ',rid_down)
                             #2. point the target reach at the reach downstream of the reach downstream of the target reach
                             self.sword_dict['rch_id_dn'][0,k] = int(rid_down_down)
                             #3. point the reach downstream of the reach downstream of the target reach, back to the target reach
                             self.sword_dict['rch_id_up'][0,k_down_down]=int(reachid)
                    
                         elif rid_down_down[-1] == '4':
                             # this happens if there are two type 4 reaches in a row, downstream of the target reach
                             if self.sword_dict['n_rch_down'][k_down_down] == 1:
#                                 if self.VerboseFlag:
#                                     print('Removing reach ',rid_down, ' and',rid_down_down)
                    
                                 #1b. find the reach downstream of the reach downstream of the reach downstream of the target reach
                                 rid_down_down_down=str(self.sword_dict['rch_id_dn'][0,k_down_down])
                                 k_down_down_down=np.argwhere(self.sword_dict['reach_id']==int(rid_down_down_down))
                                 k_down_down_down=k_down_down_down[0,0]        
                        
                                 if rid_down_down_down[-1]=='1':
                                     #2b. point the target reach at the reach downstream of the reach downstream of the reach downstream of the target reach
                                     self.sword_dict['rch_id_dn'][0,k] = int(rid_down_down_down)
                                     #3b. point the reach downstream of the reach downstream of the reach downstream of the target reach, back to the target reach
                                     self.sword_dict['rch_id_up'][0,k_down_down_down]=int(reachid)
                    

     def MOI_ObjectiveFunc(self,Q,Qbar,sigmaQ):
         # Q - value of discharge vector at which to evaluate objective function
         # Qbar - prior value of Q e.g. stage 1 McFLI
         # sigmaQ - vector of Q uncertainty
         n=np.size(Q,0)
         C=np.diag( np.reshape(sigmaQ**-1,[n,]) )
         d=Qbar*sigmaQ**(-1)
         res=C@Q-d
         y=np.linalg.norm(res,2)
         return y


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
          #keep this equation exactly how this function is written in riverobs 
          qbam = ((d_x_area+bam_Abar)**(5/3) * reach_width**(-2/3) * \
                    (reach_slope)**(1/2)) / bam_n
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
          #keep this equation exactly how this function is written in riverobs 
          hivdi_n_inv = hivdi_alpha * (
               (d_x_area+hivdi_Abar)/reach_width)**hivdi_beta
          qhivdi = (
               (d_x_area+hivdi_Abar)**(5/3) * reach_width**(-2/3) *
               (reach_slope)**(1/2)) * hivdi_n_inv
          qhivdi=np.reshape(qhivdi,(1,len(d_x_area)))
          return qhivdi

     def metroman_objfun(self,params,obs,qbar_target,q33_target): 
          q=self.metroman_flowlaw(params,obs)
          qbar=np.nanmean(q)
          #y=(qbar-qbar_target)**2
          y=abs(qbar-qbar_target)
          if not np.isnan(q33_target):
             q33_alg=np.nanquantile(q,.33)
             #y+=(q33_alg-q33_target)**2 
             y+=abs(q33_alg-q33_target)
          return y

     def metroman_flowlaw(self,params,obs):
          d_x_area=obs['dA']
          reach_width=obs['w']
          reach_slope=obs['S']
          metro_ninf=params[0]
          metro_p=params[1]
          metro_Abar=params[2]
          #keep this equation exactly how this function is written in riverobs 
          metro_n = metro_ninf * (
               (d_x_area+metro_Abar) / reach_width)**metro_p
          metro_q = (
               (d_x_area+metro_Abar)**(5/3) * reach_width**(-2/3) *
               (reach_slope)**(1/2)) / metro_n

          metro_q=np.reshape(metro_q,(1,len(d_x_area)))
          return metro_q

     def momma_objfun(self,params,obs,qbar_target,q33_target,aux_var): 
          q=self.momma_flowlaw(params,obs,aux_var)
          if np.all(np.isnan(q)):
              return 1e9
          qbar=np.nanmean(q)
          y=(qbar-qbar_target)**2

          if not np.isnan(q33_target):
             q33_alg=np.nanquantile(q,.33)
             y+=(q33_alg-q33_target)**2 

          #impose a penalty if bankfull depth gets too low
          B=params[0]
          H=params[1]
          Db=H-B #bankfull elevation - bottom elevation
          if Db<0.2 and Db >= 0.1:
              yfac=2.
          elif Db < 0.1:
              yfac=10.
          else: 
              yfac=1.
          y*=yfac

          return y

     def momma_flowlaw(self,params,obs,aux_var):
          reach_height=obs['h']
          reach_width=obs['w']
          reach_slope=obs['S']
          momma_B=params[0]
          momma_H=params[1]
          momma_Save=aux_var
          #keep this part exactly how this function is written in riverobs 
          momma_r = 2
          momma_nb = 0.11 * momma_Save**0.18

          momma_q=np.empty( (obs['nt'],)) 

          if momma_H <= momma_B+0.1:
               momma_q=np.inf
          else:
               for t in range(obs['nt']):
                    if momma_B > reach_height[t]:
                        print('MOMMA flow law got B > Hobs')
                        print('reach height=',reach_height[t])
                    log_factor = np.log10((momma_H-momma_B)/(reach_height[t]-momma_B))

                    if reach_height[t] <= momma_H:
                         momma_n = momma_nb*(1+log_factor)
                         log_check = log_factor > -1
                    else:
                         momma_n = momma_nb*(1-log_factor)
                         log_check = log_factor < 1

                    momma_q[t] = (
                         ((reach_height[t] - momma_B)*(momma_r/(1+momma_r)))**(5/3) *
                         reach_width[t] * reach_slope[t]**(1/2)) / momma_n

               momma_q=np.reshape(momma_q,(1,len(reach_height)))
          return momma_q

     def sad_objfun(self,params,obs,qbar_target,q33_target): 
          qsad=self.sad_flowlaw(params,obs)
          if np.all(np.isnan(qsad)):
              return 1e9
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
          #keep this equation exactly how this function is written in riverobs 
          qsad = ((d_x_area+sad_Abar)**(5/3) * reach_width**(-2/3) * \
                    (reach_slope)**(1/2)) / sad_n
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
          #keep this equation exactly how this function is written in riverobs 
          qsic4dvar = ((d_x_area+sic4dvar_Abar)**(5/3) * reach_width**(-2/3) * \
                    (reach_slope)**(1/2)) / sic4dvar_n
          qsic4dvar=np.reshape(qsic4dvar,(1,len(d_x_area)))
          return qsic4dvar

     def calcG(self,m,n):
        #define G matrix
        G=np.zeros((m,n))

        for junction in self.junctions:
            row=junction['row_num']
            upcols=list()
            for upflow in junction['upflows']:
                try:
                     kup=self.basin_dict['reach_ids_all'].index(str(upflow))
                     upcols.append(kup)
                except: 
                    print('did not find reach:',upflow)
                    print('... in junction',junction)

            downcols=list()
            for downflow in junction['downflows']:
                try:
                     kdn=self.basin_dict['reach_ids_all'].index(str(downflow))
                     downcols.append(kdn)
                except:
                     print('did not find reach',downflow)
                     print('... in junction',junction)

            for upcol in upcols:
                G[row,upcol]=1
            for downcol in downcols:
                G[row,downcol]=-1

        return G

     def initialize_integration_vars(self,alg,FlowLevel,PreviousResiduals,n):

         self.GoodFLPE[alg]=True
         Qbar=np.empty([n,])
         sigQ=np.empty([n,])
         facc=np.empty([n,])
         runoff=np.empty([n,])
         datasource=[]

         i=0
         for reach in self.basin_dict['reach_ids_all']:
            # assign drainage area
            reachint=np.int64(reach)
            k=np.argwhere(self.sword_dict['reach_id'] == reachint)
            k=k[0,0]
            sword_data_reach=self.pull_sword_attributes_for_reach(k) 
            facc[i]=sword_data_reach['facc']

            if reach in self.alg_dict[alg].keys():

                # if this reach is gaged using the mean flow in the sos, rather than the algorithm
                nrt_gaged_reach=(self.sos_dict[reach]['overwritten_indices']==1) and \
                                  (self.sos_dict[reach]['overwritten_source']!='grdc') and \
                                  (self.sos_dict[reach]['cal_status']==1 ) and \
                                  ('Qbar' in self.sos_dict[reach]['gage'].keys()) and \
                                  ('q33' in  self.sos_dict[reach]['gage'].keys())


                if (self.Branch == 'constrained') and nrt_gaged_reach:
                    if FlowLevel == 'Mean':
                        #Qbar[i]=self.sos_dict[reach]['Qbar']
                        Qbar[i]=self.sos_dict[reach]['gage']['Qbar']
                    elif FlowLevel == 'q33':
                        #Qbar[i]=self.sos_dict[reach]['q33']
                        Qbar[i]=self.sos_dict[reach]['gage']['q33']

                    sigQ[i]=Qbar[i]*self.params_dict['Gage_Uncertainty']
                    datasource.append('Gage')
                else:
                    if FlowLevel == 'Mean':
                        if np.ma.is_masked(self.alg_dict[alg][reach]['qbar']):
                            Qbar[i]=np.nan #
                        else:

                            nstdev=10.
                            if abs(self.alg_dict[alg][reach]['qbar']-self.sos_dict[reach]['Qbar']) > \
                                    self.sos_dict[reach]['Qbar']*self.params_dict['FLPE_Uncertainty']*nstdev:
                                Qbar[i]=np.nan
                            else:
                                Qbar[i]=self.alg_dict[alg][reach]['qbar']
                    elif FlowLevel == 'q33':
                        try:
                            if np.ma.is_masked(self.alg_dict[alg][reach]['q33']):
                                Qbar[i]=np.nan
                            else:
                                nstdev=10.
                                if abs(self.alg_dict[alg][reach]['qbar']-self.sos_dict[reach]['Qbar']) > \
                                        self.sos_dict[reach]['Qbar']*self.params_dict['FLPE_Uncertainty']*nstdev:
                                    Qbar[i]=np.nan
                                else:
                                    Qbar[i]=self.alg_dict[alg][reach]['q33']
                        except:
                            print('did not find q33. reach=',reach)
                            Qbar[i]=np.nan

                    if np.isnan(PreviousResiduals[alg][i]):
                        sigQ[i]=Qbar[i]*self.params_dict['FLPE_Uncertainty']
                    else:
                        if (self.Branch == 'constrained') and nrt_gaged_reach:
                           sigQ[i]=Qbar[i]*self.params_dict['Gage_Uncertainty']
                        else:
                           #sigQ[i]=max(abs(PreviousResiduals[alg][i]),Qbar[i]*self.params_dict['FLPE_Uncertainty'])
                           #sigQ[i]=max(abs(PreviousResiduals[alg][i]),Qbar[i]*self.params_dict['FLPE_Uncertainty'])**self.params_dict['norm']
                           #sigQ[i]=max(abs(PreviousResiduals[alg][i]),Qbar[i]*self.params_dict['FLPE_Uncertainty'])
                           #sigQ[i]=max(abs(PreviousResiduals[alg][i]),Qbar[i]*.01)
                           sigQ[i]=max(abs(PreviousResiduals[alg][i]),Qbar[i]*.01)**(-(self.params_dict['norm']-2.0))
                    datasource.append('FLPE')
            else:
                 Qbar[i]=np.nan
                 sigQ[i]=np.nan
                 datasource.append('None')
            i+=1
 
         # compute runoff and average runoff
         for i in range(n):
             if not np.isnan(Qbar[i]):
                 runoff[i]=Qbar[i]/facc[i]/1000**2*86400*365
             else:
                 runoff[i]=np.nan

         runoff_avg=np.nanmean(runoff)
         #if self.VerboseFlag:
         #    print('average runoff=',runoff_avg,'m/yr')

         # this handles accidental nans still in the flow estimates
         #   setting to zero should let these get reset
         #Qbar[np.isnan(Qbar)]=0.
         #Qbar[np.isinf(Qbar)]=0.

         # fill initial average discharge with average runoff
         for i in range(n):
             if np.isnan(Qbar[i]) or np.isinf(Qbar[i]):
                 Qbar[i]=runoff_avg*facc[i]*1000**2/86400/365
                 sigQ[i]=Qbar[i]*self.params_dict['Fill_Uncertainty']

         bignumber=1e9
         # for any values of zero in FLPE Qbar where we don't have residuals, set uncertainty to a big number
         #   (note - this should now be mostly obsolete)
         sigQmin=10.
         for i in range(n):
             if Qbar[i]==0. and np.isnan(PreviousResiduals[alg][i]) :
                 sigQ[i]=bignumber
             if sigQ[i] < sigQmin and not datasource[i]=='Gage':
                sigQ[i] = sigQmin

         for i in range(n):
             if np.isinf(PreviousResiduals[alg][i]):
                 Qbar[i]=5.
                 sigQ[i]=0.1

         #check for whether FLPE data are ok
         iFLPE=np.where(np.array(datasource)=='FLPE')
         if np.all(Qbar[iFLPE]==0):
             FLPE_Data_OK=False
             self.GoodFLPE[alg]=False
         else:
             FLPE_Data_OK=True

         #i=0
         #for reach in self.basin_dict['reach_ids_all']:
         #    print(i,reach,'Qbar=',Qbar[i],'stdQ=',sigQ[i])
         #    i+=1
         #sys.exit('Stopping at dev point')

         return Qbar,sigQ,FLPE_Data_OK,facc
        

     def integrator_optimization_calcs(self,m,n,FlowLevel,PreviousResiduals):

          #0 initialize dictionary of residuals, to be returned and passed back in for next iteration
          residuals={}
          self.GoodFLPE={}

          #alg_list=['geobam']
          alg_list=self.alg_dict

          for alg in alg_list:
               #1. compute "integrated" discharge. 
               print('    RUNNING MOI for ',alg)

               #initialize integration variables
               Qbar,sigQ,FLPE_Data_OK,facc = self.initialize_integration_vars(alg,FlowLevel,PreviousResiduals,n)

               #print('Prior Q[51]=',Qbar[51])

               # compute the G matrix, which defines mass conservation points
               G=self.calcG(m,n)
               
               '''
               import csv
               with open('G.csv','w',newline='') as csvfile:
                   Gwriter = csv.writer(csvfile, delimiter=' ',
                          quotechar='|', quoting=csv.QUOTE_MINIMAL)
                   Gwriter.writerow(self.basin_dict['reach_ids_all'])
                   for i in range(m):
                      Gwriter.writerow(G[i,:])
               '''
 
               # solve integrator problem
               if not FLPE_Data_OK or not self.junctions_valid:
                   print('FLPE data not ok for ',alg,'. setting Qintegrator = Qprior here')
                   Qintegrator=Qbar
                   residuals[alg]=np.full((n,),np.nan)
               else:
                   UncertaintyMethod='Linear' 
                   if self.params_dict['method'] == 'nonlinear':
                       cons_massbalance=optimize.LinearConstraint(G,np.zeros(m,),np.zeros(m,))
                       Qmin=0.
                       bignumber=1.0e9
                       cons_positive=optimize.LinearConstraint(np.eye(n),np.ones(n,)*Qmin,np.ones(n,)*bignumber)
                       Q0,covQ=self.compute_linear_Qhat(alg,m,n,sigQ,Qbar,G)
                       np.clip(Q0,1.,np.inf,out=Q0)

                       res=optimize.minimize(fun=self.MOI_ObjectiveFunc,x0=Q0,args=(Qbar,sigQ),method='SLSQP',                      
                           options={'maxiter':500},
                           constraints=(cons_massbalance,cons_positive))

                       if res.success:
                           Qintegrator=res.x
                           Success=True
                       else:
                           if self.VerboseFlag:
                               print('      Used linear solution :(...')
                               #print(res)
                               #sys.exit('stopping at dev point')
    
                           Qintegrator=Q0
                           res.success=True
                           Success=True
                   elif self.params_dict['method'] == 'linear': 
                       Qintegrator,covQ=self.compute_linear_Qhat(alg,m,n,sigQ,Qbar,G)
                       Success=True

                   stdQc_rel=self.compute_integrator_uncertainty(alg,m,n,covQ,Qintegrator,UncertaintyMethod,G)

                   if type(stdQc_rel) == bool:
                    if stdQc_rel == False:
                        Success=False
                   if not Success:
                       print('Optimization failed for ', alg)
                       if self.VerboseFlag: 
                           print(res)
                           print('Qbar=',Qbar)
                           #sys.exit('stopping at dev point')
                       Qintegrator=Qbar

                   #compute residuals
                   if Success:
                       residuals[alg]= Qbar-Qintegrator
                       for i in range(n):
                           if Qintegrator[i]<0.:
                               residuals[alg][i]=np.inf#this is a code to how to treat uncertainty on next iteration
                   else:
                       residuals[alg]=np.full((n,),np.nan)

                   if self.params_dict['quit_before_flpe']:
                       # write out data if we are quitting before flpe, debug mode
                       if FlowLevel == 'Mean':
                          df=pd.DataFrame(list(self.basin_dict['reach_ids_all']),columns=['reachids'])
                          df['Qbar']=Qbar
                          df['sigQ']=sigQ
                          df['stdQc_rel']=stdQc_rel
                          #df['data source']=datasource
                          df['Qintegrator']=Qintegrator
                          df['facc']=facc
                          fname=alg+'integrator_init.csv'
                          df.to_csv(fname)


               # end of calcs for this alg: 1. compute integrated discharge


               #2. save data
               i=0
               for reach in self.basin_dict['reach_ids_all']:
                   if reach in self.alg_dict[alg].keys():
                       if 'integrator' not in self.alg_dict[alg][reach]:
                           self.alg_dict[alg][reach]['integrator']={}
                           self.alg_dict[alg][reach]['integrator']['qbar']=np.nan
                           self.alg_dict[alg][reach]['integrator']['sbQ_rel']=np.nan
                       if FlowLevel == 'Mean':
                           self.alg_dict[alg][reach]['integrator']['qbar']=Qintegrator[i]
                           if  FLPE_Data_OK and self.junctions_valid:
                               if Success:
                                   self.alg_dict[alg][reach]['integrator']['sbQ_rel']=stdQc_rel[i]
                               else:
                                   warnings.warn('Topology probelm encountered, using prior uncertainty for sbQ_rel')
                                   self.alg_dict[alg][reach]['integrator']['sbQ_rel']=self.params_dict['FLPE_Uncertainty']
                           else:
                               self.alg_dict[alg][reach]['integrator']['sbQ_rel']=self.params_dict['FLPE_Uncertainty']

                       elif FlowLevel == 'q33':
                           self.alg_dict[alg][reach]['integrator']['q33']=Qintegrator[i]
                   #if reach == '73120000521':
                   #    print('        reach=',reach,'i=',i)
                   #    if FlowLevel == 'Mean':
                   #        print('        qbar=',self.alg_dict[alg][reach]['integrator']['qbar'])
                   i+=1

          # there is a for loop that goes over all algorithms

          return residuals

     def compute_linear_Qhat(self,alg,m,n,sigQ,Qbar,G):
          # using the Adjustments formulation 

          # compute covariance matrix: from compute_integrator_uncertainty
          sigQ0=sigQ 
          sigQmin=1.
          np.clip(sigQ0,sigQmin,np.inf,out=sigQ0) #prevent any zero values in sigQ
          sigQv=np.reshape(sigQ0,(n,1))
          sigQv=sigQv**(self.params_dict['norm']/2.)
          rho=self.params_dict['rho']
          covQ = np.matmul(sigQv,  sigQv.transpose()) * (rho* np.ones((n,n)) + (np.eye(n)-rho*np.eye(n) )   )  
          
          try:
              xhat= (np.eye(n)-covQ @ G.T @ np.linalg.inv(G@covQ@G.T) @ G ) @ Qbar 
          except:
              warnings.warn('adjustment calculation failed. returning Qhat=Qbar')
              return Qbar,covQ

          Q0=np.empty( (n,) )
          for i in range(n):
              Q0[i]=xhat[i]

          return Q0,covQ

     def compute_integrator_uncertainty(self,alg,m,n,covQ,Qbar,UncertaintyMethod,G):

          if UncertaintyMethod == 'Ensemble':
              if alg == 'metroman_ignore':
                  nEnsemble=20
                  #covQind=sigQ**2*np.eye(n)
                  Qens=random.multivariate_normal(Qbar,covQ,nEnsemble)
                  Qmin=10.
                  Qens[Qens<Qmin]=Qmin
      
                  Qensc=np.empty((nEnsemble,n))
                  for i in range(nEnsemble): 
                       res=optimize.minimize(fun=self.MOI_ObjectiveFunc,x0=np.reshape(Qens[i,:],[n,]),
                               args=(Qens[i,:],sigQ0),method='SLSQP',                      
                               constraints=(cons_massbalance,cons_positive))
                       Qensc[i,:]=res.x

                  stdQc=Qensc.std(axis=0)
                  stdQc_rel=stdQc/Qintegrator 
              else:
                  stdQc_rel=np.full(n,self.params_dict['FLPE_Uncertainty'])
          elif UncertaintyMethod == 'Linear':
              try:
                  σ0=np.sqrt(np.mean(covQ))
                  Q=covQ/σ0**2 #note this is the co-factor matrix from Kyle snow's book... not discharge
                  covQc=σ0**2 * (Q - Q@G.T@np.linalg.inv(G@Q@G.T)@G@Q) 
                  #covQc=covQ-covQ @ G.T @ np.linalg.inv(G @ covQ @ G.T) @ G @ covQ
                  stdQc=np.sqrt(np.diagonal(covQc))
                  stdQc_rel=stdQc/np.abs(Qbar)
              except:
                  warnings.warn('adjustment uncertainty calculation failed. returning prior uncertainty')
                  return np.reshape(np.sqrt(np.diagonal(covQ)),(n,1))
      
          return stdQc_rel 
 
     def GetM(self,sigQv,covQ,G,m,n):
          # m: number of junctions
          # n: number of reaches
          # G: mxn
          E=np.zeros((n,n)) #nxn
          np.fill_diagonal(E,-2*np.reciprocal(sigQv))

          #E= -2 * np.lingalg.inv(covQ)    
          #E= -2 * np.lingalg.inv(covQ)    

          F=np.transpose(G) #nxm
          H=np.zeros((m,m)) #mxm
          A=np.block([
              [E,F],
              [G,H]
          ])  # A is n+m x n+m
  
          I=np.zeros((n,m))
          J=np.zeros((m,n))

          B=np.block([
              [E,I],
              [J,H]
          ]) # B is n+m x n+m
          if np.linalg.det(A) == 0:
            raise ValueError('Singular Matrix found, indicative of SWORD topology problems')

          # using inv
          M=B@np.linalg.inv(A)

          return M,A
          

     def compute_FLPs(self):         
          #2.1 geobam   
          print('CALCULATING GeoBAM FLPs')
          for reach in self.alg_dict['neobam']:
               #print('CALCULATING FLPs:',reach)
               '''
               try:
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       print('good')
               except:
                   continue
               '''
               try: 
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       datagood=True
               except:
                   continue
               
               if reach not in self.basin_dict['reach_ids']:
                   # reach is not observed. do not calculate FLPs
                   continue

               with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    nhat=np.nanmean(self.alg_dict['neobam'][reach]['n'])

               if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                    
                    Abar_min=-min(self.obs_dict[reach]['dA'])+1
                    
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)

                         if not np.isnan(nhat):
                             init_params=(nhat,np.nanmean(self.alg_dict['neobam'][reach]['a0']))
                         else:
                             init_params=(0.03,Abar_min+10.)
                    #param_bounds=( (0.001,np.inf),(-min(self.obs_dict[reach]['dA'])+1,np.inf))
                    param_bounds=( (0.001,np.inf),(Abar_min,np.inf))
                    qbar=self.alg_dict['neobam'][reach]['integrator']['qbar'] 
                    if 'q33' in self.alg_dict['neobam'][reach]['integrator']:
                        q33=self.alg_dict['neobam'][reach]['integrator']['q33']
                    else:
                        q33=nan 
                    res = optimize.minimize(fun=self.bam_objfun,
                                        x0=init_params,
                                        args=(self.obs_dict[reach],qbar,q33),
                                        bounds=param_bounds )
                    param_est=res.x

                    #store output
                    self.alg_dict['neobam'][reach]['integrator']['n']=param_est[0]
                    self.alg_dict['neobam'][reach]['integrator']['a0']=param_est[1]
                    self.alg_dict['neobam'][reach]['integrator']['q']=self.bam_flowlaw(param_est,self.obs_dict[reach])


               else: 
                    #print('geobam FLP calcs failed, reach',reach)
                    self.alg_dict['neobam'][reach]['integrator']['n']=np.nan
                    self.alg_dict['neobam'][reach]['integrator']['a0']=np.nan
                    self.alg_dict['neobam'][reach]['integrator']['q']=np.full( (1,self.obs_dict[reach]['nt']),np.nan)

          #2.2 hivdi
          print('CALCULATING HiVDI FLPs')
          for reach in self.alg_dict['hivdi']:
               '''
               try:
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       print('good')
               except:
                   continue
               '''
               try: 
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       datagood=True
               except:
                   continue

               if reach not in self.basin_dict['reach_ids']:
                   # reach is not observed. do not calculate FLPs
                   continue

               with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    alphaflpe=np.nanmean(self.alg_dict['hivdi'][reach]['alpha'])


               if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                    Abar_min=-min(self.obs_dict[reach]['dA'])+1
                    if not np.isnan(alphaflpe):
                         with warnings.catch_warnings():
                              warnings.simplefilter("ignore", category=RuntimeWarning)
                              init_params=(np.nanmean(self.alg_dict['hivdi'][reach]['alpha']), \
                                   np.nanmean(self.alg_dict['hivdi'][reach]['beta']),\
                                   np.nanmean(self.alg_dict['hivdi'][reach]['a0']))
                    else:
                          init_params=(33.3,1.0,Abar_min+10.)
                    #param_bounds=( (0.001,np.inf),(-1e2,1e2),(-min(self.obs_dict[reach]['dA'])+1,np.inf))
                    param_bounds=( (0.001,np.inf),(-1e1,1.e1),(Abar_min,np.inf))
                    qbar=self.alg_dict['hivdi'][reach]['integrator']['qbar']
                    if 'q33' in self.alg_dict['hivdi'][reach]['integrator']:
                        q33=self.alg_dict['hivdi'][reach]['integrator']['q33']
                    else:
                        q33=nan 
                    res = optimize.minimize(fun=self.hivdi_objfun,
                                        x0=init_params,
                                        args=(self.obs_dict[reach],qbar,q33),
                                        bounds=param_bounds )

                    param_est=res.x

                    #store output
                    self.alg_dict['hivdi'][reach]['integrator']['alpha']=param_est[0]
                    self.alg_dict['hivdi'][reach]['integrator']['beta']=param_est[1]
                    self.alg_dict['hivdi'][reach]['integrator']['Abar']=param_est[2]
                    self.alg_dict['hivdi'][reach]['integrator']['q']=self.hivdi_flowlaw(param_est,self.obs_dict[reach])
               else: 
                    self.alg_dict['hivdi'][reach]['integrator']['alpha']=np.nan
                    self.alg_dict['hivdi'][reach]['integrator']['beta']=np.nan
                    self.alg_dict['hivdi'][reach]['integrator']['Abar']=np.nan
                    self.alg_dict['hivdi'][reach]['integrator']['q']=np.full( (1,self.obs_dict[reach]['nt']),np.nan)


          #2.3 MetroMan
          print('CALCULATING MetroMan FLPs')
          for reach in self.alg_dict['metroman']:
              
               '''
               try:
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       print('good')
               except:
                   continue
               '''
               try: 
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       datagood=True
               except:
                   continue

               if reach not in self.basin_dict['reach_ids']:
                   # reach is not observed. do not calculate FLPs
                   continue

               with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    naflpe=np.nanmean(self.alg_dict['metroman'][reach]['na'])


               if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                    with warnings.catch_warnings():
                         Abar_min=-min(self.obs_dict[reach]['dA'])+1
                         if not np.isnan(naflpe):
                             warnings.simplefilter("ignore", category=RuntimeWarning)
                             init_params=(np.nanmean(self.alg_dict['metroman'][reach]['na']), \
                                  np.nanmean(self.alg_dict['metroman'][reach]['x1']),\
                                  np.nanmean(self.alg_dict['metroman'][reach]['a0']))
                         else:
                             init_params=(0.03,-1.,Abar_min+10.)
                    #param_bounds=( (0.001,np.inf),(-1e2,1e2),(-min(self.obs_dict[reach]['dA'])+1,np.inf))
                    param_bounds=( (0.001,np.inf),(-1e1,1e1),(Abar_min,np.inf))
                    qbar=self.alg_dict['metroman'][reach]['integrator']['qbar']
                    if 'q33' in self.alg_dict['metroman'][reach]['integrator']:
                        q33=self.alg_dict['metroman'][reach]['integrator']['q33']
                    else:
                        q33=nan 
                    res = optimize.minimize(fun=self.metroman_objfun,
                                        x0=init_params,
                                        args=(self.obs_dict[reach],qbar,q33),
                                        bounds=param_bounds )
                    param_est=res.x

                    #store output
                    self.alg_dict['metroman'][reach]['integrator']['na']=param_est[0]
                    self.alg_dict['metroman'][reach]['integrator']['x1']=param_est[1]
                    self.alg_dict['metroman'][reach]['integrator']['a0']=param_est[2]
                    self.alg_dict['metroman'][reach]['integrator']['q']=self.metroman_flowlaw(param_est,self.obs_dict[reach])
               else: 
                    self.alg_dict['metroman'][reach]['integrator']['na']=np.nan
                    self.alg_dict['metroman'][reach]['integrator']['x1']=np.nan
                    self.alg_dict['metroman'][reach]['integrator']['a0']=np.nan
                    self.alg_dict['metroman'][reach]['integrator']['q']=np.full((1,self.obs_dict[reach]['nt']),np.nan)

          #2.4 MOMMA
          print('CALCULATING MOMMA FLPs')
          # params are (B,HB) == (river bottom elevation, bankfull elevation)
          for reach in self.alg_dict['momma']:
               '''
               try:
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       print('good')
               except:
                   continue
               '''
               try: 
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       datagood=True
               except:
                   continue

               #print('.... calculating MOMMA FLPs for reach',reach)
               if reach not in self.basin_dict['reach_ids']:
                   # reach is not observed. do not calculate FLPs
                   continue
               with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    Bflpe=np.nanmean(self.alg_dict['momma'][reach]['B'])
               if self.obs_dict[reach]['nt'] > 0:
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)

                         Bmax=np.min(self.obs_dict[reach]['h'])-0.1

                         if not np.isnan(Bflpe):
                              init_params=(np.nanmean(self.alg_dict['momma'][reach]['B']), \
                                   np.nanmean(self.alg_dict['momma'][reach]['H']))
                         else:
                              init_params=(Bmax-1.0,Bmax+1.0)

                    #put a limit on the initial guess for depth
                    min_H_obs=np.min(self.obs_dict[reach]['h'])
                    max_H_obs=np.max(self.obs_dict[reach]['h'])

                    if min_H_obs - init_params[0] > 10.:
                         B=min_H_obs - 10.
                         init_params=(B,B+10.)
                              
                    #param_bounds=( (0.1,np.min(self.obs_dict[reach]['h'])-0.1),(0.1,np.inf))
                    #param_bounds=( (0.1,Bmax),(0.1,np.inf))
                    param_bounds=( (0.1,Bmax),(Bmax+0.1,np.inf))


                    aux_var=self.alg_dict['momma'][reach]['Save']

                    if np.isnan(aux_var):
                        aux_var=20e-5

                    qbar=self.alg_dict['momma'][reach]['integrator']['qbar']
                    if 'q33' in self.alg_dict['momma'][reach]['integrator']:
                        q33=self.alg_dict['momma'][reach]['integrator']['q33']
                    else:
                        q33=nan 

                    try:
                        # the minimize is not just failing to minimize and returnning res.success=fale
                        # it is failing to minimize and raising an error, so we implement try excepts here.
                        res = optimize.minimize(fun=self.momma_objfun,
                                            x0=init_params,
                                            args=(self.obs_dict[reach],qbar,q33,aux_var ),
                                            bounds=param_bounds )
                    except:
                        res.success = False

                    if not res.success:
                        try:
                            param_bounds=( (.1,np.min(self.obs_dict[reach]['h'])-0.1),(max_H_obs-1.,max_H_obs+1.)   )


                            res = optimize.minimize(fun=self.momma_objfun,
                                            x0=init_params,
                                            args=(self.obs_dict[reach],qbar,q33,aux_var ),
                                            bounds=param_bounds )
                        except:
                            pass
                    if not res.success:
                        print('Could not estimate MOMMA flow law parameters to fit MOI flow estimates for reach ',reach,\
                                '. Revert to reach-scale FLPE estimates')
                        param_est= self.alg_dict['momma'][reach]['B'], self.alg_dict['momma'][reach]['H']
                    else:
                        param_est=res.x

                    #store output
                    self.alg_dict['momma'][reach]['integrator']['B']=param_est[0]
                    self.alg_dict['momma'][reach]['integrator']['H']=param_est[1]
                    self.alg_dict['momma'][reach]['integrator']['Save']=aux_var
                    self.alg_dict['momma'][reach]['integrator']['q']=self.momma_flowlaw(param_est,self.obs_dict[reach],aux_var)
               else: 
                    self.alg_dict['momma'][reach]['integrator']['B']=np.nan
                    self.alg_dict['momma'][reach]['integrator']['H']=np.nan
                    self.alg_dict['momma'][reach]['integrator']['Save']=np.nan
                    self.alg_dict['momma'][reach]['integrator']['q']=np.full( (1,self.obs_dict[reach]['nt']),np.nan)

          #2.5 SAD
          print('CALCULATING SAD FLPs')
          for reach in self.alg_dict['sad']:
               '''
               try:
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       print('good')
               except:
                   continue
               '''
               try: 
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       datagood=True
               except:
                   continue

               if reach not in self.basin_dict['reach_ids']:
                   # reach is not observed. do not calculate FLPs
                   continue
               with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    nflpe=np.nanmean(self.alg_dict['sad'][reach]['n'])
               #if self.obs_dict[reach]['nt'] > 0 and (not np.isnan(nflpe)):


               if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)
                         Abar_min=-min(self.obs_dict[reach]['dA'])+1
                         if not np.isnan(nflpe):
                             init_params=(np.nanmean(self.alg_dict['sad'][reach]['n']), \
                                  np.nanmean(self.alg_dict['sad'][reach]['a0']))
                         else:
                             init_params=(0.03,Abar_min+10.)

                    #param_bounds=( (0.001,np.inf),(-min(self.obs_dict[reach]['dA'])+1,np.inf))
                    param_bounds=( (0.001,np.inf),(Abar_min,np.inf))

                    qbar=self.alg_dict['sad'][reach]['integrator']['qbar']
                    if 'q33' in self.alg_dict['sad'][reach]['integrator']:
                        q33=self.alg_dict['sad'][reach]['integrator']['q33']
                    else:
                        q33=nan 

                    res = optimize.minimize(fun=self.sad_objfun,
                                        x0=init_params,
                                        args=(self.obs_dict[reach],qbar,q33),
                                        bounds=param_bounds )

                    param_est=res.x

                    #store output
                    self.alg_dict['sad'][reach]['integrator']['n']=param_est[0]
                    self.alg_dict['sad'][reach]['integrator']['a0']=param_est[1]
                    self.alg_dict['sad'][reach]['integrator']['q']=self.sad_flowlaw(param_est,self.obs_dict[reach])
               else: 
                    self.alg_dict['sad'][reach]['integrator']['n']=np.nan
                    self.alg_dict['sad'][reach]['integrator']['a0']=np.nan
                    self.alg_dict['sad'][reach]['integrator']['q']=np.full( (1,self.obs_dict[reach]['nt']),np.nan)

          #2.6 SIC4DVar
          print('CALCULATING SIC4DVar FLPs')
          for reach in self.alg_dict['sic4dvar']:
               '''
               try:
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       print('good')
               except:
                   continue
               '''
               try: 
                   if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                       datagood=True
               except:
                   continue

               if reach not in self.basin_dict['reach_ids']:
                   # reach is not observed. do not calculate FLPs
                   continue
               with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    nflpe=np.nanmean(self.alg_dict['sic4dvar'][reach]['n'])
               #if self.obs_dict[reach]['nt'] > 0 and (not np.isnan(nflpe)):


               if self.obs_dict[reach]['nt'] > 0 and self.obs_dict[reach]['dA'].size > 0:
                    with warnings.catch_warnings():
                         warnings.simplefilter("ignore", category=RuntimeWarning)

                         Abar_min=-min(self.obs_dict[reach]['dA'])+1
                         if not np.isnan(nflpe):
                              init_params=(np.nanmean(self.alg_dict['sic4dvar'][reach]['n']), \
                                   np.nanmean(self.alg_dict['sic4dvar'][reach]['a0']))
                         else:
                              init_params=(0.03,Abar_min+10.)

                    #param_bounds=( (0.001,np.inf),(Abar_min,np.inf))
                    param_bounds=( (0.001,10.),(Abar_min,np.inf))

                    res = optimize.minimize(fun=self.sic4dvar_objfun,
                                        x0=init_params,
                                        args=(self.obs_dict[reach],self.alg_dict['sic4dvar'][reach]['integrator']['qbar'] ),
                                        bounds=param_bounds )

                    param_est=res.x

                    #store output
                    self.alg_dict['sic4dvar'][reach]['integrator']['n']=param_est[0]
                    self.alg_dict['sic4dvar'][reach]['integrator']['a0']=param_est[1]
                    self.alg_dict['sic4dvar'][reach]['integrator']['q']=self.sic4dvar_flowlaw(param_est,self.obs_dict[reach])
               else: 
                    self.alg_dict['sic4dvar'][reach]['integrator']['n']=np.nan
                    self.alg_dict['sic4dvar'][reach]['integrator']['a0']=np.nan
                    self.alg_dict['sic4dvar'][reach]['integrator']['q']=np.full( (1,self.obs_dict[reach]['nt']),np.nan)

     def integrate_prior(self):
          """Mimic the integrate function but apply only to the prior data"""

          #Gage_Uncertainty=0.05

          #0 create list of junctions, and figure out problem dimensions
          #0.1 remove type 4 reaches from topology
          #self.RemoveDamReaches()
          #0.2 create junction list
          self.CreateJunctionList()

          #0.3 set number of flow levels to run
          FlowLevels=['Mean']

          #0.4 get sizes of the matrix sizes m & n
          m=0 #number of junctions
          for junction in self.junctions:
              m+=1
              junction['row_num']=m-1

          n=0 #number of reaches
          #for reach in reaches:
          for reach in self.basin_dict['reach_ids']:
              n+=1

          if self.VerboseFlag:
              print('Number of junctions = ',m)
              print('Number of reaches= ',n)


          #1 integration calculations
          for FlowLevel in FlowLevels:
              if self.VerboseFlag:
                  print('Running flow level',FlowLevel)
              residuals={}
              for alg in self.alg_dict:
                  residuals[alg]=np.full((n,),np.nan)
              #for i in range(0,niter):
              for i in range(0,self.params_dict['niter']):
                  if self.VerboseFlag:
                       print('  Running iteration',i,'/',self.params_dict['niter'])
                  residuals=self.integrator_optimization_calcs(m,n,FlowLevel,residuals)


     def integrate(self):
          """Integrate reach-level FLPE data."""

          #0 create list of junctions, and figure out problem dimensions
          #0.1 remove type 4 reaches from topology
          #self.RemoveDamReaches()
          #0.2 create junction list
          print('creating junction list')
          self.CreateJunctionList()       

          #0.3 set number of flow levels to run
          FlowLevels=['Mean','q33'] 

          #0.4 get sizes of the matrix sizes m & n
          m=0 #number of junctions
          for junction in self.junctions:
              m+=1
              junction['row_num']=m-1    
    
          n=0 #number of reaches
          #for reach in reaches:
          for reach in self.basin_dict['reach_ids_all']:
              n+=1

          if self.VerboseFlag:
              print('Number of junctions = ',m)
              print('Number of reaches= ',n)

          #1 integration calculations
          for FlowLevel in FlowLevels:
              print('Running flow level',FlowLevel)
              residuals={} 
              for alg in self.alg_dict:
                  residuals[alg]=np.full((n,),np.nan)
              #for i in range(0,niter):
              for i in range(0,self.params_dict['niter']):
                  print('  Running iteration',i+1,'/',self.params_dict['niter'])
                  residuals=self.integrator_optimization_calcs(m,n,FlowLevel,residuals)

          if self.params_dict['quit_before_flpe']:
              sys.exit('done with integration... exiting')

          #2 compute optimal parameters for each algorithm's flow law
          print('computing all flps')
          self.compute_FLPs()


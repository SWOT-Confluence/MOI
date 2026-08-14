import csv
from pathlib import Path
import warnings
import sys
import pandas as pd
import numpy as np
from moi.FlowLaws import MWAPN,MWACN,MWHFN
from moi.Domain import Domain
from moi.FlowLawCalibration import FlowLawCalibration


class Corridors:
    """Extracts and formats CORRIDORS data from CSV files."""
    
    def __init__(self, corridors_dir, basin_dict, obs_dict,verbose=False):
        self.corridors_dir = Path(corridors_dir)
        self.basin_dict=basin_dict
        self.obs_dict=obs_dict
        self.verbose = verbose
        self.corridors_dict = {}

    def integrate_corridors_data(self):
        """
        Reads CSV files from the corridors directory, interacts with
        existing MOI input data, and populates corridors_dict.
        
        Returns
        -------
        corridors_dict
            A dictionary structured similarly to gage_dict.
        """

        # 1. grab csv files, and read those and the translation file: klugey
        if self.verbose:
            print(f"  -> Scanning for CORRIDORS CSV files in: {self.corridors_dir}")
             
        csv_files = list(self.corridors_dir.glob("*.csv"))
        trans_fname='SWORD_v16_v17_translation_reach.csv'
        trans_file=None
        for f in csv_files:
            if f.name==trans_fname:
                trans_file=f
        csv_files=[f for f in csv_files if f.name != trans_fname] 

        # 2. read corridors data and merge into a df
        if not csv_files:
            warnings.warn(f"No CSV files found in {self.corridors_dir}.")
            return None

        corridors_dfs=[]
        for csv_file in csv_files:
            if self.verbose:
                print(f"  -> Processing {csv_file.name}...")
            try:
                corridors_dfs.append(pd.read_csv(csv_file))
                        
            except Exception as e:
                print(f"  -> Error reading {csv_file.name}: {e}")

        if corridors_dfs:
            self.corridors_df=pd.concat(corridors_dfs,ignore_index=True)
        else:
            return None

        # 3. add sword17 rids via translation
        self.add_sword_17_ids(trans_file) # add sword17 to dataframe
        
        # 4. check whether there are any corridors data in this basin
        self.find_corridors_in_basin()
        if not self.rids_in_basin:
            print('did not find any corridors reaches in basin')
            return None

        # 5. for each reach, fit flow law; evaluate Q; compute flow level metrics
        for rid in self.rids_in_basin:
            swotdf,reachdf=self.create_reach_df(rid)
            flow_law_cal=self.fit_flow_law(reachdf)
            Qhat=self.evaluate_flow_law(swotdf,flow_law_cal)
            self.corridors_dict[str(rid)]={
                    'source': 'corridors',
                    'station_id': None,
                    'station_index': None,
                    'reach_id_variable': 'sword_17c',
                    't': swotdf['t'].map(pd.Timestamp.toordinal),
                    'Q': Qhat
                 }

            #print('rid=',rid)
            #print(self.corridors_dict[str(rid)])
        
        return self.corridors_dict

    def add_sword_17_ids(self,trans_file):
        self.corridors_df['reach_id_17']=None
        try:
            transdf=pd.read_csv(trans_file)
        except Exception as e:
            print(f"  -> Error reading SWORD16-17 translation file {trans_file.name}: {e}")
            return

        rids_16=list(set(list(self.corridors_df['Reach_ID'])))
        for rid in rids_16:
            rid17=int(transdf[transdf['v16_reach_id']==rid]['v17_reach_id'].iloc[0])
            self.corridors_df.loc[self.corridors_df['Reach_ID']==rid,'reach_id_17']=rid17

    def find_corridors_in_basin(self):
        allrids = list(self.corridors_df['reach_id_17'])

        self.rids_in_basin=list(set([rid for rid in allrids if str(rid)[:4]==str(self.basin_dict['basin_id'])]))

    def create_reach_df(self,rid):
        # 1 initialize with swot data
        fields_to_keep=['h','w','S','dA']
        swotdf=pd.DataFrame(data= {k: self.obs_dict[str(rid)][k] for k in fields_to_keep })
        swotdf['time_str']= np.delete(self.obs_dict[str(rid)]['time_str'], self.obs_dict[str(rid)]['iDelete'], 0) #TODO fix time_str handling
        swotdf['t'] = pd.to_datetime(swotdf['time_str'],utc=True).dt.tz_convert("America/Anchorage") #TODO automate time zone handling
        swotdf['time_str_local']=swotdf['t'].dt.strftime('%Y-%m-%d %H:%M')

        # 2 handle corridors dataframe time
        raw_dates = self.corridors_df["Time_('dd-mm-yyyy')"].astype(str).str.rstrip("'")
        parsed_dt = pd.to_datetime(raw_dates + " 12:00:00", format="%d-%m-%Y %H:%M:%S")
        self.corridors_df["t"] = parsed_dt.dt.tz_localize("America/Anchorage") 

        # 3 merge swot and corridors
        reachdf = pd.merge_asof(
            self.corridors_df[self.corridors_df['reach_id_17']==int(rid)].sort_values(by='t'),
            swotdf,
            on='t',
            direction="nearest",  # Finds closest timestamp (past or future) TODO add a limit
            suffixes=("_corridors", "_swot"),  # Handles duplicate column names if any exist
        )

        # 4 drop unwanted rows
        cols_to_drop=['Node_ID','SWORD_Version','Reach_ID','X','Y','Qu_(m^3/s_daily)','WSE_(m)',\
             'WSEu_(m)','W_(m)','Wu_(m)','Cross-sectionalArea_(m^2)','Cross-sectionalAreau_(m^2)',\
             'MaxV_(m/s)','MaxVu_(m/s)','MeanV_(m/s)','MeanVu_(m/s)','MaxD_(m)','MaxDu_(m)',\
             'MeanD_(m)','MeanDu_(m)']
        reachdf.drop(columns=cols_to_drop,inplace=True)

        return swotdf,reachdf

    def fit_flow_law(self,reachdf):
        # initialize flow law TODO: switch flow laws depending how many observations are available
        #flow_law=MWAPN(
        #flow_law=MWACN(
        flow_law=MWHFN(
                np.array(reachdf['dA']),
                np.array(reachdf['w']),
                np.array(reachdf['S']),
                np.array(reachdf['h'])
                )

        D=Domain(
                {
                    'nR':1,
                    'xkm':np.nan,
                    'L':np.nan,
                    'nt':len(reachdf),
                    't':reachdf['t'],
                    'dt':np.nan
        })

        flow_law_cal=FlowLawCalibration(D,np.array(reachdf['Q_(m^3/s_daily)']),flow_law)
        flow_law_cal.CalibrateReach(verbose=False,suppress_warnings=True)
        if self.verbose:
            flow_law_cal.Performance.ShowKeyErrorMetrics()

        return flow_law_cal

    def evaluate_flow_law(self,swotdf,flow_law_cal):
        # initialize flow law TODO: switch flow laws depending how many observations are available
        #flow_law=MWAPN(
        #flow_law=MWACN(
        flow_law=MWHFN(
                np.array(swotdf['dA']),
                np.array(swotdf['w']),
                np.array(swotdf['S']),
                np.array(swotdf['h'])
                )

        return flow_law.CalcQ(flow_law_cal.param_est)









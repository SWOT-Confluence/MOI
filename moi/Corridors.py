import csv
from pathlib import Path
import warnings
import sys
import pandas as pd

class Corridors:
    """Extracts and formats CORRIDORS data from CSV files."""
    
    def __init__(self, corridors_dir, verbose=False):
        self.corridors_dir = Path(corridors_dir)
        self.verbose = verbose
        self.corridors_dict = {}

    def integrate_corridors_data(self, input_obj):
        """
        Reads CSV files from the corridors directory, interacts with
        existing MOI input data, and populates corridors_dict.
        
        Parameters
        ----------
        input_obj : Input
            The fully populated Input object containing sos_dict, obs_dict, etc.
            
        Returns
        -------
        dict
            A dictionary structured similarly to sos_dict.
        """

        if self.verbose:
            print(f"  -> Scanning for CORRIDORS CSV files in: {self.corridors_dir}")
             
        # grab csv file, and exclude the translation file: klugey
        csv_files = list(self.corridors_dir.glob("*.csv"))
        trans_fname='SWORD_v16_v17_translation_reach.csv'
        trans_file=None
        for f in csv_files:
            if f.name=trans_fname:
                trans_file=f
        csv_files=[f for f in csv_files if f.name != trans_fname] 

        # read corridors data
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

        # add sword17 to dataframe
        self.add_sword_17_ids(trans_file) 

        print(self.corridors_df['reach_id_17'])
        
        print('end of corridors setup. quitting')
        sys.exit(1)
        

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



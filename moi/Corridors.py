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
             
        # grab csv file, and exclude the translation file
        csv_files = list(self.corridors_dir.glob("*.csv"))
        csv_files=[f for f in csv_files if f.name != 'SWORD_v16_v17_translation_reach.csv']

        if not csv_files:
            warnings.warn(f"No CSV files found in {self.corridors_dir}.")
            return self.corridors_dict

        corridors_dfs=[]
        for csv_file in csv_files:
            if self.verbose:
                print(f"  -> Processing {csv_file.name}...")
            try:
                corridors_dfs.append(pd.read_csv(csv_file))
                        
            except Exception as e:
                print(f"  -> Error reading {csv_file.name}: {e}")
        corridors_df=pd.concat(corridors_dfs,ignore_index=True)

        print(corridors_df)

        
        print('end of corridors setup. quitting')
        sys.exit(1)
        

        return self.corridors_dict

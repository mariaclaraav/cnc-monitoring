import dask.dataframe as dd
import logging
import pandas as pd
from tqdm import tqdm
import time
import sys
import os
import gc

import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)

# Constants and Configuration
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

from src.utils.utilities import order_unique_code
from src.utils.feature_engineering.rms import CustomRMS
from src.utils.constants.process_constats import CONFIG

# Configure Logging
logging.basicConfig(level=logging.INFO)

SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'processed')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'ETL', 'ETL_final.parquet')

FS = 2000
BAND = 10
    

if __name__ == "__main__":
    os.system('cls')
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading data from '{DATA_PATH}'...")
    df = dd.read_parquet(DATA_PATH)
    
    df_rms = []
    for op_code, op_details in tqdm(CONFIG.items(), desc="Processing operations"):
        vel = op_details['VELOCITY']

        # Filter the DataFrame for the current operation code
        df_filtered = df[df.Process == op_code].compute()
        if df_filtered.empty:
            logger.warning(f"No data found for operation code '{op_code}'. Skipping.")
            continue
        
        start_time = time.time()
        
        custom_rms = CustomRMS(fs=FS, vel=vel, band=BAND)
        aux = custom_rms.get_rms(df_filtered)
        
        end_time = time.time()
        order_unique_code(aux)
        
        # Add an 'Operation' column to identify the operation code in results
        aux['Operation'] = op_code
        
        # Append the result to the list
        df_rms.append(aux)
        end_time = time.time()
        logger.info(f"Operation '{op_code}' processed in {end_time - start_time:.2f} seconds.")
        del aux, df_filtered
        gc.collect()
    
    # Concatenate all results into a final DataFrame
    if df_rms:
        final_df = pd.concat(df_rms, axis=0, ignore_index=True)
        final_df.to_parquet(os.path.join(SAVING_PATH, 'rms.parquet'))
        logger.info(f"RMS results saved to '{SAVING_PATH}/rms.parquet'")
    else:
        logger.warning("No RMS results to save.")
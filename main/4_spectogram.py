import os
import sys
import numpy as np
import pywt
import logging
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import gc
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()


# Clear terminal output
os.system('cls')

CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

from src.features.cwt import CustomCWT
# Paths
SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'spectrogram')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'ETL', 'ETL_final.parquet')

os.makedirs(SAVING_PATH, exist_ok=True)

# Parameters
WAVELET = 'shan'  # Options: gaus8, mexh
FREQUENCIES = [600, 100]  # Desired frequencies
SAMPLE_RATE = 2000  # Sampling rate
OPERATIONS = ['OP06']


# Function to load data
def load_data(data_path):
    try:
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {data_path}")
        raise e

# Function to filter and process data
def filter_and_process(df, operation):
    df_filtered = df[df['Process'] == operation][['Time', 'X_axis', 'Y_axis', 'Z_axis', 'Unique_Code']].copy()
    scaler = StandardScaler()
    df_filtered[['X_axis', 'Y_axis', 'Z_axis']] = scaler.fit_transform(df_filtered[['X_axis', 'Y_axis', 'Z_axis']])
    df_filtered.reset_index(drop=True, inplace=True)
    return df_filtered



# Main script
if __name__ == '__main__':
    data_dict = {}
    df = load_data(DATA_PATH)
    min_length = float('inf')  # Inicializa o comprimento mínimo
    
    for op in tqdm(OPERATIONS, desc="Processing operations"):
        df_filtered = filter_and_process(df, operation=op)

        for unique_code in tqdm(df_filtered['Unique_Code'].unique(), desc=f"Processing unique codes for {op}", leave=False):
            subset = df_filtered[df_filtered['Unique_Code'] == unique_code]

            for axis in ['X_axis', 'Y_axis', 'Z_axis']:
                signal = subset[axis].values
                cwt_data = CustomCWT(frequencies=FREQUENCIES, 
                                       wavelet=WAVELET, 
                                       sampling_rate= SAMPLE_RATE).run(signal)
                data_dict[f"{unique_code}_{axis}"] = cwt_data
                min_length = min(min_length, len(cwt_data))
                
                # Clear memory
                del signal, cwt_data
                gc.collect()

        # Clear memory
        del df_filtered
        gc.collect()
    
        truncated_data = {key: value[:min_length] for key, value in data_dict.items()}

        df_cwt = pd.DataFrame(truncated_data)
        parquet_path = os.path.join(SAVING_PATH, f"cwt_data_{op}.parquet")
        df_cwt.to_parquet(parquet_path, index=False)
        
        del df_cwt
        gc.collect()

        print(f"Data saved to {parquet_path} as a Parquet file.\n")

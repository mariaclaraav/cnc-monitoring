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
BASE_DIR = os.path.join(CURRENT_DIR, 'figure', 'cwt')
os.makedirs(SAVING_PATH, exist_ok=True)

# Parameters
WAVELET = 'cmor0.5-1.0'  # Options: gaus8, mexh
FREQUENCIES = [600, 100]  # Desired frequencies
SAMPLE_RATE = 2000  # Sampling rate
OPERATIONS = ['OP06', 'OP07', 'OP10']


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
    data = load_data(DATA_PATH)
    
    df_cwt = pd.DataFrame()
    for op in tqdm(OPERATIONS, desc="Processing operations"):
        
        df = data[data['Process'] == op][['Time', 'X_axis', 'Y_axis', 'Z_axis', 'Unique_Code', 'Label']].copy()
        path = os.path.join(BASE_DIR, op)
        os.makedirs(path, exist_ok=True)
        
        for unique_code in tqdm(df['Unique_Code'].unique(), desc=f"Processing unique codes for {op}", leave=False):
            subset = df[df['Unique_Code'] == unique_code]
            label = subset['Label'].iloc[0]
            cwt_transform = CustomCWT(frequencies=FREQUENCIES, 
                                       wavelet=WAVELET, 
                                       sampling_rate= SAMPLE_RATE)
            for axis in ['X_axis', 'Y_axis', 'Z_axis']:
                time = subset['Time'].values
                signal = subset[axis].values
                
                # feat_coef = cwt_data.get_cwt_features(signal)
                
                # if axis == 'X_axis':                 
                #     X_data = feat_coef.add_suffix('_X_axis')
                # elif axis == 'Y_axis':
                #     Y_data = feat_coef.add_suffix('_Y_axis')
                # elif axis == 'Z_axis':
                #     Z_data = feat_coef.add_suffix('_Z_axis')
                
                cwt_transform.save_cwt_img(
                time=time,
                signal=signal,
                save_path=os.path.join(path, f'{unique_code}_{axis[0]}.png')  # Caminho dinâmico com inicial do eixo
            )
                    
            # data_dict['Unique_Code'] = [unique_code]*len(feat_coef)
            # data_dict['Label'] = [label]*len(feat_coef)
            data_dict['Unique_Code'] = [unique_code]
            data_dict['Label'] = [label]
            #features = pd.concat([X_data, Y_data, Z_data], axis=1)
            # Clear memory
            del signal, time
            gc.collect()
            df_aux = pd.DataFrame(data_dict)
            #df_aux = pd.concat([df_aux, features], axis=1)
            if 'df_cwt' in locals():
                df_cwt = pd.concat([df_cwt, df_aux], axis=0)
            else:
                df_cwt = df_aux
        # Clear memory
        del df_aux
        gc.collect()

        parquet_path = os.path.join(path, f"label_{op}.parquet")
        df_cwt.to_parquet(parquet_path, index=False)
        
        del df_cwt
        gc.collect()

        print(f"Data saved to {parquet_path} as a Parquet file.\n")

import os
import pandas as pd
import sys
from pathlib import Path
import logging
from tqdm import tqdm

# Clear terminal output
os.system('cls')

# Set up the current directory and system path for module imports
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

from utils.data_processing.processor import OperationProcessor
from utils.data_processing.save import DataSaver

############################# Constants and configuration #################################

# Define paths for data and saving
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'wavelet')
SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'processed')

# List of operations to process

OPERATIONS = ['OP00', 'OP01', 'OP02', 'OP03', 'OP04', 
                'OP05', 'OP06', 'OP07', 'OP08', 'OP09', 
                'OP10', 'OP11', 'OP12', 'OP13', 'OP14']

# List of features used in the dataset
FEATURES = ['X_axis', 'X_node_aaa', 'X_node_aad', 'X_node_add', 'X_node_ada',
            'X_node_dda', 'X_node_ddd', 'X_node_dad', 'X_node_daa', 'Y_axis',
            'Y_node_aaa', 'Y_node_aad', 'Y_node_add', 'Y_node_ada', 'Y_node_dda',
            'Y_node_ddd', 'Y_node_dad', 'Y_node_daa', 'Z_axis', 'Z_node_aaa',
            'Z_node_aad', 'Z_node_add', 'Z_node_ada', 'Z_node_dda', 'Z_node_ddd',
            'Z_node_dad', 'Z_node_daa']

# Define the scaler type and whether to include 'Unique_Code'
SCALER = None
INCLUDE_CODES = True
VAL_SIZE = 0.8 # Acctually I'm inverting that so it means that only 30% Validation size for the train-validation split


# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


TRAIN_PERIOD = ['02-2019', '08-2019', '02-2020', '08-2020']                  
TEST_PERIOD = ['02-2021', '08-2021']                     

    
# Run the preprocessing function when this script is executed
if __name__ == "__main__":
    
    train_dfs, val_dfs, test_dfs = OperationProcessor(
                                operations=OPERATIONS,
                                features=FEATURES,
                                path=DATA_PATH,
                                scaler_type=SCALER,
                                include_codes=INCLUDE_CODES,
                                train_period=TRAIN_PERIOD,
                                test_period=TEST_PERIOD,
                                n_val=VAL_SIZE,
                                print_codes=False,
                                parquet=False
                                 ).run()
    
    DataSaver(train_dfs, val_dfs, test_dfs, SAVING_PATH).run()

import os
import sys
import logging
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Set up the current directory and system path for module imports
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

# Import required modules
from src.utils.data_processing.processor import OperationProcessor
from src.utils.data_processing.env_config import configure_environment
from src.utils.data_processing.save import DataSaver

logging.basicConfig(level=logging.INFO)

GROUPS = ['OP02','OP05', 'OP07']
# Constants and configuration
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'adaptative_filter')
SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'operation')
SCALER = None
INCLUDE_CODES = True
VAL_SIZE = 0.8  # 20% validation size for train-validation split
TRAIN_PERIOD = ['02-2019', '08-2019', '02-2020', '08-2020']
TEST_PERIOD = ['02-2021', '08-2021']


def processing(group):
    # Clear terminal output
    os.system('cls')

    # Configure operations and features
    operations, features = configure_environment(group, 
                                                 set="frequency_features", 
                                                 data_path=DATA_PATH)

    # Log selected operations and features
    logger.info(f"OPERATIONS: {operations}")
    logger.info(f"FEATURES: {features}")

    # Run preprocessing
    logger.info("Starting preprocessing...")
    train_dfs, val_dfs, test_dfs = OperationProcessor(
        operations=operations,
        features=features,
        path=DATA_PATH,
        scaler_type=SCALER,
        include_codes=INCLUDE_CODES,
        train_period=TRAIN_PERIOD,
        test_period=TEST_PERIOD,
        n_val=VAL_SIZE,
        print_codes=False,
        parquet=False
    ).run()
    logger.info("Preprocessing completed.")

    # Save results
    logger.info("Saving processed data...")
    
    DataSaver(train_dfs=val_dfs, val_dfs=train_dfs, test_dfs=test_dfs, output_dir=SAVING_PATH, group=group).run()
    logger.info(f"Done for {group}!!")

if __name__ == "__main__":
    logger = logging.getLogger()  # Initialize logger
    for GROUP in tqdm(GROUPS, desc="Processing Groups", unit="group"):
        processing(group=GROUP)

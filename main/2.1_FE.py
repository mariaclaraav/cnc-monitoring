import logging
import os
import sys
from typing import List
import pandas as pd
from tqdm import tqdm

# Constants and Configuration
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

# Import custom modules
from src.features.build_features import TimeSeriesProcessor
from src.features.custom_processor import CustomProcessor
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
from src.utils.data_processing.etl import load_data

# Paths
SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'features', 'window_feat')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'ETL', 'ETL_final.parquet')

# ParametersC
OPERATIONS = [ 'OP07']

FEATURE_TYPES = ['statistical', 'energy','jerk','filter']
WINDOW_SIZE = 200
SAMPLING_RATE = 2000
STEP_SIZE = WINDOW_SIZE // 2
MIN_PERIODS = 200

# Configure Logging
logging.basicConfig(level=logging.INFO)

def process_and_save_operations(
    df: pd.DataFrame,
    operations: List[str],
    saving_path: str,
    processor: TimeSeriesProcessor,
    feature_types: List[str],
    filter_list: List[tuple] = None
) -> None:
    """
    Process and save data for each operation.

    Args:
        df (pd.DataFrame): Input DataFrame.
        operations (List[str]): List of operations to process.
        saving_path (str): Directory to save processed data.
        processor (TimeSeriesProcessor): Processor for feature engineering.
        feature_types (List[str]): Types of features to process.
    """
    for operation in tqdm(operations, desc="Processing Operations"):
        logger.info(f"Processing operation '{operation}'...")
        
        # Initialize the custom processor
        custom_processor = CustomProcessor(processor)
        
        # Set filter_list based on operation
        if operation in ['OP05', 'OP02', 'OP07']:
            filter_list = [(185, 215), (385, 405), (475, 515)]
        elif operation in ['OP13']:
            filter_list = [(300, 350)]
        else:
            filter_list = [(235, 265), (385, 405), (475, 515)]
        
        # Process the operation
        df_processed = custom_processor.filter_and_process(df, operation, feature_types, filter_list)
        
        # Save the processed data
        save_path = os.path.join(saving_path, f"{operation}.parquet")
        logger.info(f"Saving processed data to '{save_path}'...")
        df_processed.to_parquet(save_path)
        del df_processed  # Free memory

    logger.info("All operations processed and saved successfully.")

def feature_engineering() -> None:
    """
    Main function to perform the feature engineering pipeline.
    """
    logger.info("Starting feature engineering...")

    # Initialize the processor
    processor = TimeSeriesProcessor(
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        min_periods=MIN_PERIODS,
        sampling_rate=SAMPLING_RATE
    )
    logger.info("TimeSeriesProcessor initialized.")

    # Load data
    logger.info(f"Loading data from '{DATA_PATH}'...")
    df = load_data(DATA_PATH)

    # Process and save features
    process_and_save_operations(
        df=df,
        operations=OPERATIONS,
        saving_path=SAVING_PATH,
        processor=processor,
        feature_types=FEATURE_TYPES
    )
    
    logger.info("Feature engineering completed.")
    del df  # Ensure cleanup

if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    feature_engineering()


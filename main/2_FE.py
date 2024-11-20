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
from src.utils.data_processing.etl import load_data

SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'features')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'ETL', 'ETL_final.parquet')

OPERATIONS = ['OP07', 'OP08', 'OP09', 
                'OP10', 'OP11', 'OP12', 'OP13', 'OP14']
FEATURE_TYPES = ['filter', 'dwt', 'emd']
WINDOW_SIZE = 1000
SAMPLING_RATE = 2000
STEP_SIZE = 1
MIN_PERIODS = 1

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()



def process_and_save_operations(
    df: pd.DataFrame,
    operations: List[str],
    saving_path: str,
    processor: TimeSeriesProcessor,
    feature_types: List[str]
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
        
        custom_processor = CustomProcessor(processor)
        df_processed = custom_processor.filter_and_process(df, operation, feature_types)
        
        save_path = os.path.join(saving_path, f"{operation}.parquet")
        logger.info(f"Saving processed data to '{save_path}'...")
        
        df_processed.to_parquet(save_path)
        del df_processed  # Free memory

    logger.info("All operations processed and saved successfully.")

def feature_engineering() -> None:
    """
    Main function to perform feature engineering pipeline.
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
    feature_engineering()

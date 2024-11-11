import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

current_dir = os.getcwd()
sys.path.append(current_dir)
print(f"Current directory: {current_dir}")

# Import the TimeSeriesProcessor and CustomProcessor classes
from src.features.build_features import TimeSeriesProcessor
from src.features.custom_processor import CustomProcessor

# Clear terminal output
os.system('cls')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Constants and configuration
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'features')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'ETL', 'ETL_final.parquet')

OPERATIONS = ['OP06']
FEATURE_TYPES = ['filter', 'wpd', 'dwt']
WINDOW_SIZE = 1000
SAMPLING_RATE = 2000
STEP_SIZE = 1
MIN_PERIODS = 1


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from a parquet file."""
    try:
        logger.info(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {data_path}")
        raise e

def process_and_save_operations(
    df: pd.DataFrame,
    operations: List[str],
    saving_path: str,
    processor: TimeSeriesProcessor,
    feature_types: List[str]
) -> None:
    """Process and save data for each operation."""
    for operation in tqdm(operations, desc='Processing Operations'):
        logger.info(f"Processing operation: {operation}")
        custom_processor = CustomProcessor(processor)
        df_processed = custom_processor.filter_and_process(df, operation, feature_types)
        final_save_path = os.path.join(saving_path, f'{operation}.parquet')
        logger.info(f"Saving the processed DataFrame to {final_save_path}...")
        df_processed.to_parquet(final_save_path)
        del df_processed
    logger.info("All operations processed successfully.")

def feature_engineering() -> None:
    """Main function to perform feature engineering."""
    # Initialize the TimeSeriesProcessor
    processor = TimeSeriesProcessor(
        window_size=WINDOW_SIZE,
        step_size=STEP_SIZE,
        min_periods=MIN_PERIODS,
        sampling_rate=SAMPLING_RATE
    )
    
    # Load the data
    df = load_data(DATA_PATH)
    
    # Process and save operations
    process_and_save_operations(df = df, 
                                operations = OPERATIONS, 
                                saving_path = SAVING_PATH, 
                                processor = processor, 
                                feature_types = FEATURE_TYPES
    )

    # Clean up
    del df

if __name__ == '__main__':
    feature_engineering()

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

# Import the TimeSeriesProcessor and CustomProcessor classes
from phm_feature_lab.features.time_series_features import TimeSeriesFeatures
from phm_feature_lab.utils.logger import Logger
from phm_feature_lab.utils.data_processing.load_files import LoadFiles
from phm_feature_lab.utils.data_processing.process_unique_code import ProcessUniqueCode

logger = Logger().get_logger()


# Constants and configuration
CURRENT_DIR = os.getcwd()

SAVING_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'features')
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'ETL', 'ETL_final.parquet')

OPERATIONS = ['OP06']
FEATURE_TYPES = ['statistical']
WINDOW_SIZE = [27, 50, 100, 200, 300, 400, 500]
SAMPLING_RATE = 2000
STEP_SIZE = 1
MIN_PERIODS = 1

def main() -> None:
    """Main function to perform feature engineering."""
    # Initialize the TimeSeriesProcessor
    for ws in WINDOW_SIZE:
        processor = TimeSeriesFeatures(
            window_size= ws,
            step_size=STEP_SIZE,
            min_periods=MIN_PERIODS,
            sampling_rate=SAMPLING_RATE
        )
        operation_processor = ProcessUniqueCode(processor, SAVING_PATH)

        
        # Load the data
        data_loader = LoadFiles(DATA_PATH)
        df = data_loader.load(format='parquet')  
        
        print(f"OPERATIONS: {OPERATIONS}, FEATURE_TYPES: {FEATURE_TYPES}")

        operation_processor.process_and_save(df=df, operations=OPERATIONS, feature_types=FEATURE_TYPES, extra_info = ws)
        
        # Clean up
        del df

if __name__ == '__main__':
    main()

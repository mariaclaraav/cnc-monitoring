import os
import pandas as pd
from tqdm import tqdm
from typing import List, Optional
from phm_feature_lab.features.time_series_features import TimeSeriesFeatures
from phm_feature_lab.features.custom_processor import CustomProcessor
from phm_feature_lab.utils.logger import Logger


logger = Logger().get_logger()


class ProcessUniqueCode:
    """
    Processes and saves data for each operation, following the Single Responsibility principle.

    Attributes:
        processor (TimeSeriesProcessor): Processor for time series data.
        saving_path (str): Path to save processed data.
    """
    
    def __init__(self, processor: TimeSeriesFeatures, saving_path: str):
        """
        Initialize the OperationProcessor with a processor and saving path.

        Args:
            processor (TimeSeriesProcessor): Instance of TimeSeriesProcessor.
            saving_path (str): Directory path to save processed files.
        """
        self.processor = processor
        self.saving_path = saving_path
        os.makedirs(saving_path, exist_ok=True)  # Ensure the directory exists

    def process_and_save(self, df: pd.DataFrame, operations: List[str], feature_types: List[str], extra_info: Optional[str]=None) -> None:
        """
        Processes and saves data for each operation in the list.

        Args:
            df (pd.DataFrame): Input DataFrame with raw data.
            operations (List[str]): List of operation identifiers (e.g., ['OP01', 'OP02']).
            feature_types (List[str]): Types of features to extract.

        Returns:
            None: Saves processed data to Parquet files.
        """
        for operation in tqdm(operations, desc='Processing Operations'):
            logger.info(f"Processing operation: {operation}")
            custom_processor = CustomProcessor(self.processor)
            df_processed = custom_processor.filter_and_process(df, operation, feature_types)
            if extra_info:
                save_path = os.path.join(self.saving_path, f'{operation}_{extra_info}.parquet')
            else:
                save_path = os.path.join(self.saving_path, f'{operation}.parquet')
            logger.info(f"Saving processed DataFrame to {save_path}...")
            df_processed.to_parquet(save_path)
        logger.info("All operations processed successfully.")
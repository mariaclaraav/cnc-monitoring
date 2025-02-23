import os
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

from phm_feature_lab.features.custom_processor import CustomProcessor
from phm_feature_lab.features.build_features import TimeSeriesProcessor
from phm_feature_lab.utils.logger import Logger

logger = Logger().get_logger()


class SaveProcessedOperations:
    """ A class to process and save operations data.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        operations: List[str],
        saving_path: str,
        processor: TimeSeriesProcessor,
        feature_types: List[str],
        filter_list: Optional[List[tuple]] = None,
    ) -> None:
        """
        Initialize the OperationProcessor.

        Args:
            df (pd.DataFrame): Input DataFrame.
            operations (List[str]): List of operations to process.
            saving_path (str): Directory to save processed data.
            processor (TimeSeriesProcessor): Processor for feature engineering.
            feature_types (List[str]): Types of features to process.
            filter_list (Optional[List[tuple]]): List of filters to apply.
        """
        self.df = df
        self.operations = operations
        self.saving_path = saving_path
        self.processor = processor
        self.feature_types = feature_types
        self.filter_list = filter_list

    def _get_filter_list(self, operation: str) -> List[tuple]:
        """
        Get the filter list based on the operation.

        Args:
            operation (str): The operation to process.

        Returns:
            List[tuple]: The filter list for the operation.
        """
        if operation in ['OP05', 'OP02', 'OP07']:
            return [(185, 215), (385, 405), (475, 515)]
        elif operation in ['OP13']:
            return [(300, 350)]
        else:
            return [(235, 265), (385, 405), (475, 515)]

    def process_and_save(
                    self, 
                    aditional_saving_parms: Optional[int] = None
                    ) -> None:
        """
        Process and save data for each operation.
        """
        for operation in tqdm(self.operations, desc="Processing Operations"):
            logger.info(f"Processing operation '{operation}'...")

            # Initialize the custom processor
            custom_processor = CustomProcessor(self.processor)

            # Set filter_list based on operation
            self.filter_list = self._get_filter_list(operation)

            # Process the operation
            df_processed = custom_processor.filter_and_process(
                self.df, operation, self.feature_types, self.filter_list
            )
            
            # Save the processed data
            if aditional_saving_parms:
                save_path = os.path.join(self.saving_path, f"{operation}_{aditional_saving_parms}.parquet")
            else:
                save_path = os.path.join(self.saving_path, f"{operation}.parquet")
                
            logger.info(f"Saving processed data to '{save_path}'...")
            df_processed.to_parquet(save_path)
            del df_processed  # Free memory

        logger.info("All operations processed and saved successfully.")
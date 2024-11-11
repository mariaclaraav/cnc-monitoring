import os
import pandas as pd
import logging
from typing import List, Optional

class DataSaver:
    def __init__(self, output_dir: str,
                 train_dfs: List[pd.DataFrame], 
                        val_dfs: Optional[List[pd.DataFrame]] = None, 
                        test_dfs: Optional[List[pd.DataFrame]] = None):
        """
        Initialize the DataSaver with an output directory for saving DataFrames.

        Parameters:
        ----------
        output_dir : str
            The directory where the final DataFrames will be saved.
        """
        self.__output_dir = output_dir
        self.__train_dfs = train_dfs
        self.__val_dfs = val_dfs
        self.__test_dfs = test_dfs
        self.logger = logging.getLogger(__name__)

        # Ensure the output directory exists
        if not os.path.exists(self.__output_dir):
            os.makedirs(self.__output_dir)

    def save_dataframes(self):
        """
        Save concatenated DataFrames for training, validation, and test sets to parquet files.

        Parameters:
        ----------
        train_dfs : List[pd.DataFrame]
            List of training DataFrames to be concatenated and saved.
        
        val_dfs : Optional[List[pd.DataFrame]], optional
            List of validation DataFrames to be concatenated and saved (default is None).
        
        test_dfs : Optional[List[pd.DataFrame]], optional
            List of test DataFrames to be concatenated and saved (default is None).
        """
        # Concatenate and save the final training DataFrame
        final_train_df = pd.concat(self.__train_dfs, ignore_index=True)
        final_train_df.to_parquet(os.path.join(self.__output_dir, 'final_val.parquet'))
        self.logger.info(f"Final train DataFrame saved to {os.path.join(self.__output_dir, 'final_val.parquet')}")
        del final_train_df
        self.__train_dfs = None  # Free up memory

        # Save the final validation DataFrame if it exists
        if self.__val_dfs:
            final_val_df = pd.concat(self.__val_dfs, ignore_index=True)
            final_val_df.to_parquet(os.path.join(self.__output_dir, 'final_train.parquet'))
            self.logger.info(f"Final validation DataFrame saved to {os.path.join(self.__output_dir, 'final_train.parquet')}")
            del final_val_df
            self.__val_dfs = None  # Free up memory

        # Save the final test DataFrame if it exists
        if self.__test_dfs:
            final_test_df = pd.concat(self.__test_dfs, ignore_index=True)
            final_test_df.to_parquet(os.path.join(self.__output_dir, 'final_test.parquet'))
            self.logger.info(f"Final test DataFrame saved to {os.path.join(self.__output_dir, 'final_test.parquet')}")
            del final_test_df
            self.__test_dfs = None  # Free up memory
            
    def run(self) -> None:
        """Calls the save_dataframes function to save the DataFrames.
        """
        return self.save_dataframes()
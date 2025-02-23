import os
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple
import logging
from phm_feature_lab.utils.data_processing.data_handler import DataHandler

class OperationProcessor:
    def __init__(self, operations: List[str], 
                 features: List[str], 
                 path: str, 
                 scaler_type: str, 
                 include_codes: bool, 
                 train_period: List[str], 
                 test_period: List[str], 
                 n_val: int, 
                 print_codes: bool = False, 
                 parquet: bool = False):
        """ 
        Initialize the OperationProcessor with parameters for processing data.

        Parameters:
        ----------
        operations : List[str]
            List of operations (files) to process.
        features : List[str]
            List of feature columns to use.
        path : str
            Path to the data files.
        scaler_type : str
            Type of scaler to apply ('StandardScaler', 'MinMaxScaler', or None).
        include_codes : bool
            Whether to include the 'Unique_Code' in the datasets.
        train_period : List[str]
            List of training periods.
        test_period : List[str]
            List of test periods.
        n_val : int
            Number of validation samples.
        print_codes : bool, optional
            Whether to print codes during processing (default is False).
        parquet : bool, optional
            Whether to save datasets as parquet files (default is False).
        """
        self.operations = operations
        self.features = features
        self.path = path
        self.scaler_type = scaler_type
        self.include_codes = include_codes
        self.train_period = train_period
        self.test_period = test_period
        self.n_val = n_val
        self.print_codes = print_codes
        self.parquet = parquet
        self.logger = logging.getLogger(__name__)

    def process_operations(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Process each operation by loading its dataset, applying transformations, and preparing
        the train, validation, and test datasets.

        Returns:
        -------
        Tuple[List[pd.DataFrame], List[pd.DataFrame], List[pd.DataFrame]]
            Lists of DataFrames for training, validation, and test data.
        """
        train_dfs = []
        val_dfs = []
        test_dfs = []

        for operation in tqdm(self.operations, desc="Processing Operations"):
            data_path = os.path.join(self.path, f'{operation}.parquet')

            if not os.path.exists(data_path):
                self.logger.warning(f"File {data_path} does not exist. Skipping {operation}.")
                continue

            data = pd.read_parquet(data_path)
            data[self.features] = data[self.features].astype('float32')
            self.logger.info(f"Processing operation {operation}")

            # Call the setup_data function to get the train, validation, and test datasets
            data_handler = DataHandler(
                data,
                train_period=self.train_period,
                test_period=self.test_period,
                n_val=self.n_val,
                features=self.features,
                scaler_type=self.scaler_type,
                print_codes=self.print_codes,
                parquet=self.parquet,
                include_codes=self.include_codes
            )

            (
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                unique_code_train,
                unique_code_test,
                unique_code_val,
                _,
            ) = data_handler.split_data()

            del data

            # Training DataFrame
            df_train = pd.DataFrame(X_train, columns=self.features)
            df_train['Label'] = y_train
            df_train['Operation'] = operation
            if self.include_codes:
                df_train['Unique_Code'] = unique_code_train
            train_dfs.append(df_train)

            # Validation DataFrame
            if X_val is not None:
                df_val = pd.DataFrame(X_val, columns=self.features)
                df_val['Label'] = y_val
                df_val['Operation'] = operation
                if self.include_codes:
                    df_val['Unique_Code'] = unique_code_val
                val_dfs.append(df_val)

            # Test DataFrame
            if X_test is not None:
                df_test = pd.DataFrame(X_test, columns=self.features)
                df_test['Label'] = y_test
                df_test['Operation'] = operation
                if self.include_codes:
                    df_test['Unique_Code'] = unique_code_test
                test_dfs.append(df_test)

            del df_train, df_val, df_test  # Free memory after each operation

        return train_dfs, val_dfs, test_dfs

    def run(self):
        return self.process_operations()

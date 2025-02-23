import os
import pandas as pd
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from phm_feature_lab.utils.data_processing.splitting import SplitData
from phm_feature_lab.utils.logger import Logger 

logger = Logger().get_logger()

class DataHandler:
    """
    Handles data preprocessing, splitting, and scaling for machine learning.
    Prepares datasets for training, validation, and testing, and optionally saves them as parquet files.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        train_period: list,
        test_period: list,
        n_val: float,
        features: list,
        scaler_type: str = "StandardScaler",
        print_codes: bool = False,
        parquet: bool = False,
        path: Optional[str] = None,
        include_codes: bool = False,
    ):
        """
        Initializes the DataHandler.

        Args:
            df: Input DataFrame with features and labels.
            train_period: List of periods for training data.
            test_period: List of periods for test data.
            n_val: Fraction of training data to use for validation.
            features: List of feature column names.
            scaler_type: Type of scaler to apply ('StandardScaler', 'MinMaxScaler', or None).
            print_codes: If True, prints unique codes for datasets.
            parquet: If True, saves datasets as parquet files.
            path: Directory to save parquet files (required if `parquet=True`).
            include_codes: If True, includes 'Unique_Code' in feature sets.
        """
        self.__df = df
        self.__train_period = train_period
        self.__test_period = test_period
        self.__n_val = n_val
        self.__features = features
        self.__scaler_type = scaler_type
        self.__print_codes = print_codes
        self.__parquet = parquet
        self.__path = path
        self.__include_codes = include_codes

    def __create_mask(self, periods: List[str], machine_types: Optional[List[str]] = None, normal: bool = True) -> pd.Series:
        """
        Creates a boolean mask for the DataFrame based on periods, machine types, and a normal flag.

        Args:
            periods: List of periods to include.
            machine_types: List of machine types to include. If None, all machines are included.
            normal: If True, only includes rows where 'Label' is 0 (normal).

        Returns:
            A boolean mask for the DataFrame.
        """
        period_mask = self.__df['Period'].isin(periods)
        machine_mask = self.__df['Machine'].isin(machine_types) if machine_types else pd.Series(True, index=self.__df.index)
        label_mask = self.__df['Label'] == 0 if normal else pd.Series(True, index=self.__df.index)
        return period_mask & machine_mask & label_mask

    def split_data(self) -> Tuple:
        """
        Prepares the dataset for training, validation, and testing.

        Returns:
            A tuple containing:
                - X_train, y_train: Training features and labels.
                - X_val, y_val: Validation features and labels (if applicable).
                - X_test, y_test: Test features and labels.
                - unique_codes_train: Unique codes for the training set.
                - unique_codes_test: Unique codes for the test set.
                - unique_codes_val: Unique codes for the validation set (if applicable).
                - scaler: The fitted scaler used for scaling the data.
        """
        # Create train and test masks
        train_mask = self.__create_mask(self.__train_period, normal=False)
        test_mask = self.__create_mask(self.__test_period, normal=False)

        # Initialize SplitData object with print_codes parameter
        split_data = SplitData(self.__df, train_mask, test_mask, self.__n_val, self.__features, 
                               print_codes=self.__print_codes, include_codes=self.__include_codes)

        # Access the prepared data
        X_train, y_train = split_data.X_train, split_data.y_train
        X_test, y_test = split_data.X_test, split_data.y_test
        unique_codes_train = split_data.unique_codes_train
        unique_codes_test = split_data.unique_codes_test

        unique_codes_val = None
        X_val, y_val = None, None
        if self.__n_val > 0:
            X_val, y_val = split_data.X_val, split_data.y_val
            unique_codes_val = split_data.unique_codes_val

        # Choose the scaler or skip if None
        if self.__scaler_type not in ['StandardScaler', 'MinMaxScaler', None]:
            raise ValueError(f"Unsupported scaler type: {self.__scaler_type}")

        if self.__scaler_type == 'StandardScaler':
            scaler = StandardScaler()
        elif self.__scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            scaler = None

        # Scale the data if scaler is not None
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if X_val is not None:
                X_val = scaler.transform(X_val)

        # Optionally add 'Unique_Code' back to the DataFrame
        if self.__include_codes:
            X_train = pd.DataFrame(X_train, columns=self.__features)
            X_train['Unique_Code'] = unique_codes_train

            X_test = pd.DataFrame(X_test, columns=self.__features)
            X_test['Unique_Code'] = unique_codes_test

            if X_val is not None:
                X_val = pd.DataFrame(X_val, columns=self.__features)
                X_val['Unique_Code'] = unique_codes_val

        # Save dataframes to parquet files if parquet is True
        if self.__parquet:
            if self.__path is None:
                raise ValueError("Path must be specified if parquet is set to True")

            os.makedirs(self.__path, exist_ok=True)

            train_df = pd.DataFrame(X_train, columns=self.__features)
            train_df['label'] = y_train
            train_df.to_parquet(os.path.join(self.__path, 'train_data.parquet'), index=False)

            test_df = pd.DataFrame(X_test, columns=self.__features)
            test_df['label'] = y_test
            test_df.to_parquet(os.path.join(self.__path, 'test_data.parquet'), index=False)

            if X_val is not None:
                val_df = pd.DataFrame(X_val, columns=self.__features)
                val_df['label'] = y_val
                val_df.to_parquet(os.path.join(self.__path, 'val_data.parquet'), index=False)

        # Log data sizes
        logger.info(f'Training set size: {X_train.shape}')
        if X_val is not None:
            logger.info(f'Validation set size: {X_val.shape}')
        logger.info(f'Test set size: {X_test.shape}')

        return X_train, y_train, X_val, y_val, X_test, y_test, unique_codes_train, unique_codes_test, unique_codes_val, scaler

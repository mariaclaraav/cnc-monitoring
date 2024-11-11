import os
import pandas as pd
import logging
from typing import List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.utils.data_processing.splitting import SplitData

class SetupData:
    def __init__(self, df: pd.DataFrame, train_period: list, test_period: list, n_val: float, 
                 features: list, scaler_type: str = 'StandardScaler', print_codes: bool = False, 
                 parquet: bool = False, path: Optional[str] = None, include_codes: bool = False):
        """
        Initialize the DataPreprocessor with parameters for processing data.

        Parameters:
        ----------
        df : pd.DataFrame
            Input DataFrame containing features and labels.
        
        train_period : list
            List of periods to use for training mask creation.
        
        test_period : list
            List of periods to use for test mask creation.
        
        n_val : float
            Fraction of training data to use for validation.
        
        features : list
            List of feature column names.
        
        scaler_type : str, optional (default='StandardScaler')
            Type of scaler to apply ('StandardScaler', 'MinMaxScaler', or None).
        
        print_codes : bool, optional (default=False)
            Whether to print the unique codes for training, validation, and testing.
        
        parquet : bool, optional (default=False)
            If True, saves the datasets as parquet files.
        
        path : str, optional
            Directory to save parquet files (required if `parquet=True`).
        
        include_codes : bool, optional (default=False)
            Whether to include 'Unique_Code' in the feature sets.
        """
        self.df = df
        self.train_period = train_period
        self.test_period = test_period
        self.n_val = n_val
        self.features = features
        self.scaler_type = scaler_type
        self.print_codes = print_codes
        self.parquet = parquet
        self.path = path
        self.include_codes = include_codes
        self.logger = logging.getLogger(__name__)

    def create_mask(self, periods: List[str], machine_types: Optional[List[str]] = None, normal: bool = True) -> pd.Series:
        """
        Create a boolean mask for the DataFrame based on specified periods, machine types, and a normal flag.
        """
        period_mask = self.df['Period'].isin(periods)
        machine_mask = self.df['Machine'].isin(machine_types) if machine_types else pd.Series(True, index=self.df.index)
        label_mask = self.df['Label'] == 0 if normal else pd.Series(True, index=self.df.index)
        return period_mask & machine_mask & label_mask

    def setup(self) -> Tuple:
        """
        Prepares the dataset for training, validation, and testing by splitting, scaling, 
        and optionally saving to parquet files.

        Returns:
        --------
        Tuple containing:
            - X_train, y_train : Training features and labels.
            - X_val, y_val : Validation features and labels (if applicable).
            - X_test, y_test : Test features and labels.
            - unique_codes : Validation unique codes.
            - scaler : The fitted scaler used for scaling the data.
        """
        # Create train and test masks
        train_mask = self.create_mask(self.train_period, normal=False)
        test_mask = self.create_mask(self.test_period, normal=False)
        
        # Initialize SplitData object with print_codes parameter
        split_data = SplitData(self.df, train_mask, test_mask, self.n_val, self.features, 
                               print_codes=self.print_codes, include_codes=self.include_codes)

        # Access the prepared data
        X_train, y_train = split_data.X_train, split_data.y_train
        X_test, y_test = split_data.X_test, split_data.y_test
        unique_codes_train = split_data.unique_codes_train
        unique_codes_test = split_data.unique_codes_test
        
        unique_codes_val = None
        X_val, y_val = None, None
        if self.n_val > 0:
            X_val, y_val = split_data.X_val, split_data.y_val
            unique_codes_val = split_data.unique_codes_val

        # Choose the scaler or skip if None
        if self.scaler_type == 'StandardScaler':
            scaler = StandardScaler()
        elif self.scaler_type == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif self.scaler_type is None:
            scaler = None
        else:
            raise ValueError(f"Unsupported scaler type: {self.scaler_type}")
        
        # Scale the data if scaler is not None
        if scaler is not None:
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            if X_val is not None:
                X_val = scaler.transform(X_val)

        # Optionally add 'Unique_Code' back to the DataFrame
        if self.include_codes:
            X_train = pd.DataFrame(X_train, columns=self.features)
            X_train['Unique_Code'] = unique_codes_train
            
            X_test = pd.DataFrame(X_test, columns=self.features)
            X_test['Unique_Code'] = unique_codes_test
            
            if X_val is not None:
                X_val = pd.DataFrame(X_val, columns=self.features)
                X_val['Unique_Code'] = unique_codes_val

        # Save dataframes to parquet files if parquet is True
        if self.parquet:
            if self.path is None:
                raise ValueError("Path must be specified if parquet is set to True")
            
            os.makedirs(self.path, exist_ok=True)
            
            train_df = pd.DataFrame(X_train, columns=self.features)
            train_df['label'] = y_train
            train_df.to_parquet(os.path.join(self.path, 'train_data.parquet'), index=False)

            test_df = pd.DataFrame(X_test, columns=self.features)
            test_df['label'] = y_test
            test_df.to_parquet(os.path.join(self.path, 'test_data.parquet'), index=False)

            if X_val is not None:
                val_df = pd.DataFrame(X_val, columns=self.features)
                val_df['label'] = y_val
                val_df.to_parquet(os.path.join(self.path, 'val_data.parquet'), index=False)

        # Log data sizes
        self.logger.info(f'Training set size: {X_train.shape}')
        if X_val is not None:
            self.logger.info(f'Validation set size: {X_val.shape}')
        self.logger.info(f'Test set size: {X_test.shape}')
        
        return X_train, y_train, X_val, y_val, X_test, y_test, unique_codes_train, unique_codes_test, unique_codes_val, scaler

    def run(self) -> Tuple:
        """
        Calls the setup_data function to process the data.
        
        Returns:
        --------
        The output from setup_data, containing prepared training, validation, test datasets, and scaler.
        """
        return self.setup()

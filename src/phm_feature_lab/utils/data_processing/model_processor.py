import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from phm_feature_lab.utils.data_processing.data_scaler import DataScaler
from phm_feature_lab.utils.logger import Logger 

logger = Logger().get_logger()


class ModelProcessor:
    def __init__(self, logger, scaler_type=None, need_val = True):
        self.logger = logger
        self.__scaler_type = scaler_type
        self.__scaler = None
        self.is_fitted = False
        self.__need_val = need_val
        
    def load_data(self, data_path, operation=None):
        try:
            self.logger.info(f"Loading data from {data_path}...")
            df = pd.read_parquet(data_path)
            
            if operation:
                # Ensure operation is always a list
                if isinstance(operation, str):
                    operation = [operation]
                df = df[df['Operation'].isin(operation)]
            
            return df
        
        except FileNotFoundError as e:
            self.logger.error(f"Data file not found: {data_path}")
            raise e
        
    
    def process_data(self, df_train, df_val, features):
        """
        Processes training and validation data by applying scaling and preparing feature and label datasets.
        
        Args:
            df_train (pd.DataFrame): Training data.
            df_val (pd.DataFrame): Validation data.

        Returns:
            tuple: Processed training and validation data, and scaler if applicable.
        """
        self.logger.info('Starting data processing for training and validation...' if self.__need_val else 'Starting data processing for training...')
        
        # Handling training and validation split
        if self.__need_val:
            X_train, y_train = df_train[features], df_train['Label']
            X_val, y_val, val_codes = df_val[features], df_val['Label'], df_val['Unique_Code']
        else:
            df_combined = pd.concat([df_train, df_val]).reset_index(drop=True)
            X_train, y_train = df_combined[features], df_combined['Label']
            val_codes = df_combined['Unique_Code']
            X_val, y_val, = None, None

        # Initialize scaler
        scaler_map = {
            'standard': StandardScaler,
            'minmax': MinMaxScaler
        }
        
        if self.__scaler_type:
            if self.__scaler_type not in scaler_map:
                raise ValueError(f"Scaler type '{self.__scaler_type}' is not supported.")
            self.__scaler = scaler_map[self.__scaler_type]()
        else:
            self.__scaler = None

        # Apply scaling if scaler is initialized
        if self.__scaler:
            scaler = DataScaler(scaler=self.__scaler, exclude_columns=["Time"])
            X_train = scaler.fit_transform(X_train)
            if X_val is not None:
                X_val = scaler.transform(X_val)
            self.__scaler = scaler
        self.is_fitted = True  # Mark scaler as fitted
        # Return processed data
        return X_train, y_train, X_val, y_val, val_codes, self.__scaler
    
    def process_test_data(self, df_test, features):
        
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted. Please process training data first.")
        
        self.logger.info('Starting data processing for test set...')
        
        X_test = df_test[features]
        y_test = df_test['Label']
        test_codes = df_test['Unique_Code']
        
        # Applying the fitted scaler to the test set
        if self.__scaler:
            X_test = self.__scaler.transform(X_test)
            X_test = pd.DataFrame(X_test, columns=features)

        return X_test, y_test, test_codes


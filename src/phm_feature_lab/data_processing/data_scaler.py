import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Union


class ScalerInterface(ABC):
    """Interface defining scaling operations."""
    
    @abstractmethod
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        pass

class ScalerWrapper:
    """Wraps a scaler object to perform scaling operations."""
    
    def __init__(self, scaler: ScalerInterface):
        """
        Initialize the ScalerWrapper.

        Args:
            scaler (ScalerInterface): An instantiated scaler object implementing ScalerInterface.

        Returns:
            None

        Raises:
            ValueError: If 'scaler' is None.
        """
        if scaler is None:
            raise ValueError("The 'scaler' parameter is required.")
        self.scaler = scaler
    
    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Fit the scaler and transform the data.

        Args:
            data (np.ndarray): Numeric data to fit and transform.

        Returns:
            np.ndarray: Scaled data.
        """
        return self.scaler.fit_transform(data)
    
    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform the data using the fitted scaler.

        Args:
            data (np.ndarray): Numeric data to transform.

        Returns:
            np.ndarray: Scaled data.
        """
        return self.scaler.transform(data)
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Revert scaled data to original values.

        Args:
            data (np.ndarray): Scaled numeric data to revert.

        Returns:
            np.ndarray: Original data.
        """
        return self.scaler.inverse_transform(data)


class ColumnSelector:
    """Identifies numeric columns in a DataFrame, excluding specified columns."""
    
    def __init__(self, exclude_columns: Optional[List[str]] = None):
        """
        Initialize the ColumnSelector.

        Args:
            exclude_columns (Optional[List[str]], optional): Columns to exclude from scaling. Defaults to None.

        Returns:
            None
        """
        self.exclude_columns = exclude_columns or []
        self.numeric_cols = None
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Identify numeric columns in the DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            None
        """
        self.numeric_cols = df.select_dtypes(include=['number']).columns.difference(self.exclude_columns)
    
    def get_numeric_columns(self) -> pd.Index:
        """
        Get the identified numeric columns.

        Args:
            None

        Returns:
            pd.Index: Index of numeric columns.

        Raises:
            ValueError: If columns have not been identified (fit not called).
        """
        if self.numeric_cols is None:
            raise ValueError("Numeric columns not identified. Call 'fit' first.")
        return self.numeric_cols


class DataFrameProcessor:
    """Processes DataFrames by combining scaled numeric data with non-numeric data."""
    
    @staticmethod
    def combine_scaled_data(df: pd.DataFrame, scaled_values: np.ndarray, numeric_cols: pd.Index) -> pd.DataFrame:
        """
        Combine scaled numeric data with the original DataFrame.

        Args:
            df (pd.DataFrame): Original DataFrame.
            scaled_values (np.ndarray): Scaled numeric values.
            numeric_cols (pd.Index): Columns to replace with scaled values.

        Returns:
            pd.DataFrame: Combined DataFrame.
        """
        scaled_df = pd.DataFrame(scaled_values, columns=numeric_cols, index=df.index)
        result_df = df.copy()
        result_df[numeric_cols] = scaled_df
        return result_df

class DataScaler:
    """A class to normalize numeric columns in a DataFrame with a reusable scaler."""
    
    def __init__(self, scaler: ScalerInterface, exclude_columns: Optional[List[str]] = None):
        """
        Initialize the DataScaler.

        Args:
            scaler (ScalerInterface): An instantiated scaler object implementing ScalerInterface.
            exclude_columns (Optional[List[str]], optional): Columns to exclude from scaling. Defaults to None.

        Returns:
            None
        """
        self.scaler_wrapper = ScalerWrapper(scaler)
        self.column_selector = ColumnSelector(exclude_columns)
        self.dataframe_processor = DataFrameProcessor()
        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler to the DataFrame and scale numeric columns.

        Args:
            df (pd.DataFrame): Input DataFrame to fit and transform.

        Returns:
            pd.DataFrame: DataFrame with numeric columns normalized.
        """
        self.column_selector.fit(df)
        numeric_cols = self.column_selector.get_numeric_columns()
        scaled_values = self.scaler_wrapper.fit_transform(df[numeric_cols].values)
        self.is_fitted = True
        return self.dataframe_processor.combine_scaled_data(df, scaled_values, numeric_cols)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the DataFrame using the fitted scaler.

        Args:
            df (pd.DataFrame): Input DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with numeric columns normalized.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if not self.is_fitted:
            raise ValueError("The scaler has not been fitted. Call 'fit_transform' first.")
        numeric_cols = self.column_selector.get_numeric_columns()
        scaled_values = self.scaler_wrapper.transform(df[numeric_cols].values)
        return self.dataframe_processor.combine_scaled_data(df, scaled_values, numeric_cols)
    
    def inverse_transform(self, df: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Revert scaled data to original values.

        Args:
            df (Union[pd.DataFrame, np.ndarray]): Scaled DataFrame or array to revert.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Data with numeric columns reverted to original values.

        Raises:
            ValueError: If the scaler has not been fitted.
            TypeError: If input is neither a DataFrame nor numpy array.
        """
        if not self.is_fitted:
            raise ValueError("The scaler has not been fitted. Call 'fit_transform' first.")
        
        if isinstance(df, pd.DataFrame):
            numeric_cols = self.column_selector.get_numeric_columns()
            original_values = self.scaler_wrapper.inverse_transform(df[numeric_cols].values)
            return self.dataframe_processor.combine_scaled_data(df, original_values, numeric_cols)
        elif isinstance(df, np.ndarray):
            return self.scaler_wrapper.inverse_transform(df)
        else:
            raise TypeError("Input must be a pandas DataFrame or a numpy array.")
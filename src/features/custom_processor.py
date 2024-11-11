import logging
from typing import List, Dict, Any
import pandas as pd

class CustomProcessor:
    """Class to filter and process time series data."""

    # Mapping of month abbreviations to numeric values
    MONTH_MAP: Dict[str, str] = {
        'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
        'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
        'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
    }

    def __init__(self, processor: Any):
        self.processor = processor
        self.logger = logging.getLogger(self.__class__.__name__)

    def filter_dataframe(
        self, df: pd.DataFrame, operation: str
    ) -> pd.DataFrame:
        """Filter the DataFrame by operation and process date columns.

        Args:
            df (pd.DataFrame): The input DataFrame.
            operation (str): The operation to filter on.

        Returns:
            pd.DataFrame: The filtered and processed DataFrame.
        """
        df_filtered = df[df['Process'] == operation].copy()

        # Map month abbreviations to numeric values and create 'Period'
        df_filtered['Month'] = df_filtered['Month'].map(self.MONTH_MAP)
        df_filtered['Period'] = (
            df_filtered['Month'] + '-' + df_filtered['Year'].astype(str)
        )
        df_filtered.drop(columns=['Month', 'Year'], inplace=True)
        df_filtered.reset_index(drop=True, inplace=True)
        
        self.logger.info(f"Data filtered successfully. Shape: {df_filtered.shape}")
        
        return df_filtered

    def process_features(
        self, df_filtered: pd.DataFrame, feature_types: List[str]
    ) -> List[pd.DataFrame]:
        """Process the time series features.

        Args:
            df_filtered (pd.DataFrame): The filtered DataFrame.
            feature_types (List[str]): List of feature types to process.

        Returns:
            List[pd.DataFrame]: List of processed DataFrames.
        """
        
        processed_data = {}
        for feature_type in feature_types:
            self.logger.info(f"Calculating {feature_type} data...")
            
            if feature_type in ['dwt', 'wpd']:
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type, wavelet='db14', level=3)
            elif feature_type == 'filter':
                print('here')
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type, order=4)
            elif feature_type == 'emd':
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type, num_imfs=5)
            else:
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type)
        
            data.reset_index(drop=True, inplace=True)
            print(f"Processed {feature_type} data shape:{data.shape}\n")
            print(data)
            processed_data[feature_type] = data
            
        del df_filtered  
        
        return processed_data

    def merge_processed_data(
        self, processed_data: List[pd.DataFrame]
    ) -> pd.DataFrame:
        """Merge all processed DataFrames into one.

        Args:
            processed_data (List[pd.DataFrame]): List of processed DataFrames.

        Returns:
            pd.DataFrame: The merged DataFrame.
        """
        self.logger.info("Merging processed data...")
        merged_data = list(processed_data.values())[0]
        print("Initial merged_data shape:", merged_data.shape)
        print("Initial merged_data columns:", merged_data.columns)

        # Columns to drop before merging
        columns_to_drop = [
            'Process', 'Machine', 'Label', 'Period',
            'X_X_axis', 'Y_Y_axis', 'Z_Z_axis',
        ]

        for i, data in enumerate(list(processed_data.values())[1:], start=2):
            print(f"\nProcessing DataFrame {i} for merge...")
            print("Original data shape:", data.shape)
            
            # Dropping specified columns from the data
            data = data.drop(columns=columns_to_drop, errors='ignore')
            print("Data shape after dropping columns:", data.shape)
            print("Data columns after dropping:", data.columns)

            # Checking for key column consistency
            if 'Unique_Code' not in data.columns or 'Time' not in data.columns:
                print("Warning: 'Unique_Code' or 'Time' columns missing in DataFrame", i)
                continue

            # Performing the merge
            try:
                merged_data = merged_data.merge(data, on=['Unique_Code', 'Time'], how='inner')
                print(f"Merged data shape after merging DataFrame {i}:", merged_data.shape)
            except Exception as e:
                print(f"Error during merging DataFrame {i}: {e}")
                break
            
            del data  # Freeing up memory

        del processed_data  # Freeing up memory after merging
        print("Final merged_data shape:", merged_data.shape)
        print("Final merged_data columns:", merged_data.columns)
        return merged_data

    def clean_final_dataframe(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Clean the final DataFrame by removing duplicates, renaming, and dropping NaNs.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: The cleaned DataFrame.
        """
        self.logger.info("Removing duplicate columns...")
        df = df.loc[:, ~df.columns.duplicated()]

        # Rename specific columns
        columns_to_rename = {
            'X_X_axis': 'X_axis',
            'Y_Y_axis': 'Y_axis',
            'Z_Z_axis': 'Z_axis',
        }
        self.logger.info("Renaming columns...")
        df.rename(columns=columns_to_rename, inplace=True)

        # Remove rows with any NaN values and reset index
        self.logger.info("Cleaning DataFrame...")
        df.dropna(inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

    def filter_and_process(
        self, df: pd.DataFrame, operation: str, feature_types: List[str]
    ) -> pd.DataFrame:
        """Filter the DataFrame and process time series features.

        Args:
            df (pd.DataFrame): The input DataFrame.
            operation (str): The operation to filter on.
            feature_types (List[str]): List of feature types to process.

        Returns:
            pd.DataFrame: The processed and cleaned DataFrame.
        """
        df_filtered = self.filter_dataframe(df, operation)
        processed_data = self.process_features(df_filtered, feature_types)
        merged_data = self.merge_processed_data(processed_data)
        
        del df_filtered
        del processed_data
        
        df_final = self.clean_final_dataframe(merged_data)
        
        del merged_data
        
        return df_final

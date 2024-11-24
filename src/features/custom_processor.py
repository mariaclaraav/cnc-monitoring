import logging
import gc
import psutil
from typing import List, Dict, Any
import pandas as pd
from src.features.frequency_analyzer import FrequencyAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
        self.logger = logging.getLogger()
        
    def log_memory_usage(self, message: str):
        """Log the current memory usage."""
        process = psutil.Process()
        mem_usage = process.memory_info().rss / (1024 ** 2)  # Memory usage in MB
        self.logger.info(f"{message} | Memory usage: {mem_usage:.2f} MB")
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
        try:
            # Check if the necessary columns exist
            required_columns = {'Process', 'Month', 'Year'}
            if not required_columns.issubset(df.columns):
                self.logger.error(f"Missing columns in DataFrame. Found: {df.columns}")
                raise ValueError(f"DataFrame must contain {required_columns}")

            # Check the unique values of the 'Process' column
            self.logger.info(f"Available operations: {df['Process'].unique()}")
            
            # Filter the DataFrame
            df_filtered = df[df['Process'] == operation].copy()
            if df_filtered.empty:
                self.logger.error(f"No rows found for operation: {operation}")
                raise ValueError(f"No data found for operation '{operation}'")

            # Debugging: Log dtypes and check filtered columns
            self.logger.debug(f"DataFrame dtypes: {df.dtypes}")
            self.logger.debug(f"Filtered columns: {df_filtered.columns}")

            # Map month abbreviations to numeric values and create 'Period'
            df_filtered['Month'] = df_filtered['Month'].map(self.MONTH_MAP)
            if df_filtered['Month'].isnull().any():
                self.logger.error("Mapping Month to numeric values resulted in NaN values.")
                raise ValueError("Invalid month abbreviations found in the data.")

            df_filtered['Period'] = (
                df_filtered['Month'].astype(str) + '-' + df_filtered['Year'].astype(str)
            )
            df_filtered.drop(columns=['Month', 'Year'], inplace=True)
            df_filtered.reset_index(drop=True, inplace=True)

            self.logger.info(f"Data filtered successfully. Shape: {df_filtered.shape}")
            del df
            gc.collect()
            return df_filtered

        except Exception as e:
            # Log error and provide details for debugging
            self.logger.error(f"Error filtering DataFrame: {e}")
            self.logger.error(f"DataFrame dtypes: {df.dtypes}")
            self.logger.error(f"Operation column unique values: {df['Process'].unique()}")
            raise
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reduz o uso de memória de um DataFrame convertendo os tipos de dados.

        Args:
            df (pd.DataFrame): O DataFrame original.

        Returns:
            pd.DataFrame: O DataFrame otimizado.
        """
        initial_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"Memória antes da otimização: {initial_memory:.2f} MB")

        for col in df.columns:
            col_type = df[col].dtypes

            if col_type == 'object':
                # Converte strings para categorias se o número de valores únicos for pequeno
                if df[col].nunique() / len(df[col]) < 0.5:
                    df[col] = df[col].astype('category')

            elif col_type in ['int64', 'int32']:
                # Converte inteiros para o menor tipo possível
                min_val, max_val = df[col].min(), df[col].max()
                if min_val >= -128 and max_val <= 127:
                    df[col] = df[col].astype('int8')
                elif min_val >= -32768 and max_val <= 32767:
                    df[col] = df[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    df[col] = df[col].astype('int32')

            elif col_type in ['float64', 'float32']:
                # Converte floats para float32
                df[col] = df[col].astype('float32')

        final_memory = df.memory_usage(deep=True).sum() / (1024 ** 2)
        print(f"Memória após a otimização: {final_memory:.2f} MB")
        print(f"Redução de memória: {100 * (initial_memory - final_memory) / initial_memory:.2f}%")
        
        return df
    def process_features(
        self, df_filtered: pd.DataFrame, feature_types: List[str], filter_list = None
    ) -> List[pd.DataFrame]:
        """Process the time series features.

        Args:
            df_filtered (pd.DataFrame): The filtered DataFrame.
            feature_types (List[str]): List of feature types to process.

        Returns:
            List[pd.DataFrame]: List of processed DataFrames.
        """
        
        frequency_by_axis = {}           
        processed_data = {}
        for feature_type in feature_types:
            self.logger.info(f"Calculating {feature_type} data...")
            
            if feature_type in ['dwt', 'wpd']:
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type, wavelet='db14', level=3)
                
            elif feature_type == 'filter':
                
                
                for col in ['X_axis', 'Y_axis', 'Z_axis']:
                    if filter_list is None:
                        analyzer = FrequencyAnalyzer(
                        df=df_filtered,
                        column=col,
                        sampling_rate=2000,
                        height_thresh=0.05,
                        plot = False)
                        frequency_by_axis[col] = analyzer.run_analysis(bin_width_refined=1, top_n_initial=10)
                        self.logger.info("Frequency by axis: {frequency_by_axis}")
                    else:
                        frequency_by_axis[col] = filter_list
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type, order=4, frequency_bands=frequency_by_axis)
                
            elif feature_type == 'emd':
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type, num_imfs=5)
            else:
                data = self.processor.process_time_series(df = df_filtered, feature_type=feature_type)
        
            data.reset_index(drop=True, inplace=True)
            data = self.optimize_dataframe(data)
            self.logger.info(f"Processed {feature_type} data shape:{data.shape}\n")
            processed_data[feature_type] = data
            
        del df_filtered
        gc.collect()
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

        # Columns to drop before merging
        columns_to_drop = [
            'Process', 'Machine', 'Label', 'Period',
            'X_X_axis', 'Y_Y_axis', 'Z_Z_axis',
        ]

        for i, data in enumerate(list(processed_data.values())[1:], start=2):
            self.logger.info(f"\nProcessing DataFrame {i} for merge...")
            self.log_memory_usage(f"Memory usage before processing DataFrame {i}")
            
            # Dropping specified columns from the data
            data = data.drop(columns=columns_to_drop, errors='ignore')

            # Checking for key column consistency
            if 'Unique_Code' not in data.columns or 'Time' not in data.columns:
                self.flogger.info("Warning: 'Unique_Code' or 'Time' columns missing in DataFrame", i)
                continue

            # Performing the merge
            try:
                merged_data = merged_data.merge(data, on=['Unique_Code', 'Time'], how='inner')
            except Exception as e:
                self.logger.info(f"Error during merging DataFrame {i}: {e}")
                break
            
            self.log_memory_usage(f"Memory usage after merging DataFrame {i}")
            del data  # Freeing up memory
            gc.collect()
            self.log_memory_usage("Final memory usage after merging")
            
        del processed_data  # Freeing up memory after merging
        gc.collect()
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
        self, df: pd.DataFrame, operation: str, feature_types: List[str], filter_list = None
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
        processed_data = self.process_features(df_filtered, feature_types, filter_list)
        merged_data = self.merge_processed_data(processed_data)
        
        del df_filtered
        del processed_data
        gc.collect()
        
        df_final = self.clean_final_dataframe(merged_data)
        
        del merged_data
        gc.collect()
        return df_final

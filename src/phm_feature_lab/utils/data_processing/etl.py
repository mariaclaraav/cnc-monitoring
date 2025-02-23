import logging
import pandas as pd

logger = logging.getLogger() 

def load_data(data_path: str) -> pd.DataFrame:
    """
    Load data from a parquet file.
    
    Args:
        data_path (str): Path to the parquet file.
        
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    logger.info(f"Attempting to load data from {data_path}...")
    try:
        df = pd.read_parquet(data_path)
        logger.info(f"Data successfully loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {data_path}. Please verify the path.")
        raise

# Function to filter the data
def filter_time_group(group):
    """
    Filters out the first and last 4 seconds of data for each group.

    Args:
        group (pd.DataFrame): The group of data to be filtered.

    Returns:
        pd.DataFrame: The filtered group.
    """
    filtered_group = group[(group['Time'] > 5) & (group['Time'] < group['Time'].max() - 5)]
    return filtered_group

# Function to convert columns to int8 and float32
def convert_columns(df, int8_columns, float32_columns):
    """
    Converts specified columns to int8 and float32.

    Args:
        df (pd.DataFrame): The DataFrame containing the columns to be converted.
        int8_columns (list): List of column names to be converted to int8.
        float32_columns (list): List of column names to be converted to float32.

    Returns:
        None: Modifies the original DataFrame.
    """
    for column in int8_columns:
        df[column] = df[column].round().astype('int8')
    for column in float32_columns:
        df[column] = df[column].astype('float32')

# Function to clean the columns names        
def clean_column_names(columns):
    cleaned_columns = []
    for col in columns:
        col = col.replace('Rolling ', '')
        col = col.replace(' ', '_')
        cleaned_columns.append(col)
    return cleaned_columns
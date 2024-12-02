import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Union

def add_period(
    data: pd.DataFrame, 
    month_column: str, 
    year_column: str, 
    new_column_name: str = "Period"
) -> pd.DataFrame:
    """
    Adds a new column to a DataFrame that combines month and year into a period format 'MM-YYYY'.

    Args:
        data (pd.DataFrame): The input DataFrame.
        month_column (str): The column name containing the month abbreviations (e.g., 'Jan', 'Feb').
        year_column (str): The column name containing the year values.
        new_column_name (str): The name of the new column to be added. Default is "Period".

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the new period column added.
    """
    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
    }
    df = data.copy()
    df[new_column_name] = df[month_column].map(month_map) + "-" + df[year_column].astype(str)
    return df

def order_unique_code(df, column_name='Unique_Code'):
    
    # Extract year, month, and last number from the Unique_Code
    df['Year'] = df[column_name].str.extract(r'_(\d{4})_')[0]
    df['Month'] = df[column_name].str.extract(r'_(Feb|Aug)_')[0]
    df['Last_Number'] = df[column_name].str.extract(r'_(\d+)$')[0]
    # Define the order for the months
    month_order = {'Feb': 1, 'Aug': 2}
    df['Month_Order'] = df['Month'].map(month_order)

    # Sort the DataFrame by Year, Month_Order, and Last_Number
    df.sort_values(by=['Year', 'Month_Order', 'Last_Number'], inplace=True)

    # Drop the helper columns
    df.drop(columns=['Year', 'Month', 'Last_Number', 'Month_Order'], inplace=True)
    df.reset_index(drop=True, inplace=True)
import pandas as pd

class Utilities:
    """ A utility class for performing common DataFrame operations"""

    MONTH_MAP = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
    }

    MONTH_ORDER = {
        "Feb": 1, "Aug": 2
    }

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """ Normalizes column names in a DataFrame to lowercase to ensure case-insensitivity.

        Args:
            df (pd.DataFrame): The input DataFrame.

        Returns:
            pd.DataFrame: A copy of the DataFrame with normalized column names.
        """
        df = df.copy()
        df.columns = df.columns.str.lower()
        return df

    @staticmethod
    def add_period(
        data: pd.DataFrame,
        month_column: str,
        year_column: str,
        new_column_name: str = "Period"
    ) -> pd.DataFrame:
        """ Adds a new column to the DataFrame combining month and year into a period ('MM-YYYY').

        Args:
            data (pd.DataFrame): The input DataFrame.
            month_column (str): Column name containing month abbreviations (e.g., 'Jan', 'Feb').
            year_column (str): Column name containing year values.
            new_column_name (str): Name of the new column to be added. Default is 'Period'.

        Returns:
            pd.DataFrame: A copy of the input DataFrame with the new period column added.

        Raises:
            KeyError: If the specified columns are not found in the DataFrame.
        """
        df = Utilities.normalize_columns(data)
        month_column = month_column.lower()
        year_column = year_column.lower()

        if month_column not in df.columns or year_column not in df.columns:
            raise KeyError(f"Columns '{month_column}' and/or '{year_column}' not found in the DataFrame.")

        df[new_column_name] = df[month_column].str.capitalize().map(Utilities.MONTH_MAP) + "-" + df[year_column].astype(str)
        return df

    @staticmethod
    def extract_unique_code_parts(df: pd.DataFrame, column_name: str):
        """ Extract parts of the unique code: year, month, and last number
        """
        df['year'] = df[column_name].str.extract(r'_(\d{4})_')[0].astype(int)
        df['month'] = df[column_name].str.extract(r'_(Feb|Aug)_')[0].str.capitalize()
        df['last_number'] = df[column_name].str.extract(r'_(\d+)$')[0].astype(int)
        
        return df

    @staticmethod
    def order_unique_code(df: pd.DataFrame, column_name: str = "Unique_Code") -> pd.DataFrame:
        """ Orders a DataFrame based on 'Unique_Code' column by Year, Month, and Last Number.

        Args:
            df (pd.DataFrame): The input DataFrame.
            column_name (str): Name of the column containing the unique code. Default is 'Unique_Code'.

        Returns:
            pd.DataFrame: A sorted copy of the input DataFrame.
        """
        df = Utilities.normalize_columns(df)
        column_name = column_name.lower()

        if column_name not in df.columns:
            raise KeyError(f"Column '{column_name}' not found in the DataFrame.")


        df = Utilities.extract_unique_code_parts(df, column_name)

        df["month_order"] = df["month"].map(Utilities.MONTH_ORDER)
        
        df.sort_values(by=["year", "month_order", "last_number"], inplace=True)
        df.drop(columns=["year", "month", "last_number", "month_order"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        return df

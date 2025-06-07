from tqdm import tqdm
from typing import List
from phm_feature_lab.utils.utilities import Utilities


class OperationFilter:
    """Class to filter and process DataFrame based on operations and unique codes."""

    def __init__(self, df):
        """Initialize the OperationFilter with a DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
        """
        self._df = df
        
    @staticmethod
    def filter_by_operation(df, operation):
        """Filter the DataFrame by a specific operation.

        Args:
            operation (str): The operation to filter by.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return df[df["Process"] == operation].reset_index(drop=True)

    def __filter_by_columns(self, columns):
        """Filter the DataFrame to include only specified columns.

        Args:
            columns (list): List of columns to include.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return self._df[columns]

    def filter_by_unique_code(self, df, unique_code):
        """Filter the DataFrame by a unique code.

        Args:
            df (pd.DataFrame): The DataFrame to filter.
            unique_code (str): The unique code to filter by.

        Returns:
            pd.DataFrame: Filtered DataFrame.
        """
        return df[df["Unique_Code"] == unique_code].reset_index(drop=True)

    def get_unique_codes(self, df):
        """Get unique codes from the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to extract unique codes from.

        Returns:
            np.ndarray: Array of unique codes.
        """
        return df["Unique_Code"].unique()
    
    def __validate_operations(self, operations):
        Utilities.ensure_list(
            operations,
            expected_type=str,
            allow_empty=False,
            error_message="`operations` must be a non-empty list of strings.",
        )
         
    def __validate_columns(self, columns):
        Utilities.ensure_list(
            columns,
            expected_type=str,
            allow_empty=False,
            error_message="`columns` must be a non-empty list of strings.",
        )
        
    def filter(self, columns: List[str], operations: List[str]):
        """Filter and process the DataFrame based on operations and columns.

        Args:
            operations (List[str]): List of operations to filter by.
            columns (List[str]): List of columns to include.

        Returns:
            pd.DataFrame: Processed DataFrame.

        Raises:
            TypeError: If `operations` is not a list of strings.
            ValueError: If `operations` is an empty list.
        """
   
        self.__validate_columns(columns)
        self.__filter_by_columns(columns)
        self.__validate_operations(operations)
        return self._df[self._df["Process"].isin(operations)].reset_index(drop=True)

       
    # def filter_and_process(self, operations: List[str], columns: List[str]):
    #     """Filter and process the DataFrame based on operations and columns.

    #     Args:
    #         operations (List[str]): List of operations to filter by.
    #         columns (List[str]): List of columns to include.

    #     Returns:
    #         pd.DataFrame: Processed DataFrame.

    #     Raises:
    #         TypeError: If `operations` is not a list of strings.
    #         ValueError: If `operations` is an empty list.
    #     """
    #     self.__validate_operations(operations)
    #     self.__validate_columns(columns)
        
    #     self._df = self.__filter_by_columns(columns)

    #     for operation in tqdm(operations, desc="Processing operations"):
    #         df_filtered = self.__filter_by_operation(operation)
    #         unique_codes = self.__get_unique_codes(df_filtered)

    #         for code in tqdm(
    #             unique_codes, desc=f"Processing unique codes for {operation}", leave=False
    #         ):
    #             df_filtered = self.__filter_by_unique_code(df_filtered, code)
    #             return df_filtered
            
        
        
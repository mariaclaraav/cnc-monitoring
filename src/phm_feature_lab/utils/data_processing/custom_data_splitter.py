import re
import numpy as np
import pandas as pd
from typing import List, Tuple, Union, Optional, Dict
from phm_feature_lab.utils.logger import Logger 
from phm_feature_lab.utils.utils_contants import UtilsConstants

logger = Logger().get_logger()


class CustomDataSplitter:
    """Split data sequentially into training, validation, and testing sets."""

    month_order = UtilsConstants.MONTH_ORDER

    def __init__(
        self,
        df: pd.DataFrame,
        train_split_param: Dict,
        test_split_param: Dict,
        n_val: float,
        features: List[str] = ["X_axis", "Y_axis", "Z_axis"],
        include_codes: bool = True,
    ):
        """ Initialize the CustomDataSplitter.

        Args:
            df (pd.DataFrame): Input DataFrame containing features, labels, and a 'Unique_Code' column.
            train_split_param (Dict): Dictionary with parameters for creating the training mask.
            test_split_param (Dict): Dictionary with parameters for creating the test mask.
            n_val (float): Fraction of training data to use for validation (0 < n_val < 1).
            features (List[str], optional): List of feature columns to include. Defaults to ["X_axis", "Y_axis", "Z_axis"].
            include_codes (bool, optional): Whether to include 'Unique_Code' in the returned feature sets. Defaults to False.

        Returns:
            None
        """
        self.__df = df
        self.__features = features
        self.__include_codes = include_codes
        self.__n_val = self.__validate_n_val(n_val)

        # Validate if the dictionaries are provided
        self.__validate_split_params(train_split_param, "train_split_param")
        self.__validate_split_params(test_split_param, "test_split_param")

        # Create masks using the provided parameters
        self.__train_mask = self._create_mask(**train_split_param)
        self.__test_mask = self._create_mask(**test_split_param)

    def _create_mask(
        self,
        periods: List[str],
        machine_types: Optional[List[str]] = None,
        normal: bool = False,
    ) -> pd.Series:
        """Create a boolean mask based on periods, machine types, and label conditions.

        Args:
            periods (List[str]): List of periods to filter the data.
            machine_types (Optional[List[str]], optional): List of machine types to filter. Defaults to None.
            normal (bool, optional): If True, filter for normal data (Label == 0). Defaults to False.

        Returns:
            pd.Series: Boolean mask indicating rows that match the criteria.
        """

        # Create mask for specified periods
        period_mask = self.__df["Period"].isin(periods)

        # Create mask for specified machine types (if provided)
        machine_mask = (
            self.__df["Machine"].isin(machine_types)
            if machine_types
            else pd.Series(True, index=self.__df.index)
        )

        # Create mask based on 'Label' column if 'normal' is True
        label_mask = (
            self.__df["Label"] == 0
            if normal
            else pd.Series(True, index=self.__df.index)
        )

        # Combine all masks
        return period_mask & machine_mask & label_mask

    def __validate_n_val(self, n_val: float) -> float:
        """Validate that n_val is between 0 and 1 (exclusive).

        Args:
            n_val (float): Fraction of training data for validation.

        Returns:
            float: Validated n_val value.

        Raises:
            ValueError: If n_val is not between 0 and 1 (exclusive).
        """
        if not 0 < n_val < 1:
            raise ValueError(
                f"n_val must be between 0 and 1 (exclusive), but got {n_val}."
            )
        return n_val

    def __validate_split_params(
        self,
        split_param: Optional[Dict],
        param_name: str,
    ) -> None:
        """ Validate the split parameter dictionary.

        Args:
            split_param (Optional[Dict]): Dictionary containing split parameters.
            param_name (str): Name of the parameter for error messaging.

        Returns:
            None

        Raises:
            ValueError: If split_param is None, lacks 'periods' key, or contains invalid keys.
        """

        if split_param is None:
            logger.error(f"The '{param_name}' dictionary was not provided.")
            raise ValueError(f"The '{param_name}' dictionary is required.")

        # Check for required key
        if "periods" not in split_param:
            logger.error(
                f"The '{param_name}' dictionary must contain the 'periods' key."
            )
            raise ValueError(
                f"The '{param_name}' dictionary must contain the 'periods' key."
            )

        # Check for allowed keys
        allowed_keys = {"periods", "machine_types", "normal"}
        for key in split_param.keys():
            if key not in allowed_keys:
                logger.error(
                    f"Invalid key '{key}' in '{param_name}'. Allowed keys are: {allowed_keys}."
                )
                raise ValueError(
                    f"Invalid key '{key}' in '{param_name}'. Allowed keys are: {allowed_keys}."
                )

    @staticmethod
    def extract_keys(code: str) -> Tuple[int, int, int, int]:
        """ Extract year, month, machine, and sequence number from a 'Unique_Code' string.

        Args:
            code (str): The 'Unique_Code' string to parse.

        Returns:
            Tuple[int, int, int, int]: Tuple containing (year, month index, machine, sequence number).

        Raises:
            ValueError: If the code format is invalid.
        """
        match = re.match(r"M(\d+)_OP\d+_(\w{3})_(\d{4})_(\d+)", code)
        if match is None:
            raise ValueError(
                f"Invalid Unique_Code format: {code}. Expected format: M<number>_OP<number>_<3-letter-month>_<year>_<number>"
            )
        machine = int(match.group(1))
        month = match.group(2)
        year = int(match.group(3))
        num = int(match.group(4))
        return (year, CustomDataSplitter.month_order[month], machine, num)

    @staticmethod
    def split_mask(codes: np.ndarray, values: List[str]) -> np.ndarray:
        """ Create a mask for rows where 'Unique_Code' matches given values.

        Args:
            codes (np.ndarray): Array of 'Unique_Code' values from the DataFrame.
            values (List[str]): List of 'Unique_Code' values to match.

        Returns:
            np.ndarray: Boolean mask indicating matching rows.
        """
        mask = np.zeros(len(codes), dtype=bool)
        for value in values:
            mask |= codes == value
        return mask

    @staticmethod
    def add_sequence_and_sort(df: pd.DataFrame, code_order: List[str]) -> pd.DataFrame:
        """" Sort the DataFrame by 'Unique_Code' and 'Time' based on a given order.

        Args:
            df (pd.DataFrame): DataFrame to be sorted.
            code_order (List[str]): Ordered list of 'Unique_Code' to follow.

        Returns:
            pd.DataFrame: Sorted DataFrame.
        """

        def extract_last_number(code: str) -> Union[int, None]:
            match = re.search(r"_(\d+)$", code)
            return int(match.group(1)) if match else None

        df = df.copy()
        df["Unique_Code"] = pd.Categorical(
            df["Unique_Code"], categories=code_order, ordered=True
        )
        df_sorted = df.sort_values(by=["Unique_Code", "Time"])
        return df_sorted

    def __get_codes(self, mask: pd.Series) -> List[str]:
        """ Get sorted unique codes from the DataFrame based on a mask.

        Args:
            mask (pd.Series): Boolean mask to filter the DataFrame.

        Returns:
            List[str]: Sorted list of unique codes.

        Raises:
            ValueError: If 'Unique_Code' column is missing.
        """
        try:
            codes = self.__df[mask].Unique_Code.unique()
            return sorted(codes, key=CustomDataSplitter.extract_keys)
        except KeyError:
            raise ValueError(
                "The column 'Unique_Code' is required to perform this operation."
            )

    def __split_into_val_codes(self, codes: List[str]) -> Tuple[List[str], List[str]]:
        """
        Split codes into training and validation sets.

        Args:
            codes (List[str]): List of unique codes to split.

        Returns:
            Tuple[List[str], List[str]]: Tuple of (train_codes, val_codes).
        """
        n_val = int(self.__n_val * len(codes))
        val_codes = codes[-n_val:] if n_val > 0 else []
        train_codes = codes[:-n_val] if n_val > 0 else codes
        return train_codes, val_codes

    def __get_train_val_mask(self, codes: List[str]) -> pd.Series:
        """
        Create a mask for training or validation codes.

        Args:
            codes (List[str]): List of codes to create a mask for.

        Returns:
            pd.Series: Boolean mask for the specified codes.
        """
        return CustomDataSplitter.split_mask(self.__df["Unique_Code"].values, codes)

    def __get_dataframe(self, mask, codes):
        """ Get a sorted DataFrame based on a mask and code order.

        Args:
            mask: Boolean mask to filter the DataFrame.
            codes: Ordered list of 'Unique_Code' values.

        Returns:
            pd.DataFrame: Filtered and sorted DataFrame.
        """
        data = self.__df[mask]
        return CustomDataSplitter.add_sequence_and_sort(data, codes)

    def __split_X_y(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """ Split DataFrame into features (X) and labels (y).

        Args:
            df (pd.DataFrame): DataFrame to split.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: Tuple of (features, labels).
        """
        X = df[self.__features].reset_index(drop=True)
        y = df["Label"].reset_index(drop=True)
        return X, y

    def __handle_unique_code(self, df: pd.DataFrame) -> pd.Series:
        """ Handle the inclusion of unique codes in the output.

        Args:
            df (pd.DataFrame): DataFrame to extract unique codes from.

        Returns:
            pd.Series: Series of unique codes if include_codes is True, else None.
        """
        if self.__include_codes:
            return df["Unique_Code"].astype(str).reset_index(drop=True)
        else:
            return None

    def get_data(
        self,
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.DataFrame,
        pd.Series,
        pd.Series,
    ]:
        """
        Prepare data splits for training, validation, and testing.

        Args:
            None

        Returns:
            Tuple containing:
            - X_train (pd.DataFrame): Training features.
            - y_train (pd.Series): Training labels.
            - X_val (pd.DataFrame): Validation features (None if empty).
            - y_val (pd.Series): Validation labels (None if empty).
            - X_test (pd.DataFrame): Test features.
            - y_test (pd.Series): Test labels.
            - unique_codes_train (pd.Series): Training unique codes (None if not included).
            - unique_codes_val (Optional[pd.Series]): Validation unique codes (None if not included or empty).
            - unique_codes_test (Optional[pd.Series]): Test unique codes (None if not included).
        """
        codes = self.__get_codes(self.__train_mask)
        test_codes = self.__get_codes(self.__test_mask)

        train_codes, val_codes = self.__split_into_val_codes(codes)

        val_mask = self.__get_train_val_mask(val_codes)
        train_mask_final = self.__get_train_val_mask(train_codes)

        df_val = self.__get_dataframe(val_mask, val_codes)
        df_train = self.__get_dataframe(train_mask_final, train_codes)
        df_test = self.__get_dataframe(self.__test_mask, test_codes)

        X_train, y_train = self.__split_X_y(df_train)
        X_val, y_val = self.__split_X_y(df_val) if not df_val.empty else (None, None)
        X_test, y_test = self.__split_X_y(df_test)

        unique_codes_train = self.__handle_unique_code(df_train)
        unique_codes_val = self.__handle_unique_code(df_val) if not df_val.empty else None
        unique_codes_test = self.__handle_unique_code(df_test)

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            unique_codes_train,
            unique_codes_val,
            unique_codes_test
        )

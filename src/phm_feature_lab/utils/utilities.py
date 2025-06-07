import pandas as pd
import numpy as np
from typing import List, Type, Any, Optional

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
    
    @staticmethod
    def ensure_numpy_array(data):
        """
        Ensure the input data is converted to a numpy array.
        """
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (pd.DataFrame, pd.Series)):
            return data.values
        elif isinstance(data, list):
            return np.array(data)
        else:
            raise ValueError("Unsupported data type. Cannot convert to numpy array.")
        
    @staticmethod
    def ensure_list(
        value: Any,
        expected_type: Type,
        allow_empty: bool = False,
        error_message: str = None,
    ) -> List[Any]:
        """Validate that a value is a list and that all its elements are of the expected type.

        Args:
            value (Any): The value to validate.
            expected_type (Type): The expected type of the list elements.
            allow_empty (bool): Whether an empty list is allowed. Defaults to False.
            error_message (str): Custom error message to raise if validation fails.

        Returns:
            List[Any]: The validated list.

        Raises:
            TypeError: If `value` is not a list or contains elements of the wrong type.
            ValueError: If `value` is an empty list and `allow_empty` is False.
        """
        # Verifica se o valor é uma lista
        if not isinstance(value, list):
            raise TypeError(error_message or f"Expected a list, but got {type(value).__name__}.")

        # Verifica se a lista está vazia
        if not allow_empty and not value:
            raise ValueError(error_message or "List cannot be empty.")

        # Verifica se todos os elementos são do tipo esperado
        if not all(isinstance(item, expected_type) for item in value):
            raise TypeError(
                error_message
                or f"All elements in the list must be of type {expected_type.__name__}."
            )

        return value
    
    @staticmethod
    def check_image_validity(image: np.ndarray) -> None:
        """
        Validates the format and properties of the image.

        Parameters:
        - image: Input image.

        Raises:
        - TypeError: If the image is not a NumPy array.
        - ValueError: If the image is not grayscale, square, or contains invalid values.
        """
        # Check if it is a NumPy array
        Utilities.ensure_numpy_array(image)

        # Check if it is a 2D matrix (grayscale)
        if len(image.shape) != 2:
            logger.error(f"Invalid format: Expected a 2D grayscale image, but received {len(image.shape)} dimensions.")
            raise ValueError("The image must be a 2D grayscale array.")

        # Check if the image is square
        if image.shape[0] != image.shape[1]:
            logger.error(f"Invalid dimensions: Expected a square image, but received {image.shape}.")
            raise ValueError("The image must be square (width equals height).")

        # Check if the data type is numeric
        if not np.issubdtype(image.dtype, np.number):
            logger.error(f"Invalid data type: {image.dtype} is not numeric.")
            raise ValueError("The image must be of a numeric type (e.g., int or float).")

        # Check for NaN or infinite values
        if np.any(np.isnan(image)) or np.any(np.isinf(image)):
            logger.error("Invalid values detected: The image contains NaN or infinite values.")
            raise ValueError("The image must not contain NaN or infinite values.")
        
    @staticmethod
    def get_anomaly_ratio_by_unique_code(y: np.ndarray, unique_codes: np.ndarray) -> pd.DataFrame:
        """
        Calculates the overall percentage of y==1 (anomalies) vs. y==0 (normal)
        for each unique Unique_Code.

        Args:
            y (np.ndarray): Labels (0 for normal, 1 for anomaly) - must be in the same order as unique_codes.
            unique_codes (np.ndarray): Array of unique Unique_Code values.
        Returns:
            pd.DataFrame: DataFrame with Unique_Code as index and columns for % Normal and % Anomaly.
        """
        # Create a Pandas DataFrame for easier manipulation
        df = pd.DataFrame({'Unique_Code': unique_codes, 'Label': y})

        # Group by 'Unique_Code' and calculate the value counts of 'Label'
        grouped = df.groupby('Unique_Code')['Label'].value_counts(normalize=True).unstack(fill_value=0)

        # Calculate percentages and store in a dictionary
        results = {}
        for code in df['Unique_Code'].unique():
            if code in grouped.index:
                normal_percentage = grouped.loc[code, 0] * 100 if 0 in grouped.columns else 0
                anomaly_percentage = grouped.loc[code, 1] * 100 if 1 in grouped.columns else 0
                results[code] = {'% Normal': normal_percentage, '% Anomaly': anomaly_percentage}
            else:
                results[code] = {'% Normal': 0, '% Anomaly': 0} # Handle cases where Unique_Code is in y but not in X

        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df.index.name = 'Unique_Code'

        return results_df

    @staticmethod
    def evaluate_anomaly_prediction(y: np.ndarray, unique_codes: np.ndarray, true_labels: np.ndarray, anomaly_threshold: Optional[float] = None) -> None:
        """
        Calculates anomaly predictions based on anomaly ratios, compares them to true_labels, and prints the results.

        Args:
            y (np.ndarray): Predicted labels (0 for normal, 1 for anomaly) - must be in the same order as unique_codes.
            unique_codes (np.ndarray): Array of unique Unique_Code values.
            true_labels (np.ndarray): Array of true_labels (strings: 'Normal' or 'Anomaly') - must be in the same order as unique_codes.
            anomaly_threshold (Optional[float]): The threshold percentage to consider a Unique_Code as an anomaly.
                                                If None, the prediction is not performed, and only the anomaly percentage and true_label are printed.
        """

        # Calculate anomaly ratios
        anomaly_results_df = Utilities.get_anomaly_ratio_by_unique_code(y, unique_codes)

        # Create a DataFrame for true_labels
        true_labels_df = pd.DataFrame({'Unique_Code': unique_codes, 'true_label': true_labels})
        true_labels_df = true_labels_df.drop_duplicates(subset=['Unique_Code']).set_index('Unique_Code') # Ensure each Unique_Code has only one true_label

        # Add the true_labels to the anomaly results DataFrame
        anomaly_results_df['true_label'] = anomaly_results_df.index.map(true_labels_df['true_label']).astype(int)
        anomaly_results_df['true_label'].replace([1, 0], ['Anomaly', 'Normal'], inplace=True)
        anomaly_results_df.sort_values(by='% Anomaly', ascending=False, inplace=True)  
        # Add a column indicating if the model predicts it to be an anomaly or not
        if anomaly_threshold is not None:  # Conditionally add the 'Predicted Anomaly' column
            anomaly_results_df['Predicted Anomaly'] = anomaly_results_df['% Anomaly'] > anomaly_threshold

        # Print the results
        for index, row in anomaly_results_df.iterrows():
            if anomaly_threshold is not None:
                predicted_anomaly = "Anomaly" if row['Predicted Anomaly'] else "Normal"
                print(f"Unique_Code: {index}, % Anomaly: {row['% Anomaly']:.2f}%, Predicted: {predicted_anomaly}, true_label: {row['true_label']}")
            else:
                print(f"Unique_Code: {index}, % Anomaly: {row['% Anomaly']:.2f}%, true_label: {row['true_label']}")
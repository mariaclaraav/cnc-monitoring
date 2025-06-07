import pandas as pd
from phm_feature_lab.utils.logger import Logger

logger = Logger().get_logger()

class LoadFiles:
    """
    Handles loading data from various file formats (Parquet, CSV, Excel), following the Single Responsibility principle.

    Attributes:
        data_path (str): Path to the data file.
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the DataLoader with a data path.

        Args:
            data_path (str): Path to the data file (e.g., Parquet, CSV, or Excel file).
        """
        self.data_path = data_path

    def load_parquet(self) -> pd.DataFrame:
        """
        Loads data from a Parquet file.

        Returns:
            pd.DataFrame: Loaded DataFrame.

        Raises:
            FileNotFoundError: If the Parquet file is not found.
            Exception: If there is an error reading the Parquet file.
        """
        try:
            logger.info(f"Loading data from Parquet file: {self.data_path}...")
            df = pd.read_parquet(self.data_path)
            logger.info(f"Parquet data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"Parquet file not found: {self.data_path}")
            raise e
        except Exception as e:
            logger.error(f"Error loading Parquet file: {str(e)}")
            raise

    def load_csv(self) -> pd.DataFrame:
        """
        Loads data from a CSV file.

        Returns:
            pd.DataFrame: Loaded DataFrame.

        Raises:
            FileNotFoundError: If the CSV file is not found.
            Exception: If there is an error reading the CSV file.
        """
        try:
            logger.info(f"Loading data from CSV file: {self.data_path}...")
            df = pd.read_csv(self.data_path)
            logger.info(f"CSV data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"CSV file not found: {self.data_path}")
            raise e
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise

    def load_excel(self) -> pd.DataFrame:
        """
        Loads data from an Excel file.

        Returns:
            pd.DataFrame: Loaded DataFrame.

        Raises:
            FileNotFoundError: If the Excel file is not found.
            Exception: If there is an error reading the Excel file.
        """
        try:
            logger.info(f"Loading data from Excel file: {self.data_path}...")
            df = pd.read_excel(self.data_path)
            logger.info(f"Excel data loaded successfully. Shape: {df.shape}")
            return df
        except FileNotFoundError as e:
            logger.error(f"Excel file not found: {self.data_path}")
            raise e
        except Exception as e:
            logger.error(f"Error loading Excel file: {str(e)}")
            raise

    def load(self, format: str = 'parquet') -> pd.DataFrame:
        """
        Loads data based on the specified format (Parquet, CSV, or Excel).

        Args:
            format (str): File format ('parquet', 'csv', or 'excel'). Default is 'parquet'.

        Returns:
            pd.DataFrame: Loaded DataFrame.

        Raises:
            ValueError: If an unsupported format is specified.
            Exception: If there is an error loading the data.
        """
        formats = {'parquet': self.load_parquet, 'csv': self.load_csv, 'excel': self.load_excel}
        if format.lower() not in formats:
            raise ValueError(f"Unsupported format: {format}. Supported formats are {list(formats.keys())}")
        try:
            return formats[format.lower()]()
        except Exception as e:
            logger.error(f"Error loading {format} file: {str(e)}")
            raise
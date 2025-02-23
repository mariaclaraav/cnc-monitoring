import re
import numpy as np
import pandas as pd
import logger
from typing import List, Tuple, Union, Optional
import logger
from phm_feature_lab.utils.logger import Logger 

logger = Logger().get_logger()

def create_mask(df: pd.DataFrame, periods: List[str], machine_types: Optional[List[str]] = None, normal: bool = True) -> pd.Series:
    """
    Create a boolean mask for the DataFrame based on specified periods, machine types, and a normal flag.    
    """
    
    # Create mask for specified periods
    period_mask = df['Period'].isin(periods)
    
    # Create mask for specified machine types (if provided)
    machine_mask = df['Machine'].isin(machine_types) if machine_types else pd.Series(True, index=df.index)
    
    # Create mask based on 'Label' column if 'normal' is True
    label_mask = df['Label'] == 0 if normal else pd.Series(True, index=df.index)
    
    # Combine all masks
    return period_mask & machine_mask & label_mask



class SplitData:
    """
    Split the data sequentially into training, validation, and testing.

    This class handles the preparation of data for machine learning models by creating
    training, validation, and test sets based on masks. It also sorts the data by 
    the 'Unique_Code' and 'Time' columns and can include 'Unique_Code' as part of the
    training, validation, and test feature sets if specified.
    
    Parameters:
    -----------
    - df : pd.DataFrame
        Input DataFrame containing features, labels, and a 'Unique_Code' column.
    
    - train_mask : pd.Series
        Boolean mask indicating the training data rows.
    
    - test_mask : pd.Series
        Boolean mask indicating the test data rows.
    
    - n_val : float, optional (default=0.3)
        Fraction of training data to be used for validation.
    
    - features : List[str], optional (default=['X_axis', 'Y_axis', 'Z_axis'])
        List of feature columns to include in the train, test, and validation sets.
    
    - print_codes : bool, optional (default=False)
        Whether to print the unique codes for the train, test, and validation sets.
    
    - include_codes : bool, optional (default=False)
        Whether to include the 'Unique_Code' column in the returned feature sets.
    """
    month_order = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    def __init__(self, df: pd.DataFrame, train_mask: pd.Series, test_mask: pd.Series, n_val: float = 0.3, features: List[str] = ['X_axis', 'Y_axis', 'Z_axis'], print_codes: bool = False, include_codes: bool = False):
        self.df = df
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.n_val = n_val
        self.features = features
        self.print_codes_ = print_codes
        self.include_codes = include_codes
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test, self.unique_codes_train, self.unique_codes_val, self.unique_codes_test = self.prepare_autoencoder_data()

    
    @staticmethod
    def extract_keys(code: str) -> Tuple[int, int, int, int]:
        """
        Extracts year, month, machine, and sequence number from a 'Unique_Code' string.
        
        Parameters:
        -----------
        - code : str
            The 'Unique_Code' to extract keys from.
        
        Returns:
        --------
        - tuple : (year, month, machine, num)
        """
        match = re.match(r'M(\d+)_OP\d+_(\w{3})_(\d{4})_(\d+)', code)
        if match is None:
            raise ValueError(f"Invalid Unique_Code format: {code}")
        machine = int(match.group(1))
        month = match.group(2)
        year = int(match.group(3))
        num = int(match.group(4))
        return (year, SplitData.month_order[month], machine, num)
    
    @staticmethod
    def split_mask(codes: np.ndarray, values: List[str]) -> np.ndarray:
        """
        Create a mask for rows where 'Unique_Code' matches the given list of values.
        
        Parameters:
        -----------
        - codes : np.ndarray
            Array of 'Unique_Code' values from the DataFrame.
        
        - values : List[str]
            List of 'Unique_Code' values to match.
        
        Returns:
        --------
        - np.ndarray : Boolean mask indicating matching rows.
        """
        mask = np.zeros(len(codes), dtype=bool)
        for value in values:
            mask |= (codes == value)
        return mask

    
    @staticmethod
    def add_sequence_and_sort(df: pd.DataFrame, code_order: List[str]) -> pd.DataFrame:
        """
        Sort the DataFrame by 'Unique_Code' and 'Time' columns based on a given order of 'Unique_Code'.
        
        Parameters:
        -----------
        - df : pd.DataFrame
            The DataFrame to be sorted.
        
        - code_order : List[str]
            Ordered list of 'Unique_Code' to follow.
        
        Returns:
        --------
        - pd.DataFrame : The sorted DataFrame.
        """
        def extract_last_number(code: str) -> Union[int, None]:
            match = re.search(r'_(\d+)$', code)
            return int(match.group(1)) if match else None

        df = df.copy()
        df['Unique_Code'] = pd.Categorical(df['Unique_Code'], categories=code_order, ordered=True)
        #df['Seq'] = df['Unique_Code'].apply(extract_last_number)
        df_sorted = df.sort_values(by=['Unique_Code', 'Time'])
        #df_sorted = df_sorted.drop(columns='Seq')
        return df_sorted

    def prepare_autoencoder_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare the data splits for training, validation, and testing.
        
        Returns:
        --------
        - Tuple containing the train, validation, and test feature sets (X_train, X_val, X_test), 
        labels (y_train, y_val, y_test), and unique codes (unique_codes).
        """
        codes = self.df[self.train_mask].Unique_Code.unique()
        codes = sorted(codes, key=SplitData.extract_keys)
        
        test_codes = self.df[self.test_mask].Unique_Code.unique()
        test_codes = sorted(test_codes, key=SplitData.extract_keys)
                
        n_val = int(self.n_val * len(codes))
        val_codes = codes[-n_val:] if n_val > 0 else []
        train_codes = codes[:-n_val] if n_val > 0 else codes
        
        val_mask = SplitData.split_mask(self.df['Unique_Code'].values, val_codes)
        train_mask_final = SplitData.split_mask(self.df['Unique_Code'].values, train_codes)

        df_val = self.df[val_mask]
        df_val = SplitData.add_sequence_and_sort(df_val, val_codes)

        df_train = self.df[train_mask_final]
        df_train = SplitData.add_sequence_and_sort(df_train, train_codes)

        df_test = self.df[self.test_mask]
        df_test = SplitData.add_sequence_and_sort(df_test, test_codes)
        
        if self.print_codes_:
            logger.info(f'Train codes: {df_train.Unique_Code.astype(str).unique()}\n')
            logger.info(f'Val codes: {df_val.Unique_Code.astype(str).unique()}\n')
            logger.info(f'Test codes: {df_test.Unique_Code.astype(str).unique()}\n')

        # # Conditionally include 'Unique_Code' in X_train, X_val, and X_test
        # if self.include_codes:
        #     X_train = df_train[self.features + ['Unique_Code']]
        #     X_val = df_val[self.features + ['Unique_Code']]
        #     X_test = df_test[self.features + ['Unique_Code']]
        # else:
        # Extract features and labels, reset index to ensure alignment
        
        X_train = df_train[self.features].reset_index(drop=True)
        y_train = df_train['Label'].reset_index(drop=True)
        
        X_val = df_val[self.features].reset_index(drop=True) if not df_val.empty else None
        y_val = df_val['Label'].reset_index(drop=True) if not df_val.empty else None
        
        X_test = df_test[self.features].reset_index(drop=True)
        y_test = df_test['Label'].reset_index(drop=True)
        
        if self.include_codes:
            unique_codes_train = df_train['Unique_Code'].astype(str).reset_index(drop=True)
            unique_codes_val = df_val['Unique_Code'].astype(str).reset_index(drop=True) if not df_val.empty else None
            unique_codes_test = df_test['Unique_Code'].astype(str).reset_index(drop=True)
        else:
            unique_codes_train = None
            unique_codes_val = None
            unique_codes_test = None

                
        # logger.info(f'OBS: Returning validation unique codes...\n')

        return X_train, y_train, X_val, y_val, X_test, y_test, unique_codes_train, unique_codes_val, unique_codes_test

  

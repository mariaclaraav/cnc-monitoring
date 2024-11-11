import dask.dataframe as dd
from dask import compute
import pandas as pd
from typing import List, Optional, Tuple
import re
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Union, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_mask(df: dd.DataFrame, periods: List[str], machine_types: Optional[List[str]] = None, normal: bool = True) -> dd.Series:
    """
    Create a boolean mask for the DataFrame based on specified periods, machine types, and a normal flag.    
    """
    
    # Create mask for specified periods
    period_mask = df['Period'].isin(periods)
    
    # Create mask for specified machine types (if provided)
    machine_mask = df['Machine'].isin(machine_types) if machine_types else dd.from_pandas(pd.Series(True, index=df.index), npartitions=df.npartitions)
    
    # Create mask based on 'Label' column if 'normal' is True
    label_mask = df['Label'] == 0 if normal else dd.from_pandas(pd.Series(True, index=df.index), npartitions=df.npartitions)
    
    # Combine all masks
    return period_mask & machine_mask & label_mask



class SplitData:
    """
    Split the data sequentially for autoencoder training, validation, and testing.
    """
    month_order = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    def __init__(self, df: dd.DataFrame, train_mask: dd.Series, test_mask: dd.Series, n_val: float = 0.3, features: List[str] = ['X_axis', 'Y_axis', 'Z_axis'], print_codes: bool = False):
        self.df = df
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.n_val = n_val
        self.features = features
        self.print_codes_ = print_codes
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.prepare_autoencoder_data()

    @staticmethod
    def extract_keys(code: str) -> Tuple[int, int, int, int]:
        match = re.match(r'M(\d+)_OP\d+_(\w{3})_(\d{4})_(\d+)', code)
        if match is None:
            raise ValueError(f"Invalid Unique_Code format: {code}")
        machine = int(match.group(1))
        month = match.group(2)
        year = int(match.group(3))
        num = int(match.group(4))
        return (year, SplitData.month_order[month], machine, num)
    
    @staticmethod
    def split_mask(codes: dd.Series, values: List[str]) -> dd.Series:
        mask = dd.from_pandas(pd.Series(False, index=codes.index), npartitions=codes.npartitions)
        for value in values:
            mask |= (codes == value)
        return mask

    @staticmethod
    def add_sequence_and_sort(df: dd.DataFrame, code_order: List[str]) -> dd.DataFrame:
        df = df.copy()
        df['Unique_Code'] = dd.Categorical(df['Unique_Code'], categories=code_order, ordered=True)
        df_sorted = df.sort_values(by=['Unique_Code', 'Time'])
        return df_sorted

    def prepare_autoencoder_data(self) -> Tuple[dd.DataFrame, dd.Series, dd.DataFrame, dd.Series, dd.DataFrame, dd.Series]:
        codes = self.df[self.train_mask].Unique_Code.unique().compute()
        codes = sorted(codes, key=SplitData.extract_keys)
        
        test_codes = self.df[self.test_mask].Unique_Code.unique().compute()
        test_codes = sorted(test_codes, key=SplitData.extract_keys)
                
        n_val = int(self.n_val * len(codes))
        val_codes = codes[-n_val:] if n_val > 0 else []
        train_codes = codes[:-n_val] if n_val > 0 else codes
        
        val_mask = SplitData.split_mask(self.df['Unique_Code'], val_codes)
        train_mask_final = SplitData.split_mask(self.df['Unique_Code'], train_codes)

        df_val = self.df[val_mask]
        df_val = SplitData.add_sequence_and_sort(df_val, val_codes)

        df_train = self.df[train_mask_final]
        df_train = SplitData.add_sequence_and_sort(df_train, train_codes)

        df_test = self.df[self.test_mask]
        df_test = SplitData.add_sequence_and_sort(df_test, test_codes)
        
        if self.print_codes_:
            logging.info(f'Train codes: {df_train.Unique_Code.astype(str).unique().compute()}\n')
            logging.info(f'Val codes: {df_val.Unique_Code.astype(str).unique().compute()}\n')
            logging.info(f'Test codes: {df_test.Unique_Code.astype(str).unique().compute()}\n')

        X_train, y_train = df_train[self.features], df_train['Label']
        X_val, y_val = df_val[self.features], df_val['Label']
        X_test, y_test = df_test[self.features], df_test['Label']

        return X_train, y_train, X_val, y_val, X_test, y_test

    
    
import dask.dataframe as dd
import dask.array as da
from dask_ml.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import re
from typing import List, Optional, Tuple
import logging

def create_mask(df: dd.DataFrame, periods: List[str], machine_types: Optional[List[str]] = None, normal: bool = True) -> dd.Series:
    """
    Create a boolean mask for the DataFrame based on specified periods, machine types, and a normal flag.
    """
    # Create mask for specified periods using Dask
    period_mask = df['Period'].isin(periods)
    
    # Create mask for specified machine types (if provided), using Dask's operations
    if machine_types:
        machine_mask = df['Machine'].isin(machine_types)
    else:
        machine_mask = df['Machine'].map(lambda x: True, meta=('Machine', 'bool'))
    
    # Create mask based on 'Label' column if 'normal' is True
    if normal:
        label_mask = df['Label'] == 0
    else:
        label_mask = df['Label'].map(lambda x: True, meta=('Label', 'bool'))
    
    # Combine all masks using Dask's bitwise operations
    combined_mask = period_mask & machine_mask & label_mask
    
    return combined_mask

class SplitData:
    """
    Split the data sequentially for autoencoder training, validation, and testing.
    """
    month_order = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

    def __init__(self, df: dd.DataFrame, train_mask: dd.Series, test_mask: dd.Series, n_val: float = 0.3, features: List[str] = ['X_axis', 'Y_axis', 'Z_axis'], print_codes: bool = False):
        self.df = df
        self.train_mask = train_mask
        self.test_mask = test_mask
        self.n_val = n_val
        self.features = features
        self.print_codes_ = print_codes
        self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test = self.prepare_autoencoder_data()

    @staticmethod
    def extract_keys(code: str) -> Tuple[int, int, int, int]:
        match = re.match(r'M(\d+)_OP\d+_(\w{3})_(\d{4})_(\d+)', code)
        if match is None:
            raise ValueError(f"Invalid Unique_Code format: {code}")
        machine = int(match.group(1))
        month = match.group(2)
        year = int(match.group(3))
        num = int(match.group(4))
        return (year, SplitData.month_order[month], machine, num)
    
    @staticmethod
    def split_mask(codes: dd.Series, values: List[str]) -> dd.Series:
        mask = codes.isin(values)
        return mask

    @staticmethod
    def add_sequence_and_sort(df: dd.DataFrame, code_order: List[str]) -> dd.DataFrame:
        def categorize_and_sort(partition, code_order):
            # Verifica e achata listas aninhadas em code_order
            if any(isinstance(i, list) for i in code_order):
                code_order = [item for sublist in code_order for item in (sublist if isinstance(sublist, list) else [sublist])]
            
            partition['Unique_Code'] = pd.Categorical(partition['Unique_Code'], categories=code_order, ordered=True)
            return partition.sort_values(by=['Unique_Code', 'Time'])

        # Passa code_order para a função categorize_and_sort dentro de map_partitions
        df_sorted = df.map_partitions(categorize_and_sort, code_order=code_order, meta=df)
        return df_sorted

    def prepare_autoencoder_data(self) -> Tuple[dd.DataFrame, dd.Series, dd.DataFrame, dd.Series, dd.DataFrame, dd.Series]:
        # Use Dask's operations to avoid computation until necessary
        codes = self.df[self.train_mask].Unique_Code.drop_duplicates()
        codes = codes.map_partitions(lambda part: sorted(part, key=SplitData.extract_keys), meta=('Unique_Code', 'str'))

        test_codes = self.df[self.test_mask].Unique_Code.drop_duplicates()
        test_codes = test_codes.map_partitions(lambda part: sorted(part, key=SplitData.extract_keys), meta=('Unique_Code', 'str'))

        n_val = int(self.n_val * len(codes.compute()))
        val_codes = codes[-n_val:] if n_val > 0 else []
        train_codes = codes[:-n_val] if n_val > 0 else codes
        print(f"Train codes: {train_codes}")
        
        train_codes = list(train_codes.compute())
        train_mask_final = SplitData.split_mask(self.df['Unique_Code'], train_codes)
        df_train = self.df[train_mask_final]
        print(f"Sample of df_train before sorting: {df_train.head().compute()}")
        df_train = SplitData.add_sequence_and_sort(df_train, train_codes)
        print(f"Sample of df_train after sorting: {df_train.head().compute()}")
        if n_val > 0:                
            val_codes = list(val_codes.compute())
            val_mask = SplitData.split_mask(self.df['Unique_Code'], val_codes)
            df_val = self.df[val_mask]
            df_val = SplitData.add_sequence_and_sort(df_val, val_codes)
            X_val, y_val = df_val[self.features], df_val['Label']
        else:
            X_val, y_val = [], []

        
        df_test = self.df[self.test_mask]
        df_test = SplitData.add_sequence_and_sort(df_test, test_codes)

        if self.print_codes_:
            logging.info(f'Train codes: {df_train.Unique_Code.drop_duplicates().compute()}\n')
            logging.info(f'Test codes: {df_test.Unique_Code.drop_duplicates().compute()}\n')
            if n_val > 0:
                logging.info(f'Val codes: {df_val.Unique_Code.drop_duplicates().compute()}\n')

        X_train, y_train = df_train[self.features], df_train['Label']
        X_test, y_test = df_test[self.features], df_test['Label']

        return X_train, y_train, X_val, y_val, X_test, y_test

def setup_data(df: dd.DataFrame, train_period: list, test_period: list, n_val: float, features: list, scaler_type: str = 'StandardScaler', print_codes: bool = False) -> tuple:
    
    # Create train and test masks
    train_mask = create_mask(df, train_period, normal=False)
    test_mask = create_mask(df, test_period, normal=False)
    
    # Initialize SplitData object with print_codes parameter
    split_data = SplitData(df, train_mask, test_mask, n_val, features, print_codes=print_codes)

    # Access the prepared data
    X_train, y_train = split_data.X_train, split_data.y_train
    X_test, y_test = split_data.X_test, split_data.y_test
    X_val, y_val = (split_data.X_val, split_data.y_val) if n_val > 0 else (None, None)
    
    # Select and apply the scaler
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler_type is None:
        scaler = None
    else:
        raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    if scaler is not None:
        # Fit and transform the training data, transform the test and validation data
        X_train = X_train.map_partitions(lambda df: scaler.fit_transform(df), meta=X_train)
        X_test = X_test.map_partitions(lambda df: scaler.transform(df), meta=X_test)
        if X_val is not None:
            X_val = X_val.map_partitions(lambda df: scaler.transform(df), meta=X_val)
    # else:
    #     # Convert to Dask Arrays more efficiently by leveraging the structure of the DataFrame
    #     if isinstance(X_train, dd.DataFrame):
    #         X_train = X_train.to_dask_array(lengths=True)
    #     if isinstance(X_test, dd.DataFrame):
    #         X_test = X_test.to_dask_array(lengths=True)
    #     if X_val is not None and isinstance(X_val, dd.DataFrame):
    #         X_val = X_val.to_dask_array(lengths=True)
            
    # Log the sizes of the datasets (deferred computation)
    logging.info(f'Training set size: {X_train.shape}')
    if X_val is not None:
        logging.info(f'Validation set size: {X_val.shape}')
    logging.info(f'Test set size: {X_test.shape}')
        
    return X_train, y_train, X_val, y_val, X_test, y_test, scaler

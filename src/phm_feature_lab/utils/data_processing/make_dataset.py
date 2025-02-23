import os
import time
import logging
from typing import List, Tuple, Dict, Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
from . import loader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Outputs to the console only
    ]
)

class DataTransform:
    """
    A class to load and transform raw dataset.
    """
    
    def __init__(self, path_to_dataset: str, machines: List[str], process_names: List[str], labels: List[str], freq: int = 2000) -> None:
        self.path_to_dataset = path_to_dataset
        self.machines = machines
        self.process_names = process_names
        self.labels = labels
        self.freq = freq

    def load_data(self) -> Tuple[List[pd.DataFrame], int]:
        """
        Loads data from the specified dataset path.
        
        Returns:
        - A tuple containing a list of DataFrames and the count of files processed.
        """
        dfs: List[pd.DataFrame] = []
        file_count: int = 0

        for process_name in tqdm(self.process_names, desc="Loading files"):
            for machine in self.machines:
                for label in self.labels:
                    data_path = os.path.join(self.path_to_dataset, machine, process_name, label)
                    data_list, label_list = loader.load_tool_research_data(data_path, label=label, add_additional_label=True, verbose=False)

                    for data, full_label in zip(data_list, label_list):
                        file_count += 1

                        parts = full_label.split('_')
                        month = parts[1]
                        year = parts[2]

                        unique_code = f"{machine}_{process_name}_{month}_{year}_{file_count}"

                        time_values = np.linspace(0, len(data) / self.freq, len(data), endpoint=False)

                        df = pd.DataFrame(data, columns=['X_axis', 'Y_axis', 'Z_axis'])
                        df['Time'] = time_values
                        df['Machine'] = machine
                        df['Process'] = process_name
                        df['Label'] = label
                        df['Month'] = month
                        df['Year'] = year
                        df['Unique_Code'] = unique_code

                        dfs.append(df)

        return dfs, file_count

    def transform_data(self, dfs: List[pd.DataFrame]) -> pd.DataFrame:
        """
        Transforms the list of DataFrames into a single DataFrame.
        
        Returns:
        - A transformed DataFrame.
        """
        final_df = pd.concat(dfs, axis=0, ignore_index=True)

        # Dictionary to map month abbreviations to two-digit numbers
        month_map: Dict[str, str] = {
            'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
            'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
            'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
        }

        # Convert 'Year' to string
        final_df['Year'] = final_df['Year'].astype(str)
        
        # Create 'Period' column in 'MM-YYYY' format
        final_df['Period'] = final_df['Month'].map(month_map) + '-' + final_df['Year']
        
        # Convert 'Year' back to integer
        final_df['Year'] = final_df['Year'].astype(int)
        
        # Sort the DataFrame by 'Period' (as datetime), 'Unique_Code', and 'Time'
        final_df['Period'] = pd.to_datetime(final_df['Period'], format='%m-%Y')
        final_df = final_df.sort_values(by=['Period', 'Unique_Code', 'Time'])
        final_df.reset_index(drop=True, inplace=True)
        final_df['Label'].replace({'good': 0, 'bad': 1}, inplace=True)

        final_df = final_df[['Time', 'Month', 'Year', 'Machine', 'Process', 'X_axis', 'Y_axis', 'Z_axis', 'Label', 'Unique_Code', 'Period']]

        return final_df

    def save_to_parquet(self, df: pd.DataFrame, path_to_save_parquet: str) -> str:
        """
        Saves the transformed DataFrame to a Parquet file.
        
        Returns:
        - The path where the file was saved.
        """
        file_path = os.path.join(path_to_save_parquet, 'ETL.parquet')
        df.to_parquet(file_path, index=False)
        return file_path

    def run(self, path_to_save_parquet: str) -> str:
        """
        Runs the entire data loading, transforming, and saving process.
        
        Returns:
        - The path to the saved parquet file.
        """
        dfs, file_count = self.load_data()
        transformed_df = self.transform_data(dfs)
        file_path = self.save_to_parquet(transformed_df, path_to_save_parquet)
        logging.info(f"Number of files read: {file_count}")
        logging.info(f"Number of unique codes: {transformed_df['Unique_Code'].nunique()}")
        return file_path


class UniqueCodeCorrector:
    """
    A class to correct the Unique_Code for each machine.
    """
    
    def __init__(self) -> None:
        pass

    def correct_unique_code(self, df: pd.DataFrame, machine: str) -> pd.DataFrame:
        """
        Corrects the Unique_Code for the specified machine.
        
        Returns:
        - The DataFrame with corrected Unique_Code.
        """
        df = df[df['Machine'] == machine].copy()
        df['Year'] = df['Year'].astype(str)
        df['Code'] = df['Machine'] + '_' + df['Process'] + '_' + df['Month'] + '_' + df['Year']
        df['count'] = 0
        
        def assign_counters(group: pd.DataFrame) -> pd.DataFrame:
            unique_code_counter: Dict[str, int] = {}
            counters: List[int] = []
            for unique_code in group['Unique_Code']:
                if unique_code not in unique_code_counter:
                    unique_code_counter[unique_code] = len(unique_code_counter) + 1
                counters.append(unique_code_counter[unique_code])
            group['count'] = counters
            return group

        df = df.groupby('Code').apply(assign_counters)
        df['Unique_Code'] = df['Code'] + '_' + df['count'].astype(str)
        df.reset_index(inplace=True, drop=True)
        df.drop(columns=['Code', 'count'], inplace=True)

        return df

    def save_to_parquet(self, df: pd.DataFrame, path_to_save_parquet: str) -> None:
        """
        Saves the corrected DataFrame to a Parquet file.
        """
        df.to_parquet(path_to_save_parquet)

    def run(self, path_to_dataset: str, machine: str, path_to_save_parquet: str) -> None:
        """
        Runs the unique code correction process for the specified machine.
        """
        df = pd.read_parquet(path_to_dataset)
        corrected_df = self.correct_unique_code(df, machine)
        self.save_to_parquet(corrected_df, path_to_save_parquet)
        logging.info(f"Number of unique codes for {machine}: {corrected_df['Unique_Code'].nunique()}")


class DataSplitter:
    """
    A class to split the data into train and test datasets based on the year
    or save everything into a single file.
    """

    def __init__(self, paths: List[str], base_path: str, test_year: Optional[int] = None, split: bool = True) -> None:
        """
        Initialize the DataSplitter.
        
        Parameters:
        - paths: List of file paths to the parquet files.
        - base_path: Base file path to save the datasets.
        - test_year: Year to split the data on. Used only if split is True.
        - split: Boolean indicating whether to split the data or not.
        """
        self.paths = paths
        self.base_path = base_path
        self.test_year = test_year
        self.split = split

    def split_data(self) -> None:
        """
        Splits the data from the given paths into train and test datasets and saves them,
        or saves everything into a single file if splitting is not required.
        """
        df_list: List[pd.DataFrame] = [pd.read_parquet(path) for path in self.paths]
        df = pd.concat(df_list, axis=0)
        
        df.reset_index(drop=True, inplace=True)        
        df.sort_values(by=['Period', 'Unique_Code', 'Time'], inplace=True)
        df.drop(['Period'], axis=1, inplace=True)
        df['Year'] = df['Year'].astype(int)

        if self.split and self.test_year is not None:
            df_test = df[df['Year'] == self.test_year]
            df_train = df[df['Year'] != self.test_year]

            df_train.reset_index(drop=True, inplace=True)
            df_test.reset_index(drop=True, inplace=True)

            # Define paths for train and test datasets
            train_path = f"{self.base_path}_train.parquet"
            test_path = f"{self.base_path}_test.parquet"

            df_train.to_parquet(train_path)
            df_test.to_parquet(test_path)
        else:
            # If splitting is not required, save everything into a single file
            final_path = f"{self.base_path}_final.parquet"
            df.reset_index(drop=True, inplace=True)
            df.to_parquet(final_path)


# import os
# import time
# import pandas as pd
# import numpy as np
# from tqdm import tqdm
# from . import data_loader_utils


# class DataTransform:
#     """
#     A class to load and transform raw dataset.
    
#     """
    
#     def __init__(self, path_to_dataset, machines, process_names, labels, freq=2000):
     
#         self.path_to_dataset = path_to_dataset
#         self.machines = machines
#         self.process_names = process_names
#         self.labels = labels
#         self.freq = freq

#     def load_data(self):
#         """
#         Loads data from the specified dataset path.
               
#         """
#         dfs = []
#         file_count = 0

#         for process_name in tqdm(self.process_names, desc="Loading files"):
#             for machine in self.machines:
#                 for label in self.labels:
#                     data_path = os.path.join(self.path_to_dataset, machine, process_name, label)
#                     data_list, label_list = data_loader_utils.load_tool_research_data(data_path, label=label, add_additional_label=True, verbose=False)

#                     for data, full_label in zip(data_list, label_list):
#                         file_count += 1

#                         parts = full_label.split('_')
#                         month = parts[1]
#                         year = parts[2]

#                         unique_code = f"{machine}_{process_name}_{month}_{year}_{file_count}"

#                         time = np.linspace(0, len(data) / self.freq, len(data), endpoint=False)

#                         df = pd.DataFrame(data, columns=['X_axis', 'Y_axis', 'Z_axis'])
#                         df['Time'] = time
#                         df['Machine'] = machine
#                         df['Process'] = process_name
#                         df['Label'] = label
#                         df['Month'] = month
#                         df['Year'] = year
#                         df['Unique_Code'] = unique_code

#                         dfs.append(df)

#         return dfs, file_count

#     def transform_data(self, dfs):
#         """
#         Transforms the list of DataFrames into a single DataFrame.

#         """
#         final_df = pd.concat(dfs, axis=0, ignore_index=True)

#         # Dictionary to map month abbreviations to two-digit numbers
#         month_map = {
#             'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
#             'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
#             'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
#         }

#         # Convert 'Year' to string
#         final_df['Year'] = final_df['Year'].astype(str)
        
#         # Create 'Period' column in 'MM-YYYY' format
#         final_df['Period'] = final_df['Month'].map(month_map) + '-' + final_df['Year']
        
#         # Convert 'Year' back to integer
#         final_df['Year'] = final_df['Year'].astype(int)
        
#         # Sort the DataFrame by 'Period' (as datetime), 'Unique_Code', and 'Time'
#         final_df['Period'] = pd.to_datetime(final_df['Period'], format='%m-%Y')
#         final_df = final_df.sort_values(by=['Period', 'Unique_Code', 'Time'])
#         final_df.reset_index(drop=True, inplace=True)
#         final_df['Label'].replace({'good': 0, 'bad': 1}, inplace=True)

#         final_df = final_df[['Time', 'Month', 'Year', 'Machine', 'Process', 'X_axis', 'Y_axis', 'Z_axis', 'Label', 'Unique_Code', 'Period']]

#         return final_df

#     def save_to_parquet(self, df, path_to_save_parquet):
#         """
#         Saves the transformed DataFrame to a Parquet file.
      
#         """
#         file_path = os.path.join(path_to_save_parquet, 'ETL.parquet')
#         df.to_parquet(file_path, index=False)
#         return file_path

#     def run(self, path_to_save_parquet):
#         """
#         Runs the entire data loading, transforming, and saving process.
       
#         """
#         dfs, file_count = self.load_data()
#         transformed_df = self.transform_data(dfs)
#         file_path = self.save_to_parquet(transformed_df, path_to_save_parquet)
#         print(f"Number of files read: {file_count}")
#         print(f"Number of unique codes: {transformed_df['Unique_Code'].nunique()}")
#         return file_path


# class UniqueCodeCorrector:
#     """
#     A class to correct the Unique_Code for each machine.
        
#     """
    
#     def __init__(self):
#         pass

#     def correct_unique_code(self, df, machine):
#         """
#         Corrects the Unique_Code for the specified machine.
       
#         """
#         df = df[df['Machine'] == machine].copy()
#         df['Year'] = df['Year'].astype(str)
#         df['Code'] = df['Machine'] + '_' + df['Process'] + '_' + df['Month'] + '_' + df['Year']
#         df['count'] = 0
        
#         def assign_counters(group):
#             unique_code_counter = {}
#             counters = []
#             for unique_code in group['Unique_Code']:
#                 if unique_code not in unique_code_counter:
#                     unique_code_counter[unique_code] = len(unique_code_counter) + 1
#                 counters.append(unique_code_counter[unique_code])
#             group['count'] = counters
#             return group

#         df = df.groupby('Code').apply(assign_counters)
#         df['Unique_Code'] = df['Code'] + '_' + df['count'].astype(str)
#         df.reset_index(inplace=True, drop=True)
#         df.drop(columns=['Code', 'count'], inplace=True)

#         return df

#     def save_to_parquet(self, df, path_to_save_parquet):
#         """
#         Saves the corrected DataFrame to a Parquet file.
       
#         """
#         df.to_parquet(path_to_save_parquet)

#     def run(self, path_to_dataset, machine, path_to_save_parquet):
#         """
#         Runs the unique code correction process for the specified machine.
                
#         """
#         df = pd.read_parquet(path_to_dataset)
#         corrected_df = self.correct_unique_code(df, machine)
#         self.save_to_parquet(corrected_df, path_to_save_parquet)
#         print(f"Number of unique codes for {machine}: {corrected_df['Unique_Code'].nunique()}")


# class DataSplitter:
#     """
#     A class to split the data into train and test datasets based on the year
#     or save everything into a single file.
#     """

#     def __init__(self, paths, base_path, test_year=None, split=True):
#         """
#         Initialize the DataSplitter.
        
#         Parameters:
#         - paths: List of file paths to the parquet files.
#         - base_path: Base file path to save the datasets.
#         - test_year: Year to split the data on. Used only if split is True.
#         - split: Boolean indicating whether to split the data or not.
#         """
#         self.paths = paths
#         self.base_path = base_path
#         self.test_year = test_year
#         self.split = split

#     def split_data(self):
#         """
#         Splits the data from the given paths into train and test datasets and saves them,
#         or saves everything into a single file if splitting is not required.
#         """
#         df_list = [pd.read_parquet(path) for path in self.paths]
#         df = pd.concat(df_list, axis=0)
        
#         df.reset_index(drop=True, inplace=True)        
#         df.sort_values(by=['Period', 'Unique_Code', 'Time'], inplace=True)
#         df.drop(['Period'], axis=1, inplace=True)
#         df['Year'] = df['Year'].astype('int')

#         if self.split and self.test_year is not None:
#             df_test = df[df['Year'] == self.test_year]
#             df_train = df[df['Year'] != self.test_year]

#             df_train.reset_index(drop=True, inplace=True)
#             df_test.reset_index(drop=True, inplace=True)

#             # Define paths for train and test datasets
#             train_path = f"{self.base_path}_train.parquet"
#             test_path = f"{self.base_path}_test.parquet"

#             df_train.to_parquet(train_path)
#             df_test.to_parquet(test_path)
#         else:
#             # If splitting is not required, save everything into a single file
#             final_path = f"{self.base_path}_final.parquet"
#             df.reset_index(drop=True, inplace=True)
#             df.to_parquet(final_path)


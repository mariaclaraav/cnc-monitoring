import pandas as pd
from src.utils.feature_engineering.creation import FeatureCreationUtils
from tqdm import tqdm
from typing import List, Tuple



class TimeSeriesProcessor:
    def __init__(self, window_size, step_size, min_periods, sampling_rate):
        
        self.__window_size = window_size
        self.__step_size = step_size if step_size else None
        self.__min_periods = min_periods
        self.__sampling_rate = sampling_rate

    def process_time_series(
        self,
        df: pd.DataFrame,
        feature_type: str = 'statistical',
        wavelet: str = 'db14',
        level: int = 3,
        num_imfs: int = 5,
        order: int = 4,
        frequency_bands: dict = None
    ) -> pd.DataFrame:
        """Process a time series DataFrame to extract specified features.

        Args:
            df (pd.DataFrame): The input DataFrame containing the time series data.
            feature_type (str, optional): Type of feature extraction to perform.
                Options: 'statistical', 'energy', 'dwt', 'wpd', 'filter', 'jerk', 'emd'.
                Defaults to 'statistical'.
            wavelet (str, optional): The wavelet type to use for wavelet-based
                decompositions. Defaults to 'db14'.
            level (int, optional): The level of decomposition for wavelet-based
                methods. Defaults to 3.
            num_imfs (int, optional): Number of Intrinsic Mode Functions for EMD.
                Defaults to 5.
            period (int, optional): The period parameter, if applicable.
                Defaults to 100.
            order (int, optional): The order for filter-based feature extraction.
                Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame containing the extracted features for each
                unique code in the dataset.
        
        Raises:
            ValueError: If an unsupported feature_type is provided.
        """
        additional_columns = [
            'Time', 'Machine', 'Process', 'Label', 'Unique_Code', 'Period'
        ]

        feature_functions = {
        'statistical': lambda data: FeatureCreationUtils.statistical_features(
            data=data, window_size=self.__window_size,
            step_size=self.__step_size, min_periods=self.__min_periods
        ),
        'energy': lambda data: FeatureCreationUtils.energy_features(
            data=data, window_size=self.__window_size,
            step_size=self.__step_size, min_periods=self.__min_periods
        ),
        'dwt': lambda data: FeatureCreationUtils.discrete_wavelet_decomposition(
            data=data, w=wavelet, level=level
        ),
        'wpd': lambda data: FeatureCreationUtils.wavelet_packet_decomposition(
            data=data, w=wavelet, level=level
        ),
        'filter': lambda data, axis: FeatureCreationUtils.filter_features(
            data=data, fs=self.__sampling_rate, order=order, 
            frequency_bands=axis
        ),
        'jerk': lambda data: FeatureCreationUtils.calculate_jerk(
            data=data, fs=self.__sampling_rate
        ),
        'emd': lambda data: FeatureCreationUtils.calculate_imf(
            data=data, num_imfs=num_imfs
        )
    }

        try:
            feature_function = feature_functions[feature_type]
        except KeyError:
            raise ValueError(
                f"Invalid feature_type '{feature_type}'. Supported types are: "
                f"{', '.join(feature_functions.keys())}"
            )
        
        processed_dfs = []
        for unique_code, group in tqdm(df.groupby('Unique_Code'), desc="Processing Unique Codes"):
            if 'X_axis' not in group or 'Y_axis' not in group or 'Z_axis' not in group:
                raise ValueError("Missing axis columns in the data.")
            
            # Apply the selected feature function to each axis
            if feature_type == 'filter':
                features_X = feature_function(group['X_axis'], frequency_bands['X_axis'])
                features_Y = feature_function(group['Y_axis'], frequency_bands['Y_axis'])
                features_Z = feature_function(group['Z_axis'], frequency_bands['Z_axis'])
            else:
                features_X = feature_function(group['X_axis'])
                features_Y = feature_function(group['Y_axis'])
                features_Z = feature_function(group['Z_axis'])

            
            # Prefix columns to identify axis source
            features_X.columns = [f'X_{col}' for col in features_X.columns]
            features_Y.columns = [f'Y_{col}' for col in features_Y.columns]
            features_Z.columns = [f'Z_{col}' for col in features_Z.columns]
            
            # Gather additional columns and concatenate features for each axis
            additional_data = group[additional_columns]
            
            if feature_type in ['dwt', 'wpd', 'jerk', 'emd', 'filter']:
                # Concat after reset index
                final_df = pd.concat([additional_data.reset_index(drop=True), features_X.reset_index(drop=True), features_Y.reset_index(drop=True), features_Z.reset_index(drop=True)], axis=1)
            else:
                # Concat before reset index
                final_df = pd.concat([additional_data, features_X, features_Y, features_Z], axis=1).reset_index(drop=True)
        
            final_df.dropna(inplace=True)
            processed_dfs.append(final_df)
            
            del final_df
            
        del features_X, features_Y, features_Z, additional_data, group
        
        final_result = pd.concat(processed_dfs)
        
        del processed_dfs
        
        final_result.sort_values(by=['Period', 'Unique_Code', 'Time'], inplace=True)
        final_result.reset_index(drop=True, inplace=True)
        return final_result
    
    

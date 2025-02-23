
from phm_feature_lab.utils.filter.filter import perform_fft
from phm_feature_lab.utils.utilities import order_unique_code

import numpy as np
import pandas as pd



class CustomRMS:
    def __init__(self, fs, vel, band, get_harmonics: bool = False):
        """
        Initialize the RMSCalculator with sampling frequency, fundamental frequency, and bandwidth.

        Parameters:
            fs (int): Sampling frequency.
            vel (float): Fundamental frequency.
            band (float): Bandwidth around each harmonic for RMS calculation.
        """
        self.__fs = fs
        self.__vel = vel
        self.__band = band
        self.__get_harmonics = get_harmonics
        
    def __get_limits(self, n: int):
        upper_lim = self.__vel * n + self.__band
        lower_lim = self.__vel * n - self.__band
        return lower_lim, upper_lim
    
    def __get_mask(self, n: int, freq: np.array):
        lower_lim, upper_lim = self.__get_limits(n)
        mask = (freq >= lower_lim) & (freq <= upper_lim)
        return mask
    
    def __check_df(self, df):
        if df.empty:
            raise ValueError("Input DataFrame is empty.")
                
    def __check_axis(self, df):
        if 'Unique_Code' not in df or 'X_axis' not in df or 'Y_axis' not in df or 'Z_axis' not in df:
            raise KeyError("Input DataFrame must contain 'Unique_Code', 'X_axis', 'Y_axis', and 'Z_axis' columns.")

    def rms(self, signal):
        """
        Calculate RMS and harmonic RMS for a single signal.

        Parameters:
            signal (np.array): Input signal to process.

        Returns:
            dict: A dictionary with RMS and harmonic RMS values.
        """
        # Perform FFT
        freq, fft_values = perform_fft(signal, self.__fs)

        # Calculate overall RMS
        rms = np.sqrt(np.mean(fft_values ** 2)) / 1000

        # Calculate RMS for harmonics
        result = {'RMS': rms}
        harmonics = int((self.__fs / 2) / self.__vel) if self.__get_harmonics else 2
        
        for i in range(1, harmonics):
            mask = self.__get_mask(i, freq)

            filtered_fft_values = fft_values[mask]

            result[f'RMS_f{i}'] = (
                np.sqrt(np.mean(filtered_fft_values ** 2)) / 1000
                if len(filtered_fft_values) > 0
                else 0
            )

        return result

    def __multiple_rms(self, df):

        results = []

        # Iterate over axes and unique codes
        for axis in ['X_axis', 'Y_axis', 'Z_axis']:
            for code in df['Unique_Code'].unique():
                try:
                    # Filter the DataFrame for the current unique code and axis
                    df_test = df[df['Unique_Code'] == code]
                    signal = df_test[axis].values

                    # Use the core function to calculate RMS and harmonics
                    result_entry = self.rms(signal)
                    result_entry.update({'Unique_Code': code, 'Axis': axis})

                    results.append(result_entry)
                except Exception as e:
                    print(f"Error processing axis {axis} and unique code {code}: {str(e)}")

        return pd.DataFrame(results)

    def __aggregate_axis(self, df, axis_list):

        # Identify RMS and harmonic RMS columns
        rms_columns = [col for col in df.columns if col.startswith('RMS') and col != 'Axis']


        try:
            aggregate_data = []
            unique_codes = df['Unique_Code'].unique()

            for code in unique_codes:
                code_data = df[df['Unique_Code'] == code]
                aggregated_entry = {'Unique_Code': code}

                # Aggregate each RMS variable
                for rms_col in rms_columns:
                    rms_values = code_data.loc[code_data['Axis'].isin(axis_list), rms_col].values
                    aggregated_entry[f'{rms_col}_aggregated'] = (
                        rms_values.mean() if len(rms_values) > 0 else None
                    )

                aggregate_data.append(aggregated_entry)

            combined_df = pd.DataFrame(aggregate_data)
            
        except Exception as e:
            raise ValueError(f"Error aggregating data for axes {axis_list}: {str(e)}")


        if combined_df.empty:
            raise ValueError("No data found for the specified axis combinations.")

        return combined_df
    
    def get_rms(self, df, aggragate_axis = False, axis_list = ['X_axis', 'Y_axis', 'Z_axis']):

        # Validate input DataFrame
        self.__check_df(df)
        self.__check_axis(df)
        order_unique_code(df)
        # Calculate RMS and harmonic RMS values for each unique code
        rms_results = self.__multiple_rms(df)

        # Aggregate RMS values across specified axes
        if aggragate_axis:
            rms_results = self.__aggregate_axis(rms_results, axis_list)

        return rms_results
    





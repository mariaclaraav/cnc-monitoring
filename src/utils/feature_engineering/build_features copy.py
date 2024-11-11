import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.seasonal import seasonal_decompose
from src.utils.filter.filter import bandpass_filter
import emd
from tqdm import tqdm



def empty_ratio(x):
    # Proportion of missing/empty values
    return np.sum(pd.isna(x)) / len(x)

def peak_intensity(x):
    # Peak intensity calculated as the distance between the 95th and 5th percentiles
    d_09 = np.percentile(x, 95) - np.percentile(x, 5)
    d_max = np.max(x) - np.min(x)
    return d_09 / d_max if d_max != 0 else 0

def linearity(x):
    # Linearity: ratio between the maximum difference and the minimum value from a linear fit
    model = np.polyfit(np.arange(len(x)), x, 1)
    line_fit = np.polyval(model, np.arange(len(x)))
    delta_xi = np.abs(x - line_fit)
    min_max_val = np.min(line_fit)
    return np.max(delta_xi) / min_max_val if min_max_val != 0 else 0

def equal_ratio(x):
    # Proportion of repeated/fixed values
    unique_counts = pd.Series(x).value_counts()
    n_equal = unique_counts[unique_counts > 1].sum()
    return n_equal / len(x)

def over_avg(x):
    # Proportion of times the series goes above the mean
    mean_value = np.mean(x)
    n_over = np.sum(x > mean_value)
    return n_over / len(x)

class TimeSeriesProcessor:
    def __init__(self, window_size, step_size, min_periods):
        self.window_size = window_size
        self.step_size = step_size if step_size else None
        self.min_periods = min_periods

    def statistical_features(self, data):
        rolling_mean = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).mean()
        rolling_std = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).std()
        rolling_max = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(lambda x: np.abs(max(x)), raw=False)
        #rolling_min = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).min()
        rolling_rms = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(lambda x: np.sqrt(np.sum(x ** 2) / len(x)), raw=False)
        #rolling_median = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).median()
        #rolling_variance = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).var()
        rolling_skewness = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).skew()
        rolling_kurtosis = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).kurt()
        rolling_impulse_factor = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(lambda x: np.max(np.abs(x)) / (np.sum(np.abs(x)) / len(x)), raw=False)
        rolling_margin_factor = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(lambda x: np.max(np.abs(x)) / ((np.sum(np.sqrt(np.abs(x))) / len(x)) ** 2), raw=False)
        #  # New features
        # rolling_empty_ratio = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(empty_ratio, raw=False)
        peak_to_peak = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(lambda x: np.max(x)- np.min(x), raw=False)
        # rolling_linearity = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(linearity, raw=False)
        # rolling_equal_ratio = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(equal_ratio, raw=False)
        #rolling_over_avg = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(over_avg, raw=False)
        
        # Concatenate all the features into a single DataFrame
        df = pd.concat([data, rolling_mean, rolling_std, rolling_max, rolling_skewness, rolling_kurtosis, rolling_rms, 
                        rolling_impulse_factor, rolling_margin_factor, peak_to_peak], axis=1)
                
        del rolling_mean, rolling_std, rolling_max, rolling_skewness, rolling_kurtosis, rolling_rms, rolling_impulse_factor, rolling_margin_factor, peak_to_peak
        
        # Name the columns
        df.columns = [data.name, 'mean', 'std', 'max', 'skew', 'kurt', 'rms', 'impulse_fact', 'margin_fact', 'peak_to_peak']
        
        #print(df)
        return df.dropna()
    
     # Define a function to calculate energy entropy for each window
    def _entropy_of_energy(self, x):
        energy = x ** 2  # Squaring the values to get the energy
        total_energy = np.sum(energy)  # Total energy in the window
        if total_energy > 0:
            energy_probability = energy / total_energy  # Probability distribution of energy
            # Calculate entropy using the probability distribution, add epsilon to avoid log(0)
            return -np.sum(energy_probability * np.log2(energy_probability + np.finfo(float).eps))
        else:
            return 0  # Entropy is zero if total energy is zero
        
    def energy_features(self, data):
        rolling_entropy = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(lambda x: self._entropy_of_energy(x), raw=False)
        rolling_normalized_energy = data.rolling(window=self.window_size, step=self.step_size, min_periods = self.min_periods).apply(lambda x: np.sum(x ** 2) / len(x), raw=False)
        
        # Concatenate all features into a DataFrame for easy analysis
        df = pd.concat([data, rolling_entropy, rolling_normalized_energy], axis=1)

        # Naming the columns to reflect the calculated features
        df.columns = [data.name, 'energy_entropy', 'norm_energy']

        return df.dropna()
    
    def filter_features(self, data, order=4, fs=2000):
        lowcut = [240, 480, 560]
        highcut = [250, 500, 580]
        name = ['240-250Hz', '480-500Hz', '560-580Hz']
        
        df = pd.DataFrame(data.values)
        for i in range(len(name)):
            filtered_signal = bandpass_filter(data.values, lowcut=lowcut[i], highcut=highcut[i], fs=fs, order=order)
            df[name[i]] = filtered_signal  # Adicionar a coluna filtrada ao DataFrame
        
        # Definir os nomes das colunas
        df.columns = [data.name] + name
        
        return df.dropna()
            
    def frequency_domain_features(self, data):
        def compute_power_spectrum(x, normalize=True):
            n = len(x)
            freq_data = np.fft.rfft(x)
            if normalize:
                freq_data = freq_data / n
            return np.abs(freq_data[1:])  # Remove DC component
        
        def compute_frequencies(n, sampling_rate):
            return np.fft.rfftfreq(n, d=1./sampling_rate)[1:]  # Remove DC component
        
        def frequency_metrics(x):
            power_spectrum = compute_power_spectrum(x)
            freqs = compute_frequencies(len(x))

            # Frequência média (Mean Frequency)
            mean_freq = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)

            # Centro de frequência (Center Frequency)
            center_freq = mean_freq

            # Frequência quadrática média (Root Mean Square Frequency)
            rms_freq = np.sqrt(np.sum(freqs ** 2 * power_spectrum) / np.sum(power_spectrum))

            # Desvio padrão da frequência (Standard Deviation of Frequency)
            std_freq = np.sqrt(np.sum((freqs - center_freq) ** 2 * power_spectrum) / np.sum(power_spectrum))

            # Assimetria espectral (Spectral Skewness)
            skewness_freq = np.sum(((freqs - center_freq) / std_freq) ** 3 * power_spectrum) / np.sum(power_spectrum)

            # Curtose espectral (Spectral Kurtosis)
            kurtosis_freq = np.sum(((freqs - center_freq) / std_freq) ** 4 * power_spectrum) / np.sum(power_spectrum)

            return mean_freq, center_freq, rms_freq, std_freq, skewness_freq, kurtosis_freq

        rolling_freq_metrics = data.rolling(window=self.window_size, step=self.step_size, min_periods=self.min_periods).apply(
        lambda x: pd.Series(frequency_metrics(x)), raw=False)

        # Split the rolling frequency metrics into individual features
        rolling_mean_freq = rolling_freq_metrics.apply(lambda x: x[0], raw=False)
        rolling_center_freq = rolling_freq_metrics.apply(lambda x: x[1], raw=False)
        rolling_rms_freq = rolling_freq_metrics.apply(lambda x: x[2], raw=False)
        rolling_std_freq = rolling_freq_metrics.apply(lambda x: x[3], raw=False)
        rolling_skewness_freq = rolling_freq_metrics.apply(lambda x: x[4], raw=False)
        rolling_kurtosis_freq = rolling_freq_metrics.apply(lambda x: x[5], raw=False)
        
        # Concatenate all features into a DataFrame
        freq_features = pd.concat([data, rolling_mean_freq, rolling_center_freq, rolling_rms_freq, rolling_std_freq, rolling_skewness_freq, rolling_kurtosis_freq], axis=1)
        # Naming the columns
        freq_features.columns = [data.name, 'MFQ', 'FC', 'RMFS', 'STDF', 'KF', 'SF']
   
        return freq_features.dropna()
        

    def discrete_wavelet_decomposition(self, data, w='db14', level=3):
        if len(data) % 2 != 0:
            data = data[:-1]
        w = pywt.Wavelet(w)
        a = data
        ca, cd = [], []
        for i in range(level):
            a, d = pywt.dwt(a, w)
            ca.append(a)
            cd.append(d)
        
        rec_a, rec_d = [], []
        for i, coeff in enumerate(ca):
            coeff_list = [coeff, None] + [None] * i
            rec_a.append(pywt.waverec(coeff_list, w))
        for i, coeff in enumerate(cd):
            coeff_list = [None, coeff] + [None] * i
            rec_d.append(pywt.waverec(coeff_list, w))

        
        last_ca = pd.Series(rec_a[-1])
        last_cd = pd.DataFrame(rec_d).T

        df_ca = pd.DataFrame(rec_a).T
        df_cd = pd.DataFrame(rec_d).T
        df_ca = df_ca.rename(columns={col: f'A{col + 1}' for col in df_ca.columns})
        df_cd = df_cd.rename(columns={col: f'D{col + 1}' for col in df_cd.columns})

        df_coef = pd.concat([df_cd, last_ca], axis=1)
        df_coef = df_coef.rename(columns={df_coef.columns[-1]: df_ca.columns[-1]})
        df_coef.dropna(axis=0, inplace=True)

        df_final = pd.concat([data.reset_index(drop=True), df_coef], axis=1)
        df_final.columns = [data.name] + [f'D{i}' for i in range(1, len(df_coef.columns))] + [df_ca.columns[-1]]

        return df_final
    
    def wavelet_packet_decomposition(self, data, w='db14', level=3):
        if len(data) % 2 != 0:
            data = data[:-1]
        
        wp = pywt.WaveletPacket(data=data, wavelet=w, mode='symmetric', maxlevel=level)
        
        # Initialize lists to store the reconstructed coefficients
        reconstructed_coeffs = []
        node_labels = []

        # Reconstruct the coefficients for each node
        nodes = [node.path for node in wp.get_level(level, 'freq')]
        
        for node in nodes:
            coeff = wp[node].data
            reconstructed_coeff = pywt.upcoef('a', coeff, w, level=level, take=len(data))
            reconstructed_coeffs.append(reconstructed_coeff)
            node_labels.append(f'node_{node}')
        
        # Convert reconstructed coefficients to DataFrame without transposing
        df_coef = pd.DataFrame(reconstructed_coeffs).T
        df_coef.columns = node_labels

        del reconstructed_coeffs, node_labels
        
        # Create the final DataFrame
        df_final = pd.concat([data.reset_index(drop=True), df_coef], axis=1)
        df_final.columns = [data.name] + df_coef.columns.tolist()
        
        del df_coef
        
        return df_final

    
    def calculate_jerk(self, data, fs):
        dt = 1 / fs
        # Use central differences
        jerk = (data.shift(-1) - data.shift(1)) / (2 * dt)
        # Remove NaN values caused by shifting
        jerk = jerk.iloc[1:-1].reset_index(drop=True)
        data_trimmed = data.iloc[1:-1].reset_index(drop=True)
        # Combine into a DataFrame
        df_final = pd.DataFrame({
            data.name: data_trimmed,
            'Jerk': jerk
        })
        return df_final

    def calculate_imf(self, data, num_imfs=5):
        # Calculate IMFs
        imfs = emd.sift.iterated_mask_sift(data.to_numpy(), sample_rate=2000, max_imfs=num_imfs)
        
        # Stack the original data and the IMFs without resetting the index
        combined = np.column_stack((data.to_numpy(), imfs))
        
        # Use the original data's name or a default name
        data_name = data.name if data.name else 'Original Data'
        
        # Create column names
        columns = [data_name] + [f'IMF{i+1}' for i in range(num_imfs)]
        
        # Create DataFrame directly from the combined array
        df_final = pd.DataFrame(combined, columns=columns)
        
        return df_final
    
    def seasonal_decomposition(self, data, period):
        decomposition = seasonal_decompose(data, period=period, model='additive')
        df_final = pd.DataFrame({
        data.name: data.values,
        'Trend': decomposition.trend.values,
        'Residual': decomposition.resid.values
        })
        return df_final.dropna()

     
    def process_time_series(self, df, feature_type='statistical', sampling_rate=2000, wavelet='db14', level=3, num_imfs=5, period=100, order = 4, plot=False):
        
        additional_columns = ['Time', 'Machine', 'Process', 'Label', 'Unique_Code', 'Period']
        processed_dfs = []

        feature_functions = {
            'statistical': self.statistical_features,
            'energy': self.energy_features,
            'frequency': self.frequency_domain_features,
            'dwt': lambda data: self.discrete_wavelet_decomposition(data, w=wavelet, level=level),
            'wpd': lambda data: self.wavelet_packet_decomposition(data, w=wavelet, level=level),
            'filter': lambda data: self.filter_features(data, fs=sampling_rate, order=order),
            'jerk': lambda data: self.calculate_jerk(data, fs=sampling_rate),
            'emd': lambda data: self.calculate_imf(data, num_imfs=num_imfs),
            'seasonal': lambda data: self.seasonal_decomposition(data, period=period)
        }

        if feature_type not in feature_functions:
            raise ValueError(f"Feature type {feature_type} is not recognized. Choose from: {', '.join(feature_functions.keys())}.")

        feature_function = feature_functions[feature_type]

        for unique_code, group in tqdm(df.groupby('Unique_Code'), desc="Processing Unique Codes"):
            if 'X_axis' not in group or 'Y_axis' not in group or 'Z_axis' not in group:
                raise ValueError("Missing axis columns in the data.")
        
            features_X = feature_function(group['X_axis'])
            features_Y = feature_function(group['Y_axis'])
            features_Z = feature_function(group['Z_axis'])
            
            features_X.columns = [f'X_{col}' for col in features_X.columns]
            features_Y.columns = [f'Y_{col}' for col in features_Y.columns]
            features_Z.columns = [f'Z_{col}' for col in features_Z.columns]
            
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
    
    
# Mapping of month abbreviations to numeric values
month_map = {
    'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04',
    'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08',
    'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'
}

def filter_and_process(df, operation, processor, feature_types):
    df_filtered = df[df['Process'] == operation].copy()
    del df
    df_filtered.loc[:, 'Month'] = df_filtered['Month'].map(month_map)
    df_filtered.loc[:, 'Period'] = df_filtered['Month'] + '-' + df_filtered['Year'].astype(str)
    df_filtered = df_filtered.drop(columns=['Month', 'Year'])
    df_filtered.reset_index(drop=True, inplace=True)


    # Process the time series features
    print(f"\nProcessing operation: {operation}...\n")
    processed_data = {}

    for feature_type in feature_types:
        print(f"Calculating {feature_type} data...")
        if feature_type in ['dwt', 'wpd']:
            data = processor.process_time_series(df_filtered, feature_type=feature_type, wavelet='db14', level=3)
        elif feature_type == 'filter':
            data = processor.process_time_series(df_filtered, feature_type=feature_type, order=4)
        elif feature_type == 'emd':
            data = processor.process_time_series(df_filtered, feature_type=feature_type, num_imfs=5)
        elif feature_type == 'seasonal':
            data = processor.process_time_series(df_filtered, feature_type=feature_type, period=100)
        else:
            data = processor.process_time_series(df_filtered, feature_type=feature_type)
        
        data.reset_index(drop=True, inplace=True)
        processed_data[feature_type] = data
        
    del df_filtered
    # Initialize merged_data with the first available DataFrame
    merged_data = list(processed_data.values())[0]

    # List of columns to drop before merging
    columns_to_drop = ['Process', 'Machine', 'Label', 'Period', 'X_X_axis', 'Y_Y_axis', 'Z_Z_axis']

    # Iterate over the remaining processed data and merge
    for data in list(processed_data.values())[1:]:
        data = data.drop(columns=columns_to_drop, errors='ignore')
        merged_data = merged_data.merge(data, on=['Unique_Code', 'Time'], how='inner')
        
        del data
        
    del processed_data

    df_final = merged_data

    print("Removing duplicate columns...")
    df_final = df_final.loc[:, ~df_final.columns.duplicated()]

    # Renaming specific columns
    columns_to_rename = {'X_X_axis': 'X_axis', 'Y_Y_axis': 'Y_axis', 'Z_Z_axis': 'Z_axis'}
    print("Renaming columns...")
    df_final.rename(columns=columns_to_rename, inplace=True)

    # Removing rows with any NaN values
    print("Removing rows with NaN values...")
    df_final = df_final.dropna()

    # Resetting the index of the cleaned DataFrame
    print("Resetting index of the cleaned DataFrame...")
    df_final.reset_index(drop=True, inplace=True)

    return df_final
    
    

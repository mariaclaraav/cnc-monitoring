import numpy as np
import pandas as pd
import pywt
import emd
from typing import Callable, Dict, List, Tuple
from scipy.signal import butter, filtfilt

class FeatureCreationUtils:
    """Utility class for feature extraction from time series data, including
    statistical, energy, filter-based, and wavelet decomposition features.
    """

    @staticmethod
    def _rolling_apply(
        data: pd.Series, 
        window_size: int, 
        step_size: int, 
        min_periods: int, 
        funcs: Dict[str, Callable[[pd.Series], float]]
    ) -> pd.DataFrame:
        """Applies multiple rolling functions to a time series

        Args:
            data (pd.Series): The time series data to apply rolling functions on.
            window_size (int): The rolling window size.
            step_size (int): The step size for rolling calculations.
            min_periods (int): Minimum number of observations required to calculate the result.
            funcs (Dict[str, Callable[[pd.Series], float]]): A dictionary of function names and their corresponding functions to apply.

        Returns:
            pd.DataFrame: DataFrame containing the results of the rolling calculations.
        """
        results = {
            func_name: data.rolling(window=window_size, step=step_size, min_periods=min_periods).apply(func, raw=False)
            for func_name, func in funcs.items()
        }
        
        df = pd.concat(results.values(), axis=1, keys=results.keys()).dropna()
        
        del results  # Free memory 
        return df
    
    @staticmethod
    def _empty_ratio(x: pd.Series) -> float:
        """Calculates the proportion of missing or empty values in a series."""
        return np.sum(pd.isna(x)) / len(x)

    @staticmethod
    def _peak_intensity(x: pd.Series) -> float:
        """Calculates peak intensity as the distance between the 95th and 5th percentiles,
        normalized by the range of values."""
        d_09 = np.percentile(x, 95) - np.percentile(x, 5)
        d_max = np.max(x) - np.min(x)
        return d_09 / d_max if d_max != 0 else 0

    @staticmethod
    def _linearity(x: pd.Series) -> float:
        """Calculates linearity as the ratio of the maximum deviation from a linear fit
        to the minimum fitted value."""
        model = np.polyfit(np.arange(len(x)), x, 1)
        line_fit = np.polyval(model, np.arange(len(x)))
        delta_xi = np.abs(x - line_fit)
        min_max_val = np.min(line_fit)
        return np.max(delta_xi) / min_max_val if min_max_val != 0 else 0

    @staticmethod
    def _equal_ratio(x: pd.Series) -> float:
        """Calculates the proportion of repeated or fixed values in the series."""
        unique_counts = pd.Series(x).value_counts()
        n_equal = unique_counts[unique_counts > 1].sum()
        return n_equal / len(x)

    @staticmethod
    def _over_avg(x: pd.Series) -> float:
        """Calculates the proportion of values in the series that are above the mean."""
        mean_value = np.mean(x)
        n_over = np.sum(x > mean_value)
        return n_over / len(x)

    @staticmethod
    def statistical_features(
        data: pd.Series, 
        window_size: int, 
        step_size: int, 
        min_periods: int
    ) -> pd.DataFrame:
        """Calculates various statistical features on rolling windows of the data.

        Args:
            data (pd.Series): The time series data to extract features from.
            window_size (int): The rolling window size.
            step_size (int): The step size for rolling calculations.
            min_periods (int): Minimum number of observations required to calculate the result.

        Returns:
            pd.DataFrame: DataFrame with statistical features including mean, std, max, rms, skewness,
                          kurtosis, impulse and margin factors, peak-to-peak, empty ratio, peak intensity,
                          linearity, equal ratio, and over-mean proportion.
        """
        funcs = {
            'mean': lambda x: np.mean(x),
            'std': lambda x: np.std(x),
            'max': lambda x: np.abs(max(x)),
            'rms': lambda x: np.sqrt(np.sum(x ** 2) / len(x)),
            'skew': lambda x: pd.Series(x).skew(),
            'kurt': lambda x: pd.Series(x).kurt(),
            'impulse_fact': lambda x: np.max(np.abs(x)) / (np.sum(np.abs(x)) / len(x)),
            'margin_fact': lambda x: np.max(np.abs(x)) / ((np.sum(np.sqrt(np.abs(x))) / len(x)) ** 2),
            'peak_to_peak': lambda x: np.max(x) - np.min(x),
            'empty_ratio': FeatureCreationUtils._empty_ratio,
            'peak_intensity': FeatureCreationUtils._peak_intensity,
            'linearity': FeatureCreationUtils._linearity,
            'equal_ratio': FeatureCreationUtils._equal_ratio,
            'over_avg': FeatureCreationUtils._over_avg
        }
        df = FeatureCreationUtils._rolling_apply(data, window_size, step_size, min_periods, funcs)
        del funcs  # Free memory used by function dictionary
        return df
    

    @staticmethod
    def _entropy_of_energy(x: pd.Series) -> float:
        """Calculates the entropy of energy for a given time series window.

        Args:
            x (pd.Series): A rolling window from the time series data.

        Returns:
            float: Entropy of energy in the window, where higher values indicate greater variability.
        """
        energy = x ** 2
        total_energy = np.sum(energy)
        if total_energy > 0:
            energy_probability = energy / total_energy
            entropy = -np.sum(energy_probability * np.log2(energy_probability + np.finfo(float).eps))
            del energy, total_energy, energy_probability  # Free memory
            return entropy
        
        del energy, total_energy  # Free memory
        return 0

    @staticmethod
    def energy_features(
        data: pd.Series, 
        window_size: int, 
        step_size: int, 
        min_periods: int
    ) -> pd.DataFrame:
        """Calculates energy-related features for each rolling window.

        Args:
            data (pd.Series): The time series data to extract features from.
            window_size (int): The rolling window size.
            step_size (int): The step size for rolling calculations.
            min_periods (int): Minimum number of observations required to calculate the result.

        Returns:
            pd.DataFrame: DataFrame with energy entropy and normalized energy features.
        """
        funcs = {
            'energy_entropy': FeatureCreationUtils._entropy_of_energy,
            'norm_energy': lambda x: np.sum(x ** 2) / len(x)
        }
        df = FeatureCreationUtils._rolling_apply(data, window_size, step_size, min_periods, funcs)
        
        del funcs  # Free memory
        return df

    @staticmethod
    def bandpass_filter(
        signal: np.ndarray, 
        lowcut: float, 
        highcut: float, 
        fs: int, 
        order: int
    ) -> np.ndarray:
        """Applies a bandpass filter to the signal

        Args:
            signal (np.ndarray): The time series data to filter
            lowcut (float): Lower bound frequency for the bandpass filter
            highcut (float): Upper bound frequency for the bandpass filter
            fs (int): Sampling rate of the data
            order (int): Order of the filter

        Returns:
            np.ndarray: The filtered signal
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, signal)
        
        del b, a  # Free memory
        return filtered_signal

    @staticmethod
    def filter_features(
        data: pd.Series, 
        frequency_bands: List[Tuple[float, float]] = [(240, 250), (480, 500), (560, 580)],  
        fs: int = 2000, 
        order: int = 4
    ) -> pd.DataFrame:
        """Applies bandpass filters to a time series for specified frequency bands.

        Args:
            data (pd.Series): The time series data to filter.
            frequency_bands (List[Tuple[float, float]], optional): List of tuples with (lowcut, highcut) frequencies for each band.
            fs (int, optional): Sampling rate of the data. Defaults to 2000.
            order (int, optional): Order of the filter. Defaults to 4.

        Returns:
            pd.DataFrame: DataFrame with columns for each filtered frequency band.
        """       
        filtered_signals = {
        f'{low}-{high}Hz': FeatureCreationUtils.bandpass_filter(data.values, low, high, fs, order)
        for low, high in frequency_bands
    }
        
        # Convert filtered results to DataFrame
        df_filtered = pd.DataFrame(filtered_signals)
        df_final = pd.concat([data.reset_index(drop=True), df_filtered], axis=1)
        del filtered_signals  # Free memory
        return df_final

    @staticmethod
    def discrete_wavelet_decomposition(
        data: pd.Series, 
        w: str = 'db14', 
        level: int = 3
    ) -> pd.DataFrame:
        """Performs discrete wavelet decomposition on a time series

        Args:
            data (pd.Series): The time series data to decompose
            wavelet (str, optional): The type of wavelet to use
            level (int, optional): The level of decomposition

        Returns:
            pd.DataFrame: DataFrame containing approximation and detail coefficients
        """
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

    @staticmethod
    def wavelet_packet_decomposition(
        data: pd.Series, 
        w: str = 'db14', 
        level: int = 3
    ) -> pd.DataFrame:
        """Performs wavelet packet decomposition on a time series

        Args:
            data (pd.Series): The time series data to decompose.
            wavelet (str, optional): The type of wavelet to use. Defaults to 'db14'.
            level (int, optional): The level of decomposition. Defaults to 3.

        Returns:
            pd.DataFrame: DataFrame containing coefficients of each node at the specified level
        """
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
    
    @staticmethod
    def calculate_jerk(
        data: pd.Series,
        fs: int
    ) -> pd.DataFrame:
        """Calculate the jerk (rate of change of acceleration) for a time series.

        Args:
            data (pd.Series): The time series data (e.g., acceleration) to 
                calculate jerk from.
            fs (int): Sampling rate of the data in Hz.

        Returns:
            pd.DataFrame: DataFrame containing the original data (trimmed) and 
                the calculated jerk values.
        """
        dt = 1 / fs
        jerk = (data.shift(-1) - data.shift(1)) / (2 * dt)
        
        # Remove NaN values caused by shifting
        jerk = jerk.iloc[1:-1].reset_index(drop=True)
        data_trimmed = data.iloc[1:-1].reset_index(drop=True)
        
        df_final = pd.DataFrame({
            data.name: data_trimmed,
            'Jerk': jerk
        })
        del data_trimmed, jerk
        return df_final
    
    @staticmethod
    def calculate_imf(data: pd.Series, num_imfs: int = 5) -> pd.DataFrame:
        """Calculate Intrinsic Mode Functions (IMFs) for a time series using 
        Empirical Mode Decomposition (EMD).

        Args:
            data (pd.Series): The time series data to decompose.
            num_imfs (int, optional): The number of IMFs to compute. Defaults to 5.

        Returns:
            pd.DataFrame: DataFrame containing the original data and each computed 
                IMF as separate columns.
        """
        # Calculate IMFs using EMD
        imfs = emd.sift.iterated_mask_sift(
            data.to_numpy(), sample_rate=2000, max_imfs=num_imfs
        )
        
        # Stack the original data and IMFs into a single array
        combined = np.column_stack((data.to_numpy(), imfs))
        
        # Determine the column name for the original data
        data_name = data.name if data.name else 'Original Data'
        
        # Create column names
        columns = [data_name] + [f'IMF{i+1}' for i in range(num_imfs)]
        
        # Create DataFrame from combined data
        df_final = pd.DataFrame(combined, columns=columns)
        return df_final
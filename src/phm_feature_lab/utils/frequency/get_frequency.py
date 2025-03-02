import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
from typing import Tuple, List, Optional

from phm_feature_lab.utils.filter.filter import perform_fft

class FrequencyProcessor:
    def __init__(
        self,
        data: pd.DataFrame,
        column_name: str,
        sampling_rate: float,
        bin_width: float = 10,
        height_threshold: float = 0.05,
        plot: bool = True,
        print_values: bool = True
    ):
        """
        Initializes the FrequencyProcessor with specified parameters.

        Parameters:
        - bin_width (float): Width of the frequency bins. Default is 10.
        - height_threshold (float): Height threshold for peak detection. Default is 0.05.
        - plot (bool): If True, generates plots. Default is True.
        - print_values (bool): If True, prints values to the console. Default is True.
        """
        self.__bin_width = bin_width
        self.__height_threshold = height_threshold
        self.__plot = plot
        self.__print_values = print_values
        self.__data = data
        self.__column_name = column_name
        self.__sampling_rate = sampling_rate

    def _merge_consecutive_bins(
        self,
        bins: np.ndarray,
        bin_width: float,
        energies: Optional[np.ndarray] = None,
    ) -> List[Tuple[float, float, Optional[float]]]:
        """
        Merges consecutive bins into intervals and optionally stores the maximum energy of each interval.

        Parameters:
        - bins (np.ndarray): Array of frequency bins.
        - bin_width (float): Width of each bin.
        - energies (np.ndarray, optional): Array of maximum energies per bin.

        Returns:
        - merged_intervals (List[Tuple[float, float, Optional[float]]]): 
          List of merged intervals with maximum energy (if provided).
        """
        if len(bins) == 0:
            return []
        
        merged_intervals = []
        start_bin = bins[0]
        end_bin = start_bin + bin_width
        max_energy = energies[0] if energies is not None else None

        for i in range(1, len(bins)):
            current_bin = bins[i]
            expected_bin = end_bin  # Next expected bin

            if energies is not None:
                current_energy = energies[i]

            if np.isclose(current_bin, expected_bin):
                # Extend the current interval
                end_bin = current_bin + bin_width
                if energies is not None:
                    max_energy = max(max_energy, current_energy)
            else:
                # Save the current interval and start a new one
                merged_intervals.append((start_bin, end_bin, max_energy))
                start_bin = current_bin
                end_bin = start_bin + bin_width
                max_energy = current_energy if energies is not None else None

        # Add the last interval
        merged_intervals.append((start_bin, end_bin, max_energy))

        return merged_intervals

    def get_frequency(self) -> Tuple[List[Tuple[float, float, Optional[float]]], List[Tuple[float, float, Optional[float]]]]:
        """
        Processes the frequency spectrum to identify peaks and bins with high energy and peak density.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the signal data with time as index.
        - column_name (str): Name of the column to analyze.
        - sampling_rate (float): Sampling rate of the data.

        Returns:
        - Tuple containing:
            - List of merged high energy frequency intervals with their maximum energies.
            - List of merged high density peak frequency intervals.
        """
        # Extract time and signal arrays
        time_array = self.__data.index.to_numpy()
        signal_array = self.__data[self.__column_name].values

        # Perform FFT
        freqs, magnitude = perform_fft(time_array, signal_array, self.__sampling_rate)

        # Detect peaks in the magnitude spectrum
        peaks, properties = find_peaks(magnitude, height=np.max(magnitude) * self.__height_threshold)
        peak_freqs = freqs[peaks]
        peak_magnitudes = magnitude[peaks]

        # Define frequency bins
        bins = np.arange(0, np.max(freqs) + self.__bin_width, self.__bin_width)

        # Calculate the sum of amplitudes per bin (for energy)
        hist_energy, bin_edges_energy = np.histogram(
            peak_freqs, bins=bins, weights=peak_magnitudes
        )

        # Calculate the maximum energy per bin
        max_energy, _, _ = binned_statistic(
            peak_freqs, peak_magnitudes, statistic='max', bins=bins
        )

        # Identify bins with high energy concentration
        threshold_energy = np.max(hist_energy) * 0.01  # Adjust as needed
        high_energy_bin_indices = np.where(hist_energy >= threshold_energy)[0]
        high_energy_bins = bin_edges_energy[:-1][high_energy_bin_indices]
        high_energy_bins = np.sort(high_energy_bins)
        high_energy_maxes = max_energy[high_energy_bin_indices]

        # Merge consecutive high energy bins
        merged_energy_intervals = self._merge_consecutive_bins(
            high_energy_bins, self.__bin_width, high_energy_maxes
        )

        if self.__print_values:
            # Print bins with highest energy concentration
            print("Bins com maior concentração de energia:")
            for start_freq, end_freq, _ in merged_energy_intervals:
                print(f"Intervalo de frequência: {start_freq:.2f} Hz - {end_freq:.2f} Hz")

        # Calculate the histogram of peak counts per bin (for density)
        hist_peaks, bin_edges = np.histogram(peak_freqs, bins=bins)

        # Identify bins with high peak density
        threshold_peaks = np.max(hist_peaks) * 0.01  # Adjust as needed
        high_density_bin_indices = np.where(hist_peaks >= threshold_peaks)[0]
        high_density_bins = bin_edges[:-1][high_density_bin_indices]
        high_density_bins = np.sort(high_density_bins)

        # Merge consecutive high density bins
        merged_density_intervals = self._merge_consecutive_bins(
            high_density_bins, self.__bin_width
        )

        if self.__print_values:
            # Print bins with high peak density
            print("\nBins com alta concentração de picos:")
            for start_freq, end_freq, _ in merged_density_intervals:
                print(f"Intervalo de frequência: {start_freq:.2f} Hz - {end_freq:.2f} Hz")

        # Optionally, plot the results
        if self.__plot:
            # Plot histogram of peak frequencies
            plt.figure(figsize=(12, 6))
            plt.hist(peak_freqs, bins=bins, edgecolor='black')
            plt.title(f"Histograma das Frequências dos Picos - ({self.__column_name})")
            plt.xlabel("Frequência (Hz)")
            plt.ylabel("Número de Picos")
            plt.grid(True)
            plt.show()

            # Plot frequency spectrum with high density intervals highlighted
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, magnitude, label='Espectro de Frequência')
            plt.plot(peak_freqs, peak_magnitudes, 'ro', label='Picos Detectados')

            for idx, (start_freq, end_freq, _) in enumerate(merged_density_intervals):
                label = 'Alta densidade' if idx == 0 else ""
                plt.axvspan(start_freq, end_freq, color='yellow', alpha=0.2, label=label)

            plt.title(f"Espectro de Frequência com Intervalos de Alta Densidade de Picos - ({self.__column_name})")
            plt.xlabel("Frequência (Hz)")
            plt.ylabel("Amplitude")
            plt.legend(loc='upper right')
            plt.grid(True)
            plt.show()

        return merged_energy_intervals, merged_density_intervals

    def run(self) -> Tuple[List[Tuple[float, float, Optional[float]]], List[Tuple[float, float, Optional[float]]]]:
        """
        Executes the frequency analysis process.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the signal data with time as index.
        - column_name (str): Name of the column to analyze.
        - sampling_rate (float): Sampling rate of the data.

        Returns:
        - Tuple containing:
            - List of merged high energy frequency intervals with their maximum energies.
            - List of merged high density peak frequency intervals.
        """
        return self.get_frequency()

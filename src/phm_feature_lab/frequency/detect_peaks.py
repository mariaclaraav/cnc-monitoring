import numpy as np
from scipy.signal import find_peaks
from phm_feature_lab.utils.filter.filter import perform_fft
from typing import Optional, Tuple, List


class DetectPeaks:
    """
    A class to analyze the frequency spectrum of a signal, identify peaks, and sort them by height.
    """

    def __init__(self, fs: float, height_threshold: float = 0.01, distance: int = 2000):
        """
        Initialize the FFTSignalAnalyzer with sampling frequency and peak detection parameters.

        Args:
            fs (float): Sampling frequency in Hz.
            height_threshold (float, optional): Minimum height threshold for peaks. Default is 0.01.
            distance (int, optional): Minimum distance between peaks in points. Default is 2000.
        """
        self.__fs = fs
        self.__height_threshold = height_threshold
        self.__distance = distance

    def find_signal_peaks(self, amplitudes: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Identifies peaks in a frequency spectrum using find_peaks.

        Args:
            amplitudes (np.ndarray): Array of amplitude values from FFT.

        Returns:
            Tuple[np.ndarray, dict]: Peak indices and their properties.
        """
        return find_peaks(
            amplitudes, height=self.__height_threshold, distance=self.__distance
        )

    def sort_peaks_by_height(
        self, peak_frequencies: np.ndarray, peak_heights: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sorts peak frequencies and heights by peak height in descending order.

        Args:
            peak_frequencies (np.ndarray): Frequencies of detected peaks.
            peak_heights (np.ndarray): Heights (amplitudes) of detected peaks.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Sorted frequencies and heights (highest to lowest).
        """
        sorted_order = np.argsort(peak_heights)[::-1]  # Descending order by height
        return peak_frequencies[sorted_order], peak_heights[sorted_order]

    def get_frequencies(self, signal: np.ndarray) -> None:
        """
        Analyzes the frequency spectrum of a signal, identifies peaks, and prints frequencies
        ordered by peak height (highest to lowest).

        Args:
            signal (np.ndarray): Input signal data (e.g., signal['Z_axis'].values).

        Returns:
            None: Prints the sorted peak frequencies and heights, but does not return values.
        """
        # Compute FFT
        freqs, amplitudes = perform_fft(signal, self.__fs)

        # Find peaks in the frequency spectrum
        peaks, properties = self.find_signal_peaks(amplitudes)

        # Extract peak frequencies and heights
        peak_frequencies = freqs[peaks]
        peak_heights = properties["peak_heights"]

        # Sort peaks by height (descending order)
        sorted_freqs, sorted_heights = self.sort_peaks_by_height(
            peak_frequencies, peak_heights
        )

        # Print sorted peaks
        print("Higher frequencies:")
        for freq, height in zip(sorted_freqs, sorted_heights):
            print(f"{freq:.2f} Hz, {height:.4f}")

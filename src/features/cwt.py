import numpy as np
import pywt
import matplotlib.pyplot as plt
from typing import List, Union

class CustomCWT:
    def __init__(self, frequencies: List[float], wavelet: str, sampling_rate: int):
        """
        Initializes the CWTAnalyzer with specified frequencies, wavelet, and sampling rate.

        Parameters:
        - frequencies (List[float]): List of frequencies for which scales will be calculated.
        - wavelet (str): Name of the wavelet to use in CWT (e.g., 'morl', 'gaus1').
        - sampling_rate (int): Sampling rate of the signal in Hz.
        """
        self.__frequencies = frequencies
        self.__wavelet = wavelet
        self.__sampling_rate = sampling_rate
        self.__scales = self.get_scales()

    def get_scales(self) -> List[float]:
        """
        Calculates the scales corresponding to the provided frequencies using the specified wavelet.

        Returns:
        - List[float]: List of scales corresponding to the frequencies.
        """
        if not isinstance(self.__sampling_rate, int):
            raise ValueError("`sampling_rate` must be an integer representing the sampling rate in Hz.")

        sampling_period = 1 / self.__sampling_rate

        try:
            scales = [pywt.scale2frequency(self.__wavelet, f) / sampling_period for f in self.__frequencies]
        except ValueError as e:
            raise ValueError(f"Error calculating scales: {e}")

        return np.arange(scales[0], scales[1])

    def get_cwt_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Generates the CWT spectrogram of the provided signal and returns it as a 2D NumPy array.

        Parameters:
        - signal (np.ndarray): Input signal for which the CWT spectrogram will be computed.

        Returns:
        - np.ndarray: 2D CWT spectrogram array.
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy.ndarray.")

        # Compute CWT using the pre-calculated scales
        try:
            cwtmatr, _ = pywt.cwt(signal, self.__scales, self.__wavelet, sampling_period=1 / self.__sampling_rate)
        except Exception as e:
            raise RuntimeError(f"Error computing CWT: {e}")

        # Take the absolute value and remove the last row and column to avoid edge effects
        cwtmatr = np.abs(cwtmatr[:-1, :-1])

        return cwtmatr

    def plot_cwt(self, time: np.ndarray, signal: np.ndarray, cmap: str = 'seismic'):
        """Plots the CWT spectrogram of the provided signal.

        Parameters:
        - signal (np.ndarray): Input signal for which the CWT spectrogram will be plotted.
        - cmap (str): Colormap to use for the spectrogram. Default is 'seismic'.
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Signal must be a numpy.ndarray.")

        # Compute CWT
        cwtmatr, freqs = pywt.cwt(signal, self.__scales, self.__wavelet, sampling_period=1 / self.__sampling_rate)

        # Plot using subplots to get figure and axes
        fig, ax = plt.subplots(figsize=(10, 6))
        cax = ax.pcolormesh(time, freqs, cwtmatr, cmap=cmap, shading='auto')

        fig.colorbar(cax, ax=ax, label='Power')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.set_title('CWT Spectrogram')
        plt.show()

    def run(self, signal: np.ndarray) -> List[float]:
        """Runs the CWT analysis on the provided signal and returns the spectrogram as a flattened list.
        """
        return self.get_cwt_spectrogram(signal).flatten().tolist()

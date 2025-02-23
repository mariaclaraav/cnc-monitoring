import numpy as np
import matplotlib.pyplot as plt
import pywt
from typing import List, Tuple, Optional
from tqdm import tqdm 
from phm_feature_lab.utils.logger import Logger

logger = Logger().get_logger()
 


class WaveletSimilarity:
    """
    Class to analyze signal similarity using different wavelets.
    """

    def __init__(self, wavlist: List[str]):
        self.__wavlist = wavlist

    def __normalize_correlation(self, correlation: np.ndarray) -> np.ndarray:
        """
        Normalizes correlation values to [0, 1].
        """
        X_min = np.min(correlation)
        X_max = np.max(correlation)

        return (
            (correlation - X_min) / (X_max - X_min) if X_max != X_min else correlation
        )

    def __process_wavelet(self, wavelet: str) -> np.ndarray:
        """
        Processes the wavelet to generate its wavelet function.
        """
        if wavelet == "shan":
            wavelet += "1-1"
        elif wavelet == "cmor":
            wavelet += "0.5-1"
        elif wavelet == "fbsp":
            wavelet += "1-1.5-1.0"

        try:
            [psi, _] = pywt.ContinuousWavelet(wavelet).wavefun(10)
            return psi
        
        except Exception as e:
            logger.error(f"Error processing wavelet {wavelet}: {e}")
            raise


    def __get_correlation(self, wavelet: str, signal: np.ndarray) -> np.ndarray:
        """
        Calculates sliding cross-correlation for a wavelet and signal.
        """
        try:
            wavelet_complex = self.__process_wavelet(wavelet)
            wavelet_length = len(wavelet_complex)

            sliding_correlation = [
                np.vdot(signal[i:i + wavelet_length], wavelet_complex)
                for i in range(len(signal) - wavelet_length + 1)
            ]

            normalized_correlation = self.__normalize_correlation(np.abs(sliding_correlation))
            return normalized_correlation
        
        except Exception as e:
            logger.error(f"Error calculating correlation for wavelet {wavelet}: {e}")
            raise

    def __get_common_time_interval(
        self, times: List[np.ndarray]
    ) -> Tuple[float, float]:
        """
        Finds the common time interval across all time arrays.
        """
        try:
            start_time = max([t[0] for t in times])  # Latest start time
            end_time = min([t[-1] for t in times])   # Earliest end time
            return start_time, end_time
        
        except Exception as e:
            logger.error(f"Error finding common time interval: {e}")
            raise

        return start_time, end_time

    def __filter_signals_for_common_interval(
        self,
        signals: List[np.ndarray],
        times: List[np.ndarray],
        start_time: float,
        end_time: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Filters signals and times to match the common interval.
        """
        try:
            filtered_signals = []
            filtered_times = []

            for signal, time in zip(signals, times):
                time = np.array(time)  # Ensure time is a numpy array
                signal = np.array(signal)  # Ensure signal is a numpy array

                mask = (time >= start_time) & (time <= end_time)

                filtered_signals.append(signal[mask])
                filtered_times.append(time[mask])

            return filtered_signals, filtered_times
        
        except Exception as e:
            logger.error(f"Error filtering signals for common interval: {e}")
            raise


    def __plot_wavelet_correlation(
        self,
        ax: plt.Axes,
        common_time: np.ndarray,
        mean_correlation: np.ndarray,
        wavelet: str,
        max_correlation: Optional[np.ndarray],
    ) -> None:
        """
        Plots the correlation for a specific wavelet.
        """
        try:
            ax.plot(common_time[:len(mean_correlation)], mean_correlation, label=f"Wavelet {wavelet}", color='blue')
            ax.set_title(f'Wavelet {wavelet} - Mean: {np.mean(mean_correlation):.2f} - Max: {max_correlation:.2f}')
            ax.set_xlabel('Time')
            ax.set_ylabel('Normalized Correlation', color='blue')
            ax.legend(loc='upper right')
            
        except Exception as e:
            logger.error(f"Error plotting correlation for wavelet {wavelet}: {e}")
            raise

    def plot_correlation(
        self, signals: List[np.ndarray], times: List[np.ndarray]
    ) -> None:
        """
        Plots sliding cross-correlation between wavelets and signals.
        """
        try:
            if len(signals) != len(times):
                logger.error("Signals and times lists must have the same length.")
                return 


            start_time, end_time = self.__get_common_time_interval(times)
            filtered_signals, filtered_times = self.__filter_signals_for_common_interval(signals, times, start_time, end_time)
            common_time = filtered_times[0]

            cols = 1
            rows = (len(self.__wavlist) + cols - 1) // cols

            # Create subplots with shared axes
            fig, axs = plt.subplots(rows, cols, figsize=(12, 20), sharex=True, sharey=True)

            # Ensure axs is always iterable (even if there's only one subplot)
            if rows == 1:
                axs = [axs]

            # Adding tqdm progress bar to the wavelet loop
            for idx, wavelet in enumerate(tqdm(self.__wavlist, desc="Processing wavelets")):
                # Calculate correlation for each signal
                correlations = [self.__get_correlation(wavelet, signal) for signal in filtered_signals]

                # Calculate mean of correlations
                mean_correlation = np.mean(correlations, axis=0)
                max_correlation = np.max(mean_correlation)

                # Plot mean correlation for the wavelet
                ax = axs[idx]
                self.__plot_wavelet_correlation(ax, common_time, mean_correlation, wavelet, max_correlation)

            logger.info("Correlation plotting complete.")
            plt.tight_layout()
            plt.show()

        except ValueError as e:
            logger.error(f"Value error in plot_correlation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in plot_correlation: {e}")
            raise
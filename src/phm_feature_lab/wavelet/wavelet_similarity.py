import numpy as np
import matplotlib.pyplot as plt
import pywt
from typing import List, Tuple, Optional
from tqdm import tqdm
from phm_feature_lab.utils.logger import Logger

logger = Logger().get_logger()


class WaveletSimilarity:
    """
    Class to analyze signal similarity using different wavelets with dynamic scaling.
    """

    def __init__(self, wavlist: List[str]):
        """
        Initialize the WaveletSimilarity class.

        Args:
            wavlist (List[str]): List of wavelet names with optional parameters (e.g., 'cmor1-1', 'cmor1-2', 'shan').
        """
        self.__wavlist = wavlist
        # self.__target_freq = target_freq

    def __normalize_correlation(self, correlation: np.ndarray) -> np.ndarray:
        """
        Normalizes correlation values to [0, 1].
        """
        X_min = np.min(correlation)
        X_max = np.max(correlation)
        return (
            (correlation - X_min) / (X_max - X_min) if X_max != X_min else correlation
        )

    def __parse_wavelet_parameters(self, 
        wavelet: str,
    ) -> str:
        """
        Parses wavelet name and extracts bandwidth frequency (fb) and central frequency (fc) parameters.

        Args:
            wavelet (str): Wavelet name or specification (e.g., 'cmor0.5-1', 'shan1-2', 'mexh').

        Returns:
            Tuple[str, Optional[float], Optional[float]]: Wavelet name, bandwidth frequency (fb), and central frequency (fc).

        Raises:
            ValueError: If the wavelet is invalid or not supported.
        """
        try:
            # Check if wavelet already has parameters or is a complete name
            if (
                "-" in wavelet
                or wavelet in ["mexh", "morl"]
                or wavelet.startswith(("cgau", "shan", "cmor"))
            ):
                wavelet_name = wavelet
            else:
                # Default parameters for wavelets without specification
                default_params = {
                    "shan": ("shan1-2", 1.0, 2.0),  # fb=1, fc=2
                    "cmor": ("cmor0.5-1", 0.5, 1.0),  # fb=0.5, fc=1
                    "fbsp": ("fbsp1-1.5-1.0", 1.5, 1.0),  # m=1, fb=1.5, fc=1.0
                    "mexh": ("mexh", None, 1.0),
                    "morl": ("morl", None, 1.0),
                }
                if wavelet in default_params:
                    wavelet_name, fb, fc = default_params[wavelet]
                elif wavelet.startswith("cgau"):
                    wavelet_name = wavelet  # e.g., cgau1, cgau2
                    # fb = None  # Not applicable for cgau, as fb is implicit
                    # fc = 1.0
                elif wavelet in pywt.wavelist(kind="continuous"):
                    wavelet_name = wavelet
                    # fb = None  # Default or implicit for some wavelets
                    # fc = 1.0
                else:
                    raise ValueError(
                        f"Wavelet {wavelet} must be from pywt.wavelist(kind='continuous')."
                    )

                return wavelet_name
            # # Extract fb and fc for parameterized wavelets (shan, cmor, fbsp) if not already set
            # fb, fc = None, None  # Default to None if not applicable
            # if (
            #     wavelet_name.startswith(("shan", "cmor", "fbsp"))
            #     and "-" in wavelet_name
            # ):
            #     parts = wavelet_name.split("-")
            #     if wavelet_name.startswith("cmor"):
            #         fb, fc = float(parts[1]), float(parts[2])
            #     elif wavelet_name.startswith("shan"):
            #         fb, fc = float(parts[1]), float(parts[2])
                    
            #     elif wavelet_name.startswith("fbsp"):
            #         if len(parts) != 3:
            #             raise ValueError("fbsp wavelet must be in format 'fbspM-fb-fc'")
            #         fb, fc = float(parts[1]), float(parts[2])  # fb is second, fc is third

            # # Validate fb and fc if provided
            # if fb is not None and fc is not None and (fb <= 0 or fc <= 0):
            #     raise ValueError(
            #         f"Bandwidth frequency (fb) and central frequency (fc) must be positive: fb={fb}, fc={fc}"
            #     )
            return wavelet_name

        except ValueError as e:
            logger.error(f"Invalid wavelet specification: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error parsing wavelet: {str(e)}")
            raise

    def __process_wavelet(
        self, wavelet: str, length: int = 10
    ) -> Tuple[np.ndarray, float]:
        """
        Processes the wavelet to generate its function and calculate the scale.

        Args:
            wavelet (str): Wavelet name (e.g., 'cmor1-1', 'cmor1-2', 'shan1-2').
        Returns:
            Tuple[np.ndarray, float]: Wavelet function values and scale factor.
        """
        try:
            wavelet_name = self.__parse_wavelet_parameters(wavelet)

            # Calcular escala para target_freq
            # a = fc / (self.__target_freq * self.__Ts)
            [psi, _] = pywt.ContinuousWavelet(wavelet_name).wavefun(length)

            return psi

        except Exception as e:
            logger.error(f"Error processing wavelet {wavelet}: {e}")
            raise

    def __get_correlation(self, wavelet: str, signal: np.ndarray) -> np.ndarray:
        """
        Calculates sliding cross-correlation for a wavelet and signal.

        Args:
            wavelet (str): Wavelet name.
            signal (np.ndarray): Input signal.

        Returns:
            np.ndarray: Normalized correlation values.
        """
        try:
            # Normalizar o sinal
            signal_norm = (signal - np.mean(signal)) / np.std(signal)

            # Processar a wavelet com escala dinâmica
            wavelet_values = self.__process_wavelet(wavelet)
            wavelet_length = len(wavelet_values)

            # Normalizar a wavelet
            wavelet_norm = (wavelet_values - np.mean(wavelet_values)) / np.std(wavelet_values)

            # Correlação cruzada deslizante
            sliding_correlation = [
                np.sum(signal_norm[i:i + wavelet_length] * wavelet_norm)
                for i in range(len(signal_norm) - wavelet_length + 1)
            ]

            # Normalizar a correlação para [0, 1]
            normalized_correlation = self.__normalize_correlation(np.abs(sliding_correlation))
            return normalized_correlation

        except Exception as e:
            logger.error(f"Error calculating correlation for wavelet {wavelet}: {e}")
            raise

    def __get_common_time_interval(
        self, times: List[np.ndarray]
    ) -> Tuple[float, float]:
        try:
            start_time = max([t[0] for t in times])
            end_time = min([t[-1] for t in times])
            return start_time, end_time
        except Exception as e:
            logger.error(f"Error finding common time interval: {e}")
            raise

    def __filter_signals_for_common_interval(
        self,
        signals: List[np.ndarray],
        times: List[np.ndarray],
        start_time: float,
        end_time: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        try:
            filtered_signals = []
            filtered_times = []
            for signal, time in zip(signals, times):
                time = np.array(time)
                signal = np.array(signal)
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
        max_correlation: Optional[float],
        line_color: str,
    ) -> None:
        try:
            ax.plot(common_time[:len(mean_correlation)], mean_correlation, label=f"Wavelet {wavelet}", color=line_color)
            ax.set_title(f'Wavelet {wavelet} - Média: {np.mean(mean_correlation):.2f}')
            print(f"Wavelet {wavelet} - Mean: {np.mean(mean_correlation):.2f} Max: {max_correlation:.2f}")
            #ax.set_xlabel('Time')
            ax.set_ylabel('Correlação', color='k')
            #ax.legend(loc='upper right', fontsize=line_color)
        except Exception as e:
            logger.error(f"Error plotting correlation for wavelet {wavelet}: {e}")
            raise

    def plot_correlation(
        self,
        signals: List[np.ndarray],
        times: List[np.ndarray],
        figsize: Tuple[int, int] = (12, 20),
        line_color: str = 'k',
    ) -> None:
        try:
            if len(signals) != len(times):
                logger.error("Signals and times lists must have the same length.")
                return

            start_time, end_time = self.__get_common_time_interval(times)
            print(f"Common time interval: {start_time:.2f} to {end_time:.2f} s")
            filtered_signals, filtered_times = self.__filter_signals_for_common_interval(signals, times, start_time, end_time)
            common_time = filtered_times[0]

            cols = 1
            rows = (len(self.__wavlist) + cols - 1) // cols

            fig, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True, sharey=True)
            if rows == 1:
                axs = [axs]

            for idx, wavelet in enumerate(tqdm(self.__wavlist, desc="Processing wavelets")):
                correlations = [self.__get_correlation(wavelet, signal) for signal in filtered_signals]
                mean_correlation = np.mean(correlations, axis=0)
                max_correlation = np.max(mean_correlation)

                ax = axs[idx]
                self.__plot_wavelet_correlation(ax, common_time, mean_correlation, wavelet, max_correlation, line_color)

            logger.info("Correlation plotting complete.")
            plt.tight_layout()
            plt.show()

        except ValueError as e:
            logger.error(f"Value error in plot_correlation: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in plot_correlation: {e}")
            raise

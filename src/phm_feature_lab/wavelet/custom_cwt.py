import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Union
from mpl_toolkits.axes_grid1 import make_axes_locatable

class CustomCWT:
    def __init__(self, frequencies: List[float], wavelet: str, sampling_rate: int, norm: str = False):
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
        self.__norm = norm
        
        self.__label = 'Coeficientes Normalizados' if norm else 'Coeficientes'
        
    def get_scales(self) -> List[float]:
        """
        Calculates the scales corresponding to the provided frequencies using the specified wavelet.

        Returns:
        - List[float]: List of scales corresponding to the frequencies.
        """
        if not isinstance(self.__sampling_rate, int):
            raise ValueError("`sampling_rate` must be an integer representing the sampling rate in Hz.")
        
        frequencies = np.array(self.__frequencies)/self.__sampling_rate
        try:
            scales = pywt.frequency2scale(self.__wavelet, frequencies)
        except ValueError as e:
            raise ValueError(f"Error calculating scales: {e}")
        return np.linspace(scales[0], scales[-1])
    
    def __coef_norm(self, cwtmatr: np.ndarray) -> np.ndarray:
        aux = np.abs(cwtmatr[:-1, :-1])
        coef = (aux - aux.min())/(aux.max() - aux.min())
        return coef 
      
    def get_cwt_features(self, signal: np.ndarray):
        
        coef, _ = self.get_cwt_spectrogram(signal)
        # max value for each frequency
        Fdom = coef.max(axis=1)
                
        # energy distribution for each scale
        Fdist = np.sum(np.abs(coef)**2, axis=1)  
        
        Fmean = np.mean(coef, axis=1)
        
        Fstd = np.std(coef, axis=1)
        Fq1 = np.quantile(coef, 0.25, axis=1)

        Fq3 = np.quantile(coef, 0.75, axis=1)
        
        feat = pd.DataFrame({'Fdom': Fdom, 'Fdist': Fdist, 'Fmean': Fmean, 'Fstd': Fstd, 'Fq1': Fq1, 'Fq3': Fq3})
        return feat
        
    def get_cwt_spectrogram(self, signal: np.ndarray) -> np.ndarray:
        """Generates the CWT spectrogram of the provided signal and returns it as a 2D NumPy array.

        Parameters:
        - signal (np.ndarray): Input signal for which the CWT spectrogram will be computed.

        Returns:
        - np.ndarray: 2D CWT spectrogram array.
        """
        if not isinstance(signal, np.ndarray):
            try:
                signal = np.array(signal)
            except Exception as e:
                raise RuntimeError(f"Error converting to numpy array: {e}")

        # Compute CWT using the pre-calculated scales
        try:
            cwtmatr, freq = pywt.cwt(signal, self.__scales, self.__wavelet, sampling_period=1 / self.__sampling_rate)
        except Exception as e:
            raise RuntimeError(f"Error computing CWT: {e}")

        # Take the absolute value and remove the last row and column to avoid edge effects
        # transform to power and apply logarithm ?!
        #coef = np.log2(coef**2+0.001) 
        # normalize coef
        
        if self.__norm:
            return self.__coef_norm(cwtmatr), freq
        else:           
            return np.abs(cwtmatr[:-1, :-1]), freq
        
    def save_cwt_img(self, 
                 time: np.ndarray, 
                 signal: np.ndarray, 
                 cmap: str = 'seismic',
                 figsize: tuple = (10, 4), 
                 save_path: str = None):
    
        # Compute the CWT spectrogram
        coef, freq = self.get_cwt_spectrogram(signal)  
        
        # Create the figure and axis
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot the spectrogram
        cax = ax.pcolormesh(time, freq, coef, cmap=cmap, shading='auto')
        
        # Remove all visual elements
        ax.axis('off')  # Remove axes
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        
        # Save the figure in RGBA and or return it
        if save_path:
            fig.savefig(save_path, dpi=500, bbox_inches='tight', pad_inches=0)  # Save without extra padding
            plt.close(fig)  # Close the figure after saving
        else:
            return fig

    def plot_cwt(self, 
                 time: np.ndarray, 
                 signal: np.ndarray, 
                 cmap: str = 'seismic', 
                 title: str = 'CWT Spectrogram', 
                 figsize: tuple = (10, 4),
                 ax=None
                 ):
        """Plots the CWT spectrogram of the provided signal.

        Parameters:
        - signal (np.ndarray): Input signal for which the CWT spectrogram will be plotted.
        - cmap (str): Colormap to use for the spectrogram. Default is 'seismic'.
        """
        # Compute CWT
        coef, freq = self.get_cwt_spectrogram(signal)  
        
        if ax is None:
           fig , ax = plt.subplots(figsize=figsize)

        cax = ax.pcolormesh(time, freq, coef, cmap=cmap, shading='auto')

        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')
        ax.set_title(title)
        
        # Add colorbar only if not using external axis
        if ax is None:
            fig.colorbar(cax, ax=ax, label = self.__label)
            plt.show()
        else:
            plt.colorbar(cax, ax=ax, label = self.__label)
            

    def plot_cwt_multi(self,
                    time: np.ndarray,
                    signal_x: np.ndarray,
                    signal_y: np.ndarray,
                    signal_z: np.ndarray,
                    titles: list = ['Sinal X', 'Sinal Y', 'Sinal Z'],
                    cmap: str = 'seismic',
                    figsize: tuple = (16, 4),
                    w_pad: float = 0.2
                    ):
        signals = [signal_x, signal_y, signal_z]
        num_plots = len(signals)

        if len(titles) != num_plots:
            raise ValueError(f"Length of titles ({len(titles)}) must match the number of signals ({num_plots}).")

        fig, axes = plt.subplots(1, num_plots, figsize=figsize, sharey=True)

        # Garantir que axes é uma lista (se tiver apenas 1 plot, não será array)
        if num_plots == 1:
            axes = [axes]

        # Pré-cálculo dos coeficientes para normalizar escala de cores
        all_coefs = [self.get_cwt_spectrogram(sig)[0] for sig in signals]
        vmin = min(np.min(c) for c in all_coefs)
        vmax = max(np.max(c) for c in all_coefs)

        for i, (ax, signal, title) in enumerate(zip(axes, signals, titles)):
            coef, freq = self.get_cwt_spectrogram(signal)
            im = ax.pcolormesh(time, freq, coef, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Tempo (s)', fontsize=14)
            if i == 0:
                ax.set_ylabel('Frequência (Hz)', fontsize=14)

            # Adiciona o colorbar ao último subplot
            if i == num_plots - 1:
                divider = make_axes_locatable(ax) # make_axes_locatable cria um espaço reservado para o colorbar, mantendo o layout dos plots | evita conflitos com o tight_layout
                cax = divider.append_axes("right", size="3%", pad=0.2)
                cbar = fig.colorbar(im, cax=cax)
                cbar.set_label(self.__label, fontsize=14)

        fig.tight_layout(w_pad=w_pad)
        plt.show()


    def run(self, signal: np.ndarray) -> List[float]:
        """Runs the CWT analysis on the provided signal and returns the spectrogram as a flattened list.
        """
        coef, _ = self.get_cwt_spectrogram(signal)
        return coef.flatten().tolist()
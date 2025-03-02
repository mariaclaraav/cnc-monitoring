from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import probplot
import pandas as pd
from phm_feature_lab.utils.filter.filter import perform_fft

class PlotIMFs:
    def __init__(self, n_imfs: int = 5, color_map: str = 'viridis') -> None:
        """
        Initializes the class with the number of IMFs and the color map.

        Args:
            n_imfs (int, optional): Number of IMFs (default: 5).
            color_map (str, optional): Name of the color map (default: 'viridis').
        """
        self.__n_imfs = n_imfs
        self.__color_map = color_map

    def __set_color_map(self, n: int) -> List[Tuple[float, float, float, float]]:
        """
        Defines the color map for the plot based on the number of IMFs.

        Args:
            n (int): Number of IMFs.

        Returns:
            List[Tuple[float, float, float, float]]: List of colors for the plot.
        """
        color_map = plt.get_cmap(self.__color_map)
        return [color_map(i / n) for i in range(n)]

    def _generate_feature_names(self, axis: str) -> List[str]:
        """
        Generates a list of feature names for a given axis and number of IMFs.

        Args:
            axis (str): Axis identifier ('X', 'Y', or 'Z').

        Returns:
            List[str]: List of feature names (e.g., ['X_IMF1', 'X_IMF2', ..., 'X_IMF5']).
        """
        return [f"{axis}_IMF{i}" for i in range(1, self.__n_imfs + 1)]

    def __get_fft(self, df: pd.DataFrame, feature: str, fs: int) -> Tuple[List[float], List[float]]:
        """
        Performs FFT analysis for a specific feature.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature (str): Name of the feature column.
            fs (int): Sampling frequency.

        Returns:
            Tuple[List[float], List[float]]: Frequencies and amplitudes.
        """
        return perform_fft(df[feature].values, fs)

    def plot_fft(
        self,
        df: pd.DataFrame,
        fs: int,
        axis: str,
        figsize: Tuple[int, int] = (10, 6),
        fontsize: int = 15,
        legend_fontsize: int = 12,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Plots the FFT for features of a specific axis in a single 2D graph.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            fs (int): Sampling frequency.
            axis (str): Axis to be plotted ('X', 'Y', or 'Z').
            figsize (Tuple[int, int], optional): Size of the figure (default: (10, 6)).
            fontsize (int, optional): Font size for labels and title (default: 15).
            extra_info (Optional[str], optional): Additional information for the title (default: None).

        Returns:
            None: Displays the FFT plot.
        """
        # Generate feature names
        features = self._generate_feature_names(axis)
        
        # Define the color map
        color_map = self.__set_color_map(len(features))
    
                
        # Configure the plot
        plt.figure(figsize=figsize)
        
        # Plot the FFT for each feature
        for i, feature in enumerate(features):
            freqs, amps = self.__get_fft(df, feature, fs)
            plt.plot(freqs, amps, label=feature, color=color_map[i])
        
        # Add title and labels
        if extra_info:
            plt.title(f"FFT IMFs - {axis}-Axis ({extra_info})", fontsize=fontsize)
        else:    
            plt.title(f"FFT IMFs - {axis}-Axis", fontsize=fontsize)
            
        plt.xlabel("Frequência (Hz)", fontsize=fontsize)
        plt.ylabel("Amplitude", fontsize=fontsize)
        plt.legend(fontsize=legend_fontsize, loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    
    def plot_single_fft(
        self,
        df: pd.DataFrame,
        fs: int,
        axis: str,
        figsize: Tuple[int, int] = (10, 6),
        fontsize: int = 15,
        legend_fontsize: int = 12,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Plots the FFT for a single feature in a 2D graph.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            fs (int): Sampling frequency.
            feature (str): Name of the feature column to plot.
            figsize (Tuple[int, int], optional): Size of the figure (default: (10, 6)).
            fontsize (int, optional): Font size for labels and title (default: 15).
            extra_info (Optional[str], optional): Additional information for the title (default: None).

        Returns:
            None: Displays the FFT plot.
        """
        # Generate feature names
        features = self._generate_feature_names(axis)
        
        # Define the color map
        color_map = self.__set_color_map(len(features))
        
        max_amplitude = 0
        for feature in features:
            _, amps = self.__get_fft(df, feature, fs)
            current_max = max(amps)
            if current_max > max_amplitude:
                max_amplitude = current_max
                
        # Plot each IMF in a separate graph
        for i, feature in enumerate(features):
            # Create a new figure with constrained_layout to ensure consistent sizing
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            
            # Perform FFT and plot
            freqs, amps = self.__get_fft(df, feature, fs)
            ax.plot(freqs, amps, label=feature, color=color_map[i])
            
            # Add title and labels
            if extra_info:
                ax.set_title(f"FFT - {feature} ({extra_info})", fontsize=fontsize)
            else:    
                ax.set_title(f"FFT - {feature}", fontsize=fontsize)
            
            # Set y-axis limits
            ax.set_ylim(0, max_amplitude * 1.1)  # 10% padding for better visualization
            
            ax.set_xlabel("Frequência (Hz)", fontsize=fontsize)
            ax.set_ylabel("Amplitude", fontsize=fontsize)
            ax.legend(fontsize=legend_fontsize, loc='upper right')
            ax.grid(True)
            
            # Ensure tight layout to avoid overlapping elements
            plt.tight_layout()
            
            # Show the plot for the current IMF
            plt.show()
    def __get_ylims(
        self,
        df: pd.DataFrame,
        features: List[str],
        padding: float = 0.1
    ) -> Tuple[float, float]:
        """
        Calculates the minimum and maximum amplitude limits for a set of features,
        with optional padding.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            features (List[str]): List of feature column names.
            padding (float, optional): Padding percentage around the min and max values (default: 0.1).

        Returns:
            Tuple[float, float]: Minimum and maximum amplitude limits with padding.
        """
        min_amplitude = float('inf')
        max_amplitude = float('-inf')
        
        for feature in features:
            current_min = df[feature].min()
            current_max = df[feature].max()
            if current_min < min_amplitude:
                min_amplitude = current_min
            if current_max > max_amplitude:
                max_amplitude = current_max
        
        # Add padding to the y-axis limits
        y_min = min_amplitude * (1 - padding) if min_amplitude < 0 else min_amplitude * (1 + padding)
        y_max = max_amplitude * (1 + padding) if max_amplitude > 0 else max_amplitude * (1 - padding)
        
        return y_min, y_max
       
    def plot_single_time_domain(
        self,
        df: pd.DataFrame,
        axis: str,
        figsize: Tuple[int, int] = (10, 6),
        fontsize: int = 15,
        legend_fontsize: int = 12,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Plots the time-domain signal for each feature of a specific axis in separate graphs.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            axis (str): Axis to be plotted ('X', 'Y', or 'Z').
            figsize (Tuple[int, int], optional): Size of the figure (default: (10, 6)).
            fontsize (int, optional): Font size for labels and title (default: 15).
            legend_fontsize (int, optional): Font size for the legend (default: 12).
            extra_info (Optional[str], optional): Additional information for the title (default: None).

        Returns:
            None: Displays the time-domain plots.
        """
        # Generate feature names
        features = self._generate_feature_names(axis)
        
        # Define the color map
        color_map = self.__set_color_map(len(features))
        
        y_min, y_max = self.__get_ylims(df, features)
                
        # Plot each feature in a separate graph
        for i, feature in enumerate(features):
            # Create a new figure with constrained_layout to ensure consistent sizing
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            
            # Plot the time-domain signal
            ax.plot(df[feature], label=feature, color=color_map[i])
            ax.set_ylim(y_min, y_max)
            
            # Add title and labels
            if extra_info:
                ax.set_title(f"Domínio do tempo - {feature} ({extra_info})", fontsize=fontsize)
            else:    
                ax.set_title(f"Domínio do tempo - {feature}", fontsize=fontsize)
            
            ax.set_xlabel("índice", fontsize=fontsize)
            ax.set_ylabel("Amplitude [a.u.]", fontsize=fontsize)
            ax.legend(fontsize=legend_fontsize, loc='upper right')
            ax.grid(True)
            
            # Ensure tight layout to avoid overlapping elements
            plt.tight_layout()
            
            # Show the plot for the current feature
            plt.show()

    def plot_time_domain(
        self,
        df: pd.DataFrame,
        axis: str,
        figsize: Tuple[int, int] = (10, 6),
        fontsize: int = 15,
        legend_fontsize: int = 12,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Plots the time-domain signal for all features of a specific axis in a single graph.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            axis (str): Axis to be plotted ('X', 'Y', or 'Z').
            figsize (Tuple[int, int], optional): Size of the figure (default: (10, 6)).
            fontsize (int, optional): Font size for labels and title (default: 15).
            legend_fontsize (int, optional): Font size for the legend (default: 12).
            extra_info (Optional[str], optional): Additional information for the title (default: None).

        Returns:
            None: Displays the time-domain plot.
        """
        # Generate feature names
        features = self._generate_feature_names(axis)
        
        # Define the color map
        color_map = self.__set_color_map(len(features))
        
        # Create a single figure for all features
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        
        # Plot the time-domain signal for each feature
        for i, feature in enumerate(features):
            ax.plot(df[feature], label=feature, color=color_map[i])
        
        # Add title and labels
            if extra_info:
                ax.set_title(f"Domínio do tempo - {feature} ({extra_info})", fontsize=fontsize)
            else:    
                ax.set_title(f"Domínio do tempo - {feature}", fontsize=fontsize)
            
            ax.set_xlabel("índice", fontsize=fontsize)
            ax.set_ylabel("Amplitude [a.u.]", fontsize=fontsize)
            ax.legend(fontsize=legend_fontsize, loc='upper right')
            ax.grid(True)
        
        # Ensure tight layout to avoid overlapping elements
        plt.tight_layout()
        
        # Show the plot
        plt.show()
        
    def qq_plots(
        self,
        df: pd.DataFrame,
        axis: str,
        figsize: Tuple[int, int] = (10, 6),
        fontsize: int = 15,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Plots Q-Q plots for each feature of a specific axis.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            axis (str): Axis to be plotted ('X', 'Y', or 'Z').
            figsize (Tuple[int, int], optional): Size of the figure (default: (10, 6)).
            fontsize (int, optional): Font size for labels and title (default: 15).
            extra_info (Optional[str], optional): Additional information for the title (default: None).

        Returns:
            None: Displays the Q-Q plots.
        """
        # Generate feature names
        features = self._generate_feature_names(axis)
        
        # Plot Q-Q plot for each feature
        for feature in features:
            # Create a new figure
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            
            # Plot the Q-Q plot
            probplot(df[feature], dist="norm", plot=ax)
            
            # Add title and labels
            if extra_info:
                ax.set_title(f"Q-Q Plot - {feature} ({extra_info})", fontsize=fontsize)
            else:
                ax.set_title(f"Q-Q Plot - {feature}", fontsize=fontsize)
            
            ax.set_xlabel("Quantis Teóricos", fontsize=fontsize)
            ax.set_ylabel("Quantis Amostrais", fontsize=fontsize)
            ax.grid(True)
            
            # Ensure tight layout to avoid overlapping elements
            plt.tight_layout()
            
            # Show the plot for the current feature
            plt.show()
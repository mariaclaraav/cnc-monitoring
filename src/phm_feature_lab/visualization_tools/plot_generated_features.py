import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Union
from scipy.stats import probplot
import math
from scipy.signal import welch
from phm_feature_lab.frequency.filter import perform_fft


class PlotGeneratedFeatures:
    def __init__(self, color_map: Union[str, List[str]] = 'viridis', line_styles: Optional[List[str]] = None) -> None:
        """
        Initializes the base class with the color map and line styles.

        Args:
            color_map (Union[str, List[str]], optional): Name of the color map (str) or list of colors (List[str]).
                                                         Defaults to 'viridis'.
            line_styles (Optional[List[str]], optional): List of line styles. Defaults to None.
        """
        self.__color_map = color_map
        self.__line_styles = line_styles

    def _set_color_map(self, n: int) -> List[Tuple[float, float, float, float]]:
        """
        Defines the color map for the plot based on the number of features.

        Args:
            n (int): Number of features.

        Returns:
            List[Tuple[float, float, float, float]]: List of colors for the plot.
        """
        if isinstance(self.__color_map, str):  # Use a colormap
            color_map = plt.get_cmap(self.__color_map)
            return [color_map(i / n) for i in range(n)]
        elif isinstance(self.__color_map, list):  # Use a list of colors
            if len(self.__color_map) < n:
                raise ValueError(f"Not enough colors in the color list.  Need at least {n}, got {len(self.__color_map)}.")
            return self.__color_map[:n] # Return the first n colors
        else:
            raise TypeError("color_map must be a string (colormap name) or a list of colors.")

    def _set_line_styles(self, n: int) -> List[str]:
        """
        Defines the line styles for the plot based on the number of features.

        Args:
            n (int): Number of features.

        Returns:
            List[str]: List of line styles for the plot.
        """
        default_line_styles = ['-', '--', '-.', ':']
        if self.__line_styles is None:
            # Repeat default line styles if n is greater than the number of default styles
            return (default_line_styles * (n // len(default_line_styles) + 1))[:n]
        elif len(self.__line_styles) < n:
            raise ValueError(f"Not enough line styles in the list. Need at least {n}, got {len(self.__line_styles)}.")
        else:
            return self.__line_styles[:n]

    def _generate_feature_names(self, axis: str) -> List[str]:
        """
        Generates a list of feature names for a given axis.
        This method should be overridden by child classes.

        Args:
            axis (str): Axis identifier ('X', 'Y', or 'Z').

        Returns:
            List[str]: List of feature names.
        """
        raise NotImplementedError("This method should be overridden by child classes.")

    def _get_fft(self, df: pd.DataFrame, feature: str, fs: int) -> Tuple[List[float], List[float]]:
        """
        Performs FFT analysis for a specific feature.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            feature (str): Name of the feature column.
            fs (int): Sampling frequency.

        Returns:
            Tuple[List[float], List[float]]: Frequencies and amplitudes.
        """
        # Assuming perform_fft is a predefined function
        return perform_fft(df[feature].values, fs)

    def plot_fft(
        self,
        df: pd.DataFrame,
        fs: int,
        axis: str,
        figsize: Tuple[int, int] = (12, 4),
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
        color_map = self._set_color_map(len(features))
        
        # Define the line styles
        line_styles = self._set_line_styles(len(features))
    
        # Configure the plot
        plt.figure(figsize=figsize)
        
        # Plot the FFT for each feature
        for i, feature in enumerate(features):
            freqs, amps = self._get_fft(df, feature, fs)
            plt.plot(freqs, amps, label=feature, color=color_map[i], linestyle=line_styles[i])
        
        # Add title and labels
        title = f"FFT - {axis}-Axis"
        if extra_info:
            title += f" ({extra_info})"
        plt.title(title, fontsize=fontsize)
            
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
        color_map = self._set_color_map(len(features))
        
        # Define the line styles
        line_styles = self._set_line_styles(len(features))
        
        max_amplitude = 0
        for feature in features:
            _, amps = self._get_fft(df, feature, fs)
            current_max = max(amps)
            if current_max > max_amplitude:
                max_amplitude = current_max
                
        # Plot each feature in a separate graph
        for i, feature in enumerate(features):
            # Create a new figure with constrained_layout to ensure consistent sizing
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            
            # Perform FFT and plot
            freqs, amps = self._get_fft(df, feature, fs)
            ax.plot(freqs, amps, label=feature, color=color_map[i], linestyle=line_styles[i])
            
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
            
            # Show the plot for the current feature
            plt.show()

    def _get_ylims(
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
        extra_info: Optional[str] = None,
        xaxis: Optional[List[int]] = None
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
        color_map = self._set_color_map(len(features))
        
        # Define the line styles
        line_styles = self._set_line_styles(len(features))
        
        y_min, y_max = self._get_ylims(df, features)
            
        # Plot each feature in a separate graph
        for i, feature in enumerate(features):
            # Create a new figure with constrained_layout to ensure consistent sizing
            fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
            if xaxis is not None:
                # Plot the time-domain signal
                if len(xaxis) == len(df[feature]):
                    ax.plot(xaxis, df[feature], label=feature, color=color_map[i], linestyle=line_styles[i])
                else:
                    raise ValueError("Length of xaxis must match the length of the feature data.")
            else:
                # Plot the time-domain signal
                ax.plot(df[feature], label=feature, color=color_map[i], linestyle=line_styles[i])
            ax.set_ylim(y_min, y_max)
            
            # Add title and labels
            if extra_info:
                ax.set_title(f"Domínio do tempo - {feature} ({extra_info})", fontsize=fontsize)
            else:    
                ax.set_title(f"Domínio do tempo - {feature}", fontsize=fontsize)
            
            ax.set_xlabel("Indice" if xaxis is None else "Time", fontsize=fontsize)
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
        figsize: Tuple[int, int] = (12, 8),  # Increased default figsize
        fontsize: int = 12,
        legend_fontsize: int = 10,
        extra_info: Optional[str] = None,
        xaxis: Optional[List[int]] = None
    ) -> None:
        """
        Plots the time-domain signal for all features of a specific axis in separate subplots,
        arranged in 2 columns per row.  Subplots share the x-axis.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            axis (str): Axis to be plotted ('X', 'Y', or 'Z').
            figsize (Tuple[int, int], optional): Size of the *entire* figure (default: (12, 8)).
            fontsize (int, optional): Font size for labels and title (default: 12).
            legend_fontsize (int, optional): Font size for the legend (default: 10).
            extra_info (Optional[str], optional): Additional information for the title (default: None).

        Returns:
            None: Displays the time-domain plot with subplots.
        """
        import numpy as np

        # Generate feature names
        features = self._generate_feature_names(axis)
        num_features = len(features)

        # Define the color map
        color_map = self._set_color_map(num_features)

        # Define the line styles
        line_styles = self._set_line_styles(num_features)

        # Determine the number of rows needed for the subplots
        num_rows = math.ceil(num_features / 2)

        # Calculate the common y-axis limits
        ymin = np.min([df[feature].min() for feature in features])
        ymax = np.max([df[feature].max() for feature in features])
        
        # Create the figure and subplots
        fig, axes = plt.subplots(num_rows, 2, figsize=figsize, constrained_layout=True, sharex=True)

        # Flatten the axes array for easier indexing if num_rows is 1
        if num_rows == 1:
            axes = axes.reshape(1, -1)  # Reshape to 2D array even with 1 row

        # Plot the time-domain signal for each feature in its own subplot
        for i, feature in enumerate(features):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            if xaxis is not None:
                # Plot the time-domain signal with xaxis
                if len(xaxis) == len(df[feature]):
                    ax.plot(xaxis, df[feature], color=color_map[i], linestyle=line_styles[i], label=feature)
                else:
                    raise ValueError("Length of xaxis must match the length of the feature data.")
            else:
                ax.plot(df[feature], color=color_map[i], linestyle=line_styles[i], label=feature) #Plot the feature

            # Add title and labels to each subplot
            #title = f"{feature}"  # Subplot Title
            #ax.set_title(title, fontsize=fontsize - 2)  # Slightly smaller font

            ax.set_xlabel("Indice" if xaxis is None else "Time", fontsize=fontsize - 2) #Reduce xlabel fontsize
            ax.set_ylabel("Amplitude [a.u.]", fontsize=fontsize - 2) #Reduce ylabel fontsize
            ax.set_ylim([ymin*1.1, ymax*1.1]) #Set y-axis limits with padding
            ax.legend(fontsize=legend_fontsize - 2, loc='upper right') #Smaller legend
            ax.grid(True)

        # Remove any unused subplots
        for i in range(num_features, num_rows * 2):
            row = i // 2
            col = i % 2
            fig.delaxes(axes[row, col])

        # Add a common title to the entire figure
        fig.suptitle(f"Domínio do tempo - {axis}-Axis" + (f" ({extra_info})" if extra_info else ""), fontsize=fontsize + 2)


        # Show the plot
        plt.show()

    def plot_psd(
        self,
        df: pd.DataFrame,
        fs: int,
        axis: str,
        figsize: Tuple[int, int] = (12, 4),
        fontsize: int = 15,
        legend_fontsize: int = 12,
        extra_info: Optional[str] = None
    ) -> None:
        """
        Plots the Power Spectral Density (PSD) for features of a specific axis in a single 2D graph.

        Args:
            df (pd.DataFrame): DataFrame containing the data.
            fs (int): Sampling frequency.
            axis (str): Axis to be plotted ('X', 'Y', or 'Z').
            figsize (Tuple[int, int], optional): Size of the figure (default: (12, 4)).
            fontsize (int, optional): Font size for labels and title (default: 15).
            legend_fontsize (int, optional): Font size for the legend (default: 12).
            extra_info (Optional[str], optional): Additional information for the title (default: None).

        Returns:
            None: Displays the PSD plot.
        """
        # Generate feature names
        features = self._generate_feature_names(axis)

        # Define the color map
        color_map = self._set_color_map(len(features))

        # Define the line styles
        line_styles = self._set_line_styles(len(features))

        # Configure the plot
        plt.figure(figsize=figsize)

        # Plot the PSD for each feature
        for i, feature in enumerate(features):
            f, psd = welch(df[feature].values, fs=fs)
            plt.plot(f, 10 * np.log10(psd), label=feature, color=color_map[i], linestyle=line_styles[i])  # Plot in dB scale

        # Add title and labels
        title = f"PSD - {axis}-Axis"
        if extra_info:
            title += f" ({extra_info})"
        plt.title(title, fontsize=fontsize)

        plt.xlabel("Frequência (Hz)", fontsize=fontsize)
        plt.ylabel("Densidade Espectral de Potência (dB/Hz)", fontsize=fontsize)  # Corrected y-axis label
        plt.legend(fontsize=legend_fontsize, loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
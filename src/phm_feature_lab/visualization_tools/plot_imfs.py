from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import probplot
import pandas as pd
from phm_feature_lab.visualization_tools.plot_generated_features import PlotGeneratedFeatures

class PlotIMFs(PlotGeneratedFeatures):
    def __init__(self, n_imfs: int = 5, color_map: Union[str, List[str]] = 'viridis', line_styles: Optional[List[str]] = None) -> None:
        """
        Initializes the class with the number of IMFs and the color map.

        Args:
            n_imfs (int, optional): Number of IMFs (default: 5).
            color_map (str, optional): Name of the color map (default: 'viridis').
        """
        super().__init__(color_map, line_styles)
        self.__n_imfs = n_imfs

    def _generate_feature_names(self, axis: str) -> List[str]:
        """
        Generates a list of feature names for a given axis and number of IMFs.

        Args:
            axis (str): Axis identifier ('X', 'Y', or 'Z').

        Returns:
            List[str]: List of feature names (e.g., ['X_IMF1', 'X_IMF2', ..., 'X_IMF5']).
        """
        return [f"{axis}_IMF{i}" for i in range(1, self.__n_imfs + 1)]
    

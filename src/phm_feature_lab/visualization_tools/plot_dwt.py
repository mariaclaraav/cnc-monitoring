from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import probplot
import pandas as pd
from phm_feature_lab.visualization_tools.plot_generated_features import PlotGeneratedFeatures

class PlotDWT(PlotGeneratedFeatures):
    def __init__(self, level: int = 3, color_map: Union[str, List[str]] = 'viridis', line_styles: Optional[List[str]] = None) -> None:
        """
        Initializes the class with the decomposition level, the color map, and line styles.

        Args:
            level (int, optional): Decomposition level (default: 3).
            color_map (str, optional): Name of the color map (default: 'viridis').
            line_styles (Optional[List[str]], optional): List of line styles. Defaults to None.
        """
        super().__init__(color_map, line_styles)  # Pass line_styles to the superclass
        self.__level = level

    def _generate_feature_names(self, axis: str) -> List[str]:
        """
        Generates a list of feature names for a given axis and decomposition level.

        Args:

        Returns:
            List[str]: List of feature names (e.g., ['X_D1', 'X_D2', ..., 'X_D3', 'X_A3']).
        """
        detail_features = [f"{axis}_D{i}" for i in range(1, self.__level + 1)]
        approximation_feature = f"{axis}_A{self.__level}"
        return detail_features + [approximation_feature]
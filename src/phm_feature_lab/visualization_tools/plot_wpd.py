
from typing import List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.stats import probplot
import pandas as pd
from phm_feature_lab.visualization_tools.plot_generated_features import PlotGeneratedFeatures

class PlotWPD(PlotGeneratedFeatures):
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
            axis (str): Axis identifier ('X', 'Y', or 'Z').

        Returns:
            List[str]: List of feature names (e.g., ['X_node_aaa', 'X_node_aad', ...]).
        """
        import itertools
        nodes = [''.join(node) for node in itertools.product('ad', repeat=self.__level)]
        return [f"{axis}_node_{node}" for node in nodes]
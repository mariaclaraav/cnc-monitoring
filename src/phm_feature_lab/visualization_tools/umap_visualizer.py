from typing import List, Optional
import pandas as pd
import umap
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt


class UMAPVisualizer:
    def __init__(self, n_neighbors: int = 25, n_components: int = 2, subsample: int = 100):
        """
        Initialize the UMAPVisualizer with default parameters.

        Parameters:
            n_neighbors (int): Number of neighbors for UMAP. Default is 25.
            n_components (int): Number of dimensions in the UMAP output. Default is 2.
            random_state (int): Random state for reproducibility. Default is 42.
            subsample (int): Subsample factor for performance. Default is 100.
        """
        self.__n_neighbors: int = n_neighbors
        self.__n_components: int = n_components
        self.__subsample: int = subsample
        self.__scaler: StandardScaler = StandardScaler()
        self.__umap_reducer: umap.UMAP = umap.UMAP(
            n_neighbors=self.__n_neighbors,
            n_components=self.__n_components,
            n_jobs=-1
        )
        self.df_subsampled: Optional[pd.DataFrame] = None

    def fit_transform(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Fit the UMAP reducer and transform the data.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data.
            features (List[str]): List of column names to use as features.

        Returns:
            pd.DataFrame: Transformed data with UMAP components.
        """
        # Scale the data
        data_scaled = self.__scaler.fit_transform(df[features])

        # Subsample for performance
        if self.__subsample > 1:
            data_scaled = data_scaled[::self.__subsample]
            self.df_subsampled = df.iloc[::self.__subsample].reset_index(drop=True)
        else:
            self.df_subsampled = df

        # Apply UMAP
        umap_result = self.__umap_reducer.fit_transform(data_scaled)

        # Return as a DataFrame
        umap_df = pd.DataFrame(umap_result, columns=[f'UMAP{i+1}' for i in range(self.__n_components)])
        return umap_df

    def visualize(self, umap_df: pd.DataFrame, color_column: str, hover_columns: Optional[List[str]] = None) -> go.Figure:
        """
        Create a Plotly visualization for the UMAP results.

        Parameters:
            umap_df (pd.DataFrame): DataFrame containing UMAP components.
            color_column (str): Column name to color the points.
            hover_columns (Optional[List[str]]): Additional columns to display on hover.

        Returns:
            go.Figure: Generated figure.
        """
        if self.df_subsampled is None:
            raise ValueError("The fit_transform method must be called before visualize.")

        # Add the color column to the DataFrame
        umap_df[color_column] = self.df_subsampled[color_column].values

        # Add hover columns if specified
        if hover_columns:
            for col in hover_columns:
                umap_df[col] = self.df_subsampled[col].values

        # Create Plotly scatter plot
        fig = px.scatter(
            umap_df,
            x='UMAP1',
            y='UMAP2',
            color=color_column,
            hover_data=hover_columns or [],
            title='UMAP Visualization',
            labels={'UMAP1': 'UMAP Component 1', 'UMAP2': 'UMAP Component 2'}
        )
        fig.update_traces(marker=dict(size=4))
        return fig
    
    def visualize_matplotlib(self, umap_df: pd.DataFrame, color_column: str, s: int = 8) -> None:
        """
        Create a Matplotlib visualization for the UMAP results.

        Parameters:
            umap_df (pd.DataFrame): DataFrame containing UMAP components.
            color_column (str): Column name to color the points.
        """
        if self.df_subsampled is None:
            raise ValueError("The fit_transform method must be called before visualize_matplotlib.")

        # Add the color column to the DataFrame
        umap_df[color_column] = self.df_subsampled[color_column].values

        # Matplotlib scatter plot
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            umap_df['UMAP1'], umap_df['UMAP2'], c=umap_df[color_column], cmap='viridis', s = s, alpha=0.8
        )
        plt.colorbar(scatter, label=color_column)
        plt.title('UMAP Visualization')
        plt.xlabel('UMAP 1')
        plt.ylabel('UMAP 2')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()


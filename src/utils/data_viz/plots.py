import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import List, Tuple, Union, Optional



def plot_original_vs_reconstructed(X_test, X_test_reconstructed, sample_index=1, data = 'Test', scaler = None):
    """
    Plot original vs reconstructed signal for a sample.
    """
    if scaler:
        X_test = scaler.inverse_transform(X_test)
        X_test_reconstructed = scaler.inverse_transform(X_test_reconstructed)
    plt.figure(figsize=(13, 5))
    plt.plot(X_test[:, sample_index], label='Original', color='blue')
    plt.plot(X_test_reconstructed[:, sample_index], label='Reconstructed', linestyle='dashed', color='red')
    plt.xlabel('Index')
    plt.ylabel('Signal Value')
    plt.title(f'Original vs Reconstructed - {data} Set')
    plt.legend(loc='upper right')
    plt.show()
    
def split_hitogram(X_train, X_val, column_index = 1, ylim = -0.1):
    
    plt.figure(figsize=(10, 4))

    sns.histplot(X_train[:,column_index], color='blue', kde=True, label='train', stat='density', common_norm=False)
    sns.histplot(X_val[:,column_index], color='orange', 
                kde=True, label='val', stat='density', common_norm=False)

    plt.title('train and val distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.ylim([ylim,None])
    plt.legend()

    plt.show()


class AnomalyDetectionVisualizer:
    def __init__(self, scaler=None):
        self.scaler = scaler

    def ensure_numpy_array(self, data):
        """
        Ensure the data is a numpy array. If not, convert it to numpy array.
        """
        if not isinstance(data, np.ndarray):
            return data.values
        return data

    def plot_anomalies(self, X_val, y_val, y_pred_val, column_index, data):
        """
        Plot the anomalies detected by the model along with the known anomalies.

        :param X_val: Validation input data (can be a DataFrame or numpy array)
        :param y_val: Known anomaly labels (1 for anomaly, 0 for normal)
        :param y_pred_val: Predicted anomaly labels by the model (1 for anomaly, 0 for normal)
        :param column_index: Index of the column to be plotted
        :param data: A string indicating the type of data (used in the plot title)
        """
        X_val = self.ensure_numpy_array(X_val)
        y_val = self.ensure_numpy_array(y_val)
        y_pred_val = np.array(y_pred_val)
        
        if self.scaler:
            X_val = self.scaler.inverse_transform(X_val)

        # Select the specified column for plotting
        X_val = X_val[:, column_index]

        # Plot the data
        plt.figure(figsize=(12, 4))
        plt.plot(X_val, color='blue', label=f'{data} data', zorder=2)

        # Plot the anomalies detected by the model
        plt.scatter(np.where(y_pred_val == 1)[0], X_val[np.where(y_pred_val == 1)[0]], color='orange', s=40, label='Model', zorder=3)

        # Highlight known anomalies
        ymin, ymax = plt.ylim()
        plt.fill_between(np.arange(len(X_val)), ymin, ymax, where=y_val == 1, color='red', alpha=0.3, label='Known', zorder=1)

        # Add labels and title
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Detected Anomalies - {data} data')
        plt.show()
        
        

def plot_features(
    df: Union[pd.DataFrame, 'dask.dataframe.DataFrame'], 
    cols: List[str], 
    is_dask: bool = False, 
    normalize: bool = False, 
    scaler_type: str = 'standard', 
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Plots X, Y, Z axes data with optional normalization.

    Parameters:
    - df: Union[pd.DataFrame, dask.dataframe.DataFrame]
        The DataFrame containing the data to plot.
    - cols: List[str]
        List of base column names to plot for each axis (X, Y, Z).
    - is_dask: bool, optional (default=False)
        Whether the DataFrame is a Dask DataFrame. If True, the data will be computed.
    - normalize: bool, optional (default=False)
        Whether to normalize the data before plotting.
    - scaler_type: str, optional (default='standard')
        Type of scaler to use if normalization is True. Options: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler).
    - figsize: Tuple[int, int], optional (default=(15, 10))
        The size of the figure for the plots.
    
    Raises:
    - ValueError: If an invalid `scaler_type` is provided.
    """

    def apply_scaler(data: pd.DataFrame) -> pd.DataFrame:
        """Apply the chosen scaler to normalize the data."""
        if scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Invalid scaler_type '{scaler_type}'. Choose 'standard' or 'minmax'.")
        
        scaled_data = scaler.fit_transform(data)
        return pd.DataFrame(scaled_data, columns=data.columns)

    def process_data() -> pd.DataFrame:
        """Process the DataFrame by computing (if Dask) and normalizing (if needed)."""
        if is_dask:
            local_df = df[all_cols].compute()
        else:
            local_df = df[all_cols]
        
        if normalize:
            local_df = apply_scaler(local_df)

        return local_df

    # Define columns to process for X, Y, Z axes
    all_cols = [f'X_{col}' for col in cols] + [f'Y_{col}' for col in cols] + [f'Z_{col}' for col in cols]

    # Process data (compute and normalize if needed)
    df_processed = process_data()

    # Create subplots for X, Y, and Z axes
    fig, axs = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Plot for each axis
    for i, axis in enumerate(['X', 'Y', 'Z']):
        for col in cols:
            axis_col = f'{axis}_{col}'
            if axis_col in df_processed.columns:
                axs[i].plot(df_processed[axis_col], label=axis_col)
        axs[i].set_title(f'{axis}-axis Features')
        axs[i].legend(loc='upper right')
        axs[i].grid(True)

    plt.tight_layout()
    plt.show()

def plot_columns(
    df: pd.DataFrame,
    columns: List[str],
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 6),
    label_column: str = "Label",  # Specify the name of the column to use for shading
    highlight_label: bool = False  # New parameter to toggle the red shading
) -> None:

    fig, ax = plt.subplots(figsize=figsize)

    # Plotting each column
    for column in columns:
        if column in df.columns:
            ax.plot(df.index, df[column], label=column)
        else:
            print(f"Warning: Column '{column}' not found in DataFrame.")

    # Adding the red shaded background
    if highlight_label and label_column in df.columns:
        # Find ranges where the label column equals 1

        is_label = df[label_column] == 1 
        for i in range(len(is_label) - 1):
            if is_label.iloc[i] and not is_label.iloc[i - 1]:
                start = df.index[i]
            if is_label.iloc[i] and not is_label.iloc[i + 1]:
                end = df.index[i + 1]
                ax.axvspan(start, end, color="red", alpha=0.2, label="_nolegend_")  # "_nolegend_" hides it from legend

    # Adding labels, title, and legend
    ax.set_title(title if title else "Plot of Columns")
    ax.set_xlabel(xlabel if xlabel else "Index")
    ax.set_ylabel(ylabel if ylabel else "Value")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    
class FeaturePlotter:
    def __init__(
        self, 
        df: Union[pd.DataFrame, 'dask.dataframe.DataFrame'], 
        features: List[str], 
        is_dask: bool = False, 
        normalize: bool = False, 
        scaler_type: str = 'standard'
    ) -> None:
        """
        Initialize the FeaturePlotter with the DataFrame and features to plot.

        Parameters:
        - df: Union[pd.DataFrame, dask.dataframe.DataFrame]
            The DataFrame (Dask or Pandas) with the data to plot.
        - features: List[str]
            List of feature names without axis prefix (e.g., 'rms', 'mean').
        - is_dask: bool, optional (default=False)
            Whether the DataFrame is a Dask DataFrame. If True, it will be computed.
        - normalize: bool, optional (default=False)
            Whether to normalize the features.
        - scaler_type: str, optional (default='standard')
            Type of scaler to use if normalization is True. Options: 'standard' or 'minmax'.
        """
        self.df = df
        self.features = features
        self.is_dask = is_dask
        self.normalize = normalize
        self.scaler_type = scaler_type
        self.x_features = [f'X_{f}' for f in features]
        self.y_features = [f'Y_{f}' for f in features]
        self.z_features = [f'Z_{f}' for f in features]
        self.all_features = self.x_features + self.y_features + self.z_features
        self._process_data()

    def _process_data(self) -> None:
        """Process the data, handling Dask computation and normalization if specified."""
        if self.is_dask:
            self.df = self.df[self.all_features].compute()  # Convert Dask to Pandas
        else:
            self.df = self.df[self.all_features]

        if self.normalize:
            self._apply_scaler()

    def _apply_scaler(self) -> None:
        """Apply the chosen scaler to normalize the data."""
        if self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            raise ValueError(f"Invalid scaler_type '{self.scaler_type}'. Choose 'standard' or 'minmax'.")

        scaled_data = scaler.fit_transform(self.df)
        self.df = pd.DataFrame(scaled_data, columns=self.all_features)

    def kde_plot(self, figsize: Tuple[int, int] = (10, 12)) -> None:
        """
        Plot the Kernel Density Estimate (KDE) for X, Y, and Z features.

        Parameters:
        - figsize: Tuple[int, int], optional (default=(10, 12))
            The size of the figure.
        """
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        self._plot_kde(axs)

        for ax in axs:
            ax.set_xlabel('Values')
            ax.set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def boxplot(self, figsize: Tuple[int, int] = (12, 10), marker: str = 'o', markersize: int = 3) -> None:
        """
        Plot the Boxplot for X, Y, and Z features.

        Parameters:
        - figsize: Tuple[int, int], optional (default=(12, 10))
            The size of the figure.
        - marker: str, optional (default='o')
            Marker style for outliers.
        - markersize: int, optional (default=3)
            Size of the markers for outliers.
        """
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        flierprops = dict(marker=marker, markersize=markersize)
        self._plot_boxplot(axs, flierprops)

        for ax in axs:
            ax.set_xlabel('Features')
            ax.set_ylabel('Values')

        plt.tight_layout()
        plt.show()

    def violin_plot(self, figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot the Violin plot for X, Y, and Z features.

        Parameters:
        - figsize: Tuple[int, int], optional (default=(12, 10))
            The size of the figure.
        """
        fig, axs = plt.subplots(3, 1, figsize=figsize)
        self._plot_violin(axs)

        for ax in axs:
            ax.set_xlabel('Features')
            ax.set_ylabel('Values')
            ax.set_xticks(range(len(self.features)))
            ax.set_xticklabels(self.features)

        plt.tight_layout()
        plt.show()

    def _plot_kde(self, axs: List[plt.Axes]) -> None:
        """Helper function to plot KDE for X, Y, Z axis features."""
        self._plot_axis(axs[0], 'X-axis features', self.x_features)
        self._plot_axis(axs[1], 'Y-axis features', self.y_features)
        self._plot_axis(axs[2], 'Z-axis features', self.z_features)

    def _plot_boxplot(self, axs: List[plt.Axes], flierprops: dict) -> None:
        """Helper function to plot Boxplot for X, Y, Z axis features."""
        sns.boxplot(data=self.df[self.x_features], ax=axs[0], flierprops=flierprops)
        axs[0].set_title('X-axis features')

        sns.boxplot(data=self.df[self.y_features], ax=axs[1], flierprops=flierprops)
        axs[1].set_title('Y-axis features')

        sns.boxplot(data=self.df[self.z_features], ax=axs[2], flierprops=flierprops)
        axs[2].set_title('Z-axis features')

    def _plot_violin(self, axs: List[plt.Axes]) -> None:
        """Helper function to plot Violin plot for X, Y, Z axis features."""
        sns.violinplot(data=self.df[self.x_features], ax=axs[0])
        axs[0].set_title('X-axis features')

        sns.violinplot(data=self.df[self.y_features], ax=axs[1])
        axs[1].set_title('Y-axis features')

        sns.violinplot(data=self.df[self.z_features], ax=axs[2])
        axs[2].set_title('Z-axis features')

    def _plot_axis(self, ax: plt.Axes, title: str, features: List[str]) -> None:
        """Helper function to plot a specific axis (X, Y, Z) features."""
        ax.set_title(title)
        for feature in features:
            sns.kdeplot(self.df[feature], fill=True, label=feature, ax=ax)
        ax.legend(loc='upper right')
        
def plot_scatter_matrix(df, machine, process, cols, sample_frac=0.05, random_state=0):
    """
    Plots a scatter matrix for specified columns for a given process and machine in a DataFrame,
    highlighting different 'Unique_Code' values.

    Parameters:
    - df: DataFrame containing the data.
    - machine: String representing the machine to filter by.
    - process: String representing the process to filter by.
    - cols: List of column names to include in the scatter matrix.
    - sample_frac: Fraction of the DataFrame to sample (default 0.05).
    - random_state: Seed for random number generation (default 0).
    """

    # Filter data by machine and process
    df_filtered = df[(df['Machine'] == machine) & (df['Process'] == process)]

    # Ensure only columns that exist in the DataFrame are used
    cols = [col for col in cols if col in df_filtered.columns]

    # Determine the column order for 'Unique_Code'
    unique_code_order = df_filtered['Unique_Code'].unique()

    # Create the scatter matrix
    fig = px.scatter_matrix(df_filtered.sample(frac=sample_frac, random_state=random_state),
                            dimensions=cols, color='Unique_Code',
                            category_orders={'Unique_Code': list(unique_code_order)})

    # Update layout
    fig.update_layout(width=1400, height=1000, legend_title_font_size=22)
  

    # Update trace characteristics
    fig.update_traces(marker=dict(size=5), diagonal_visible=False, showupperhalf=False)

    # Display the figure
    fig.show()
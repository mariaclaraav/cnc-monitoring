from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import umap
from scipy.stats import ks_2samp


class CompareData:
    """
    A base class for comparing and visualizing data across different groups (e.g., periods or machines).
    """

    def __init__(self, filter_column: str, filter_value: str, df: pd.DataFrame) -> None:
        """
        Initialize CompareData.

        Parameters:
        - filter_column (str): Column to filter by (e.g., 'Machine' or 'Period').
        - filter_value (str): Value to filter the column by (e.g., a specific machine or period).
        - df (pd.DataFrame): DataFrame containing the data.
        """
        self.filter_column = filter_column
        self.filter_value = filter_value
        self.df = df

    def __filter_and_downsample(self, downsample_factor: int) -> pd.DataFrame:
        """
        Filter the DataFrame by the specified filter column and value, and downsample if needed.

        Parameters:
        - downsample_factor (int): Factor by which to downsample the DataFrame.

        Returns:
        - pd.DataFrame: Filtered and downsampled DataFrame.
        """
        df_filtered = self.df[self.df[self.filter_column] == self.filter_value]
        if downsample_factor > 1:
            df_filtered = df_filtered[::downsample_factor]
        return df_filtered

    def cdf_kde(
        self, cols: List[str], group_column: str, downsample_factor: int = 1
    ) -> None:
        """
        Plot CDF and KDE plots for specified columns across groups.

        Parameters:
        - cols (List[str]): Columns to plot.
        - group_column (str): Column to group by (e.g., 'Period' or 'Machine').
        - downsample_factor (int): Downsampling factor for the DataFrame. Default is 1 (no downsampling).
        """
        df_filtered = self.__filter_and_downsample(downsample_factor)
        nrows = len(cols)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(13, 5 * nrows))

        # Ensure axes is iterable for a single row
        if nrows == 1:
            axes = [axes]

        for i, col in enumerate(cols):
            # CDF Plot
            sns.ecdfplot(data=df_filtered, x=col, hue=group_column, ax=axes[i][0])
            axes[i][0].set_title(f'CDF by {group_column} - {self.filter_value}')
            axes[i][0].set_xlabel(col)
            axes[i][0].set_ylabel('CDF')

            # KDE Plot
            sns.kdeplot(
                data=df_filtered,
                x=col,
                hue=group_column,
                common_norm=False,
                linewidth=2,
                ax=axes[i][1],
            )
            axes[i][1].set_title(f'KDE by {group_column} - {self.filter_value}')
            axes[i][1].set_xlabel(col)
            axes[i][1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def ks_test(
        self, groups: List[str], data_column: str, group_column: str, downsample_factor: int = 1
    ) -> None:
        """
        Perform the Kolmogorov-Smirnov test between specified groups for a column.

        Parameters:
        - groups (List[str]): List of groups to compare (e.g., periods or machines).
        - data_column (str): Column to analyze.
        - group_column (str): Column representing the groups (e.g., 'Period' or 'Machine').
        - downsample_factor (int): Downsampling factor for the DataFrame. Default is 1 (no downsampling).
        """
        df_filtered = self.__filter_and_downsample(downsample_factor)

        if len(groups) < 2:
            print("Please provide at least two groups to compare.")
            return

        print(f'Kolmogorov-Smirnov test results for {self.filter_value}:\n')

        for i, group1 in enumerate(groups[:-1]):
            for group2 in groups[i + 1:]:
                data1 = df_filtered[df_filtered[group_column] == group1][data_column]
                data2 = df_filtered[df_filtered[group_column] == group2][data_column]

                ks_stat, p_value = ks_2samp(data1, data2)

                print(f'{group1} vs {group2}: KS Statistic = {ks_stat:.3f}, p-value = {p_value:.5f}')

    def scatter_plot(self, cols: List[str], group_column: str, downsample_factor: int = 1) -> None:
        """
        Plot a scatter matrix for specified columns.

        Parameters:
        - cols (List[str]): Columns to include in the scatter plot.
        - group_column (str): Column representing the groups (e.g., 'Period' or 'Machine').
        - downsample_factor (int): Downsampling factor for the DataFrame. Default is 1 (no downsampling).
        """
        df_filtered = self.__filter_and_downsample(downsample_factor)

        fig = px.scatter_matrix(df_filtered, dimensions=cols, color=group_column)
        fig.update_layout(width=1000, height=600, legend_title_font_size=20)
        fig.update_traces(marker=dict(size=4), diagonal_visible=False, showupperhalf=False)
        fig.show()
        
    def scatter_plot_umap(
        self,
        group_column: str,
        features: List[str],
        downsample_factor: int = 1,
        n_neighbors: int = 25,
        n_components: int = 2,
        title: str = "UMAP Visualization",
    ) -> None:
        """
        Create a UMAP scatter plot using Plotly.

        Parameters:
        - group_column (str): Column to color by in the plot (e.g., 'Period' or 'Machine').
        - features (List[str]): List of feature columns to use for UMAP.
        - downsample_factor (int): Factor by which to downsample the DataFrame.
        - title (str): Title of the plot.
        """
        # Filter and scale the data
        df_filtered = self.__filter_and_downsample(downsample_factor)
        data_scaled = df_filtered[features]

        # UMAP dimensionality reduction
        umap_reducer = umap.UMAP(n_neighbors, n_components, random_state=42)
        data_umap = umap_reducer.fit_transform(data_scaled)

        # Create UMAP DataFrame
        umap_df = pd.DataFrame(data_umap, columns=["UMAP1", "UMAP2"])
        umap_df[group_column] = df_filtered[group_column].values
        umap_df["Unique_Code"] = df_filtered["Unique_Code"].values
        umap_df["Machine"] = df_filtered["Machine"].values
        umap_df["Period"] = df_filtered["Period"].values

        # Create Plotly scatter plot
        fig = px.scatter(
            umap_df,
            x="UMAP1",
            y="UMAP2",
            color=group_column,
            hover_data=["UMAP1", "UMAP2"],
            title=title,
            labels={"UMAP1": "UMAP 1", "UMAP2": "UMAP 2"},
        )
        fig.update_traces(marker=dict(size=4))
        fig.show()


class ComparePeriods(CompareData):
    """
    A class to compare and visualize data across different periods for a specific machine.
    """

    def __init__(self, machine: str, df: pd.DataFrame) -> None:
        super().__init__(filter_column="Machine", filter_value=machine, df=df)

    def cdf_kde(self, cols: List[str], downsample_factor: int = 1) -> None:
        super().cdf_kde(cols=cols, group_column="Period", downsample_factor=downsample_factor)

    def ks_test(self, periods: List[str], data_column: str, downsample_factor: int = 1) -> None:
        super().ks_test(groups=periods, data_column=data_column, group_column="Period", downsample_factor=downsample_factor)

    def scatter_plot(self, cols: List[str] = ["X_axis", "Y_axis", "Z_axis"], downsample_factor: int = 1) -> None:
        super().scatter_plot(cols=cols, group_column="Period", downsample_factor=downsample_factor)
    
    def scatter_plot_umap(self, downsample_factor: int = 1, n_neighbors: int = 25,
        n_components: int = 2) -> None:
        """
        Highlight Periods for a specific Machine.
        """
        features = ["X_axis", "Y_axis", "Z_axis", "Time"]
        title = f"{self.filter_value}"
        super().scatter_plot_umap(group_column="Period", features=features, downsample_factor=downsample_factor, title=title, n_neighbors= n_neighbors, n_components= n_components)
        
    


class CompareMachines(CompareData):
    """
    A class to compare and visualize data across different machines for a specific period.
    """

    def __init__(self, period: str, df: pd.DataFrame) -> None:
        super().__init__(filter_column="Period", filter_value=period, df=df)

    def cdf_kde(self, cols: List[str], downsample_factor: int = 1) -> None:
        super().cdf_kde(cols=cols, group_column="Machine", downsample_factor=downsample_factor)

    def ks_test(self, machines: List[str], data_column: str, downsample_factor: int = 1) -> None:
        super().ks_test(groups=machines, data_column=data_column, group_column="Machine", downsample_factor=downsample_factor)

    def scatter_plot(self, cols: List[str] = ["X_axis", "Y_axis", "Z_axis"], downsample_factor: int = 1) -> None:
        super().scatter_plot(cols=cols, group_column="Machine", downsample_factor=downsample_factor)
        
    def scatter_plot_umap(self, downsample_factor: int = 1, n_neighbors: int = 25,
        n_components: int = 2) -> None:
        """
        Highlight Machines for a specific Period.
        """
        features = ["X_axis", "Y_axis", "Z_axis", "Time"]
        title = f"{self.filter_value}"
        super().scatter_plot_umap(group_column="Machine", features=features, downsample_factor=downsample_factor, title=title, n_neighbors= n_neighbors, n_components= n_components)


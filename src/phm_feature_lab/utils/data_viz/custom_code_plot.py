import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.colors import to_hex
from typing import Optional, Union, List
from phm_feature_lab.utils.logger import Logger 

logger = Logger().get_logger()

class CustomCodePlot:
    """
    Class for generating plots with support for DataFrame manipulation and color grouping.

    Args:
        df (pd.DataFrame): DataFrame containing the data to be analyzed.
        cmap (str): Name of the colormap to be used in the plots. Default: 'tab20b'.
        group_set (str): Group to be analyzed ('train', 'test', or None for all groups).
        combine_axis_var (bool): If True, combines 'axis' and 'var' to form column names (e.g., 'X_acc').
                                 If False, treats 'var' and 'axis' as separate columns.
    """

    TRAIN_MONTHS = ["Feb_2020", "Aug_2020", "Feb_2021", "Aug_2021"]
    TEST_MONTHS = ["Feb_2019", "Aug_2019"]

    def __init__(
        self,
        df: pd.DataFrame,
        cmap: str = "tab20b",
        group_set: str = None,
        combine_axis_var: bool = True,
    ):
        self.__cmap = cmap
        self.__group_set = group_set
        self.__df = self.__normalize_columns(df)
        
        self.__combine_axis_var = combine_axis_var

    def __normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts all column names in the DataFrame to lowercase."""
        df = df.copy()
        df.columns = df.columns.str.lower()
        df = self.__filter_by_group(df)
        return df

    @staticmethod
    def __validate_dataframe(df: pd.DataFrame) -> None:
        """Validates that the DataFrame is not empty."""
        if df.empty:
            raise ValueError("The provided DataFrame is empty.")
    
    def __filter_by_group(self, df):
        """Filters the DataFrame based on the selected group_set and logs if 2019 data is missing for test."""
        

        if self.__group_set is None:
            return df

        # Extrai 'year_month' para filtrar
        df = self.__extract_year_month(df)

        if self.__group_set == "train":
            # Filtra para os meses de treinamento disponíveis
            train_months = [month for month in self.TRAIN_MONTHS if month in df["year_month"].unique()]
            
            filtered_df = df[df["year_month"].isin(train_months)]
            
        elif self.__group_set == "test":
            # Verifica se há dados para 2019
            test_months = [month for month in self.TEST_MONTHS if month in df["year_month"].unique()]
            if not test_months:
                logger.info("No data for 2019 in this operation. Stopping execution.")
                return None
            filtered_df = df[df["year_month"].isin(test_months)]
            
        else:
            raise ValueError("Invalid group_set. Choose 'train', 'test', or None.")

        return filtered_df
    
    @staticmethod
    def __ensure_column_exists(df: pd.DataFrame, column_name: str) -> None:
        """Ensures that the specified column exists in the DataFrame."""
        if column_name not in df.columns:
            raise KeyError(f"The DataFrame must contain the column '{column_name}'.")
    
    @staticmethod
    def __extract_year_month(df: pd.DataFrame) -> pd.DataFrame:
        """Adds 'year_month' column based on 'unique_code'."""
        df = df.copy()
        df["month"] = df["unique_code"].str.extract(r"(Feb|Aug)")[0]
        df["year"] = df["unique_code"].str.extract(r"_(\d{4})_")[0]
        df["year_month"] = df["month"].astype(str) + "_" + df["year"].astype(str)
        return df
    
    def __get_year_month_labels(self) -> List[str]:
        """Returns the 'Year_Month' labels based on the selected group."""
        return self.__df['year_month'].unique()    

    def __generate_color_map(self) -> dict:
        """Generates a color mapping for the 'Year_Month' values."""
        year_month_labels = self.__get_year_month_labels()
        colormap = cm.get_cmap(self.__cmap, 256)  # 256 available colors
        indices = np.linspace(0, 255, len(year_month_labels), dtype=int)
        return {label: to_hex(colormap(idx)) for label, idx in zip(year_month_labels, indices)}

    def __apply_color_map(self, df: pd.DataFrame) -> pd.Series:
        """Maps colors to the 'Year_Month' column."""
        color_map = self.__generate_color_map()
        logger.info(f"Color map: {color_map}")
        logger.info(f"Year_Month: {df['year_month']}")
        return df["year_month"].map(color_map)

    def __filter_data(
        self,
        machine: str,
        operation: str,
        axis: Optional[Union[str, List[str]]],
        var: str,
    ) -> pd.DataFrame:
        """Filters the DataFrame by machine, operation, and axis (supporting multiple axes)."""
        self.__validate_dataframe(self.__df)
        self.__ensure_column_exists(self.__df, "unique_code")

        # Se 'axis' for uma string, converta para lista
        if isinstance(axis, str):
            axis = [axis]

        filtered_df = self.__df.copy()

        if self.__combine_axis_var:
            # Modo combinado: usa 'axis_var' como nome da coluna
            column_names = [f"{ax}_{var}" for ax in axis]
            for col in column_names:
                self.__ensure_column_exists(filtered_df, col)
        else:
            # Modo separado: usa 'axis' e 'var' como colunas separadas
            self.__ensure_column_exists(filtered_df, "axis")
            self.__ensure_column_exists(filtered_df, var)
            filtered_df = filtered_df[filtered_df["axis"].isin(axis)]

        return filtered_df[filtered_df["unique_code"].str.contains(f"{machine}_{operation}")]

    @staticmethod
    def __add_year_month_metadata(df: pd.DataFrame, machine: str, operation: str) -> pd.DataFrame:
        """
        Adds auxiliary columns ('year_month', 'month', 'year', etc.) for sorting and color mapping.
        """
        prefix = f"{machine}_{operation}_"

        df = df.copy()
        df["month"] = df["unique_code"].str.extract(r"(Feb|Aug)")[0]
        df["year"] = df["unique_code"].str.extract(r"_(\d{4})_")[0]
        df["last_number"] = df["unique_code"].str.extract(r"_(\d+)$")[0].astype(int)
        df["year_month"] = df["month"].astype(str) + "_" + df["year"].astype(str)
        df["unique_code"] = df["unique_code"].str.replace(f"^{prefix}", "", regex=True)
        logger.info(f"Unique code: {df['unique_code']}")
        logger.info(f"Year_Month: {df['year_month'].unique()}")
        return df

    @staticmethod
    def __order_data_by_unique_code(df: pd.DataFrame) -> pd.DataFrame:
        """Sorts the DataFrame based on 'Year', 'Month', and 'Last_Number'."""
        month_order = {"Feb": 1, "Aug": 2}
        df["month_order"] = df["month"].map(month_order)
        ordered_df = df.sort_values(by=["year", "month_order", "last_number"]).reset_index(drop=True)
        return ordered_df

    @staticmethod
    def __calculate_y_axis_limits(df: pd.DataFrame, column_name: str, y_min: Optional[float], y_max: Optional[float]) -> tuple:
        """
        Calculates Y-axis limits with a 10% margin if not provided.

        Args:
            df (pd.DataFrame): DataFrame with the variable to be plotted.
            column_name (str): Name of the column to be used for Y-axis limits.
            y_min (float): Minimum Y-axis value.
            y_max (float): Maximum Y-axis value.

        Returns:
            tuple: Calculated (y_min, y_max) values.
        """
        if y_min is None or y_max is None:
            y_min_local = df[column_name].min()
            y_max_local = df[column_name].max()
            margin = 0.1 * (y_max_local - y_min_local)
            return y_min_local - margin, y_max_local + margin
        return y_min, y_max

    def __create_subplots(self, num_axes: int):
        """Creates subplots based on the number of axes."""
        fig, axes = plt.subplots(num_axes, 1, figsize=(12, 4 * num_axes), sharex=True)
        if num_axes == 1:
            axes = [axes]
        return fig, axes

    def __plot_axis(
        self,
        ax,
        ax_name: str,
        var: str,
        machine: str,
        operation: str,
        y_min: Optional[float],
        y_max: Optional[float],
        y_axis_label: Optional[str],
        title: Optional[str],
        anomaly_column: Optional[str],
    ):
        """Plots the data for a single axis."""
        
        if self.__combine_axis_var:
            # Modo combinado: usa 'axis_var' como nome da coluna
            column_name = f"{ax_name}_{var}"
        else:
            # Modo separado: usa 'var' como nome da coluna
            column_name = var

        df_filtered = self.__filter_data(machine, operation, ax_name, var)
        df_metadata = self.__prepare_data(df_filtered, machine, operation, column_name)
        y_min_local, y_max_local = self.__calculate_y_axis_limits(
            df_metadata, column_name, y_min, y_max  # Passa o nome da coluna corretamente
        )

        for j in range(len(df_metadata) - 1):
            ax.plot(
                df_metadata["unique_code"].iloc[j : j + 2],
                df_metadata[column_name].iloc[j : j + 2],
                linestyle="-",
                color=df_metadata["color"].iloc[j + 1],
            )
            self.__plot_point(
                ax,
                df_metadata,
                index=j,
                var=column_name,
                anomaly_column=anomaly_column,
            )

        self.__plot_last_point(ax, df_metadata, column_name, anomaly_column)
        ax.set_ylim(y_min_local, y_max_local)
        ax.set_ylabel(y_axis_label if y_axis_label else f"{column_name}")
        ax.set_title(title if title else f"{machine}-{operation} ({ax_name})")
        ax.grid(True)

    def __prepare_data(self, df, machine, operation, column_name):
        """Filters and prepares the data with metadata for plotting."""
        df_metadata = self.__add_year_month_metadata(df, machine, operation)
        df_metadata = self.__order_data_by_unique_code(df_metadata)
        df_metadata["color"] = self.__apply_color_map(df_metadata)
        self.__validate_dataframe(df_metadata)
        return df_metadata

    def __plot_point(self, ax, df_metadata, index, var, anomaly_column):
        """Plots a single point on the graph."""
        is_anomaly = anomaly_column and df_metadata[anomaly_column].iloc[index] == "Anomaly"
        ax.scatter(
            df_metadata["unique_code"].iloc[index],
            df_metadata[var].iloc[index],
            color="red" if is_anomaly else df_metadata["color"].iloc[index],
            marker="x" if is_anomaly else "o",
        )

    def __plot_last_point(self, ax, df_metadata, var, anomaly_column):
        """Plots the last point in the series."""
        is_anomaly_last = anomaly_column and df_metadata[anomaly_column].iloc[-1] == "Anomaly"
        ax.scatter(
            df_metadata["unique_code"].iloc[-1],
            df_metadata[var].iloc[-1],
            color="red" if is_anomaly_last else df_metadata["color"].iloc[-1],
            marker="x" if is_anomaly_last else "o",
        )

    def __format_x_axis(self, ax):
        """Formats the X-axis to display only the final part of unique codes."""
        ax.set_xlabel("Unique_Code")
        plt.xticks(rotation=25)
        xticks = ax.get_xticks()
        xtick_labels = ax.get_xticklabels()
        new_labels = [label.get_text().split("_")[-1] for label in xtick_labels]
        ax.set_xticks(xticks)
        ax.set_xticklabels(new_labels)

    def __add_legend(self, fig):
        """Adds a legend to the plot."""
        handles = [
            plt.Line2D([0], [0], color=color, lw=2)
            for label, color in self.__generate_color_map().items()
        ]
        fig.legend(
            handles,
            self.__get_year_month_labels(),
            title="Período",
            loc="upper left",
            fontsize=11,
            title_fontsize=12,
            bbox_to_anchor=(1, 1),
            borderaxespad=1.5,
        )

    def plot(
        self,
        var: str,
        machine: str,
        operation: str,
        axis: Optional[Union[str, List[str]]] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        title: Optional[str] = None,
        y_axis_label: Optional[str] = None,
        anomaly_column: Optional[str] = None,
    ) -> None:
        """
        Plots the graph based on the provided DataFrame, with support for multiple axes 
        (X, Y, Z) sharing the same X-axis.

        Args:
            var (str): Name of the variable to be plotted (e.g., 'num_edges').
            machine (str): Machine name.
            operation (str): Operation name.
            axis (str | List[str]): Axis of interest (e.g., ['X', 'Y', 'Z']).
            y_min (float): Minimum Y-axis value (optional).
            y_max (float): Maximum Y-axis value (optional).
            anomaly_column (str): Column name indicating anomalies (optional).
        """
        if isinstance(axis, str):
            axis = [axis]
            
        if self.__combine_axis_var:
            axis = [ax.lower() for ax in axis]

        fig, axes = self.__create_subplots(len(axis))
        
        for i, ax_name in enumerate(axis):
            self.__plot_axis(
                ax=axes[i],
                ax_name=ax_name,
                var=var,
                machine=machine,
                operation=operation,
                y_min=y_min,
                y_max=y_max,
                y_axis_label=y_axis_label,
                title=title,
                anomaly_column=anomaly_column,
            )

        self.__format_x_axis(axes[-1])
        self.__add_legend(fig)
        plt.tight_layout()
        plt.show()
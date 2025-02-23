import os
from typing import List, Dict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm

import hiplot as hip

from phm_feature_lab.utils.logger import Logger 

logger = Logger().get_logger()

def label_histograms(df, axis, machine):
    """
    Plots histograms of a specified axis for different periods, comparing normal and anomaly labels for a given machine.

    """
    periods = df['Period'].unique()
    def reorder_periods(periods: np.ndarray) -> np.ndarray:
        return np.array(sorted(periods, key=lambda x: (x.split('-')[1], x.split('-')[0])))
    
    periods = reorder_periods(periods)
    num_periods = len(periods)
    rows = (num_periods + 1) // 2  # Calculate the number of rows needed

    fig, axes = plt.subplots(rows, 2, figsize=(15, rows * 5))

    for idx, period in enumerate(periods):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # Filter the DataFrame based on the machine, period, and label
        x1 = df[(df['Machine'] == machine) & (df['Period'] == period) & (df['Label'] == 0)][axis]
        y1 = df[(df['Machine'] == machine) & (df['Period'] == period) & (df['Label'] == 1)][axis]

        # Plot the KDE for normal and anomaly data
        sns.kdeplot(x1, fill=True, color='blue', label='Normal', ax=ax)
        sns.kdeplot(y1, fill=True, color='red', label='Anomaly', ax=ax)
        ax.set_xlabel('Distribution')
        ax.set_ylabel('Probability')
        ax.set_ylim([-0.0001, None])
        ax.set_title(f'Period: {period}')
        ax.legend(fontsize=12)

    # Remove any empty subplots if the number of periods is odd
    if num_periods % 2 != 0:
        fig.delaxes(axes[rows - 1, 1])
        
    fig.suptitle(f'{axis} distribution for {machine}', fontsize=24)
    plt.tight_layout()
    plt.show()


def read_files_to_dict(base_path: str, op_list: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Reads multiple parquet files based on a list of OPs and returns a dictionary of DataFrames.

        """
    dataframes: Dict[str, pd.DataFrame] = {}
    
    for op in op_list:
        file_path = os.path.join(base_path, f'OP{op}.parquet')
        if os.path.exists(file_path):
            logger.info(f"Reading file: {file_path}")
            df = pd.read_parquet(file_path)
            dataframes[op] = df  # Store the DataFrame in the dictionary
        else:
            logger.error(f"File not found: {file_path}")

    return dataframes
    
    
def plot_violinplot(df, axis, machines, periods):
    

    # Determine the number of subplots needed
    num_machines = len(machines)
    
    # Find the global min and max for the y-axis
    global_min = df[axis].min()
    global_max = df[axis].max()
    
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(num_machines, 1, figsize=(15, 4 * num_machines), sharex=True)
    
    # If only one machine is provided, axs will not be an array, so we convert it to an array
    if num_machines == 1:
        axs = [axs]
    
    for ax, machine in zip(axs, machines):
        subset = df[df['Machine'] == machine]
        
        sns.violinplot(
            data=subset,
            x='Period',
            y=axis,
            hue='Label',
            palette={0: 'blue', 1: 'red'},
            split=False,
            dodge=True,
            ax=ax,
            order=periods
        )
        
        ax.set_xlabel('Period')
        ax.set_ylabel(f'{axis} distribution')
        ax.set_title(f'Machine {machine}')
        ax.set_ylim(global_min*1.1, global_max*1.1)
        ax.legend(title='Label', loc='upper right')
    
    plt.tight_layout()
    plt.show()

def plot_label_signals(df_anomalous, df_non_anomalous):
    """
    Plots three subplots (X, Y, Z axes) for anomalous and non-anomalous signals.
    """

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, subplot_titles=("X Axis", "Y Axis", "Z Axis"))
    for axis, row in zip(['X_axis', 'Y_axis', 'Z_axis'], range(1, 4)):       
        fig.add_trace(go.Scatter(x=df_anomalous['Time'], y=df_anomalous[axis], mode='lines', name=f'Anomalous {axis}', line=dict(color='orange')), row=row, col=1)
        fig.add_trace(go.Scatter(x=df_non_anomalous['Time'], y=df_non_anomalous[axis], mode='lines', name=f'Non-Anomalous {axis}', line=dict(color='blue')), row=row, col=1)

    machine = df_anomalous['Machine'].values[0]

    fig.update_layout(height=600, width=1000, title_text=f"Anomaly and normal - {machine}")
    fig.update_xaxes(title_text="Time (s)", row=3, col=1)
    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_yaxes(title_text="Amplitude", row=2, col=1)
    fig.update_yaxes(title_text="Amplitude", row=3, col=1)

    fig.show()

# Definindo uma paleta de cores fixa
color_palette = [
    '#1f77b4',  # Azul
    '#bcbd22',  # Verde-oliva
    '#e377c2',  # Rosa
    '#2ca02c',  # Verde
    '#d62728',  # Vermelho
    '#9467bd',  # Roxo
    '#ff7f0e',  # Laranja
    '#8c564b',  # Marrom
    '#7f7f7f',  # Cinza
    '#17becf'   # Ciano-azulado
]

def plot_all_axis(df, process, machine, by_code = False, decimation_factor=100):

    """
    Plots data for X_axis, Y_axis, and Z_axis in separate subplots for each unique code or year-month combination,
    for a specified process and machine, with a fixed color scale for consistency.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - by_code: Boolean, true to group by unique code, false to group only by year and month.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)].copy()

    # Create a subplot for each axis
    fig = make_subplots(rows=3, cols=1, subplot_titles=('X_axis', 'Y_axis', 'Z_axis'))

    # Determine group by columns
    if by_code:
        # Use Unique_Code for coloring
        unique_codes = filtered_df['Unique_Code'].unique()
        color_mapping = {code: color_palette[i % len(color_palette)] for i, code in enumerate(unique_codes)}
        
        for i, code in enumerate(unique_codes):
            df_plot = filtered_df[filtered_df['Unique_Code'] == code][::decimation_factor]
            if not df_plot.empty:
                for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis'], start=1):
                    fig.add_trace(go.Scatter(
                        x=df_plot['Time'],
                        y=df_plot[axis],  # Plot each axis
                        mode='markers',
                        name=f'Code: {code} - {axis}',
                        marker=dict(color=color_mapping[code])
                    ), row=j, col=1)
    else:
        # Use Period for coloring
        filtered_df['Period'] = filtered_df['Year'].astype(str) + '-' + filtered_df['Month']
        periods = filtered_df['Period'].unique()
        color_mapping = {period: color_palette[i % len(color_palette)] for i, period in enumerate(periods)}
        
        for i, period in enumerate(periods):
            df_plot = filtered_df[filtered_df['Period'] == period][::decimation_factor]
            if not df_plot.empty:
                for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis'], start=1):
                    fig.add_trace(go.Scatter(
                        x=df_plot['Time'],
                        y=df_plot[axis],  # Plot each axis
                        mode='markers',
                        name=f'Period: {period} - {axis}',
                        marker=dict(color=color_mapping[period])
                    ), row=j, col=1)

    # Update layout of the plot
    fig.update_layout(
        title=f'{process} on Machine {machine}',
        xaxis_title='Time',
        yaxis_title='Axis Value',
        template='plotly_white',
        legend_title="Group",
        height=900  # Increase the height to accommodate three subplots
    )

    # Show the plot
    fig.show()

def plot_all_axis_matplotlib(df, process, machine, by_code=False, decimation_factor=100):
    """
    Plots data for X_axis, Y_axis, and Z_axis in separate subplots for each unique code or year-month combination,
    for a specified process and machine, with a fixed color scale for consistency using matplotlib.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - by_code: Boolean, true to group by unique code, false to group only by year and month.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)].copy()

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(3, 1, figsize=(12,10), sharex=True)

    # Titles for each subplot
    axis_titles = ['X_axis', 'Y_axis', 'Z_axis']

    # Determine group by columns
    groups = filtered_df['Period'].unique() 
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))  # Generate colors from a colormap

    for i, group in enumerate(groups):
        group_df = filtered_df[filtered_df['Period'] == group][::decimation_factor]
        for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis']):
            axes[j].scatter(group_df['Time'], group_df[axis], label=f'{group} - {axis}', color=colors[i % len(colors)], s=10)
            axes[j].set_title(axis_titles[j])
            axes[j].set_xlabel('Time')
            axes[j].set_ylabel('Value')
            axes[j].legend(title="Period", loc='upper right',bbox_to_anchor=(1.1, 1.05))
            axes[j].grid(True)

    fig.suptitle(f'{process} on Machine {machine}', fontsize=18)
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_scatter_matrix(df, machine, process, sample_frac=0.05, random_state=0):
    """
    Plota uma matriz de dispersão para as colunas 'X_axis', 'Y_axis', e 'Z_axis' para um dado
    processo e máquina em um DataFrame.

    Parâmetros:
    - df: DataFrame contendo os dados.
    - machine: String representando a máquina a ser filtrada.
    - process: String representando o processo a ser filtrado.
    - sample_frac: Fração do DataFrame para amostragem (default 0.05).
    - random_state: Semente para a geração de números aleatórios (default 0).
    """

    # Filtrar dados por máquina e processo
    df_filtered = df[(df['Machine'] == machine) & (df['Process'] == process)]

    # Ordenar as categorias de período se a coluna 'Period' existe no DataFrame
    if 'Period' in df_filtered.columns:
        code_order = df_filtered['Period'].unique()
    else:
        code_order = None

    # Definir as colunas para a matriz de dispersão
    cols = ['X_axis', 'Y_axis', 'Z_axis']

    # Criar a matriz de dispersão
    fig = px.scatter_matrix(df_filtered.sample(frac=sample_frac, random_state=random_state),
                            dimensions=cols, color='Period',
                            category_orders={'Period': code_order} if code_order is not None else None)

    # Atualizar layout do gráfico
    fig.update_layout(width=1200, height=800, legend_title_font_size=22)

    # Atualizar características dos traços
    fig.update_traces(marker=dict(size=5), diagonal_visible=False, showupperhalf=False)

    # Exibir o gráfico
    fig.show()


def plot_by_code_index_matplotlib(df, process, machine, axis='Z_axis', decimation_factor=100):
    """
    Plots data for a specified axis for groups of unique codes using Matplotlib,
    with each subplot containing up to 5 unique codes, for a specified process and machine,
    with a manually defined color scale for consistency, and grid enabled for better visualization.
    
    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - axis: String, the axis to plot (e.g., 'X_axis', 'Y_axis', 'Z_axis').
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)]
    filtered_df.reset_index(drop=True, inplace=True)

    # Get unique codes
    unique_codes = filtered_df['Unique_Code'].unique()
    num_subplots = np.ceil(len(unique_codes) / 5).astype(int)

    # Calculate global y limits
    global_y_min = filtered_df[axis].min()
    global_y_max = filtered_df[axis].max()
    
    # Create the subplots
    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 4 * num_subplots), sharey=True)
    
    # Check if axs is iterable
    if not hasattr(axs, '__iter__'):
        axs = [axs]
    
    # Define a custom color palette
    color_palette = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if len(unique_codes) > len(color_palette):
        # Extend the palette with random colors if there are more unique codes than the default palette length
        extra_colors = np.random.rand(len(unique_codes) - len(color_palette), 3)
        color_palette.extend(extra_colors)

    # Plot each unique code in its corresponding subplot
    for i, ax in enumerate(axs):
        codes_to_plot = unique_codes[i*5:(i+1)*5]
        for idx, code in enumerate(codes_to_plot):
            df_plot = filtered_df[filtered_df['Unique_Code'] == code][::decimation_factor]
            if not df_plot.empty:
                ax.plot(df_plot.index, df_plot[axis], label=f'{code}', color=color_palette[idx % len(color_palette)])
        
        # Set the same y-axis limit for all subplots
        ax.set_ylim(global_y_min, global_y_max)
        
        # Set legend, title, and labels
        ax.legend(loc='lower right')
        ax.set_title(f'{process} on {machine} - {axis}')
        ax.set_xlabel('Index')
        ax.set_ylabel(axis)

        # Enable grid
        ax.grid(True)

    # Adjust layout
    plt.tight_layout()

    # Show the plot
    plt.show()

def plot_selected_columns(df, process, machine, columns_to_plot, max_codes_per_plot=5, decimation_factor=100):
    """
    Plots specified columns with a maximum of 5 unique codes per plot.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - columns_to_plot: List of column names to be plotted.
    - max_codes_per_plot: int, maximum number of unique codes per plot.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)]
    filtered_df.reset_index(drop=True, inplace=True)

    # Get unique codes
    unique_codes = filtered_df['Unique_Code'].unique()

    # Ensure the columns to plot exist in the DataFrame
    available_columns = filtered_df.columns
    columns_to_plot = [col for col in columns_to_plot if col in available_columns]
    num_columns = len(columns_to_plot)

    # Calculate the number of plots needed per column
    num_plots = np.ceil(len(unique_codes) / max_codes_per_plot).astype(int)

    # Create subplots with one column per row
    fig, axs = plt.subplots(num_columns * num_plots, 1, figsize=(12, 4 * num_columns * num_plots), sharey=False)

    # Ensure axs is iterable
    if len(axs) == 1:
        axs = [axs]

    # Define a custom color palette
    color_palette = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if len(unique_codes) > len(color_palette):
        extra_colors = np.random.rand(len(unique_codes) - len(color_palette), 3)
        color_palette.extend(extra_colors)

    # Track subplot index
    subplot_idx = 0

    # Loop through columns and plots
    for col_idx, column in enumerate(columns_to_plot):
        for i in range(num_plots):
            ax = axs[subplot_idx]
            codes_to_plot = unique_codes[i * max_codes_per_plot:(i + 1) * max_codes_per_plot]

            # Get the y-limits for the current column
            global_y_min = filtered_df[column].min()
            global_y_max = filtered_df[column].max()

            for idx, code in enumerate(codes_to_plot):
                df_plot = filtered_df[filtered_df['Unique_Code'] == code][::decimation_factor]
                if not df_plot.empty:
                    ax.plot(df_plot.index, df_plot[column], label=f'{code}', color=color_palette[idx % len(color_palette)])
            
            # Set the y-axis limits
            ax.set_ylim(global_y_min, global_y_max)

            # Set legend, title, and labels
            ax.legend(loc='lower right')
            ax.set_title(f'{process} on {machine} - {column}')
            ax.set_xlabel('Index')
            ax.set_ylabel(column)

            # Enable grid
            ax.grid(True)

            # Increment subplot index for the next plot
            subplot_idx += 1

    # Adjust layout
    plt.tight_layout()
    plt.show()

def visualize_with_hiplot(df):
    """
    Visualizes a Pandas DataFrame using HiPlot.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data to visualize.
    
    Returns:
    - Displays an interactive HiPlot visualization.
    """
    # Convert the DataFrame to a HiPlot experiment
    data = df.to_dict(orient='records')
    exp = hip.Experiment.from_iterable(data)
    
    # Display the HiPlot visualization
    exp.display()

def plot_scatter_matrix_FE(df, machine, process, cols, sample_frac=0.05, random_state=0):
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

def plot_all_axis_matplotlib(df, process, machine, by_code=False, decimation_factor=100):
    """
    Plots data for X_axis, Y_axis, and Z_axis in separate subplots for each unique code or year-month combination,
    for a specified process and machine, with a fixed color scale for consistency using matplotlib.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - by_code: Boolean, true to group by unique code, false to group only by year and month.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Filter the DataFrame based on process and machine
    filtered_df = df[(df['Process'] == process) & (df['Machine'] == machine)].copy()

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(3, 1, figsize=(12,12), sharex=True)

    # Titles for each subplot
    axis_titles = ['X_axis', 'Y_axis', 'Z_axis']

    # Determine group by columns
    groups = filtered_df['Period'].unique() 
    colors = plt.cm.viridis(np.linspace(0, 1, len(groups)))  # Generate colors from a colormap

    for i, group in enumerate(groups):
        group_df = filtered_df[filtered_df['Period'] == group][::decimation_factor]
        for j, axis in enumerate(['X_axis', 'Y_axis', 'Z_axis']):
            axes[j].scatter(group_df['Time'], group_df[axis], label=f'{group} - {axis}', color=colors[i % len(colors)], s=10)
            axes[j].set_title(axis_titles[j])
            axes[j].set_xlabel('Time')
            axes[j].set_ylabel('Value')
            axes[j].legend(title="Period", loc='upper right',bbox_to_anchor=(1.1, 1.05), fontsize=12)
            axes[j].grid(True)

    fig.suptitle(f'{process} on Machine {machine}', fontsize=18)
    # Adjust layout
    plt.tight_layout()
    plt.show()


def plotly_scattermatrix(
        df, cols, category_order=None, symbol=None, color="Unique_Code", upload=False,
        width=1200, height=800, label_fontsize=16, legend_fontsize=14):
    """Create a scatter matrix plot using Plotly Express.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data.
        cols (list[str]): The columns to plot in the scatter matrix.
        category_order (dict[str, list], optional): The order of categories for the color parameter. Defaults to None.
        symbol (str, optional): The column name to use as symbols. Defaults to None.
        color (str, optional): The column name to use for coloring the points. Defaults to "Month".
        upload (bool, optional): If True, the plot is uploaded to Plotly's online platform. Defaults to False.
        filename (str, optional): The filename for the uploaded plot. Defaults to 'Scattermatrix'.
        width (int, optional): The width of the plot in pixels. Defaults to 1800.
        height (int, optional): The height of the plot in pixels. Defaults to 1100.
        label_fontsize (int, optional): The font size of x and y axis labels. Defaults to 16.
        legend_fontsize (int, optional): The font size of the legend. Defaults to 14.

    Returns:
        None
    """

    # Create the scatter matrix plot
    fig = px.scatter_matrix(df, dimensions=cols, category_orders=category_order, color=color, symbol=symbol)

    # Update the layout of the plot
    fig.update_layout(
        title='Pairplot',
        width=width,
        height=height,
        hovermode='closest',
        font=dict(size=label_fontsize),
        legend=dict(font=dict(size=legend_fontsize))
    )

    # Update the trace properties
    fig.update_traces(showupperhalf=False,diagonal_visible=False, marker=dict(size=2.5))

    # Show the plot
    fig.show()

def plot_kde(df, hue_col, sample_step=1000, columns_to_plot = None, fill= False):
    """
    Plots data for X_axis, Y_axis, and Z_axis in separate subplots for each unique code or year-month combination,
    for a specified process and machine, with a fixed color scale for consistency.

    Parameters:
    - df: DataFrame containing the data.
    - process: String, the process type to filter by (e.g., 'OP00').
    - machine: String, the machine to filter by (e.g., 'M01').
    - by_code: Boolean, true to group by unique code, false to group only by year and month.
    - decimation_factor: int, factor by which to thin the data for clarity.
    """
    # Function implementation goes here
    if columns_to_plot is None:
        raise ValueError("columns_to_plot must be provided")
    fig, ax = plt.subplots(len(columns_to_plot), 2, figsize=(22, 5*len(columns_to_plot)))
    ax = ax.ravel()
    hue_order = df[hue_col].unique()

    for i, col in enumerate(columns_to_plot):
        sns.kdeplot(data=df[::sample_step], x=col, hue=hue_col, ax=ax[2 * i], palette='Set1', hue_order=hue_order, 
        common_norm=True,common_grid=True, fill=fill )
        sns.kdeplot(data=df[::sample_step],x=col, hue=hue_col, ax=ax[2 * i + 1], palette='Set1', cumulative=True, hue_order=hue_order, 
        common_norm=False, common_grid=True)
        ax[2 * i + 1].set_ylabel('Cumulative Density')

    plt.tight_layout()
    plt.show()


def plot_histograms(df, period, column, label_condition=True, compare_periods=False, use_plotly=False):
    """
    Plots normalized histograms of the specified column for each 'Machine' in the dataset within the specified period.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'Machine', the specified column, 'Label', and 'Period' columns.
    period (str or list): The period(s) to filter the DataFrame. If compare_periods is True, this should be a list of periods.
    column (str): The column to be plotted in the histogram.
    label_condition (bool): If True, plots histograms for each label (0 and 1) separately.
                            If False, plots a single histogram for each machine.
    compare_periods (bool): If True, compares different periods by plotting histograms for each period in different colors.
    use_plotly (bool): If True, uses Plotly for plotting. If False, uses Matplotlib.
    """
    if compare_periods:
        df_period = df[df['Period'].isin(period)]
    else:
        df_period = df[df['Period'] == period]
    
    machines = df_period['Machine'].unique()
    colors = ['dodgerblue','deeppink', 'orange', 'green']  # Updated colors
    
    # Determine global x-axis limits
    global_min = df_period[column].min()
    global_max = df_period[column].max()
    
    if use_plotly:
        df_period = df_period[::100]
        fig = make_subplots(rows=len(machines), cols=1, subplot_titles=[f'Machine {machine}' for machine in machines])

        for i, machine in enumerate(machines):
            machine_data = df_period[df_period['Machine'] == machine]
            
            if compare_periods:
                periods = df_period['Period'].unique()
                
                for period, color in zip(periods, colors):
                    subset = machine_data[machine_data['Period'] == period]
                    hist_data = subset[column]
                    fig.add_trace(go.Histogram(x=hist_data, histnorm='probability', name=f'Period {period}', marker=dict(color=color, line=dict(color=color)), opacity=0.5), row=i+1, col=1)
                    
            elif label_condition:
                labels = machine_data['Label'].unique()
                
                for label, color in zip(labels, colors):
                    subset = machine_data[machine_data['Label'] == label]
                    hist_data = subset[column]
                    fig.add_trace(go.Histogram(x=hist_data, histnorm='probability', name=f'Label {label}', marker=dict(color=color, line=dict(color=color)), opacity=0.5), row=i+1, col=1)
                    
            else:
                hist_data = machine_data[column]
                fig.add_trace(go.Histogram(x=hist_data, histnorm='probability', name='All Labels', marker=dict(color='olive', line=dict(color='olive')), opacity=0.5), row=i+1, col=1)
            
            fig.update_xaxes(range=[global_min, global_max], row=i+1, col=1)
        
        fig.update_layout(height=500*len(machines), width=1000, title_text=f'Histograms for {period if not compare_periods else "Comparison of Periods"}')
        fig.show()
    else:
        df_period = df_period[::10]
        plt.figure(figsize=(12, 12))

        for i, machine in enumerate(machines, 1):
            plt.subplot(len(machines), 1, i)
            machine_data = df_period[df_period['Machine'] == machine]
            
            if compare_periods:
                periods = df_period['Period'].unique()
                
                for period, color in zip(periods, colors):
                    subset = machine_data[machine_data['Period'] == period]
                    sns.kdeplot(subset[column], fill=True, color=color, label=f'Period {period}', alpha=0.5, stat="density", edgecolor=None)
            elif label_condition:
                labels = machine_data['Label'].unique()
                
                for label, color in zip(labels, colors):
                    subset = machine_data[machine_data['Label'] == label]
                    sns.histplot(subset[column], bins=100, kde=True, color=color, label=f'Label {label}', alpha=0.5, stat="density", edgecolor=None)
            else:
                sns.histplot(machine_data[column], bins=100, kde=True, color='olive', label='All Labels', alpha=0.5, stat="density", edgecolor=None)
            
            plt.xlim(global_min, global_max)
            plt.ylim(-0.0001, None) 
            plt.legend()
            plt.title(f'Machine {machine} - Period {period if not compare_periods else "Comparison"}')
            plt.xlabel(column)
            plt.ylabel('Density')

        plt.tight_layout()
        plt.show()

class DataAnalyzer:
    def __init__(self, df, window_size, machine, period, num_unique_codes_to_plot):
        self.df = df
        self.window_size = window_size
        self.machine = machine
        self.period = period
        self.num_unique_codes_to_plot = num_unique_codes_to_plot
        self.filtered_df = self._filter_data()
        self.unique_code_windows, self.min_windows = self._calculate_max_time_instances()
        self._report()

    def _filter_data(self):
        filtered_df = self.df[(self.df['Machine'] == self.machine) & (self.df['Period'] == self.period)]
        unique_codes = filtered_df['Unique_Code'].unique()[:self.num_unique_codes_to_plot]
        return filtered_df[filtered_df['Unique_Code'].isin(unique_codes)]

    def _calculate_max_time_instances(self):
        unique_codes = self.filtered_df['Unique_Code'].unique()[:self.num_unique_codes_to_plot]
        min_windows = float('inf')
        unique_code_windows = {}

        for unique_code in unique_codes:
            unique_code_df = self.filtered_df[self.filtered_df['Unique_Code'] == unique_code]
            total_points = len(unique_code_df)
            total_windows = total_points // self.window_size + (1 if total_points % self.window_size != 0 else 0)
            unique_code_windows[unique_code] = total_windows
            if total_windows < min_windows:
                min_windows = total_windows

        return unique_code_windows, min_windows

    def _report(self):
        start_time = self.filtered_df['Time'].min()
        end_time = self.filtered_df['Time'].max()
        instance_size = self.window_size
        num_instances = self.min_windows

        report = (f"{'-'*40}\n"
                  f"Report for Data Analyzer:\n"
                  f"Machine: {self.machine}\n"
                  f"Period: {self.period}\n" 
                  f"Instance Size: {instance_size}\n"                 
                  f"Max number of instances for this parameters: {num_instances}\n"
                  f"{'-'*40}\n")
        
        logger.info(report)

    def plot_instances(self, column_to_plot, instance_to_plot, plot_type='matplotlib'):
        self.filtered_df = self.filtered_df.sort_values(by=['Seq', 'Time'])
        unique_seqs = self.filtered_df['Seq'].unique()[:self.num_unique_codes_to_plot]
        unique_seqs = [str(seq) for seq in unique_seqs]

        if plot_type == 'matplotlib':
            plt.figure(figsize=(15, 4))

            for seq in unique_seqs:
                seq_df = self.filtered_df[self.filtered_df['Seq'] == int(seq)]
                total_points = len(seq_df)
                total_windows = total_points // self.window_size + (1 if total_points % self.window_size != 0 else 0)
                time_instances = [(i * self.window_size, min((i + 1) * self.window_size, total_points)) for i in range(total_windows)]
                
                if 1 <= instance_to_plot <= total_windows:
                    start_index, end_index = time_instances[instance_to_plot - 1]
                    plt.plot(seq_df['Time'].iloc[start_index:end_index], 
                             seq_df[column_to_plot].iloc[start_index:end_index], 
                             label=f'{seq}')
                else:
                    logger.error(f'Invalid instance number for unique code {seq}. Please choose a number between 1 and {total_windows}.')
            
            init = round(min(seq_df['Time'].iloc[start_index:end_index]), 2)
            end = round(max(seq_df['Time'].iloc[start_index:end_index]), 2)
            
            plt.xlabel('Time')
            plt.ylabel(column_to_plot)
            plt.title(f'{column_to_plot} for {self.machine} ({init} to {end}s)')
            plt.legend()
            plt.show()
        
        elif plot_type == 'plotly':
            fig = go.Figure()

            for seq in unique_seqs:
                seq_df = self.filtered_df[self.filtered_df['Seq'] == int(seq)]
                total_points = len(seq_df)
                total_windows = total_points // self.window_size + (1 if total_points % self.window_size != 0 else 0)
                time_instances = [(i * self.window_size, min((i + 1) * self.window_size, total_points)) for i in range(total_windows)]
                
                if 1 <= instance_to_plot <= total_windows:
                    start_index, end_index = time_instances[instance_to_plot - 1]
                    fig.add_trace(go.Scatter(x=seq_df['Time'].iloc[start_index:end_index], 
                                             y=seq_df[column_to_plot].iloc[start_index:end_index], 
                                             mode='lines', 
                                             name=f'{seq}'))
                else:
                    logger.error(f'Invalid instance number for unique code {seq}. Please choose a number between 1 and {total_windows}.')
            
            init = round(min(seq_df['Time'].iloc[start_index:end_index]), 2)
            end = round(max(seq_df['Time'].iloc[start_index:end_index]), 2)
            
            fig.update_layout(
                title=f'{column_to_plot} for {self.machine} ({init} to {end}s)',
                xaxis_title='Time',
                yaxis_title=column_to_plot,
                legend_title='Process Number'
            )

            fig.show()

    def plot_fft_for_unique_codes(self, column_to_plot, instance_to_plot, plot_type='matplotlib'):
        self.filtered_df = self.filtered_df.sort_values(by=['Seq', 'Time'])
        unique_seqs = self.filtered_df['Seq'].unique()[:self.num_unique_codes_to_plot]
        unique_seqs = [str(seq) for seq in unique_seqs]

        if plot_type == 'matplotlib':
            plt.figure(figsize=(15, 4))

            for seq in unique_seqs:
                seq_df = self.filtered_df[self.filtered_df['Seq'] == int(seq)]
                total_points = len(seq_df)
                total_windows = total_points // self.window_size + (1 if total_points % self.window_size != 0 else 0)
                time_instances = [(i * self.window_size, min((i + 1) * self.window_size, total_points)) for i in range(total_windows)]
                
                if 1 <= instance_to_plot <= total_windows:
                    start_index, end_index = time_instances[instance_to_plot - 1]
                    signal_data = seq_df[column_to_plot].iloc[start_index:end_index]
                    N = len(signal_data)
                    T = (seq_df['Time'].iloc[1] - seq_df['Time'].iloc[0])
                    yf = np.fft.fft(signal_data)
                    xf = np.fft.fftfreq(N, T)[5:N//2]
                    plt.plot(xf[5:], 2.0/N * np.abs(yf[:N//2][5:]), label=f'{seq}')
                else:
                    logger.info(f'Invalid instance number for unqiue code {seq}. Please choose a number between 1 and {total_windows}.')
            
            init = round(min(seq_df['Time'].iloc[start_index:end_index]), 2)
            end = round(max(seq_df['Time'].iloc[start_index:end_index]), 2)
            
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
            plt.title(f'FFT of {column_to_plot} for {self.machine} ({init} to {end}s)')
            plt.legend()
            plt.grid(True)
            plt.show()

        elif plot_type == 'plotly':
            fig = go.Figure()

            for seq in unique_seqs:
                seq_df = self.filtered_df[self.filtered_df['Seq'] == int(seq)]
                total_points = len(seq_df)
                total_windows = total_points // self.window_size + (1 if total_points % self.window_size != 0 else 0)
                time_instances = [(i * self.window_size, min((i + 1) * self.window_size, total_points)) for i in range(total_windows)]
                
                if 1 <= instance_to_plot <= total_windows:
                    start_index, end_index = time_instances[instance_to_plot - 1]
                    signal_data = seq_df[column_to_plot].iloc[start_index:end_index]
                    N = len(signal_data)
                    T = (seq_df['Time'].iloc[1] - seq_df['Time'].iloc[0])
                    yf = np.fft.fft(signal_data)
                    xf = np.fft.fftfreq(N, T)[:N//2]
                    fig.add_trace(go.Scatter(x=xf[5:], 
                                             y=2.0/N * np.abs(yf[0:N//2][5:]), 
                                             mode='lines', 
                                             name=f'{seq}'))
                else:
                    logger.info(f'Invalid instance number for Unique Code {seq}. Please choose a number between 1 and {total_windows}.')
            
            init = round(min(seq_df['Time'].iloc[start_index:end_index]), 2)
            end = round(max(seq_df['Time'].iloc[start_index:end_index]), 2)
            
            fig.update_layout(
                title=f'FFT of {column_to_plot} for {self.machine} ({init} to {end}s)',
                xaxis_title='Frequency (Hz)',
                yaxis_title='Amplitude',
                legend_title='Process Number',
                showlegend=True
            )

            fig.show()

    def plot_scatter_matrix(self, instance_to_plot, cols):
        unique_codes = self.filtered_df['Unique_Code'].unique()[:self.num_unique_codes_to_plot]
        instance_data = []

        for unique_code in unique_codes:
            unique_code_df = self.filtered_df[self.filtered_df['Unique_Code'] == unique_code]
            total_points = len(unique_code_df)
            total_windows = total_points // self.window_size + (1 if total_points % self.window_size != 0 else 0)
            time_instances = [(i * self.window_size, min((i + 1) * self.window_size, total_points)) for i in range(total_windows)]
            
            if 1 <= instance_to_plot <= total_windows:
                start_index, end_index = time_instances[instance_to_plot - 1]
                instance_data.append(unique_code_df.iloc[start_index:end_index])
            else:
                logger.info(f'Invalid instance number for unique code {unique_code}. Please choose a number between 1 and {total_windows}.')

        instance_df = pd.concat(instance_data)
        fig = px.scatter_matrix(instance_df, dimensions=cols, color='Unique_Code')
        fig.update_layout(width=1200, height=800, legend_title_font_size=22)
        fig.update_traces(marker=dict(size=1.5), diagonal_visible=False, showupperhalf=False)
        fig.show()




class StationarityChecker:
    def __init__(self, df):
        """
        Initialize the StationarityChecker with a DataFrame.
        """
        self.df = df

    def check_stationarity(self, series):
        """
        Perform the Augmented Dickey-Fuller test on a series.
        """
        result = adfuller(series.values)
        is_stationary = (result[1] <= 0.05) & (result[0] < result[4]['5%'])
        return is_stationary, result

    def filter_data(self, machine, period):
        """
        Filter the DataFrame by Machine and Period.
        """
        return self.df[(self.df['Machine'] == machine) & (self.df['Period'] == period)]

    def generate_report(self, machine, period):
        """
        Generate a report on the stationarity of each axis for each unique code.
        """
        filtered_df = self.filter_data(machine, period)
        unique_codes = filtered_df['Unique_Code'].unique()
        
        simple_report = []
        detailed_report = []

        for code in tqdm(unique_codes, desc="Processing unique codes"):
            detailed_report.append('-' * 40 + '\n')
            detailed_report.append(f"Results for Unique Code: {code}\n")
            code_df = filtered_df[filtered_df['Unique_Code'] == code]

            for axis in ['X_axis', 'Y_axis', 'Z_axis']:
                is_stationary, result = self.check_stationarity(code_df[axis])
                
                # Append to simple report
                simple_report.append(f"Unique Code: {code}, Axis: {axis}, Stationary: {'Yes' if is_stationary else 'No'}\n")
                
                # Append to detailed report
                detailed_report.append(f"Axis: {axis}\n")
                detailed_report.append(f"ADF Statistic: {result[0]:.6f}\n")
                detailed_report.append(f"p-value: {result[1]:.6f}\n")
                detailed_report.append('Critical Values:\n')
                for key, value in result[4].items():
                    detailed_report.append(f"\t{key}: {value:.3f}\n")
                if is_stationary:
                    detailed_report.append("\u001b[32mStationary\u001b[0m\n")
                else:
                    detailed_report.append("\x1b[31mNon-stationary\x1b[0m\n")
                detailed_report.append("\n")
        
        simple_report_str = "".join(simple_report)
        detailed_report_str = "".join(detailed_report)
        
        return simple_report_str, detailed_report_str
    
    
class UMAP_Visualize:
    def __init__(self, df):
        """
        Initialize the UMAP_Visualize with a DataFrame.
        """
        self.df = df

    def filter_data(self, machine, periods):
        """
        Filter the DataFrame by Machine and a list of Periods.
        """
        if isinstance(periods, list):
            return self.df[(self.df['Machine'] == machine) & (self.df['Period'].isin(periods))]
        else:
            raise ValueError("Periods should be provided as a list")

    def perform_umap(self, machine, periods, n_neighbors=15, features=['X_axis', 'Y_axis', 'Z_axis'], filter = True):
        """
        Perform UMAP transformation on the filtered data and plot the results.

        Parameters:
        machine (str): The machine identifier to filter the data.
        periods (list): A list of periods to filter the data.
        n_neighbors (int, optional): The number of neighbors to use for UMAP. Default is 15.
        """
        # Filter the data
        filtered_df = self.filter_data(machine, periods)

        # Define the features to use for UMAP
        features = features
        X = filtered_df[features]

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select the target labels (assuming 'Label' column exists in the dataset)
        target = filtered_df['Label'].astype(int).values

        # Perform UMAP
        reducer = umap.UMAP(random_state=42, n_jobs=1, n_neighbors=n_neighbors)
        X_umap = reducer.fit_transform(X_scaled)
        
        # Create a custom colormap
        cmap = ListedColormap(['blue', 'orange'])
        norm = plt.Normalize(0, 1)

        # Plot the UMAP results
        plt.figure(figsize=(14, 5))
        scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=target, cmap=cmap, norm=norm, s=5)
        
        # Create a colorbar with custom labels
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Normal', 'Anomalous'])
        cbar.set_label('Target')

        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('UMAP - Embeddings')
        plt.grid(True)
        plt.show()
        
class PCA_Visualize:
    def __init__(self, df):
        """
        Initialize the PCA_Visualize with a DataFrame.
        """
        self.df = df

    def filter_data(self, machine, periods):
        """
        Filter the DataFrame by Machine and a list of Periods.
        """
        if isinstance(periods, list):
            return self.df[(self.df['Machine'] == machine) & (self.df['Period'].isin(periods))]
        else:
            raise ValueError("Periods should be provided as a list")

    def perform_pca(self, machine, periods, features=['X_axis', 'Y_axis', 'Z_axis']):
        """
        Perform PCA transformation on the filtered data and plot the first two principal components.
        """
        # Filter the data
        filtered_df = self.filter_data(machine, periods)

        # Define the features to use for PCA
        features = features
        X = filtered_df[features]

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Select the target labels (assuming 'Label' column exists in the dataset)
        target = filtered_df['Label'].astype(int).values
        
        # Perform PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Calculate variance explained by the first two components
        explained_variance = pca.explained_variance_ratio_ * 100
        logger.info(f"Variance explained by the first component: {explained_variance[0]:.2f}%")
        logger.info(f"Variance explained by the second component: {explained_variance[1]:.2f}%")
        
        cmap = ListedColormap(['blue', 'orange'])
        norm = plt.Normalize(0, 1)

        # Plot the PCA results
        plt.figure(figsize=(14, 5))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap=cmap, norm=norm, s=5)
        
        # Create a colorbar with custom labels
        cbar = plt.colorbar(scatter, ticks=[0, 1])
        cbar.ax.set_yticklabels(['Normal', 'Anomalous'])
        cbar.set_label('Target')

        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA - First Two Components')
        plt.grid(True)
        plt.show()
        
        
        
def plot_correlation_matrix(data: pd.DataFrame, title: str, method: str = "pearson") -> None:
    """
    Plots a heatmap of the Pearson correlation matrix for the given dataset.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset for which the correlation matrix will be calculated.
    title : str
        The title for the heatmap plot.
    """
    plt.figure(figsize=(16, 14))
    correlation_matrix = data.corr(method)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size": 12},
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
        square=True
    )
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    cbar = plt.gcf().axes[-1]
    cbar.tick_params(labelsize=12)
    plt.title(title, size=16)
    plt.show()
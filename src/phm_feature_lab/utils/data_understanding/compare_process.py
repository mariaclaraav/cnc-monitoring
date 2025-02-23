from typing import List, Optional, Union, Dict, Any
import dask.dataframe as dd
from dask import compute
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import ks_2samp, entropy
from sklearn.preprocessing import StandardScaler
import itertools

class CompareProcess:
    """
    A class for comparing and visualizing process distributions using
    statistical and graphical methods.
    """

    def __init__(self, df: Union[pd.DataFrame, Any]) -> None:
        """
        Initialize the CompareProcess class.

        Parameters:
        - df: Input data containing 'Process' and feature columns. Can be a pandas or dask DataFrame.
        """
        self.__df = df.copy()

    def __compute_and_filter(self, col: str, processes: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Helper method to compute and filter the data.
        
        Parameters:
        - col: Column to analyze.
        - processes: List of processes to filter. Default is None.

        Returns:
        - pandas.DataFrame: Filtered and computed DataFrame.
        """
        data = self.__df[['Process', col]].compute()
        data['Process'] = data['Process'].astype(str)
        if processes:
            data = data[data['Process'].isin(processes)]
        return data

    def cdf(self, col: str, plot_type: str = 'matplotlib', processes: Optional[List[str]] = None) -> None:
        """
        Plot the Cumulative Distribution Function (CDF) of a specified column.

        Parameters:
        - col: Column to analyze.
        - plot_type: Type of plot ('matplotlib' or 'plotly'). Default is 'matplotlib'.
        - processes: List of processes to filter. Default is None.
        """
        data = self.__compute_and_filter(col, processes)

        if plot_type == 'matplotlib':
            plt.figure(figsize=(10, 7))
            sns.ecdfplot(data=data, x=col, hue='Process', palette='tab10')
            plt.title(f'CDF by Process for {col}')
            plt.xlabel(col)
            plt.ylabel('CDF')
            plt.legend(title='Process')
            plt.tight_layout()
            plt.show()
        elif plot_type == 'plotly':
            fig = px.ecdf(data, x=col, color='Process', title=f'CDF by Process for {col}')
            fig.update_layout(
                xaxis_title=col,
                yaxis_title='CDF',
                legend_title='Process',
                template='plotly_white'
            )
            fig.show()
        else:
            raise ValueError("plot_type must be either 'matplotlib' or 'plotly'")

    def kde(self, col: str, plot_type: str = 'matplotlib', processes: Optional[List[str]] = None) -> None:
        """
        Plot the Kernel Density Estimate (KDE) of a specified column.

        Parameters:
        - col: Column to analyze.
        - plot_type: Type of plot ('matplotlib' or 'plotly'). Default is 'matplotlib'.
        - processes: List of processes to filter. Default is None.
        """
        data = self.__compute_and_filter(col, processes)

        if plot_type == 'matplotlib':
            plt.figure(figsize=(12, 6))
            sns.kdeplot(data=data, x=col, hue='Process', fill=True, palette='tab10')
            plt.title(f'KDE by Process - {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            plt.tight_layout()
            plt.show()
        elif plot_type == 'plotly':
            fig = px.density_contour(data, x=col, color='Process', title=f'KDE by Process - {col}')
            fig.update_layout(
                xaxis_title=col,
                yaxis_title='Density',
                legend_title='Process',
                template='plotly_white'
            )
            fig.show()
        else:
            raise ValueError("plot_type must be either 'matplotlib' or 'plotly'")

    def histogram(self, col: str, plot_type: str = 'matplotlib', processes: Optional[List[str]] = None, bins: int = 10, log_scale: bool = False) -> None:
        """
        Plot the histogram of a specified column.

        Parameters:
        - col: Column to analyze.
        - plot_type: Type of plot ('matplotlib' or 'plotly'). Default is 'matplotlib'.
        - processes: List of processes to filter. Default is None.
        - bins: Number of bins for the histogram. Default is 10.
        - log_scale: Whether to use a logarithmic scale for the y-axis. Default is False.
        """
        data = self.__compute_and_filter(col, processes)

        if plot_type == 'matplotlib':
            plt.figure(figsize=(12, 6))
            sns.histplot(data=data, x=col, hue='Process', kde=True, bins=bins, palette='tab10')
            plt.title(f'Histogram and KDE by Process - {col}')
            plt.xlabel(col)
            plt.ylabel('Density')
            if log_scale:
                plt.yscale('log')
            plt.tight_layout()
            plt.show()
        elif plot_type == 'plotly':
            fig = px.histogram(data, x=col, color='Process', nbins=bins, title=f'Histogram by Process for {col}')
            if log_scale:
                fig.update_layout(yaxis_type='log')
            fig.update_layout(
                xaxis_title=col,
                yaxis_title='Density',
                legend_title='Process',
                template='plotly_white'
            )
            fig.show()
        else:
            raise ValueError("plot_type must be either 'matplotlib' or 'plotly'")

    def violin(self, col: str, processes: Optional[List[str]] = None) -> None:
        """
        Plot a violin plot of a specified column.

        Parameters:
        - col: Column to analyze.
        - processes: List of processes to filter. Default is None.
        """
        data = self.__compute_and_filter(col, processes)

        plt.figure(figsize=(15, 6))
        sns.violinplot(data=data, x='Process', y=col, palette='tab10')
        plt.title(f'Violin Plot - {col}')
        plt.xlabel('Process')
        plt.ylabel(col)
        plt.tight_layout()
        plt.show()

    def ks_test(self, col: str, processes: List[str], normalize: bool = False) -> List[Dict[str, Union[str, float, bool]]]:
        """
        Perform the Kolmogorov-Smirnov (KS) test between process pairs.

        Parameters:
        - col: Column to analyze.
        - processes: List of processes to compare.
        - normalize: Whether to normalize the data. Default is False.

        Returns:
        - List[Dict]: Results of the KS test for each process pair.
        """
        data = self.__compute_and_filter(col, processes)
        process_pairs = list(itertools.combinations(processes, 2))
        results = []

        for p1, p2 in process_pairs:
            data1 = data[data['Process'] == p1][col].values
            data2 = data[data['Process'] == p2][col].values

            if normalize:
                scaler = StandardScaler()
                data1 = scaler.fit_transform(data1.reshape(-1, 1)).ravel()
                data2 = scaler.fit_transform(data2.reshape(-1, 1)).ravel()

            statistic, p_value = ks_2samp(data1, data2)
            results.append({
                'Process 1': p1,
                'Process 2': p2,
                'Statistic': statistic,
                'P-value': p_value,
                'Different': p_value < 0.05
            })

        return results

    def kl_divergence(self, col: str, processes: List[str], bins: int = 100) -> List[Dict[str, Union[str, float]]]:
        """
        Compute the Kullback-Leibler (KL) divergence between process pairs.

        Parameters:
        - col: Column to analyze.
        - processes: List of processes to compare.
        - bins: Number of bins for histogram estimation. Default is 100.

        Returns:
        - List[Dict]: Results of the KL divergence for each process pair.
        """
        data = self.__compute_and_filter(col, processes)
        process_pairs = list(itertools.permutations(processes, 2))
        results = []

        for p1, p2 in process_pairs:
            data1 = data[data['Process'] == p1][col]
            data2 = data[data['Process'] == p2][col]

            p1_hist, _ = np.histogram(data1, bins=bins, density=True)
            p2_hist, _ = np.histogram(data2, bins=bins, density=True)

            p1_hist += 1e-10
            p2_hist += 1e-10

            kl_div = entropy(p1_hist, p2_hist)
            results.append({'Process 1': p1, 'Process 2': p2, 'KL Divergence': kl_div})

        return results

    def kl_divergence_heatmap(self, col: str, processes: List[str], bins: int = 100, cmap: str = 'magma', kl_dict = None) -> None:
        """
        Plot a heatmap of the KL divergence between process pairs.

        Parameters:
        - col: Column to analyze.
        - processes: List of processes to compare.
        - bins: Number of bins for histogram estimation. Default is 100.
        - cmap: Colormap for the heatmap. Default is 'magma'.
        """
        data = self.__compute_and_filter(col, processes)
        n = len(processes)
        kl_matrix = np.zeros((n, n))

        for i, p1 in enumerate(processes):
            for j, p2 in enumerate(processes):
                if i != j:
                    data1 = data[data['Process'] == p1][col]
                    data2 = data[data['Process'] == p2][col]

                    p1_hist, _ = np.histogram(data1, bins=bins, density=True)
                    p2_hist, _ = np.histogram(data2, bins=bins, density=True)

                    p1_hist += 1e-10
                    p2_hist += 1e-10


                    kl_matrix[i, j] = entropy(p1_hist, p2_hist)
        df_kl = pd.DataFrame(kl_matrix, index=processes, columns=processes)
    
        # Initialize the dictionary if it is None
        if kl_dict is None:
            kl_dict = {}
        
        # Store the DataFrame in the dictionary with 'col' as the key
        kl_dict[col] = df_kl

        # Create the heatmap
        plt.figure(figsize=(15, 12))
        sns.heatmap(df_kl, xticklabels=processes, yticklabels=processes, annot=True, cmap=cmap)
        plt.title(f'KL Divergence - {col}')
        plt.grid(False)
        plt.show()
            
        return kl_dict 
   
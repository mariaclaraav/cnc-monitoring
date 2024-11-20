import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import ks_2samp

class ComparePeriods:
    def __init__(self, machine, df):
        self.machine = machine
        self.df = df

    def cdf_kde(self, cols, downsample_factor=1):
        """
        Plot CDF and KDE plots for the specified machine and columns by period.

        Parameters:
        cols (list): The list of columns to include in the CDF and KDE plots.
        downsample_factor (int): Factor by which to downsample the DataFrame.
        """
        # Filter the DataFrame by the specified machine
        df_filtered = self.df[self.df['Machine'] == self.machine]

        # Downsample the DataFrame if needed
        if downsample_factor > 1:
            df_filtered = df_filtered[::downsample_factor]

        # Create subplots
        nrows = len(cols)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(13, 5 * nrows))

        if nrows == 1:
            axes = [axes]

        for i, col in enumerate(cols):
            # Plot CDF using Seaborn
            sns.ecdfplot(data=df_filtered, x=col, hue='Period', ax=axes[i][0])
            axes[i][0].set_title(f'CDF by Period - {self.machine}')
            axes[i][0].set_xlabel(col)
            axes[i][0].set_ylabel('CDF')

            # Plot KDE using Seaborn
            sns.kdeplot(data=df_filtered, x=col, hue='Period', fill=False, common_norm=False, linewidth=2, ax=axes[i][1])
            axes[i][1].set_title(f'KDE by Period - {self.machine}')
            axes[i][1].set_xlabel(col)
            axes[i][1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def KS_test(self, periods, data_column='data_column', downsample_factor=1):
        """
        Compare data distributions between specified periods for a given machine using the Kolmogorov-Smirnov test.

        Parameters:
        periods (list): List of periods to compare.
        data_column (str): The data column to use for the KS test.
        downsample_factor (int): Factor by which to downsample the DataFrame.
        """
        # Filter the DataFrame by the specified machine
        df_filtered = self.df[self.df['Machine'] == self.machine]

        # Downsample the DataFrame if needed
        if downsample_factor > 1:
            df_filtered = df_filtered[::downsample_factor]

        if len(periods) < 2:
            print("Please provide at least two periods to compare.")
            return
        
        print(f'Kolmogorov-Smirnov test for: {self.machine}\n')
        # Iterate over all unique pairs of periods
        for i in range(len(periods) - 1):
            for j in range(i + 1, len(periods)):
                period1, period2 = periods[i], periods[j]
                data1 = df_filtered[df_filtered['Period'] == period1][data_column]
                data2 = df_filtered[df_filtered['Period'] == period2][data_column]

                # Perform the Kolmogorov-Smirnov test
                ks_stat, p_value = ks_2samp(data1, data2)

                # Print the results
                print(f'Comparing Period {period1} and {period2}: KS Statistic={np.round(ks_stat, 3)}, p-value={np.round(p_value, 5)}')

    def scatter_plot(self, cols=['X_axis', 'Y_axis', 'Z_axis'], downsample_factor=1):
        """
        Plot a scatter matrix for the specified machine and columns. 
        
        """
        # Filter the DataFrame by the specified machine
        df_filtered = self.df[self.df['Machine'] == self.machine]

        # Downsample the DataFrame if needed
        if downsample_factor > 1:
            df_filtered = df_filtered[::downsample_factor]

        # Create the scatter matrix
        fig = px.scatter_matrix(df_filtered, dimensions=cols, color='Period')

        # Update the layout of the plot
        fig.update_layout(width=1000, height=600, legend_title_font_size=20)

        # Update trace characteristics
        fig.update_traces(marker=dict(size=2), diagonal_visible=False, showupperhalf=False)

        # Show the plot
        fig.show()

class CompareMachines:
    def __init__(self, period, df):
        self.period = period
        self.df = df

    def cdf_kde(self, cols, downsample_factor=1):
        """
        Plot CDF and KDE plots for the specified period and columns by machine.

        Parameters:
        cols (list): The list of columns to include in the CDF and KDE plots.
        downsample_factor (int): Factor by which to downsample the DataFrame.
        """
        # Filter the DataFrame by the specified period
        df_filtered = self.df[self.df['Period'] == self.period]

        # Downsample the DataFrame if needed
        if downsample_factor > 1:
            df_filtered = df_filtered[::downsample_factor]

        # Create subplots
        nrows = len(cols)
        fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(13, 5 * nrows))

        if nrows == 1:
            axes = [axes]

        for i, col in enumerate(cols):
            # Plot CDF using Seaborn
            sns.ecdfplot(data=df_filtered, x=col, hue='Machine', ax=axes[i][0])
            axes[i][0].set_title(f'CDF by Machine - {self.period}')
            axes[i][0].set_xlabel(col)
            axes[i][0].set_ylabel('CDF')

            # Plot KDE using Seaborn
            sns.kdeplot(data=df_filtered, x=col, hue='Machine', fill=False, common_norm=False, linewidth=2, ax=axes[i][1])
            axes[i][1].set_title(f'KDE by Machine - {self.period}')
            axes[i][1].set_xlabel(col)
            axes[i][1].set_ylabel('Density')

        plt.tight_layout()
        plt.show()

    def KS_test(self, machines, data_column='data_column', downsample_factor=1):
        """
        Compare data distributions between specified machines for a given period using the Kolmogorov-Smirnov test.

        Parameters:
        machines (list): List of machines to compare.
        data_column (str): The data column to use for the KS test.
        downsample_factor (int): Factor by which to downsample the DataFrame.
        """
        # Filter the DataFrame by the specified period
        df_filtered = self.df[self.df['Period'] == self.period]

        # Downsample the DataFrame if needed
        if downsample_factor > 1:
            df_filtered = df_filtered[::downsample_factor]

        if len(machines) < 2:
            print("Please provide at least two machines to compare.")
            return
        
        print(f'Kolmogorov-Smirnov test for period: {self.period}\n')
        # Iterate over all unique pairs of machines
        for i in range(len(machines) - 1):
            for j in range(i + 1, len(machines)):
                machine1, machine2 = machines[i], machines[j]
                data1 = df_filtered[df_filtered['Machine'] == machine1][data_column]
                data2 = df_filtered[df_filtered['Machine'] == machine2][data_column]

                # Perform the Kolmogorov-Smirnov test
                ks_stat, p_value = ks_2samp(data1, data2)

                # Print the results
                print(f'Comparing Machine {machine1} and {machine2}: KS Statistic={np.round(ks_stat, 3)}, p-value={np.round(p_value, 3)}')

    def scatter_plot(self, cols=['X_axis', 'Y_axis', 'Z_axis'], downsample_factor=1):
        """
        Plot a scatter matrix for the specified period and columns. 
        
        Parameters:
        cols (list): List of columns to include in the scatter plot matrix.
        downsample_factor (int): Factor by which to downsample the DataFrame.
        """
        # Filter the DataFrame by the specified period
        df_filtered = self.df[self.df['Period'] == self.period]

        # Downsample the DataFrame if needed
        if downsample_factor > 1:
            df_filtered = df_filtered[::downsample_factor]

        # Create the scatter matrix
        fig = px.scatter_matrix(df_filtered, dimensions=cols, color='Machine')

        # Update the layout of the plot
        fig.update_layout(width=1000, height=600, legend_title_font_size=20)

        # Update trace characteristics
        fig.update_traces(marker=dict(size=2), diagonal_visible=False, showupperhalf=False)

        # Show the plot
        fig.show()
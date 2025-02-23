import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from matplotlib.colors import to_hex
from typing import Optional, Union, List

## TODO: REFATORAR ESSA CLASSE 
class PlotRMS:
    def __init__(self, df: pd.DataFrame, cmap: str = 'tab20b', group_set: str = None):
        self.__df = df.copy()
        self.__cmap = cmap
        self.__group_set = group_set
        
    def __check_df(self, df: pd.DataFrame) -> None:
        if df.empty:
            raise ValueError("Input DataFrame is empty. Cannot process.")
        
    def __check_unique_code(self, df: pd.DataFrame) -> None:
        if 'Unique_Code' not in df.columns:
            raise KeyError("Input DataFrame must contain a 'Unique_Code' column.")
    
    def __get_ym(self):
        if self.__group_set == 'train': 
            unique_year_month = ['Feb_2019', 'Aug_2019', 'Feb_2020', 'Aug_2020']
        elif self.__group_set == 'test':
            unique_year_month = ['Feb_2021', 'Aug_2021']
        else:
            unique_year_month = ['Feb_2019', 'Aug_2019', 'Feb_2020', 'Aug_2020', 'Feb_2021', 'Aug_2021']
        return unique_year_month
    
    def __color_map(self):
        unique_year_month = self.__get_ym()
        num_colors = len(unique_year_month)
        
        # Use a colormap with a larger number of colors
        colormap = cm.get_cmap(self.__cmap, 256)  # 256 colors in the colormap
        
        # Select colors at evenly spaced intervals to maximize differences
        indices = np.linspace(0, 255, num_colors, dtype=int)
        
        # Map the selected colors to the Year_Month values
        return {ym: to_hex(colormap(i)) for i, ym in zip(indices, unique_year_month)}
    
    def __get_cmap(self, df: pd.DataFrame):
        return df['Year_Month'].map(self.__color_map())
    
    def __filter_process(self, df: pd.DataFrame, machine: str, operation: str, axis: str) -> pd.DataFrame:
        self.__check_df(df)
        self.__check_unique_code(df)
        df_filtered = df[df['Axis'] == axis]
        return df_filtered[df_filtered['Unique_Code'].str.contains(f"{machine}_{operation}")]
    
    def __get_axis(self, df: pd.DataFrame, y_min: float, y_max: float, var: str) -> List[str]:
        if y_min is None or y_max is None:
            y_min_local = df[var].min()
            y_max_local = df[var].max()

            # Add margin
            y_range = y_max_local - y_min_local
            y_margin = y_range * 0.1  # 10% margin
            y_min_local -= y_margin
            y_max_local += y_margin
            
        else:
            y_min_local = y_min
            y_max_local = y_max
        
        return y_min_local, y_max_local
    
    def __order_unique_code(self, df: pd.DataFrame, column_name: str ='Unique_Code'):
    
        # Extract year, month, and last number from the Unique_Code
        df['Year'] = df[column_name].str.extract(r'_(\d{4})_')[0].astype(int)
        df['Month'] = df[column_name].str.extract(r'(Feb|Aug)_')[0]
        df['Last_Number'] = df[column_name].str.extract(r'_(\d+)$')[0].astype(int)
        # Define the order for the months
        month_order = {'Feb': 1, 'Aug': 2}
        df['Month_Order'] = df['Month'].map(month_order)

        # Sort the DataFrame by Year, Month_Order, and Last_Number
        df.sort_values(by=['Year', 'Month_Order', 'Last_Number'], inplace=True)

        # Drop the helper columns
        df.reset_index(drop=True, inplace=True)
    
    
    def __get_metadata(self, df: pd.DataFrame, machine: str, operation: str, axis: str) -> pd.DataFrame:
        df_filtered = self.__filter_process(df, machine, operation, axis)
        
        # Remove the prefix from 'Unique_Code'
        prefix = f"{machine}_{operation}_"
                
        # Extract 'Month' and 'Year' from 'Unique_Code'
        df_filtered['Month'] = df_filtered['Unique_Code'].str.extract(r'(Feb|Aug)')[0]
        df_filtered['Year'] = df_filtered['Unique_Code'].str.extract(r'_(\d{4})_')[0]
        df_filtered['Last_Number'] = df_filtered['Unique_Code'].str.extract(r'_(\d+)$')[0]

        # Create 'Year_Month' column
        df_filtered['Year_Month'] = df_filtered['Month'].astype(str) + "_" + df_filtered['Year'].astype(str)
        
        df_filtered['Unique_Code'] = df_filtered['Unique_Code'].str.replace(f'^{prefix}', '', regex=True)
        return df_filtered

    
    def plot_rms(
        self,
        var: str,
        machine: str,
        operation: str,
        axis: Optional[Union[str, List[str]]] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        anomaly_column: Optional[str] = None,
        
    ) -> None:
        
        df_metadata = self.__get_metadata(self.__df, machine, operation, axis)
        self.__check_df(df_metadata)
        
        unique_year_month = self.__get_ym()
        df_metadata['Color'] = self.__get_cmap(df_metadata)
        y_min_local, y_max_local = self.__get_axis(df_metadata, y_min, y_max, var)
        self.__order_unique_code(df_metadata)
        
        plt.figure(figsize= (12,4))

        try:
            # Plot lines and points
            for i in range(len(df_metadata) - 1):
                plt.plot(
                    df_metadata['Unique_Code'].iloc[i:i+2],
                    df_metadata[var].iloc[i:i+2],
                    linestyle='-',
                    color=df_metadata['Color'].iloc[i+1]
                )
                
                if anomaly_column:
                    is_anomaly = df_metadata[anomaly_column].iloc[i] == 'Anomaly'
                else: 
                    is_anomaly = False
                plt.scatter(
                    df_metadata['Unique_Code'].iloc[i],
                    df_metadata[var].iloc[i],
                    color= "red" if is_anomaly else df_metadata['Color'].iloc[i],
                    marker= 'x' if is_anomaly else 'o'
                )
                
            if anomaly_column:
                is_anomaly_last = df_metadata[anomaly_column].iloc[-1] == 'Anomaly'
            else:
                is_anomaly_last = False
            plt.scatter(
                df_metadata['Unique_Code'].iloc[-1],
                df_metadata[var].iloc[-1],
                color= "red" if is_anomaly_last else df_metadata['Color'].iloc[-1],
                marker= 'x' if is_anomaly else'o'
            )
            
            plt.ylim(y_min_local, y_max_local)
            # Add labels, title, and legend
            plt.xlabel('Unique_Code')
            plt.ylabel(f'{var} Value')
            plt.title(f'{var} for {machine}-{operation} ({axis or "combined"})')
            plt.xticks(
                ticks=range(len(df_metadata['Last_Number'])),
                labels=df_metadata['Last_Number'],
                rotation=25
            )

            # Create legend
            handles = [plt.Line2D([0], [0], color=self.__color_map()[ym], lw=2) for ym in unique_year_month]
            plt.legend(handles, unique_year_month, title='Month_Year', loc='upper right', fontsize=10, title_fontsize=10)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            raise RuntimeError(f"An error occurred while plotting: {str(e)}")        
   


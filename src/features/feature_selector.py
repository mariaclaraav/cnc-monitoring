import pandas as pd

class FeatureSelector:
    """
    A class to filter and categorize features based on their axis (X, Y, Z) and a frequency threshold.
    """

    def __init__(self, df, frequency_threshold=90):
        """
        Initializes the FeatureFilter.

        Parameters:
            df (pd.DataFrame): DataFrame containing feature columns.
            frequency_threshold (float): Minimum frequency value (in Hz) for feature inclusion.
        """
        self.__df = df
        self.__frequency_threshold = frequency_threshold

    def __is_above_threshold(self, col):
        """
        Determines if a feature meets the frequency threshold.

        Parameters:
            col (str): Column name representing a feature.

        Returns:
            bool: True if the feature meets the frequency threshold, False otherwise.
        """
        try:
            # Extract frequency range (e.g., "18.0-41.0Hz" -> [18.0, 41.0])
            freq_range = col.split('_')[-1].replace('Hz', '').split('-')
            min_freq = float(freq_range[0])  # Get the minimum frequency
            return min_freq >= self.__frequency_threshold
        except (ValueError, IndexError):
            # If parsing fails, assume the column is not frequency-related
            return False

    def get_features(self):
        """
        Returns all features (X, Y, Z) combined into a single list, filtered by the frequency threshold.

        Returns:
            list: A list of features that meet the frequency threshold.
        """
        feat = [col for col in self.__df.columns if col.endswith("Hz") and self.__is_above_threshold(col)]
        return feat

    def get_separated_features(self):
        """
        Returns features categorized by axis (X, Y, Z), filtered by the frequency threshold.

        Returns:
            dict: A dictionary with keys 'X', 'Y', 'Z' and their corresponding feature lists.
        """
        feat = [col for col in self.__df.columns if col.endswith("Hz") and self.__is_above_threshold(col)]
        x_feat = [name for name in feat if name.startswith('X')]
        y_feat = [name for name in feat if name.startswith('Y')]
        z_feat = [name for name in feat if name.startswith('Z')]

        return {
            'X': x_feat,
            'Y': y_feat,
            'Z': z_feat
        }

import contextlib
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import entropy
from tqdm import tqdm
from xgboost import XGBRegressor
import ruptures as rpt
from scipy.stats import entropy, ks_2samp, kurtosis

@contextlib.contextmanager
def tqdm_joblib(tqdm_object: tqdm):
    """
    Context manager to enable tqdm progress bar for joblib parallel processing.

    Args:
        tqdm_object (tqdm): tqdm progress bar object.

    Yields:
        tqdm_object: The same tqdm progress bar object, updated by joblib tasks.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class FRUFS(TransformerMixin, BaseEstimator):
    """
    Feature Ranking Using Feature Subsets (FRUFS) class for feature selection.

    The FRUFS class implements a feature selection method by training a regression model
    to predict each feature from all other features, and uses the feature importances
    and R^2 scores from these models to rank the features.

    Attributes:
        random_state (Optional[int]): Random state for reproducibility.
        k (Union[int, float]): Number of features to select; can be an integer or a fraction (float).
        verbose (int): Verbosity level (0: progress bar, >0: detailed output).
        columns_ (Optional[np.ndarray]): Array of feature names or indices.
        feat_imps_ (Optional[np.ndarray]): Feature importance scores without weighting.
        r2_scores_ (Optional[np.ndarray]): R^2 scores for each feature prediction.
        feat_imps_r2_ (Optional[np.ndarray]): Feature importance scores weighted by R^2 scores.
        importance_matrix_ (Optional[np.ndarray]): Matrix of feature importances for each feature prediction.
        model_name (str): Internal model identifier.
        model_display_name (str): Display name of the model for plotting.
    """

    MODEL_DISPLAY_NAMES = {
        'dt': 'Decision Tree',
        'rf': 'Random Forest',
        'lgbm': 'LightGBM',
        'elasticnet': 'ElasticNet',
        'xgb': 'XGBoost',
    }

    def __init__(
        self,
        model: str = 'dt',
        k: Union[int, float] = 1.0,
        verbose: int = 0,
        random_state: Optional[int] = None
    ):
        """
        Initialize the FRUFS feature selection class.

        Args:
            model (str): Model to use for regression tasks ('dt', 'rf', 'lgbm', 'elasticnet', 'xgb').
            k (Union[int, float]): Number of features to select; can be an integer or a fraction (float).
            verbose (int): Verbosity level (0: progress bar, >0: detailed output).
            random_state (Optional[int]): Random state for reproducibility.
        """
        self.random_state = random_state
        self.k = k
        self.verbose = verbose
        self.columns_ = None
        self.feat_imps_ = None
        self.r2_scores_ = None
        self.feat_imps_r2_ = None
        self.model = self._get_model(model)
        self.model_name = model
        self.model_display_name = self.MODEL_DISPLAY_NAMES.get(model, model)

    def _get_model(self, model: str) -> BaseEstimator:
        """
        Return the appropriate regression model based on the input string.

        Args:
            model (str): Internal model identifier.

        Returns:
            BaseEstimator: An instance of the regression model.

        Raises:
            ValueError: If the model identifier is not supported.
        """
        if model == 'dt':
            return DecisionTreeRegressor(random_state=self.random_state)
        elif model == 'rf':
            return RandomForestRegressor(random_state=self.random_state, bootstrap=True, n_jobs=-1)
        elif model == 'lgbm':
            return LGBMRegressor(random_state=self.random_state, n_jobs=-1)
        elif model == 'elasticnet':
            return ElasticNet(random_state=self.random_state)
        elif model == 'xgb':
            return XGBRegressor(random_state=self.random_state, n_jobs=-1)
        else:
            raise ValueError(
                f"Model '{model}' is not supported. Choose from 'dt', 'rf', 'lgbm', 'elasticnet', 'xgb'."
            )

    def cal_feat_imp_(self, index: int, X: np.ndarray, model: BaseEstimator) -> Tuple[np.ndarray, float]:
        """
        Calculate feature importance for predicting the feature at the given index and compute R^2 score.

        Args:
            index (int): Index of the feature being predicted.
            X (np.ndarray): Input data matrix.
            model (BaseEstimator): Model to use for prediction.

        Returns:
            Tuple[np.ndarray, float]: Feature importance array and R^2 score for the prediction.
        """
        feat_imp = np.zeros(X.shape[1])

        # Get the feature name using the index
        feature_name = self.columns_[index]

        # Print the feature name before R^2 calculation
        if self.verbose > 0:
            print(f"Predicting feature '{feature_name}'...")

        # Remove the feature at 'index' and use the remaining features to predict it
        x_train = np.concatenate((X[:, :index], X[:, index + 1:]), axis=1)
        y_train = X[:, index]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_train)

        # Calculate R^2 score for the prediction
        r2 = r2_score(y_train, y_pred)

        # Print the R^2 score after the prediction
        if self.verbose > 0:
            print(f"R^2 for predicting feature '{feature_name}': {r2:.4f}")

        # Handle model's feature importance (tree-based or linear models)
        inds = np.concatenate((np.arange(index), np.arange(index + 1, X.shape[1])))
        try:
            feat_imp[inds] += model.feature_importances_
        except AttributeError:
            feat_imp[inds] += model.coef_

        return feat_imp, r2

    def fit(self, X: Union[pd.DataFrame, np.ndarray]):
        """
        Fit the FRUFS model and compute the feature importance for all features,
        along with the R^2 score for each feature prediction.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data.
        """
        # Handle pandas DataFrame: Convert to numpy array and track column names
        if isinstance(X, pd.DataFrame):
            self.columns_ = np.array(X.columns)
            X = X.values
        else:
            self.columns_ = np.arange(X.shape[1])

        # Store the original columns before any sorting
        original_columns = self.columns_.copy()

        # Initialize the importance matrix (features x features)
        self.importance_matrix_ = np.zeros((X.shape[1], X.shape[1]))

        # Parallel computation of feature importances and R^2 scores
        parallel = Parallel(n_jobs=-1, verbose=self.verbose)
        if self.verbose == 0:
            with tqdm_joblib(tqdm(desc="Progress bar", total=X.shape[1])) as progress_bar:
                results = parallel(
                    delayed(self.cal_feat_imp_)(i, X, self.model) for i in range(X.shape[1])
                )
        else:
            results = [self.cal_feat_imp_(i, X, self.model) for i in range(X.shape[1])]

        # Separate feature importance and R^2 scores
        feat_imps, r2_scores = zip(*results)

        # Store the R^2 scores and their corresponding feature names in the same order
        self.r2_scores_ = np.array(r2_scores)
        self.r2_scores_ordered_ = original_columns

        # Convert feat_imps to NumPy array
        feat_imps = np.array(feat_imps)

        # Fill the importance matrix
        for i, imp in enumerate(feat_imps):
            self.importance_matrix_[i, :] = imp

        # Standard feature importance (without weighting)
        self.feat_imps_ = np.mean(feat_imps, axis=0)

        # Average feature importance scores, weighted by the R^2 scores
        self.feat_imps_r2_ = np.average(feat_imps, axis=0, weights=self.r2_scores_)

        # Sort features based on unweighted importance
        inds = np.argsort(np.abs(self.feat_imps_))[::-1]
        self.columns_ = self.columns_[inds]
        self.feat_imps_ = self.feat_imps_[inds]

        # Sort features based on R^2-weighted importance
        inds_r2 = np.argsort(np.abs(self.feat_imps_r2_))[::-1]
        self.columns_r2_ = original_columns[inds_r2]
        self.feat_imps_r2_ = self.feat_imps_r2_[inds_r2]

        # Adjust 'k' if it's a float representing a fraction of features
        if isinstance(self.k, float):
            self.k = round(self.k * X.shape[1])

    def display_r2_scores(self):
        """
        Display the R^2 scores along with the feature names in the order they were computed.
        """
        print("R^2 Scores for each feature:")
        for feature, r2_score_value in zip(self.r2_scores_ordered_, self.r2_scores_):
            print(f"Feature '{feature}': R^2 = {r2_score_value:.4f}")

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform the data by selecting the top 'k' features.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Transformed dataset with only the top 'k' features.
        """
        cols = self.columns_[:self.k]
        return X[cols] if isinstance(X, pd.DataFrame) else X[:, :self.k]

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit the model and then transform the data (select top 'k' features).

        Args:
            X (Union[pd.DataFrame, np.ndarray]): Input data.

        Returns:
            Union[pd.DataFrame, np.ndarray]: Transformed data with selected features.
        """
        self.fit(X)
        return self.transform(X)

    def feature_importance(
        self, top_x_feats: Optional[int] = None, figsize: Tuple[int, int] = (12, 16)
    ):
        """
        Plot the traditional feature importance scores.

        Args:
            top_x_feats (Optional[int]): Number of top features to display. If None, display all.
            figsize (Tuple[int, int]): Size of the figure.
        """
        y_axis, x_axis = self._get_axes_for_plot(top_x_feats, self.feat_imps_)

        plt.figure(figsize=figsize)
        sns.lineplot(x=x_axis, y=[self.k] * len(y_axis), linestyle='--', color='k')
        sns.barplot(
            x=x_axis,
            y=y_axis,
            orient="h",
            hue=y_axis,
            palette=sns.color_palette("tab10"),
            legend=False
        )
        title = f"Traditional Feature Importance - {self.model_display_name}"
        self._set_plot_labels(y_axis, title, self.columns_)

    def feature_importance_r2(
        self, top_x_feats: Optional[int] = None, figsize: Tuple[int, int] = (12, 16)
    ):
        """
        Plot the feature importance scores weighted by the R^2 scores.

        Args:
            top_x_feats (Optional[int]): Number of top features to display. If None, display all.
            figsize (Tuple[int, int]): Size of the figure.
        """
        y_axis, x_axis = self._get_axes_for_plot(top_x_feats, self.feat_imps_r2_)

        plt.figure(figsize=figsize)
        sns.lineplot(x=x_axis, y=[self.k] * len(y_axis), linestyle='--', color='k')
        sns.barplot(
            x=x_axis,
            y=y_axis,
            orient="h",
            hue=y_axis,
            palette=sns.color_palette("tab10"),
            legend=False
        )
        title = f"$R^2$ Weighted Feature Importance - {self.model_display_name}"
        self._set_plot_labels(y_axis, title, self.columns_r2_)

    def _get_axes_for_plot(
        self, top_x_feats: Optional[int], feat_imps: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Helper function to extract x and y axes for the plotting functions.

        Args:
            top_x_feats (Optional[int]): Number of top features to display.
            feat_imps (np.ndarray): Feature importance scores.

        Returns:
            Tuple[np.ndarray, np.ndarray]: y_axis (indices of features), x_axis (importance scores).
        """
        if top_x_feats is not None:
            y_axis = np.arange(top_x_feats)
            x_axis = feat_imps[:top_x_feats]
        else:
            y_axis = np.arange(len(feat_imps))
            x_axis = feat_imps
        return y_axis, x_axis

    def _set_plot_labels(
        self, y_axis: np.ndarray, title: str, columns: np.ndarray
    ):
        """
        Helper function to set labels and titles for the plotting functions.

        Args:
            y_axis (np.ndarray): y-axis values (feature indices).
            title (str): Title for the plot.
            columns (np.ndarray): Array of feature names.
        """
        if isinstance(columns[0], str):
            plt.yticks(y_axis, columns[:len(y_axis)], fontsize=15)
        else:
            plt.yticks(
                y_axis,
                [f"Feature {i}" for i in columns[:len(y_axis)]],
                fontsize=15
            )
        plt.xlabel("Importance Scores")
        plt.ylabel("Features")
        plt.title(title)
        plt.show()

    def process_feature_name(self, feature_name: str) -> str:
        """
        Process the feature name to remove 'node' and keep only the prefix (X_, Y_, Z_) and the third part.

        Args:
            feature_name (str): Original feature name.

        Returns:
            str: Processed feature name.
        """
        parts = feature_name.split('_')
        if len(parts) >= 3:
            return f"{parts[0]}_{parts[2]}"  # Keep the first and third part
        return feature_name  # Return unchanged if it doesn't follow the expected pattern

    def plot_importance_heatmap(
        self,
        figsize: Tuple[int, int] = (10, 8),
        fontsize: int = 25,
        titlesize: int = 25,
        ticksize: int = 20,
        rotation: int = 90,
        linewidth: float = 0.5,
        fmt: str = '.1f',
        cmap: str = 'crest'
    ):
        """
        Plot a heatmap showing the feature importance of each feature in predicting others,
        normalized per row so that each row sums to 1.

        Args:
            figsize (Tuple[int, int]): Size of the figure.
            fontsize (int): Font size for labels.
            titlesize (int): Font size for the title.
            ticksize (int): Font size for the tick labels.
            rotation (int): Rotation angle for x-tick labels.
            linewidth (float): Line width for the heatmap.
            fmt (str): Format string for annotations.
            cmap (str): Colormap to use for the heatmap.
        """
        plt.figure(figsize=figsize)
        # Apply the feature name processing to all feature names
        processed_feature_names = [self.process_feature_name(f) for f in self.columns_]

        # Create a mask to hide the upper triangle
        mask = np.triu(np.ones_like(self.importance_matrix_, dtype=bool))

        # Normalize the importance matrix per row so that each row sums to 1
        importance_matrix_normalized = self.importance_matrix_.copy()
        row_sums = importance_matrix_normalized.sum(axis=1, keepdims=True)

        # Avoid division by zero
        importance_matrix_normalized = np.divide(
            importance_matrix_normalized,
            row_sums,
            out=np.zeros_like(importance_matrix_normalized),
            where=row_sums != 0
        )

        # Plot the heatmap with the mask
        sns.heatmap(
            importance_matrix_normalized,
            annot=True,
            fmt=fmt,
            cmap=cmap,
            xticklabels=processed_feature_names,
            yticklabels=processed_feature_names,
            cbar_kws={"shrink": 0.8},
            linewidth=linewidth
        )

        title = f"Feature Importance Heatmap {self.model_display_name}"
        plt.title(title, fontsize=titlesize)
        plt.xlabel("Features", fontsize=fontsize)
        plt.ylabel("Target", fontsize=fontsize)

        plt.xticks(fontsize=ticksize, rotation=rotation)
        plt.yticks(fontsize=ticksize, rotation=rotation)

        cbar = plt.gcf().axes[-1]
        cbar.tick_params(labelsize=20)
        plt.show()
        
        
################# Ruptures - Detect changes in time series data #################


def detect_changes(
    df: pd.DataFrame,
    feature: str,
    width: int,
    pen: float = 1.0,
    model: str = "rbf",
    jump: Optional[int] = None,
    figsize: Tuple[int, int] = (15, 8)
) -> Tuple[List[int], int]:
    """
    Detects change points in a feature using the Window-based algorithm and plots the results.

    Parameters:
    - df: DataFrame containing the time series data.
    - feature: Name of the feature to analyze.
    - width: Window size for the sliding windows.
    - pen: Penalty used in the Window algorithm. Controls the number of breakpoints.
    - model: Model to use for change detection (e.g., "rbf" for variance-based changes).
    - jump: Step size for jumping over points (optional). If None, jump is not used.
    - figsize: Size of the figure (width, height).

    Returns:
    - A tuple containing:
        - A list of detected breakpoints.
        - The number of breakpoints.
    """
    
    # Extract the signal (feature values) from the DataFrame
    signal = df[feature].values
    
    # Apply the Window algorithm with or without jump
    algo = rpt.Window(width=width, model=model, jump=jump).fit(signal)
    bkps = algo.predict(pen=pen)

    # Plot the detected change points
    rpt.display(signal, bkps, figsize=figsize)
    plt.title(f'Change Point Detection for {feature}', size=13)
    plt.show()
    
    # Output the number of breakpoints detected (excluding the last point)
    num_breakpoints = len(bkps) - 1
    print(f"Feature: {feature}")
    print(f"Breakpoints detected: {bkps}")
    print(f"Number of breakpoints: {num_breakpoints}")
    
    return bkps, num_breakpoints


def detect_changes_for_all_features(
    df: pd.DataFrame,
    features: List[str],
    width: int,
    pen: float = 1.0,
    model: str = "rbf",
    jump: Optional[int] = None
) -> pd.DataFrame:
    """
    Detects change points for all specified features in the DataFrame and stores the results in a new DataFrame.

    Parameters:
    - df: DataFrame containing the time series data.
    - features: List of feature names to analyze.
    - width: Window size for the sliding windows.
    - pen: Penalty used in the Window algorithm.
    - model: Model to use for change detection.
    - jump: Step size for jumping over points (optional).

    Returns:
    - A DataFrame with the number of breakpoints detected for each feature.
    """
    results = []

    # Loop through each feature and apply the change detection
    for feature in features:
        print(f"Analyzing feature: {feature}")
        _, num_breakpoints = detect_changes(df, feature, width=width, pen=pen, model=model, jump=jump, figsize=(9, 3))
        results.append({'Feature': feature, 'Num_Breakpoints': num_breakpoints})

    # Convert the list of results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df


################################## Detect statistical Changes between unique codes ##################################

class StatisticalChangeDetector:
    """
    A class to detect statistical changes in time series data based on KL divergence,
    statistical measures (mean, variance, kurtosis), and distribution comparisons
    using the KS test.
    """

    def __init__(self, df: pd.DataFrame, unique_code_col: str = 'Unique_Code'):
        """
        Initialize the StatisticalChangeDetector with the data.
        """
        self.df = df.copy()
        self.unique_code_col = unique_code_col
        self.df[self.unique_code_col] = self.df[self.unique_code_col].astype(str)
        self.unique_codes = self.df[self.unique_code_col].unique().tolist()

    ### General Statistical Methods ###

    def compute_statistic_per_code(
        self, col: str, statistic: str
    ) -> List[Dict[str, float]]:
        """
        Computes a specified statistic for each unique code in the specified column.

        Args:
            col (str): Column name to analyze.
            statistic (str): The statistical measure to compute ('mean', 'variance',
                             or 'kurtosis').

        Returns:
            List[Dict[str, float]]: A list of dictionaries with the statistic
            for each unique code.
        """
        results = []

        # Map statistic names to functions
        stat_functions = {
            'mean': np.mean,
            'variance': np.var,
            'kurtosis': kurtosis
        }

        if statistic not in stat_functions:
            raise ValueError(f"Statistic '{statistic}' is not supported.")

        stat_func = stat_functions[statistic]

        for code in self.unique_codes:
            data = self.df[self.df[self.unique_code_col] == code][col]
            value = stat_func(data)
            results.append({
                'Code': code,
                'Statistic': value
            })

        return results

    def mark_significant_statistic_codes(
        self, stat_results: List[Dict[str, float]], statistic: str,
        threshold: Optional[float] = None
    ) -> List[str]:
        """
        Marks the unique codes as significant based on the specified statistic.

        Args:
            stat_results (List[Dict[str, float]]): A list of dictionaries with
                statistics for each code.
            statistic (str): The statistical measure used ('mean', 'variance',
                             or 'kurtosis').
            threshold (Optional[float]): Threshold for the statistic.
                If None, use mean + 3*std.

        Returns:
            List[str]: A list of unique codes marked as significant.
        """
        stat_values = [entry['Statistic'] for entry in stat_results]

        if threshold is None:
            threshold = np.mean(stat_values) + 3 * np.std(stat_values)

        significant_codes = [
            entry['Code'] for entry in stat_results
            if entry['Statistic'] > threshold
        ]

        return significant_codes

    def statistic_multiple_features(
        self, features: List[str], statistic: str
    ) -> pd.DataFrame:
        """
        Computes the number of significant unique codes for each feature
        based on the specified statistic.

        Args:
            features (List[str]): List of features to analyze.
            statistic (str): The statistical measure to compute ('mean', 'variance',
                             or 'kurtosis').

        Returns:
            pd.DataFrame: A DataFrame with the feature names and the number
            of significant unique codes for each feature.
        """
        results = []

        for feature in tqdm(features, desc=f"Processing features ({statistic.title()})"):
            stat_results = self.compute_statistic_per_code(feature, statistic)
            significant_codes = self.mark_significant_statistic_codes(
                stat_results, statistic)
            results.append({
                'Feature': feature,
                'Num_Significant_Codes': len(significant_codes)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            by='Num_Significant_Codes', ascending=False)

        return results_df

    ### KL Divergence Methods ###

    def kl_divergence_consecutive(
        self, col: str, bins: int = 1000
    ) -> List[Dict[str, float]]:
        """
        Computes the KL divergence between consecutive segments of unique codes
        for a specific column.

        Args:
            col (str): Column name to analyze.
            bins (int): Number of bins for histograms.

        Returns:
            List[Dict[str, float]]: A list of dictionaries with KL divergence
            between consecutive unique codes.
        """
        results = []
        num_codes = len(self.unique_codes)

        for i in range(num_codes - 1):
            code1 = self.unique_codes[i]
            code2 = self.unique_codes[i + 1]

            data1 = self.df[self.df[self.unique_code_col] == code1][col]
            data2 = self.df[self.df[self.unique_code_col] == code2][col]

            # Create histograms with density=True to normalize
            p1, _ = np.histogram(data1, bins=bins, density=True)
            p2, _ = np.histogram(data2, bins=bins, density=True)

            # Avoid division by zero and log of zero by adding a small constant
            p1 += 1e-10
            p2 += 1e-10

            kl_div = entropy(p1, p2)

            results.append({
                'Codes Compared': f'{code1} vs {code2}',
                'KL Divergence': kl_div,
                'Code 1': code1,
                'Code 2': code2
            })

        return results

    def kl_divergence_to_total(
        self, col: str, bins: int = 1000
    ) -> List[Dict[str, float]]:
        """
        Computes the KL divergence between the overall distribution of the entire
        series and each unique code.

        Args:
            col (str): Column name to analyze.
            bins (int): Number of bins for histograms.

        Returns:
            List[Dict[str, float]]: A list of dictionaries with KL divergence
            between each unique code and the total series.
        """
        results = []
        total_data = self.df[col]
        total_hist, bin_edges = np.histogram(total_data, bins=bins, density=True)
        total_hist += 1e-10  # Avoid zeros

        for code in self.unique_codes:
            code_data = self.df[self.df[self.unique_code_col] == code][col]
            code_hist, _ = np.histogram(code_data, bins=bin_edges, density=True)
            code_hist += 1e-10  # Avoid zeros

            kl_div = entropy(code_hist, total_hist)

            results.append({
                'Code': code,
                'KL Divergence': kl_div
            })

        return results

    def mark_significant_kl_codes(
        self, kl_results: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        comparison_type: str = 'consecutive'
    ) -> List[str]:
        """
        Marks the unique codes as having significant changes based on KL divergence.

        Args:
            kl_results (List[Dict[str, Any]]): A list of dictionaries with
                KL divergence information.
            threshold (Optional[float]): Threshold for marking significant codes.
                If None, use mean + 3*std.
            comparison_type (str): Type of comparison ('consecutive' or 'total').

        Returns:
            List[str]: A list of unique codes marked as having significant changes.
        """
        kl_values = [entry['KL Divergence'] for entry in kl_results]

        if threshold is None:
            kl_mean = np.mean(kl_values)
            kl_std = np.std(kl_values)
            threshold = kl_mean + 3 * kl_std

        significant_codes = set()

        if comparison_type == 'consecutive':
            for entry in kl_results:
                if entry['KL Divergence'] > threshold:
                    significant_codes.add(entry['Code 2'])  # Code 2 is after the change
        elif comparison_type == 'total':
            for entry in kl_results:
                if entry['KL Divergence'] > threshold:
                    significant_codes.add(entry['Code'])
        else:
            raise ValueError("comparison_type must be 'consecutive' or 'total'.")

        return list(significant_codes)

    def kl_divergence_multiple_features(
        self, features: List[str], bins: int = 1000,
        comparison_type: str = 'consecutive'
    ) -> pd.DataFrame:
        """
        Computes the number of significant unique codes for each feature
        based on KL divergence.

        Args:
            features (List[str]): List of features to analyze.
            bins (int): Number of bins for histograms.
            comparison_type (str): Type of comparison ('consecutive' or 'total').

        Returns:
            pd.DataFrame: A DataFrame with the feature names and the number
            of significant unique codes for each feature.
        """
        results = []

        for feature in tqdm(features, desc=f"Processing features (KL Divergence - {comparison_type})"):
            if comparison_type == 'consecutive':
                kl_results = self.kl_divergence_consecutive(feature, bins=bins)
            elif comparison_type == 'total':
                kl_results = self.kl_divergence_to_total(feature, bins=bins)
            else:
                raise ValueError("comparison_type must be 'consecutive' or 'total'.")

            significant_codes = self.mark_significant_kl_codes(
                kl_results, comparison_type=comparison_type)

            results.append({
                'Feature': feature,
                'Num_Significant_Codes': len(significant_codes)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            by='Num_Significant_Codes', ascending=False)

        return results_df


    ### KS Test Methods ###

    def compare_distributions(
        self, col: str, significance_level: float = 0.05
    ) -> List[Dict[str, Any]]:
        """
        Compares the distribution of each unique code to the distribution
        of the entire series for a specific feature.
        """
        results = []
        # Data from the entire series
        total_data = self.df[col]

        for code in self.unique_codes:
            code_data = self.df[self.df[self.unique_code_col] == code][col].dropna()

            # Perform the K-S test
            ks_stat, p_value = ks_2samp(code_data, total_data)

            # Determine if the result is significant - Rejects H0 if p-value < alpha
            is_significant = p_value < significance_level

            results.append({
                'Code': code,
                'KS_Statistic': ks_stat,
                'p_value': p_value,
                'Significant': is_significant
            })

        return results

    def significant_ks_codes(
        self, comparison_results: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extracts the codes that have significantly different distributions
        based on the KS test.
        """
        significant_codes = [
            entry['Code'] for entry in comparison_results if entry['Significant']
        ]

        return significant_codes

    def ks_test_multiple_features(
        self, features: List[str], significance_level: float = 0.05
    ) -> pd.DataFrame:
        """
        Computes the number of codes with significantly different distributions
        for multiple features.
        """
        results = []

        for feature in tqdm(features, desc="Processing features (KS Test)"):
            comparison_results = self.compare_distributions(
                feature, significance_level)
            significant_codes = self.significant_ks_codes(comparison_results)

            results.append({
                'Feature': feature,
                'Num_Significant_Codes': len(significant_codes)
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            by='Num_Significant_Codes', ascending=False)

        return results_df

    ### Plotting Method ###

    def plot_significant_codes_bar_chart(
        self, results_df: pd.DataFrame,
        metric_name: str = 'Num_Significant_Codes',
        top_x_feats: Optional[int] = None,
        figsize: Tuple[int, int] = (15, 8)
    ) -> None:
        """
        Plots a horizontal bar chart showing the number of significant codes
        for each feature.
        """
        if top_x_feats is not None:
            results_df = results_df.head(top_x_feats)

        # Modify the 'Feature' column to process feature names
        results_df = results_df.copy()

        def process_feature_name(feature_name):
            parts = feature_name.split('_')
            if len(parts) >= 3:
                return f"{parts[0]}_{parts[2]}"  # Keep the first and third part
            return feature_name  
        
        results_df['Feature'] = results_df['Feature'].apply(process_feature_name)
        x_axis = results_df[metric_name]
        y_axis = results_df['Feature']


        plt.figure(figsize=figsize)

        sns.barplot(
            x=x_axis,
            y=y_axis,
            orient="h",
            hue=y_axis,
            dodge=False,
            legend=False,
            palette=sns.color_palette("Set2", n_colors=len(results_df))
        )
        
        

        plt.xlabel(metric_name.replace('_', ' '))
        plt.ylabel("Features")
        plt.title(f"Number of Significant Codes by Feature ({metric_name})")

        plt.tight_layout()
        plt.show()

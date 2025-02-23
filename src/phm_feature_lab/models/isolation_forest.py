import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Dict
from joblib import parallel_backend


def ensure_numpy_array(data):
    """
    Ensure the input data is converted to a numpy array.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise ValueError("Unsupported data type. Cannot convert to numpy array.")


class IsolationForestModel:
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, random_state: int = 42, bootstrap: bool = False, verbose =1):
        """
        Initialize the Isolation Forest model.
        
        Args:
            contamination (float): The proportion of anomalies expected in the data.
            random_state (int): Random state for reproducibility.
        """
        self.__n_estimators = n_estimators
        self.__contamination = contamination
        self.__random_state = random_state
        self.__bootstrap = bootstrap
        self.__verbose = verbose
        self.__model = IsolationForest(
            n_estimators=self.__n_estimators, contamination=self.__contamination, random_state=self.__random_state, bootstrap=self.__bootstrap, verbose = self.__verbose, n_jobs=-1
        )

    def train(self, X_train: np.ndarray):
        """
        Train the Isolation Forest model on the given data.
        
        Args:
            X_train (np.ndarray): Training data features.
        """
        X_train = ensure_numpy_array(X_train)
        self.__model.fit(X_train)
        
        
    def predict(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict and evaluate the model on validation data.
        
        Args:
            X_val (np.ndarray): Validation data features.
            y_val (np.ndarray): Validation data labels.
        
        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Predicted labels and evaluation metrics.
        """
        X_val = ensure_numpy_array(X_val)
        y_val = ensure_numpy_array(y_val)

        y_pred = self.__model.predict(X_val)
        y_pred = np.where(y_pred == -1, 1, 0)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred, zero_division=1),
            'precision': precision_score(y_val, y_pred, zero_division=1),
            'recall': recall_score(y_val, y_pred, zero_division=1),
        }
        return y_pred, metrics

    def plot_anomalies(self, X_val: np.ndarray, y_val: np.ndarray, y_pred_val: np.ndarray, column_index: int = 0, s: int = 5):
        """
        Plot anomalies detected by the model.
        
        Args:
            X_val (np.ndarray): Validation data features.
            y_val (np.ndarray): Validation data labels.
            y_pred_val (np.ndarray): Predicted labels.
            column_index (int): The index of the column to plot.
        """
        plt.figure(figsize=(12, 6))
        X_val = ensure_numpy_array(X_val)[:, column_index]
        plt.plot(X_val, color='blue', label='Validation Data', zorder=2)
        plt.scatter(
            np.where(y_pred_val == 1)[0],
            X_val[y_pred_val == 1],
            color='orange',
            s=s,
            label='Anomalies (Predicted)',
            zorder=3,
        )

        # Highlight known anomalies
        ymin, ymax = plt.ylim()
        plt.fill_between(
            np.arange(len(X_val)),
            ymin,
            ymax,
            where=y_val == 1,
            color='red',
            alpha=0.3,
            label='True Anomalies',
            zorder=1,
        )

        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.title('Anomaly Detection with Isolation Forest')
        plt.show()

    def plot_scores(self, X_val: np.ndarray, save_dir: str = None, group: str = None):
        """
        Plot histogram of prediction scores.
        
        Args:
            X_val (np.ndarray): Validation data features.
        """
        X_val = ensure_numpy_array(X_val)
        with parallel_backend("threading", n_jobs=4):
            y_scores = self.__model.decision_function(X_val)
        limiar = np.percentile(y_scores, self.__contamination if self.__contamination != 'auto' else 0.1) 
        plt.figure(figsize=(10, 4))
        sns.histplot(y_scores, kde=True, stat='density')
        plt.axvline(x=limiar, color='red', linestyle='--', label=f'Limiar: {limiar:.4f}')
        plt.ylim([-0.5, None])
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.title('Prediction Scores Distribution')
        plt.grid(True)
        plt.legend(loc='upper left')

        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
            save_path = os.path.join(save_dir, f'scores_{group}.png')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to '{save_path}'")
        plt.show()
        print(f'Limiar de Anomalia: {limiar}')

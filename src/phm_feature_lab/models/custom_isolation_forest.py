import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Dict
from phm_feature_lab.utils.utilities import Utilities


class CustomIsolationForest:
    def __init__(self, n_estimators: int = 100, max_samples: float = 0.5, max_features: float = 0.5, contamination: float = 0.1, random_state: int = 42, bootstrap: bool = False, verbose: int = 1):
        """
        Initialize the Isolation Forest model.

        Args:
            n_estimators (int): The number of base estimators in the ensemble.
            contamination (float): The proportion of anomalies expected in the data (between 0 and 0.5).
            random_state (int): Random state for reproducibility.
            bootstrap (bool): Whether to use bootstrap sampling.
            verbose (int): Verbosity level.
        """
        self.__n_estimators = n_estimators
        self.__max_samples = max_samples
        self.__max_features = max_features
        self.__contamination = contamination
        self.__random_state = random_state
        self.__bootstrap = bootstrap
        self.__verbose = verbose
        self.__model = self.get_sklearn_model()

    def get_sklearn_model(self):
        """Create and return an IsolationForest instance."""
        model = IsolationForest(
            n_estimators=self.__n_estimators,
            max_samples=self.__max_samples,
            max_features=self.__max_features,
            contamination=self.__contamination,
            random_state=self.__random_state,
            bootstrap=self.__bootstrap,
            verbose=self.__verbose,
            n_jobs=-1,
        )
        return model

    def train(self, X_train: np.ndarray):
        """
        Train the Isolation Forest model on the given data.

        Args:
            X_train (np.ndarray): Training data features.
        """
        X_train = Utilities.ensure_numpy_array(X_train)
        self.__model.fit(X_train)
        return self.__model
    
    def predict(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict and evaluate the model on validation data.

        Args:
            X_val (np.ndarray): Validation data features.
            y_val (np.ndarray): Validation data labels.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Predicted labels and evaluation metrics.
        """
        X_val = Utilities.ensure_numpy_array(X_val)
        y_val = Utilities.ensure_numpy_array(y_val)

        y_pred = self.__model.predict(X_val)
        y_pred = np.where(y_pred == -1, 1, 0)

        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred, zero_division=1),
            'precision': precision_score(y_val, y_pred, zero_division=1),
            'recall': recall_score(y_val, y_pred, zero_division=1),
        }
        return y_pred, metrics

    def plot_anomalies(self, X_val: np.ndarray, y_val: np.ndarray, y_pred_val: np.ndarray, column_index: int = 0, s: int = 6):
        """
        Plot anomalies detected by the model.

        Args:
            X_val (np.ndarray): Validation data features.
            y_val (np.ndarray): Validation data labels.
            y_pred_val (np.ndarray): Predicted labels.
            column_index (int): The index of the column to plot.
            s (int): Size of the scatter points.
        """
        plt.figure(figsize=(12, 5))
        X_val = Utilities.ensure_numpy_array(X_val)[:, column_index]
        plt.plot(X_val, color='blue', label='Sinal', zorder=2)
        plt.scatter(
            np.where(y_pred_val == 1)[0],
            X_val[y_pred_val == 1],
            color='orange',
            s=s,
            label='Anomalias (Predição)',
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
            label='Anomalias (Real)',
            zorder=1,
        )

        plt.xlabel('Índice')
        plt.ylabel('Aceleração [a.u.]')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.title('Anomaly Detection with Isolation Forest')
        plt.show()

    def plot_scores(self, X_val: np.ndarray, save_dir: str = None, group: str = None):
        """
        Plot histogram of prediction scores.

        Args:
            X_val (np.ndarray): Validation data features.
            save_dir (str, optional): Directory to save the plot.
            group (str, optional): Group name for the plot file.
        """
        X_val = Utilities.ensure_numpy_array(X_val)
        y_scores = self.__model.decision_function(X_val)
        limiar = np.percentile(y_scores, 100 * self.__contamination) #between 0 and 100 - The lower, the more abnormal. Negative scores represent outliers, positive scores represent inliers.
        plt.figure(figsize=(10, 4))
        sns.histplot(y_scores, kde=True, stat='density')
        plt.axvline(x=limiar, color='red', linestyle='--', label=f'Limiar: {limiar:.4f}')
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.title('Prediction Scores Distribution')
        plt.grid(True)
        plt.legend(loc='upper left')

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'scores_{group}.png')
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Plot saved to '{save_path}'")
        plt.show()
        print(f'Limiar de Anomalia: {limiar}')
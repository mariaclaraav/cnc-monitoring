import optuna
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Dict
import numpy as np
from phm_feature_lab.utils.utilities import Utilities

class ObjectiveFunctions:
    
    @staticmethod
    def IsolationForest(trial: optuna.Trial, 
                        X_train: np.ndarray, 
                        X_val: np.ndarray, 
                        y_val: np.ndarray, 
                        metric: str = 'precision') -> float:
        """
        Objective function for Optuna to optimize Isolation Forest model.

        """
        contamination = trial.suggest_float('contamination', 0.000001, 0.2)
        max_samples = trial.suggest_float('max_samples', 0.1, 1)
        max_features = trial.suggest_float('max_features', 0.1, 1)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)

        model = IsolationForest(
            contamination=contamination,
            max_samples=max_samples, 
            max_features=max_features,
            n_estimators=n_estimators,
            random_state=42, 
            bootstrap=False,
        )
        
        X_train = Utilities.ensure_numpy_array(X_train)
        X_val = Utilities.ensure_numpy_array(X_val)
        y_val = Utilities.ensure_numpy_array(y_val)
        
        model.fit(X_train)
        
        y_pred = model.predict(X_val)
        y_pred = np.where(y_pred == -1, 1, 0)
        
        if metric == 'precision':
            return precision_score(y_val, y_pred)
        elif metric == 'f1':
            return f1_score(y_val, y_pred, zero_division=1)
        else:
            raise ValueError("Metric must be either 'precision' or 'f1'.")
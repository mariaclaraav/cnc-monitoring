import optuna
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Tuple, Dict
import numpy as np
from phm_feature_lab.utils.utilities import Utilities

import optuna
import numpy as np
from sklearn.metrics import precision_score, f1_score, accuracy_score, recall_score
from sklearn.ensemble import IsolationForest

class ObjectiveFunctions:
    
    @staticmethod
    def IsolationForest(trial: optuna.Trial, 
                        X_train: np.ndarray, 
                        X_val: np.ndarray, 
                        y_val: np.ndarray, 
                        metrics: list = ['precision']) -> tuple:
        """
        Objective function for Optuna to optimize Isolation Forest model.
        
        Args:
            trial: Optuna trial object.
            X_train: Training data.
            X_val: Validation data.
            y_val: Validation labels.
            metrics: List of metrics to optimize. Supported metrics: 'precision', 'f1', 'accuracy', 'recall'.
        
        Returns:
            Tuple with the values of the selected metrics.
        """
        # Defina os hiperparâmetros a serem otimizados
        contamination = trial.suggest_float('contamination', 0.001, 0.1)
        max_samples = trial.suggest_float('max_samples', 0.1, 1)
        max_features = trial.suggest_float('max_features', 0.1, 1)
        n_estimators = trial.suggest_int('n_estimators', 10, 1000)

        # Crie o modelo Isolation Forest
        model = IsolationForest(
            contamination=contamination,
            max_samples=max_samples, 
            max_features=max_features,
            n_estimators=n_estimators,
            random_state=42, 
            bootstrap=False,
        )
        
        # Garanta que os dados sejam arrays numpy
        X_train = np.asarray(X_train)
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        
        # Treine o modelo
        model.fit(X_train)
        
        # Faça previsões
        y_pred = model.predict(X_val)
        y_pred = np.where(y_pred == -1, 1, 0)  # Converta -1 para 1 (anomalias) e 1 para 0 (normais)
        
        # Calcule as métricas selecionadas
        results = []
        for metric in metrics:
            if metric == 'precision':
                results.append(precision_score(y_val, y_pred))
            elif metric == 'f1':
                results.append(f1_score(y_val, y_pred, zero_division=1))
            elif metric == 'accuracy':
                results.append(accuracy_score(y_val, y_pred))
            elif metric == 'recall':
                results.append(recall_score(y_val, y_pred))
            else:
                raise ValueError(f"Metric '{metric}' is not supported. Use 'precision', 'f1', 'accuracy', or 'recall'.")
        
        # Retorne uma tupla com os valores das métricas
        return tuple(results)
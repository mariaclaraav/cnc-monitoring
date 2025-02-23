# Standard library imports
import logging
import os
from typing import Any, Dict, List, Tuple, Union

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from src.model_lib.anomaly_rate import sliding_window_anomalies, expand_anomaly_index_with_threshold



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure consistent style
#sns.set_theme()

def ensure_numpy_array(data: Union[np.ndarray, pd.DataFrame, pd.Series, List[Any]]) -> np.ndarray:
    """
    Ensure the input data is converted to a numpy array.
    
    Args:
        data: The input data, which could be a numpy array, pandas DataFrame, Series, or list.
    
    Returns:
        A numpy array representing the input data.
    
    Raises:
        ValueError: If the data type is unsupported.
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        return data.values
    elif isinstance(data, list):
        return np.array(data)
    else:
        raise ValueError("Unsupported data type. Cannot convert to numpy array.")
    
def IF_objective(trial: optuna.Trial, X_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, metric: str = 'precision') -> float:
    """
    Objective function for Optuna to optimize Isolation Forest model.

    """
    contamination = trial.suggest_float('contamination', 0.000001, 0.2)

    model = IsolationForest(
        contamination=contamination, 
        random_state=42, 
        bootstrap=False
    )
    
    X_train = ensure_numpy_array(X_train)
    X_val = ensure_numpy_array(X_val)
    y_val = ensure_numpy_array(y_val)
    
    model.fit(X_train)
    
    y_pred = model.predict(X_val)
    y_pred = np.where(y_pred == -1, 1, 0)
    
    if metric == 'precision':
        return precision_score(y_val, y_pred)
    elif metric == 'f1':
        return f1_score(y_val, y_pred, zero_division=1)
    else:
        raise ValueError("Metric must be either 'precision' or 'f1'.")
    
class AblationStudy:
    def __init__(self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        unique_codes: np.ndarray,
        storage_path: str,
        metric: str,
        seed: int,
        n_trials: int,
        n_warmup_steps: int,
        logger):
        
        self.X_train = X_train
        self.X_val = X_val
        self.y_val = y_val
        self.unique_codes = unique_codes
        self.storage_path = storage_path
        self.metric = metric
        self.seed = seed
        self.n_trials = n_trials
        self.n_warmup_steps = n_warmup_steps
        self.logger = logger
        self.sampler = TPESampler(seed=self.seed)
        self.pruner = MedianPruner(n_warmup_steps=self.n_warmup_steps)

    def ablation_process(self,
        ablation_type: str,
        fraction_start: float,
        fraction_end: float,
        decrement: float,
        n_decrement: int):
        
        if ablation_type == 'fraction':
            fraction = fraction_start
            while fraction >= fraction_end:
                num_labels = int(len(self.y_val) * fraction)
                X_selected, _, y_selected, _ = train_test_split(
                    self.X_val, self.y_val, train_size=num_labels, stratify=self.y_val, random_state=self.seed
                )
                value = f'{fraction:.2f}'
                study_name = f'{ablation_type}_{value}_{self.metric}'
                log_message = f'\n{"-"*15} Study with {fraction*100:.2f}% of labels {"-"*15}\n'
                yield X_selected, y_selected, study_name, log_message
                fraction -= decrement

        elif ablation_type == 'operation':
            defective_operations = np.unique(self.unique_codes[self.y_val == 1])
            num_operations = len(defective_operations)

            for i in range(num_operations, 0, -n_decrement):
                if i == num_operations:
                    X_selected = self.X_val
                    y_selected = self.y_val
                else:
                    j = num_operations - i
                    removed_operations = defective_operations[-j:]
                    mask = np.isin(self.unique_codes, removed_operations, invert=True)
                    X_selected = self.X_val[mask]
                    y_selected = self.y_val[mask]
                    
                value = f'{i}'
                study_name = f'{ablation_type}_{value}_{self.metric}'
                log_message = f'\n{"-"*15} Study with {i} defective operations {"-"*15}\n'
                yield X_selected, y_selected, study_name, log_message
                
        elif ablation_type == 'label_change':
            defective_operations = np.unique(self.unique_codes[self.y_val == 1])
            num_operations = len(defective_operations)
            
            for i in range(num_operations, 0, -n_decrement):
                if i == num_operations:
                    # Use all data without changing labels
                    X_selected = self.X_val
                    y_selected = self.y_val
                else:
                    j = num_operations - i
                    changed_operations = defective_operations[-j:]
                    y_selected = self.y_val.copy()
                    mask = np.isin(self.unique_codes, changed_operations)
                    y_selected[mask] = 0
                    X_selected = self.X_val
                    
                value = f'{i}'
                study_name = f'{ablation_type}_{value}_{self.metric}'
                log_message = f'\n{"-"*15} Study with {i} defective operations {"-"*15}\n'
                yield X_selected, y_selected, study_name, log_message
        else:
            raise ValueError(f"Unsupported ablation type: {ablation_type}")
        

    def optimize_study(self,
        X_train_subset: np.ndarray,
        X_selected: np.ndarray,
        y_selected: np.ndarray,
        study_name: str):
        
        study_path = os.path.join(self.storage_path, f'{study_name}.db')
        IF_study = create_study(
            study_name=study_name,
            direction='maximize',
            sampler=self.sampler,
            pruner=self.pruner,
            storage=f'sqlite:///{study_path}',
            load_if_exists=True
        )
        IF_study.optimize(
            lambda trial: IF_objective(trial, X_train_subset, X_selected, y_selected, metric=self.metric),
            n_trials=self.n_trials,
            n_jobs=-1
        )

    def run_studies(self, fraction_start: float, fraction_end: float, decrement: float,ablation_type: str = 'fraction', n_decrement: int = 1):
        
        for X_selected, y_selected, study_name, log_message in self.ablation_process(
                                                                        ablation_type,
                                                                        fraction_start,
                                                                        fraction_end,
                                                                        decrement, 
                                                                        n_decrement):
            self.logger.info(log_message)
            self.optimize_study(X_train_subset=self.X_train[::100],
                                X_selected=X_selected,
                                y_selected=y_selected,
                                study_name=study_name)  


class RetrainBestTrial:
    def __init__(self, study: optuna.Study, X_train: np.ndarray):
        self.study_ = study
        self.X_train_ = ensure_numpy_array(X_train)
        self.model_ = None

    def train(self) -> None:
        """
        Train the model using the best trial from the study.
        """
        best_trial = self.study_.best_trial
        self.model_ = IsolationForest(        
            contamination=best_trial.params['contamination'],
            random_state=42,
            bootstrap=False
        )
        self.model_.fit(self.X_train_)
    
    def predict(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict and evaluate the model on validation data.
        """
        X_val = ensure_numpy_array(X_val)
        y_val = ensure_numpy_array(y_val)
        
        if self.model_ is None:
            raise Exception('Model has not been trained yet. Call the train method first.')
        
        y_pred = self.model_.predict(X_val)
        y_pred = np.where(y_pred == -1, 1, 0)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred, zero_division=1),
            'precision': precision_score(y_val, y_pred, zero_division=1),
            'recall': recall_score(y_val, y_pred, zero_division=1)
        }
        return y_pred, metrics
    
    def plot_anomalies(self, X_val: np.ndarray, y_val: np.ndarray, y_pred_val: np.ndarray, column_index: int, data: str, figsize: Tuple = (12, 6)) -> None:
        """
        Plot anomalies detected by the model.
        """
        plt.figure(figsize=figsize)
        X_val = ensure_numpy_array(X_val)[:, column_index]
        plt.plot(X_val, color='blue', label=f'{data} data', zorder=2)
        plt.scatter(np.where(y_pred_val == 1)[0], X_val[y_pred_val == 1], color='orange', s=25, label='Model', zorder=3)
        
        # Highlight known anomalies
        ymin, ymax = plt.ylim()
        plt.fill_between(np.arange(len(X_val)), ymin, ymax, where=y_val == 1, color='red', alpha=0.3, label='Known', zorder=1)

        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Detected Anomalies - {data} data')
        
        plt.show()
        
    def plot_scores(self, X_val: np.ndarray) -> None:
        """
        Plot histogram of prediction scores.
        """
        X_val = ensure_numpy_array(X_val)
        y_scores = self.model_.decision_function(X_val)
        plt.figure(figsize=(10, 4))
        sns.histplot(y_scores, kde=True, stat='density')
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.title('Prediction Scores')
        plt.show()
        
class IF_testing:
    def __init__(self, X_train, X_val, y_val, X_test, y_test, val_codes, storage_path):
        """
        Initializes the IF_testing class with training and testing data.

        Args:
            X_train (np.ndarray): Training data features.
            X_test (np.ndarray): Testing data features.
            y_test (np.ndarray): Testing data labels.
            val_codes (np.ndarray): Validation codes corresponding to the test data.
            storage_path (str): Path to the storage directory for Optuna studies.
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_test = y_test
        self.X_val = X_val
        self.y_val = y_val
        self.val_codes = val_codes
        self.storage_path = storage_path
        self.results = {}  # Initialize the results dictionary
    
    def get_window_size(self, sample_rate, reducer):
        if reducer is not None:
            window_size = self.sample_rate/self.reducer
        else:
            window_size = sample_rate
        return window_size
    
    def retrainer(self, ablation_type, metric, n_decrement, n=None, figsize = (12, 6)):
        """
        Retrains models using the best hyperparameters from previous Optuna studies.

        Args:
            ablation_type (str): The type of ablation ('operation', 'fraction', etc.).
            metric (str): The evaluation metric used ('precision', 'recall', etc.).
            n_decrement (int): Number of operations to decrement each iteration.
            n (float, optional): Fraction of samples to use for partial prediction (between 0 and 1).
        """
        # Get unique defective operations from validation codes where y_test == 1
        defective_operations = np.unique(self.val_codes[self.y_val == 1])
        num_operations = len(defective_operations)
        
        # Loop over the number of defective operations, decreasing by one each time
        for i in range(num_operations, 0, -n_decrement):
            # Define the study name with the current number of remaining operations
            value = f'{i}'
            study_name = f'{ablation_type}_{value}_{metric}'
            
            try:
                # Load the Optuna study from storage
                IF_study = optuna.load_study(
                    study_name=study_name,
                    storage=f'sqlite:///{self.storage_path}/{study_name}.db'
                )
            except Exception as e:
                print(f"Error while loading {study_name}: {e}")
                continue  # Skip to the next iteration if loading fails
            
            # Save the best contamination level for this number of operations
            best_contamination = IF_study.best_trial.params['contamination']
            if best_contamination not in self.results:
                self.results[best_contamination] = []
            self.results[best_contamination].append(i)
            
            # Retrain the best model from the study using the training data
            model = RetrainBestTrial(IF_study, self.X_train)
            model.train()
            
            print(f'\n{"-"*15} Model trained with {i} defective operations {"-"*15}\n')
            
            # Perform prediction on the test data and evaluate the model
            y_pred, metrics = model.predict(self.X_test, self.y_test)
            
            print('Prediction:\n')
            for metric_name, metric_value in metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            print()
                
            # Plot anomalies detected by the model on the test data
            model.plot_anomalies(self.X_test,
                                self.y_test,
                                y_pred,
                                column_index=1,
                                data='Test',
                                figsize=figsize)
            
            if n is not None:
                # n is expected to be a fraction between 0 and 1
                num_samples = int(len(self.X_test) * n)
                print(f'Prediction with {n*100:.0f}% of samples:\n')
                # Use a fraction of the test data for partial prediction
                X_partial = self.X_test[:num_samples]
                y_partial = self.y_test[:num_samples]
                
                # Perform prediction on the partial test data
                y_pred_partial, metrics_partial = model.predict(X_partial, y_partial)
    
                for metric_name, metric_value in metrics_partial.items():
                    print(f"{metric_name}: {metric_value:.4f}")
                print()
                
                # Plot anomalies detected by the model on the partial test data
                model.plot_anomalies(X_partial,
                                    y_partial,
                                    y_pred_partial,
                                    column_index=1,
                                    data=f'{n*100:.0f}% of Test')
        return self.results, y_pred


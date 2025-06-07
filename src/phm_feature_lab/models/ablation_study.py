# Standard library imports
import logging
import os
import joblib
from typing import Any, Dict, List, Tuple, Union, Optional, Generator

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from optuna import create_study
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from typing import List, Tuple, Dict
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from phm_feature_lab.utils.logger import Logger
from phm_feature_lab.utils.utilities import Utilities
from phm_feature_lab.models.objective_functions import ObjectiveFunctions

logger = Logger().get_logger()

class AblationStudy:
    def __init__(self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        unique_codes: np.ndarray,
        storage_path: str,
        metrics: List[str],
        seed: int,
        n_trials: int,
        n_warmup_steps: int):
        
        self.X_train = X_train
        self.X_val = X_val
        self.y_val = y_val
        self.unique_codes = unique_codes
        self.storage_path = storage_path
        self.metrics = metrics
        self.seed = seed
        self.n_trials = n_trials
        self.n_warmup_steps = n_warmup_steps
        self.sampler = TPESampler(seed=self.seed)
        self.pruner = MedianPruner(n_warmup_steps=self.n_warmup_steps)

    def __ablation_process(
        self,
        ablation_type: str,
        fraction_start: float,
        fraction_end: float,
        decrement: float,
        n_decrement: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray, str, str], None, None]:
        """
        Generate data splits for ablation studies.

        Args:
            ablation_type (str): Type of ablation ('fraction', 'operation', 'label_change').
            fraction_start (float): Starting fraction of labels for 'fraction' ablation.
            fraction_end (float): Ending fraction of labels for 'fraction' ablation.
            decrement (float): Decrement value for 'fraction' ablation.
            n_decrement (int): Number of operations to decrement for 'operation' ablation.

        Yields:
            Tuple[np.ndarray, np.ndarray, str, str]: Selected data, study name, and log message.
        """
        if ablation_type == "fraction":
            yield from self.__fraction_split(fraction_start, fraction_end, decrement)
        elif ablation_type in ["operation", "label_change"]:
            yield from self.__operation_split(ablation_type, n_decrement)
        else:
            raise ValueError(f"Unsupported ablation type: {ablation_type}")

    def __fraction_split(
        self, fraction_start: float, fraction_end: float, decrement: float
    ) -> Generator[Tuple[np.ndarray, np.ndarray, str, str], None, None]:
        """
        Generate data splits for fraction-based ablation.

        Args:
            fraction_start (float): Starting fraction of labels.
            fraction_end (float): Ending fraction of labels.
            decrement (float): Decrement value.

        Yields:
            Tuple[np.ndarray, np.ndarray, str, str]: Selected data, study name, and log message.
        """
        fraction = fraction_start
        while fraction >= fraction_end:
            num_labels = int(len(self.y_val) * fraction)
            X_selected, _, y_selected, _ = train_test_split(
                self.X_val, self.y_val, train_size=num_labels, stratify=self.y_val, random_state=self.seed
            )
            value = f"{fraction:.2f}"
            study_name = f"fraction_{value}_{'_'.join(self.metrics)}"
            log_message = f"\n{'-'*15} Study with {fraction*100:.2f}% of labels {'-'*15}\n"
            yield X_selected, y_selected, study_name, log_message
            fraction -= decrement

    def __operation_split(
        self, ablation_type: str, n_decrement: int
    ) -> Generator[Tuple[np.ndarray, np.ndarray, str, str], None, None]:
        """
        Generate data splits for operation-based ablation.

        Args:
            ablation_type (str): Type of ablation ('operation', 'label_change').
            n_decrement (int): Number of operations to decrement.

        Yields:
            Tuple[np.ndarray, np.ndarray, str, str]: Selected data, study name, and log message.
        """
        defective_operations = np.unique(self.unique_codes[self.y_val == 1])
        num_operations = len(defective_operations)

        for i in range(num_operations, 0, -n_decrement):
            if i == num_operations:
                X_selected = self.X_val
                y_selected = self.y_val
            else:
                j = num_operations - i
                if ablation_type == "operation":
                    removed_operations = defective_operations[-j:]
                    mask = np.isin(self.unique_codes, removed_operations, invert=True)
                    X_selected = self.X_val[mask]
                    y_selected = self.y_val[mask]
                    
                elif ablation_type == "label_change":
                    changed_operations = defective_operations[-j:]
                    y_selected = self.y_val.copy()
                    mask = np.isin(self.unique_codes, changed_operations)
                    y_selected[mask] = 0
                    X_selected = self.X_val

            value = f"{i}"
            study_name = f"{ablation_type}_{value}_{'_'.join(self.metrics)}"
            log_message = f"\n{'-'*15} Study with {i} defective operations {'-'*15}\n"
            yield X_selected, y_selected, study_name, log_message
        

    def __optimize_study(self,
        X_train_subset: np.ndarray,
        X_selected: np.ndarray,
        y_selected: np.ndarray,
        study_name: str):
        
        study_path = os.path.join(self.storage_path, f'{study_name}.db')
                
         # Define a direção de otimização
        if len(self.metrics) == 1:
            direction = "maximize"  # Single-objective
            IF_study = create_study(
                study_name=study_name,
                direction=direction,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=f'sqlite:///{study_path}',
                load_if_exists=True
            )
        else:
            direction = ["maximize"] * len(self.metrics)  # Multiobjetivo
            IF_study = create_study(
                study_name=study_name,
                directions=direction,
                sampler=self.sampler,
                pruner=self.pruner,
                storage=f'sqlite:///{study_path}',
                load_if_exists=True
            )
        IF_study.optimize(
            lambda trial: ObjectiveFunctions.IsolationForest(trial, X_train_subset, X_selected, y_selected, metrics=self.metrics),
            n_trials=self.n_trials,
            n_jobs=-1
        )

    def run_studies(self, fraction_start: float, fraction_end: float, decrement: float,ablation_type: str = 'fraction', n_decrement: int = 1):
        
        for X_selected, y_selected, study_name, log_message in self.__ablation_process(
                                                                        ablation_type,
                                                                        fraction_start,
                                                                        fraction_end,
                                                                        decrement, 
                                                                        n_decrement):
            logger.info(log_message)
            self.__optimize_study(X_train_subset=self.X_train,
                                X_selected=X_selected,
                                y_selected=y_selected,
                                study_name=study_name)  

import os
import optuna
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict, List, Optional, Tuple
from phm_feature_lab.utils.logger import Logger
from phm_feature_lab.utils.utilities import Utilities

logger = Logger().get_logger()


class OptunaRetrainer:
    """
    A class to retrain a model using the best trial from an Optuna study.

    Attributes:
        study_ (optuna.Study): The Optuna study.
        X_train_ (np.ndarray): Training data features.
        model_ (Optional[IsolationForest]): The trained model.
    """

    def __init__(self, study: optuna.Study, X_train: np.ndarray) -> None:
        """
        Initialize the OptunaRetrainer class.

        Args:
            study (optuna.Study): The Optuna study.
            X_train (np.ndarray): Training data features.
        """
        self.study_ = study
        self.X_train_ = Utilities.ensure_numpy_array(X_train)
        self.model_ = None

    def train(self, trial_index: Optional[int] = None) -> None:
        """
        Train the model using the best trial from the study.

        For single-objective studies, the best trial is used.
        For multi-objective studies, a trial from the Pareto front is used.

        Args:
            trial_index (Optional[int]): Index of the trial to use from the Pareto front (for multi-objective studies).
                                         If None, the best trial is used for single-objective studies.
        """
        if len(self.study_.directions) == 1:
            # Single-objective study
            if trial_index is not None:
                logger.warning("trial_index is ignored for single-objective studies.")
            params = self.study_.best_params
        else:
            # Multi-objective study: Use the specified trial from the Pareto front
            if not self.study_.trials:
                raise RuntimeError("No feasible trials found in the study.")
            if trial_index is None:
                raise ValueError("trial_index must be provided for multi-objective studies.")
            if trial_index >= len(self.study_.trials):
                raise ValueError(f"trial_index {trial_index} is out of range for the Pareto front.")
            params = self.study_.trials[trial_index].params

        logger.info(f"Retraining model with params: {params}")
        self.model_ = IsolationForest(
            **params,
            n_jobs=-1,
            random_state=42,
            bootstrap=False
        )
        self.model_.fit(self.X_train_)

    def predict(self, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Predict and evaluate the model on validation data.

        Args:
            X_val (np.ndarray): Validation data features.
            y_val (np.ndarray): Validation data labels.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Predictions and evaluation metrics.
        """
        if self.model_ is None:
            raise RuntimeError("Model has not been trained yet. Call the train method first.")

        y_pred = self.model_.predict(X_val)
        y_pred = np.where(y_pred == -1, 1, 0)  # Convert -1 (anomaly) to 1, 1 (normal) to 0

        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "f1_score": f1_score(y_val, y_pred, zero_division=1),
            "precision": precision_score(y_val, y_pred, zero_division=1),
            "recall": recall_score(y_val, y_pred, zero_division=1),
        }
        return y_pred, metrics

    def plot_anomalies(self, X_val: np.ndarray, y_val: np.ndarray, y_pred_val: np.ndarray, column_index: int, data: str, label_name: str, figsize: Tuple = (13, 5)) -> None:
        """
        Plot anomalies detected by the model.

        Args:
            X_val (np.ndarray): Validation data features.
            y_val (np.ndarray): Validation data labels.
            y_pred_val (np.ndarray): Predicted labels.
            column_index (int): Index of the feature column to plot.
            data (str): Description of the data being plotted.
            label_name (str): Label for the plot.
            figsize (Tuple): Size of the plot.
        """
        plt.figure(figsize=figsize)
        X_val = X_val[:, column_index]
        plt.plot(X_val, color='blue', label=f'{label_name}', zorder=2)
        plt.scatter(np.where(y_pred_val == 1)[0], X_val[y_pred_val == 1], color='orange', s=20, label='Detectada', zorder=3)

        # Highlight known anomalies
        ymin, ymax = plt.ylim()
        plt.fill_between(np.arange(len(X_val)), ymin, ymax, where=y_val == 1, color='red', alpha=0.3, label='Anomalia real', zorder=1)

        plt.xlabel('Índice', size=14)
        plt.ylabel('Aceleração [a.u.]', size=14)
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    def plot_scores(self, X_val: np.ndarray) -> None:
        """
        Plot histogram of prediction scores.

        Args:
            X_val (np.ndarray): Validation data features.
        """
        y_scores = self.model_.decision_function(X_val)
        plt.figure(figsize=(10, 4))
        sns.histplot(y_scores, kde=True, stat='density')
        plt.xlabel('Prediction Score')
        plt.ylabel('Frequency')
        plt.title('Prediction Scores')
        plt.show()
        
class IsolationForestModel:
    """
    A class to manage Isolation Forest models, including loading studies, retraining models,
    and evaluating predictions.

    Attributes:
        X_train (np.ndarray): Training data features.
        X_val (np.ndarray): Validation data features.
        y_val (np.ndarray): Validation data labels.
        X_test (np.ndarray): Testing data features.
        y_test (np.ndarray): Testing data labels.
        val_codes (np.ndarray): Validation codes corresponding to the test data.
        results (List[Dict]): List to store results from each study.
    """

    def __init__(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        val_codes: np.ndarray,
    ) -> None:
        self.__X_train = Utilities.ensure_numpy_array(X_train)
        self.__X_val = Utilities.ensure_numpy_array(X_val)
        self.__y_val = Utilities.ensure_numpy_array(y_val)
        self.__X_test = Utilities.ensure_numpy_array(X_test)
        self.__y_test = Utilities.ensure_numpy_array(y_test)
        self.__val_codes = Utilities.ensure_numpy_array(val_codes)
        self.__results = []
        
    @staticmethod
    def load_study(storage_path: str, study_name: str) -> optuna.Study:
        """
        Load an Optuna study from storage.

        Args:
            storage_path (str): Path to the storage directory for Optuna studies.
            study_name (str): Name of the study.

        Returns:
            optuna.Study: The loaded Optuna study.
        """
        study_path = os.path.join(storage_path, f"{study_name}.db")
        return optuna.load_study(
            study_name=study_name,
            storage=f"sqlite:///{study_path}",
        )
        
    @staticmethod
    def plot_pareto_front(
        study: optuna.Study, 
        metrics: List[str]
    ) -> None:
        """
        Plot the Pareto front for a multi-objective study.

        Args:
            study (optuna.Study): The Optuna study.
            metrics (List[str]): List of metric names used in the study.
                                Must have at least two elements.
        """
        if len(study.directions) == 1:
            logger.info(f"Study is single-objective. No Pareto front to plot.")
            return

        # Verifica se há métricas suficientes
        if len(metrics) < 2:
            raise ValueError("At least two metrics are required to plot the Pareto front.")

        # Extrai os trials da frente de Pareto
        pareto_trials = study.best_trials
        if not pareto_trials:
            logger.info(f"No feasible trials found for this study.")
            return

        try:
            # Tenta usar a visualização do Optuna
            optuna.visualization.plot_pareto_front(
                study,
                target_names=[metrics[0].capitalize(), metrics[1].capitalize()]
            ).show()
        except ImportError:
            # Fallback manual se a visualização do Optuna não estiver disponível
            logger.info("Optuna visualization module not found. Using manual plotting.")

            # Prepara os dados para o plot manual
            metric1_values = [trial.values[0] for trial in pareto_trials]
            metric2_values = [trial.values[1] for trial in pareto_trials]

            # Plota a frente de Pareto manualmente
            plt.figure(figsize=(8, 6))
            plt.scatter(metric1_values, metric2_values, color="blue", label="Pareto Front")
            plt.title(f"Pareto Front Plot")
            plt.xlabel(metrics[0].capitalize())
            plt.ylabel(metrics[1].capitalize())
            plt.legend()
            plt.grid(True)
            plt.show()
            
    @staticmethod
    def plot_hyperparameter_importance(
        study: optuna.Study,
        metrics: List[str]
    ) -> None:
        """
        Plot the hyperparameters importance.

        Args:
            study (optuna.Study): The Optuna study.
        """
        for i, metric in enumerate(metrics):  # Use enumerate para obter índice e valor
            # Cria o gráfico de importância dos parâmetros
            fig = optuna.visualization.plot_param_importances(
                study, 
                target=lambda t: t.values[i],  
                target_name=metric  
            )
            
            # Adiciona um título ao gráfico
            fig.update_layout(
                title=f"Hyperparameter Importances for {metric.capitalize()}",  # Título com o nome da métrica
            )
        
            # Exibe o gráfico
            fig.show()
        
    def __retrain_with_selected_trial(
        self, storage_path: str, study_name: str, trial_index: Optional[int] = None
    ) -> IsolationForest:
        """
        Retrain the model using a chosen trial from the study.

        Args:
            study_name (str): Name of the study.
            trial_index (Optional[int]): Index of the trial to use from the Pareto front.
                                         If None, the best trial is used for single-objective studies.

        Returns:
            IsolationForest: The retrained model.
        """
        study = IsolationForestModel.load_study(storage_path, study_name)

        retrainer = OptunaRetrainer(study, self.__X_train)
        if trial_index:
            retrainer.train(trial_index)
        else:
            retrainer.train()
        
        return retrainer # Retorna o retrainer, que é um objeto da classe OptunaRetrainer
    
    

    def __evaluate_and_plot(
        self,
        retrainer: OptunaRetrainer,  # Recebe o OptunaRetrainer
        study_name: str,
        column_index: int = 1,
        label_name: str = "Aceleração",
        figsize: Tuple = (13, 5),
    ) -> Dict[str, float]:
        y_pred, metrics = retrainer.predict(self.__X_test, self.__y_test)  
        logger.info(f"Evaluation metrics for study '{study_name}':")
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name}: {metric_value:.4f}")

        # Plot anomalies
        retrainer.plot_anomalies(
            self.__X_test,
            self.__y_test,
            y_pred,
            column_index=column_index,
            data="Test",
            label_name=label_name,
            figsize=figsize,
        )
        return y_pred, metrics
    
    def get_pareto_fronts(
        self,
        storage_path: str,
        ablation_type: str,
        metrics: List[str],
        n_decrement: int,
    ) -> None:
        """
        Show the Pareto front for all studies.

        Args:
            storage_path (str): Path to the storage directory for Optuna studies.
            ablation_type (str): The type of ablation ('operation', 'fraction', etc.).
            metrics (List[str]): List of evaluation metrics used ('precision', 'recall', etc.).
            n_decrement (int): Number of operations to decrement each iteration.
        """
        defective_operations = np.unique(self.__val_codes[self.__y_val == 1])
        num_operations = len(defective_operations)
        for i in range(num_operations, 0, -n_decrement):
            value = f"{i}"
            study_name = f"{ablation_type}_{value}_{'_'.join(metrics)}"
            try:
                study = IsolationForestModel.load_study(storage_path, study_name)
                logger.info(f"Pareto Front for Study: {study_name}")
                IsolationForestModel.plot_pareto_front(study, metrics)
            except Exception as e:
                logger.error(f"Error loading study '{study_name}': {e}")
    
    def get_hyperparameter_importances(
        self,
        storage_path: str,
        ablation_type: str,
        metrics: List[str],
        n_decrement: int,
    ) -> None:
        """
        Show the Hyperparameter Importances plot for all studies.

        Args:
            storage_path (str): Path to the storage directory for Optuna studies.
            ablation_type (str): The type of ablation ('operation', 'fraction', etc.).
            metrics (List[str]): List of evaluation metrics used ('precision', 'recall', etc.).
            n_decrement (int): Number of operations to decrement each iteration.
        """
        defective_operations = np.unique(self.__val_codes[self.__y_val == 1])
        num_operations = len(defective_operations)
        for i in range(num_operations, 0, -n_decrement):
            value = f"{i}"
            study_name = f"{ablation_type}_{value}_{'_'.join(metrics)}"
            try:
                study = IsolationForestModel.load_study(storage_path, study_name)
                logger.info(f"Hyperpameter Importance for Study: {study_name}")
                IsolationForestModel.plot_hyperparameter_importance(study, metrics)
            except Exception as e:
                logger.error(f"Error loading study '{study_name}': {e}")
                
    def retrain_models(
        self,
        storage_path: str,
        ablation_type: str,
        metrics: List[str],
        n_decrement: int,
        label_name: str,
        trial_indices: Optional[Dict[str, int]] = None,  # Dicionário com {study_name: trial_index}
        model_storage_path: Optional[str] = None,
        column_index: int = 1,
        n: Optional[float] = None,
        figsize: Tuple = (13, 5),
    ) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
        """
        Retrain models using the best hyperparameters from previous Optuna studies.

        Args:
            storage_path (str): Path to the storage directory for Optuna studies.
            ablation_type (str): The type of ablation ('operation', 'fraction', etc.).
            metrics (List[str]): List of evaluation metrics used ('precision', 'recall', etc.).
            n_decrement (int): Number of operations to decrement each iteration.
            label_name (str): Label for the anomaly plots.
            trial_indices (Dict[str, int]): Dictionary mapping study names to trial indices.
            model_storage_path (Optional[str]): Path to save the retrained models.
            column_index (int): Index of the feature column to plot.
            n (Optional[float]): Fraction of samples to use for partial prediction (between 0 and 1).
            figsize (Tuple): Size of the plots.

        Returns:
            Tuple[List[Dict], Dict[str, np.ndarray]]: Results and predictions for each study.
        """
        defective_operations = np.unique(self.__val_codes[self.__y_val == 1])
        num_operations = len(defective_operations)
        y_pred_dict = {}
        self.__results = []  # Reinicializa a lista de resultados

        for i in range(num_operations, 0, -n_decrement):
            value = f"{i}"
            study_name = f"{ablation_type}_{value}_{'_'.join(metrics)}"

            try:
                study = IsolationForestModel.load_study(storage_path, study_name)  # Chamada estática
                
                trial_index = None  # Initialize trial_index to None
                if trial_indices and study_name in trial_indices:
                    trial_index = trial_indices[study_name]
                    logger.info(f"Using provided trial index {trial_index} for study '{study_name}'.")
                else:
                     logger.info(f"No trial index provided or found in `trial_indices` for study '{study_name}'. Using best trial instead.")


                # Retrain the model
                retrainer = self.__retrain_with_selected_trial(storage_path, study_name, trial_index)

                if model_storage_path:
                    model_filename = os.path.join(model_storage_path, f"IF_model_{study_name}.pkl")
                    joblib.dump(retrainer.model_, model_filename)
                    logger.info(f"Model saved to {model_filename}")

                # Evaluate and plot
                y_pred, metrics_dict = self.__evaluate_and_plot(retrainer, study_name, column_index, label_name, figsize)
                y_pred_dict[f"operation_{i}"] = y_pred
                if trial_index is not None:
                    operation_results = {"num_operations": i, "best_params": study.trials[trial_index].params, **metrics_dict}
                else:
                    operation_results = {"num_operations": i, "best_params": study.best_params, **metrics_dict}

                self.__results.append(operation_results)


            except Exception as e:
                logger.error(f"Error processing study '{study_name}': {e}", exc_info=True)  # Capture traceback
                continue


        return self.__results, y_pred_dict

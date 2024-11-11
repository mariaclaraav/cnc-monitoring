import torch
import os
import shutil
import optuna
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# PyTorch Lightning specific imports
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# Custom model import
from src.models.VanillaAE import VanillaAE

# COPPIED FROM SOURCE CODE: https://optuna.readthedocs.io/en/v2.0.0/_modules/optuna/integration/pytorch_lightning.html

class PyTorchLightningPruningCallback(EarlyStopping):
    """
    PyTorch Lightning callback to prune unpromising trials.

    Args:
        trial: A :class:`~optuna.trial.Trial` corresponding to the current evaluation of the objective function.
        monitor: An evaluation metric for pruning, e.g., ``val_loss`` or ``val_acc``.
    """

    def __init__(self, trial: optuna.trial.Trial, monitor: str) -> None:
        super(PyTorchLightningPruningCallback, self).__init__(monitor=monitor)
        self._trial = trial

    def _process(self, trainer: Trainer, pl_module: LightningModule) -> None:
        logs = trainer.callback_metrics
        epoch = pl_module.current_epoch
        current_score = logs.get(self.monitor)

        if current_score is None:
            return

        self._trial.report(current_score, step=epoch)

        if self._trial.should_prune():
            message = f"Trial was pruned at epoch {epoch}."
            raise optuna.TrialPruned(message)

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._process(trainer, pl_module)

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._process(trainer, pl_module)

class AutoencoderTrainerSetup:
    """ 
    Class responsible for setting up the logger, checkpoint callback, and the PyTorch Lightning trainer
    """
    def __init__(self, save_dir, experiment_name, monitor_metric, min_delta, patience, max_epochs):

        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.monitor_metric = monitor_metric
        self.min_delta = min_delta
        self.patience = patience
        self.max_epochs = max_epochs
        
      
    def setup_trainer(self, trial):
        # Setup the logger using the previously defined method
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor=self.monitor_metric)
        trainer = pl.Trainer(
            logger= False,  # Logger 
            max_epochs=self.max_epochs,  # Maximum number of training epochs
            devices=1,  # Use 1 device (either CPU or GPU)
            accelerator='gpu' if torch.cuda.is_available() else 'cpu',  # Use GPU if available, else fallback to CPU
            callbacks=[
                # Pruning callback 
                pruning_callback,
                # Early stopping 
                EarlyStopping(monitor=self.monitor_metric, min_delta=self.min_delta, patience=self.patience)                
            ]
        )
        return trainer


class AutoencoderObjective:
    """
    Class responsible for defining the objective function for hyperparameter optimization
    """
    def __init__(self, input_dim, save_dir, experiment_name, min_delta, patience, max_epochs, monitor_metric='val_r2'):

        self.input_dim = input_dim
        self.save_dir = save_dir
        self.experiment_name = experiment_name
        self.min_delta = min_delta
        self.patience = patience
        self.max_epochs = max_epochs
        self.monitor_metric = monitor_metric
        
        self.best_value = float('inf') if monitor_metric != 'val_r2' else -float('inf')
        self.best_model_path = None
        
        # Initialize the trainer setup class 
        self.trainer_setup = AutoencoderTrainerSetup(
            save_dir, experiment_name, monitor_metric, min_delta, patience, max_epochs
        )
    # Set training and validation data - Since batch size is an hyperparameter
    def set_data(self, X_train, X_val, num_workers=4, persistent_workers=True):
        self.X_train = torch.tensor(X_train, dtype=torch.float32)
        self.X_val = torch.tensor(X_val, dtype=torch.float32)
        self.num_workers = num_workers  # Number of worker threads 
        self.persistent_workers = persistent_workers  # Keep worker threads alive after data loading is complete
        
    # Set hyperparameters 
    def set_hyperparameters(self, hyperparameters):
        self.hyperparameters = hyperparameters
        
    # Create data loaders for training and validation  
    def create_dataloaders(self, batch_size):
        train_loader = DataLoader(self.X_train, batch_size=batch_size, shuffle=False, 
                                  num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        val_loader = DataLoader(self.X_val, batch_size=batch_size, shuffle=False, 
                                num_workers=self.num_workers, persistent_workers=self.persistent_workers)
        return train_loader, val_loader
    
    # Suggest hyperparameters for the current trial
    def __call__(self, trial):
        dim1 = self.hyperparameters['dim1'](trial)
        dim2 = self.hyperparameters['dim2'](trial)
        latent_dim = self.hyperparameters['latent_dim'](trial)
        learning_rate = self.hyperparameters['learning_rate'](trial)
        batch_size = self.hyperparameters['batch_size'](trial)

        # Initialize the autoencoder model with the suggested hyperparameters
        model = VanillaAE(self.input_dim, dim1, dim2, latent_dim, learning_rate, logger=False)

        # Setup the trainer and checkpoint callback using the AutoencoderTrainerSetup class
        trainer = self.trainer_setup.setup_trainer(trial)
    
        # Create data loaders for training and validation
        train_dataloader, val_dataloader = self.create_dataloaders(batch_size)

        # Train the model using the PyTorch Lightning trainer
        trainer.fit(model, train_dataloader, val_dataloader)

        # Retrieve the value of the monitored metric from the best model
        metric_value = trainer.callback_metrics[self.monitor_metric].item()

        # Ensure the directory exists
        os.makedirs(self.save_dir, exist_ok=True)

         # Check if this is the best model across all trials
        if (self.monitor_metric == 'val_r2' and metric_value > self.best_value) or \
        (self.monitor_metric != 'val_r2' and metric_value < self.best_value):
            self.best_value = metric_value
            
            # Manually save the model
            self.best_model_path = f"{self.save_dir}/best_model.pth"
            # Save the entire model and the hyperparameters
            torch.save({
                'model_state_dict': model.state_dict(),
                'hyperparameters': {
                    'dim1': dim1,
                    'dim2': dim2,
                    'latent_dim': latent_dim,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                }
            }, self.best_model_path)
        
        # Return the metric value for Optuna's optimization process
        return metric_value
    
# Optional: Method to save the best model path after all trials
    def save_best_model(self, final_save_dir):
        if self.best_model_path:
            os.makedirs(final_save_dir, exist_ok=True)
            shutil.copy(self.best_model_path, final_save_dir)
            print(f"Best model saved to: {os.path.join(final_save_dir, os.path.basename(self.best_model_path))}")
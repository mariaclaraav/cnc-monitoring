import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import r2_score
from torchsummary import summary
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

class VanillaAE(pl.LightningModule):
    def __init__(self, input_dim, dim1, dim2, latent_dim, learning_rate):
        super(VanillaAE, self).__init__()
        self.save_hyperparameters()  # Save hyperparameters to checkpoint
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.training_outputs = []
        self.training_r2_scores = []

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim1),
            nn.ReLU(),
            nn.Linear(dim1, dim2),
            nn.ReLU(),
            nn.Linear(dim2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, dim2),
            nn.ReLU(),
            nn.Linear(dim2, dim1),
            nn.ReLU(),
            nn.Linear(dim1, input_dim)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_latent_space(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, inputs)
        r2 = r2_score(outputs, inputs)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.training_outputs.append(loss)
        self.training_r2_scores.append(r2)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, inputs)
        r2 = r2_score(outputs, inputs)
        self.log('val_r2', r2, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_outputs).mean()
        avg_r2 = torch.stack(self.training_r2_scores).mean()
        self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_r2_epoch', avg_r2, on_step=False, on_epoch=True, prog_bar=True)
        self.training_outputs.clear()
        self.training_r2_scores.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def summarize_model(self, input_size):
        summary(self, input_size)
        
    def freeze_weights(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.parameters():
            param.requires_grad = True


class AutoencoderCallbacks:
    def __init__(self, monitor_metric='val_loss', min_delta=0.001, patience=5, verbose=True,
                 log_dir='lightning_logs', experiment_name='autoencoder_experiment', version=0, model_prefix='autoencoder'):
        self.monitor_metric = monitor_metric
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.version = version
        self.model_prefix = model_prefix

        self.experiment_path = os.path.join(self.log_dir, self.experiment_name, str(self.version))
        self.checkpoint_path = os.path.join(self.experiment_path, 'checkpoints')
        self.tensorboard_path = os.path.join(self.experiment_path, 'tensorboard')

        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.tensorboard_path, exist_ok=True)

    def get_callbacks(self):
        # Since we are only working with loss, mode is 'min'
        mode = 'min'

        early_stopping = EarlyStopping(
            monitor=self.monitor_metric,
            min_delta=self.min_delta,
            patience=self.patience,
            verbose=self.verbose,
            mode=mode
        )
        checkpoint = ModelCheckpoint(
            monitor=self.monitor_metric,
            save_top_k=1,
            mode=mode,
            dirpath=self.checkpoint_path,
            filename=f'{self.model_prefix}-model-{{epoch}}-{{{self.monitor_metric}:.6f}}',
            save_weights_only=True
        )
        return [early_stopping, checkpoint]

    def get_logger(self):
        logger = TensorBoardLogger(
            save_dir=self.tensorboard_path,
            name=self.experiment_name,
            version=self.version
        )
        return logger
    
class AnomalyDetector:
    def __init__(self, scaler=None, method='MSE'):
        """
        Initialize the AnomalyDetector with a scaler and a method for calculating reconstruction error.
        Args:
            scaler: Scaler object to normalize/denormalize data.
            method: Method for calculating reconstruction error ('MSE' or 'MAE').
        """
        if method not in ['MSE', 'MAE']:
            raise ValueError("The method must be either 'MSE' or 'MAE'")
        
        self.scaler = scaler
        self.method = method

    @staticmethod
    def ensure_numpy_array(data):
        """
        Ensure the data is a numpy array. If not, convert it to numpy array.
        """
        if not isinstance(data, np.ndarray):
            return data.values
        return data

    def calculate_reconstruction_error(self, inputs, outputs):
        """
        Calculate the reconstruction error for each sample.
        """
        if self.scaler:
            inputs = self.scaler.inverse_transform(inputs)  # true
            outputs = self.scaler.inverse_transform(outputs)  # predicted

        # Ensure the inputs and outputs are numpy arrays
        inputs = self.ensure_numpy_array(inputs)
        outputs = self.ensure_numpy_array(outputs)

        # Calculate the reconstruction error per feature for each sample
        if self.method == 'MSE':
            errors = np.mean((inputs - outputs) ** 2, axis=1)  # MSE per feature
        else: 
            errors = np.mean(np.abs(inputs - outputs), axis=1)  # MAE per feature            

        return errors

    @staticmethod
    def define_threshold(train_errors, c=8, xmax=None, library='seaborn'):
        """
        Define a threshold for anomaly detection based on the reconstruction errors.
        """
        train_errors = AnomalyDetector.ensure_numpy_array(train_errors)
        
        threshold = train_errors.mean() + c * train_errors.std()
        
        # Convert train_errors to a DataFrame if it's not already
        if isinstance(train_errors, np.ndarray):
            train_errors = pd.DataFrame(train_errors, columns=['error'])
        
        if library == 'seaborn':
            # Plot the error distribution for the train data using Seaborn
            plt.figure(figsize=(12, 4))
            plt.plot(train_errors['error'], color ='#1f77b4')
            plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
            plt.xlabel('Index')
            plt.ylabel('Reconstruction Error')
            plt.legend(fontsize=12)
        
            # Set x-axis limits if provided
            if xmax:
                plt.xlim(right=xmax)
        
            plt.show()
            
        elif library == 'plotly':
            # Plot the error distribution for the train data using Plotly
            fig = px.histogram(train_errors, x='error', nbins=800, title='Error Distribution',
                            labels={'error': 'Reconstruction Error'}, histnorm='density')
            fig.add_vline(x=threshold, line=dict(color='red', dash='dash'), annotation_text='Threshold', annotation_position='top right')
            
            fig.update_layout(
                xaxis_title='Reconstruction Error',
                yaxis_title='Density',
                showlegend=False,
                template='plotly_white'
            )
            
            # Set x-axis limits if provided
            if xmax:
                fig.update_xaxes(range=[0, xmax])
            
            fig.show()
        else:
            raise ValueError("plot_lib should be either 'seaborn' or 'plotly'")
        
        return threshold
    
    def plot_anomalies(self, test_data, decoded_data, threshold, column_index=1, range=None):
        """
        Plot the original and reconstructed data for a sample, highlighting anomalies.
        Args:
            test_data (np.ndarray): Original test data.
            decoded_data (np.ndarray): Reconstructed data.
            threshold (float): Threshold for anomaly detection.
            sample_index (int): Index of the sample to plot.
            t (np.ndarray): Time or sample index array for plotting.
            column_index (int): Index of the column to plot.
        """
        test_data = AnomalyDetector.ensure_numpy_array(test_data)
        decoded_data = AnomalyDetector.ensure_numpy_array(decoded_data)
        
        if self.scaler:
            test_data = self.scaler.inverse_transform(test_data)
            decoded_data = self.scaler.inverse_transform(decoded_data)

        # Calculate reconstruction error between the test data and reconstructed data using the specified method
        if self.method == 'MSE':
            val_error = np.mean(np.square(test_data - decoded_data), axis=1)
        else:
            val_error = np.mean(np.abs(test_data - decoded_data), axis=1)
        
        # Identify anomalies (boolean matrix)
        anomalies = val_error > threshold
        
        # Check if there are any anomalies in the series
        if np.any(anomalies):
            print('Anomalies detected in the series.')
        else:
            print('No anomalies detected in the series.')
            
        # Determine the range to plot
        if range is None:
            range = slice(None)
            
        plt.figure(figsize=(12, 4))
        plt.plot(test_data[range, column_index], 'b', label='Input', zorder=1)
        plt.plot(decoded_data[range, column_index], color = '#d62728', linestyle='dashed', label='Reconstruction', zorder=2)
            
        # Highlight anomalies
        anomaly_indices = np.where(anomalies)[0]
        if anomaly_indices.size > 0:
            plt.scatter(anomaly_indices[range], test_data[range, column_index][anomaly_indices], color='orange', s=20, label='Anomaly', zorder=3)
        plt.legend()
        plt.show()

    def detected_anomalies(self, test_data, decoded_data, y_test, threshold, column_index=1, plot_range=None):
        """
        Plot the original and reconstructed data for a sample, highlighting anomalies.
        Args:
            test_data (np.ndarray): Original test data.
            decoded_data (np.ndarray): Reconstructed data.
            y_test (np.ndarray): Array indicating known anomalies (1 for anomaly, 0 for normal).
            threshold (float): Threshold for anomaly detection.
            column_index (int): Index of the column to plot.
            plot_range (slice): Range of indices to plot.
        """
        test_data = AnomalyDetector.ensure_numpy_array(test_data)
        decoded_data = AnomalyDetector.ensure_numpy_array(decoded_data)
        y_test = AnomalyDetector.ensure_numpy_array(y_test)
        
        if self.scaler:
            test_data = self.scaler.inverse_transform(test_data)
            decoded_data = self.scaler.inverse_transform(decoded_data)
            
        # Calculate reconstruction error between the test data and reconstructed data using the specified method
        if self.method == 'MSE':
            val_error = np.mean(np.square(test_data - decoded_data), axis=1)
        else:
            val_error = np.mean(np.abs(test_data - decoded_data), axis=1)
        
        # Identify anomalies (boolean matrix)
        anomalies = val_error > threshold
        
        # Check if there are any anomalies in the series
        if np.any(anomalies):
            print('Anomalies detected in the series.')
        else:
            print('No anomalies detected in the series.')
            
        # Determine the range to plot
        if plot_range is None:
            plot_range = slice(None)
        
        # Plot the original and reconstructed data
        plt.figure(figsize=(12, 4))
   
        plt.plot(test_data[plot_range, column_index], 'b', label='Input', zorder=2)
        #plt.plot(decoded_data[plot_range, column_index], 'r', label='Reconstruction')
        
        # Highlight anomalies detected by the model
        anomaly_indices = np.where(anomalies)[0]
        if anomaly_indices.size > 0:
            plt.scatter(anomaly_indices, test_data[anomaly_indices, column_index], color='orange', s=20, label='Model', zorder=3)
        
        ymin, ymax = plt.ylim()
        plt.fill_between(np.arange(len(y_test)), ymin, ymax, where=y_test == 1, color='red', alpha=0.3, label='Known', zorder=1)
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Detected Anomalies')
        plt.legend(loc='upper right')
        plt.show()

        
def reconstruct_predicted_data(loader, model, device):
    model.eval()
    model.freeze()
    outputs_collected = []
    latent_spaces = []

    with torch.no_grad():
        for data in loader:
            inputs = data.to(device)
            outputs = model(inputs)

            #inputs_collected.append(inputs.cpu().numpy())
            outputs_collected.append(outputs.cpu().numpy())
            latent_space = model.get_latent_space(inputs)
            latent_spaces.append(latent_space.cpu().numpy())

    #inputs_concatenated = np.concatenate(inputs_collected, axis=0)
    outputs_reconstructed = np.concatenate(outputs_collected, axis=0)
    latent_spaces_concatenated = np.concatenate(latent_spaces, axis=0)

    return outputs_reconstructed, latent_spaces_concatenated


def load_model(checkpoint_path, model_class, **kwargs):
    """
    Load a trained model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model_class (pl.LightningModule): The class of the model to be loaded.
        **kwargs: Additional keyword arguments to be passed to the model's __init__ method.
        
    Returns:
        pl.LightningModule: The loaded model.
    """
    model = model_class.load_from_checkpoint(checkpoint_path, **kwargs)
    model.eval()  # Set the model to evaluation mode
    model.freeze()  # Ensure the model parameters are not modified
    return model

def train_model(model, train_loader, val_loader, num_epochs, trainer_logger, callbacks_list, logger):
    logger.info('Training the model...')
    trainer = Trainer(
        max_epochs=num_epochs,
        logger=trainer_logger,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=callbacks_list,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
def create_dataloaders(X_train, X_val, X_test, BATCH_SIZE, NUM_WORKERS, logger):
    """
    Creates DataLoaders for training, validation, and test datasets.

    Args:
        X_train (array-like): Training data.
        X_val (array-like): Validation data.
        X_test (array-like): Test data.
        BATCH_SIZE (int): Number of samples per batch to load.
        NUM_WORKERS (int): Number of subprocesses to use for data loading.
        logger (logging.Logger): Logger for logging information.

    Returns:
        tuple: A tuple containing three DataLoaders (train_loader, val_loader, test_loader).
    """
    logger.info('Creating DataLoaders...')
    train_loader = DataLoader(
        torch.tensor(X_train, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    val_loader = DataLoader(
        torch.tensor(X_val, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    test_loader = DataLoader(
        torch.tensor(X_test, dtype=torch.float32),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True
    )
    return train_loader, val_loader, test_loader
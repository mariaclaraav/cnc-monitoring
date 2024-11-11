import os
import torch
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import torch.nn as nn
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.functional import r2_score
import torchmetrics
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, data, window_size):
        """
        Initializes a SequenceDataset object.
        Parameters:
        - data (ndarray): The input data.
        - window_size (int): The size of the sliding window.
        """
        self.data = torch.tensor(data, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self):
        """
        Returns the length of the dataset.
        Returns:
        - length (int): The length of the dataset.
        """
        return len(self.data) - self.window_size + 1
    
    def __getitem__(self, idx):
        """
        Returns the item at the given index.
        Parameters:
        - idx (int): The index of the item.
        Returns:
        - x (Tensor): The input data.
        - x (Tensor): The target data (same as input for autoencoders).
        """
        x = self.data[idx:idx + self.window_size]
        return x, x  # Input and target are the same for autoencoders


class DataLoaderManager:
    def __init__(self, train_data, val_data, test_data, window_size, batch_size, num_workers=0, shuffle = False, persistent_workers=False):
        self.window_size = window_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.persistent_workers = persistent_workers
        
        # Initialize datasets
        self.train_dataset = SequenceDataset(train_data, self.window_size)
        self.val_dataset = SequenceDataset(val_data, self.window_size)
        self.test_dataset = SequenceDataset(test_data, self.window_size)

        # Create DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            persistent_workers=self.persistent_workers
        )
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_test_loader(self):
        return self.test_loader
    
    def print_batch_info(self):
       # Obtém a primeira batch do train_loader
        first_batch = next(iter(self.train_loader))
        input_batch, target_batch = first_batch

        # Imprime as formas das entradas e saídas
        print(f"Shape of the first batch (input): {input_batch.shape}")


        # Imprime o número de batches
        num_batches = len(self.train_loader)
        print(f'Number of batches in train_loader: {num_batches}')

class LSTMAE(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim1, latent_dim, learning_rate):
        super(LSTMAE, self).__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()
        self.training_outputs = []

        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.encoder_fc = nn.Linear(hidden_dim1, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim1)
        self.decoder_lstm = nn.LSTM(hidden_dim1, input_dim, batch_first=True)

    def forward(self, x):        
        # Encoder
        _, (h_n, _) = self.encoder_lstm(x) # Returns Output, hidden state and cell state (The memory cell)
        x = h_n[-1]  # Taking the hidden state from the last LSTM layer, shape [batch_size, hidden_dim]
        x = self.encoder_fc(x)  # Projecting into the latent space, shape [batch_size, latent_dim] --> you lose the dimension window_size


        # Decoder
        x = self.decoder_fc(x)  # Projects latent_dim to hidden_dim, shape [batch_size, hidden_dim]
        x = x.unsqueeze(1).repeat(1, 2048, 1)  #unsqueeze(1) specifically adds a dimension at position 1 of the tensor --> it goes from [batch_size, hidden_dim], to [batch_size, 1, hidden_dim].
        x, _ = self.decoder_lstm(x)  # Decode to [batch_size, window_size, input_dim]
        return x

    def get_latent_space(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch  # Unpacking the batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True)
        self.training_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch  # Unpacking the batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_outputs).mean()
        #avg_r2 = torch.stack(self.training_r2_scores).mean()
        self.log('train_loss_epoch', avg_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.training_outputs.clear()


    def configure_optimizers(self):
        """
        Configure the optimizer for the model.
        
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def summarize_model(self, input_size):
        """
        Print the summary of the model architecture.

        """
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
            filename=f'{self.model_prefix}-model-{self.monitor_metric}-{{epoch}}-{{{self.monitor_metric}:.6f}}',
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
    def __init__(self, scaler=None):
        self.scaler = scaler

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
            inputs = self.scaler.inverse_transform(inputs)
            outputs = self.scaler.inverse_transform(outputs)
        errors = np.mean((inputs - outputs) ** 2, axis=1)
        
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
            plt.figure(figsize=(10, 6))
            sns.histplot(train_errors['error'], kde=True, stat='density')
            plt.axvline(x=threshold, color='red', linestyle='--', label='Threshold')
            plt.title('Error Distribution')
            plt.xlabel('Reconstruction Error')
            plt.ylabel('Density')
            plt.legend()
        
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
    
    def plot_anomalies(self, test_data, decoded_data, threshold, column_index=1, range = None):
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

        # Calculate mean squared error between the test data and reconstructed data
        val_mse = np.mean(np.square(test_data - decoded_data), axis=1)
        
        # Identify anomalies (boolean matrix)
        anomalies = val_mse > threshold
        
        # Check if there are any anomalies in the series
        if np.any(anomalies):
            print('Anomalies detected in the series.')
        else:
            print('No anomalies detected in the series.')
            
        # Determine the range to plot
        if range is None:
            range = slice(None)
            
        plt.figure(figsize=(12, 4))
        plt.plot(test_data[range, column_index], 'b', label='Input')
        plt.plot(decoded_data[range, column_index], 'r', label='Reconstruction')
            
        # Highlight anomalies
        anomaly_indices = np.where(anomalies)[0]
        if anomaly_indices.size > 0:
            plt.scatter(anomaly_indices[range], test_data[range, column_index][anomaly_indices], color='orange', s=40, label='Anomaly')
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
            
        # Calculate mean squared error between the test data and reconstructed data
        val_mse = np.mean(np.square(test_data - decoded_data), axis=1)
        
        # Identify anomalies (boolean array)
        anomalies = val_mse > threshold
        
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
        plt.plot(test_data[plot_range, column_index], 'b', label='Input')
        #plt.plot(decoded_data[plot_range, column_index], 'r', label='Reconstruction')
        
        # Highlight anomalies detected by the model
        anomaly_indices = np.where(anomalies)[0]
        if anomaly_indices.size > 0:
            plt.scatter(anomaly_indices, test_data[anomaly_indices, column_index], color='orange', s=40, label='Model')
        
         # Shade regions where known anomalies are present
        ymin, ymax = plt.ylim()
        plt.fill_between(np.arange(len(y_test)), ymin, ymax, where=y_test == 1, color='red', alpha=0.3, label='Known')
        
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title('Detected Anomalies')
        plt.legend(loc='upper right')
        plt.show()

def reconstruct_predicted_data(loader, model, device):
    model.eval()
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
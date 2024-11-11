import os
import sys
from pathlib import Path
import logging
import pandas as pd
import torch
import matplotlib.pyplot as plt
import tracemalloc

# Set float32 matmul precision
torch.set_float32_matmul_precision('medium')

CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

# Set plot style
style = os.path.join(CURRENT_DIR, 'src', 'utils', 'visualization', 'plot.mplstyle')
plt.style.use(style)

# Import project-specific modules
from utils.data_processing.splitting import DataProcessor
from src.models.VanillaAE import VanillaAE, AutoencoderCallbacks, train_model, create_dataloaders

# Constants and configuration
BATCH_SIZE = 128*4
NUM_WORKERS = 4
NUM_EPOCHS = 200
LEARNING_RATE = 0.005
DIM_1 = 100
DIM_2 = 32
DIM_LATENT = 16
PATIENCE = 20
FEATURES = ['X_axis', 'X_node_aad', 'X_node_add', 'X_node_ada',
            'X_node_dda', 'X_node_ddd', 'X_node_dad', 'X_node_daa', 'Y_axis',
            'Y_node_aaa', 'Y_node_aad', 'Y_node_add', 'Y_node_ada', 'Y_node_dda',
            'Y_node_ddd', 'Y_node_dad', 'Y_node_daa', 'Z_axis', 'Z_node_aaa',
            'Z_node_aad', 'Z_node_add', 'Z_node_ada', 'Z_node_dda', 'Z_node_ddd',
            'Z_node_dad', 'Z_node_daa']

SCALER_TYPE = 'standard'
OPERATIONS = ['OP00', 'OP01','OP04','OP14']

LOG_DIR = os.path.join(CURRENT_DIR, 'code', 'modeling', 'lightning_logs')
EXPERIMENT_NAME = 'VanillaAE_exp_5'
VERSION = 'version_0'
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def configure_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f'Number of GPUs available: {num_gpus}')
        for i in range(num_gpus):
            logger.info(f'GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        logger.warning('CUDA is not available. No GPUs found.')

def main():
    # Clear console (optional)
    os.system('cls' if os.name == 'nt' else 'clear')

    # Initialize DataProcessor
    processor = DataProcessor(logger,
                              features=FEATURES, 
                              scaler_type=SCALER_TYPE)

    # Load data
    df_train = processor.load_data(os.path.join(DATA_PATH, 'final_train.parquet'), operation=OPERATIONS)
    df_val = processor.load_data(os.path.join(DATA_PATH, 'final_val.parquet'), operation=OPERATIONS)
    df_test = processor.load_data(os.path.join(DATA_PATH, 'final_test.parquet'), operation=OPERATIONS)

    # Process data
    X_train, _, X_val, _, _, scaler = processor.process_data(
        df_train=df_train,
        df_val=df_val
    )
    X_test, _, _ = processor.process_test_data(df_test=df_test)
    X_train = X_train.values
    X_test = X_test.values
    X_val = X_val.values
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(X_train = X_train, 
                                                           X_val=X_val, 
                                                           X_test=X_test, BATCH_SIZE=BATCH_SIZE, NUM_WORKERS=NUM_WORKERS, logger=logger)
    logger.info(f'Training set size: {X_train.shape}')
    logger.info(f'Validation set size: {X_val.shape}')
    logger.info(f'Test set size: {X_test.shape}')

    # Configure GPU
    configure_gpu()

    # Initialize model
    model = VanillaAE(
        input_dim=X_train.shape[1],
        dim1=DIM_1,
        dim2=DIM_2,
        latent_dim=DIM_LATENT,
        learning_rate=LEARNING_RATE
    )
    model.summarize_model(input_size=[(BATCH_SIZE, X_train.shape[1])])

    # Set up callbacks and logger
    callbacks = AutoencoderCallbacks(
        monitor_metric='val_loss',
        min_delta=1e-5,
        patience=PATIENCE,
        verbose=True,
        log_dir=LOG_DIR,
        experiment_name=EXPERIMENT_NAME,
        version=VERSION,
        model_prefix='VanillaAE'
    )

    callbacks_list = callbacks.get_callbacks()
    trainer_logger = callbacks.get_logger()

    logger.info(f'Logger Version: {trainer_logger.version}')

    # Train model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        trainer_logger=trainer_logger,
        callbacks_list=callbacks_list, 
        logger=logger
    )

if __name__ == '__main__':
    tracemalloc.start()
    snapshot_before = tracemalloc.take_snapshot()
    
    main()
    
    # After running the code
    snapshot_after = tracemalloc.take_snapshot()
    
    stats = snapshot_after.compare_to(snapshot_before, 'lineno')

    # Show differences
    logger.info(f'Top 10 differences in memory allocations:\n')
    for stat in stats[:10]:
        print(stat)

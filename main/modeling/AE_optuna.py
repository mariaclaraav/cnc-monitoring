import os
import sys
import optuna
import torch
import matplotlib.pyplot as plt
import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, PatientPruner

torch.set_float32_matmul_precision('medium')


def main():
    # Set current directory
    os.system('cls')
    current_dir = os.getcwd()
    sys.path.append(current_dir)

    style = os.path.join(current_dir, 'src','utils','visualization','plot.mplstyle')
    plt.style.use(style)

    # Import project-specific modules
    from utils.data_processing.splitting import create_mask, SplitData
    from src.model_lib.Optuna import AutoencoderObjective

    # Define paths
    data_path = os.path.join(current_dir, 'data', 'processed', 'wavelet', 'OP04.parquet')
    storage_path = os.path.join(current_dir, 'code', 'modeling', 'AE_study')
    df = pd.read_parquet(data_path)

    # Create train and test masks
    train_mask = create_mask(df, ['02-2021', '08-2021'], normal=True)
    test_mask = create_mask(df, ['02-2019','08-2019'], normal = False)

    # Split data
    split_data = SplitData(df, train_mask, test_mask, n_val=0.2, 
                           features=['X_node_aad','Y_node_aad', 'Z_node_aad'])

    # Access the prepared data
    X_train, y_train = split_data.X_train, split_data.y_train
    X_val, y_val = split_data.X_val, split_data.y_val
    X_test, y_test = split_data.X_test, split_data.y_test

    print(f'Training set size: {X_train.shape}')
    print(f'Validation set size: {X_val.shape}')
    print(f'Test set size: {X_test.shape}')


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Constants
    TRAIN_DATA = X_train
    VAL_DATA = X_val
    NUM_WORKERS = 4
    NUM_EPOCHS = 200
    INPUT_DIM = TRAIN_DATA.shape[1]
    MIN_DELTA = 0.0001
    PATIENCE = 20
    SAVE_DIR = 'optuna_logs'
    EXPERIMENT_NAME = 'OptunaAE_experiment_2'
    MONITOR_METRIC = 'val_loss'  # or 'val_r2'


    # Instanciate Objective function 
    objective = AutoencoderObjective(
        input_dim = INPUT_DIM, 
        save_dir = SAVE_DIR, 
        experiment_name = EXPERIMENT_NAME, 
        min_delta = MIN_DELTA, 
        patience = PATIENCE, 
        max_epochs = NUM_EPOCHS,
        monitor_metric = MONITOR_METRIC)

    objective.set_data(X_train = TRAIN_DATA, 
                       X_val = VAL_DATA, 
                       num_workers = NUM_WORKERS, 
                       persistent_workers = True)


    objective.set_hyperparameters({
        'dim1': lambda trial: trial.suggest_int('dim1', 32, 128),
        'dim2': lambda trial: trial.suggest_int('dim2', 16, 64),
        'latent_dim': lambda trial: trial.suggest_int('latent_dim', 8, 32),
        'learning_rate': lambda trial: trial.suggest_float('learning_rate', 1e-4, 1e-2),
        'batch_size': lambda trial: trial.suggest_categorical('batch_size', [12, 32, 64, 128, 256, 512])
    })

    torch.set_float32_matmul_precision('medium')

    # Create the Optuna study
    direction = 'minimize'
    study_name = 'AE_study'
    storage_name = f'sqlite:///{storage_path}//{study_name}.db'

    sampler = TPESampler(seed=10)
    pruner = PatientPruner(MedianPruner(n_warmup_steps=5), patience=10) # Wait 5 epochs to check the prunning condition and wait for 10 epochs after a pruning condition is met before actually pruning the trial.


    # Create the Optuna study with persistent storage
    study = optuna.create_study(study_name=study_name, storage=storage_name, 
                                direction=direction, sampler = sampler, 
                                pruner = pruner, load_if_exists=True)


    study.optimize(objective, n_trials=20) #Using TPEssampler

    best_trial = study.best_trial

    print(f"Best trial: {best_trial.value}")
    print(f"Best hyperparameters: {best_trial.params}")

if __name__ == '__main__':
    main()
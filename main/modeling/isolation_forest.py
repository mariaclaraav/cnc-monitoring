import os
import sys
import logging

from tqdm import tqdm
import pickle
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)

# Set up the current directory and system path for module imports
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

# Import required modules
from src.utils.data_processing.model_processor import ModelProcessor
from src.utils.data_processing.predicted_anoamlies import PredictedAnomalies
from src.model_lib.isolation_forest import IsolationForestModel
from src.features.feature_selector import FeatureSelector

GROUPS = ['OP01', 'OP02', 'OP03', 'OP04', 'OP05', 'OP06', 'OP07' 'OP08', 'OP09', 'OP10', 'OP11', 'OP12', 'OP14']

SCALER_TYPE = None
DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed', 'split_train_test', 'all_features')
SAVING_PATH = os.path.join(CURRENT_DIR, 'data','predicted', 'filtro_adaptavel')
MODEL_PATH = os.path.join(CURRENT_DIR, 'src','models', 'isolation_forest')
IMAGE_PATH = os.path.join(CURRENT_DIR, 'figure', 'isolation_forest')

def main(group):
    # Clear terminal output
    os.system('cls' if os.name == 'nt' else 'clear')
                        
    # Initialize DataProcessor
    processor = ModelProcessor(logger, 
                                scaler_type=SCALER_TYPE,
                                need_val=False)

    # Load data
    data_files = {
        'train': f'final_train_{group}.parquet',
        'val': f'final_val_{group}.parquet',
        'test': f'final_test_{group}.parquet'  
    }

    data = {
        key: processor.load_data(os.path.join(DATA_PATH, file), operation=group)
        for key, file in data_files.items()
    }
    features = FeatureSelector(data['train']).get_features()
    # Process data
    X_train, _, _, _, _, scaler = processor.process_data(df_train=data['train'], df_val=data['val'], features=features)
    X_test, y_test, test_codes = processor.process_test_data(df_test=data['test'], features=features)

    # Initialize and train the Isolation Forest model
    model = IsolationForestModel(n_estimators= 200, 
                                    contamination=0.05, 
                                    random_state=42, 
                                    bootstrap=False
                                    )
    model.train(X_train)

    # Predict and evaluate
    y_pred, metrics = model.predict(X_test, y_test)
    
    file_path = os.path.join(MODEL_PATH, f"{group}_model.pkl")
    
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print("Model saved...")

    # Visualize prediction scores
    model.plot_scores(X_val = X_test, 
                      save_dir=IMAGE_PATH, 
                      group=group)

    # Initialize the evaluator with the saving path
    evaluator = PredictedAnomalies(saving_path=SAVING_PATH, 
                                    group=group, 
                                    df_test = data['test'])

    # Evaluate anomalies
    validation, anomaly_scores = evaluator.evaluate(X_test = X_test, 
                                                    test_codes= test_codes, 
                                                    y_pred = y_pred, 
                                                    y_test = y_test)

    # Save results to JSON
    evaluator.save_results(validation, anomaly_scores)

if __name__ == "__main__":
    logger = logging.getLogger()
    for GROUP in tqdm(GROUPS, desc="Processing Groups", unit="group"):
        main(group=GROUP)


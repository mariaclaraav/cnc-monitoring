import os
import sys
import logging
import matplotlib.pyplot as plt



# Constants and configuration
CURRENT_DIR = os.getcwd()
sys.path.append(CURRENT_DIR)

# Import project-specific modules
from src.models.IsolationForest import AblationStudy
from utils.data_processing.splitting import DataProcessor

style = os.path.join(CURRENT_DIR, 'src','utils','visualization','plot.mplstyle')
plt.style.use(style)

DATA_PATH = os.path.join(CURRENT_DIR, 'data', 'processed')
STORAGE_PATH = os.path.join(CURRENT_DIR, 'code', 'modeling', 'IF_study')

###################### Train, val and test split #####################
FEATURES = ['Z_axis','Y_node_aad', 'Z_node_aad']
SCALER_TYPE = 'standard' # 'standard' or 'minmax'
OPERATIONS = ['OP01','OP02']

######################## Ablation study ##############################
FRACTION_START = 1.0
FRACTION_END = 0.1
N = 1
DECREMENT = 0.1
METRIC = 'precision'
ABLATION_TYPE = 'operation'  # 'fraction' or 'operation' or 'label_change'

######################## Optuna configuration ########################
SEED = 42
N_WARMUP_STEPS = 5
N_TRIALS = 20
#######################################################################


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


        
def isolation_forest():
    os.system('cls')

    # Initialize DataProcessor
    processor = DataProcessor(logger,
                              features=FEATURES, 
                              scaler_type=SCALER_TYPE)

    # Load data
    df_train = processor.load_data(os.path.join(DATA_PATH, 'final_train.parquet'), operation=OPERATIONS)
    df_val = processor.load_data(os.path.join(DATA_PATH, 'final_val.parquet'), operation=OPERATIONS)
    

    # Process data
    X_train, _, X_val, y_val, val_codes, scaler = processor.process_data(df_train=df_train,
                                                                            df_val=df_val)        
    study = AblationStudy(X_train=X_train, X_val=X_val, y_val=y_val, unique_codes=val_codes, 
                          storage_path=STORAGE_PATH, metric=METRIC, seed=SEED,n_trials=N_TRIALS, n_warmup_steps=N_WARMUP_STEPS, logger=logger)
    
    study.run_studies(fraction_start=FRACTION_START, fraction_end=FRACTION_END, decrement=DECREMENT,ablation_type=ABLATION_TYPE, n_decrement=N)  

    logger.info('\nOptimization process completed.')
    
if __name__ == "__main__":
    isolation_forest()
  

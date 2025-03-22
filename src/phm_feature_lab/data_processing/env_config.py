import os
import pandas as pd
from phm_feature_lab.utils.constants.process_constats import CONFIG

def configure_environment(group, configuration = CONFIG, set: str = "frequency_features", data_path: str = None):
    """
    Configures the environment and retrieves operations and features based on the selected group.

    Parameters:
    - group (int or str): Group ID to configure settings.
    - set (str): The feature set to configure ("frequency_features" or "all_features").
    - df (pd.DataFrame, optional): DataFrame to extract features from if using "all_features".

    Returns:
    - operations (list): List of operations for the selected group.
    - features (list): List of features to use in the analysis.
    """

    if set == "frequency_features":
        config = configuration[group]
        operations = config["OPERATIONS"]
        freq = config["FREQ"]

        # Feature creation
        base_features = ['Time']
        bands = [freq, '385-405Hz']  # Add other bands if needed
        features = base_features + [f'{axis}_{band}' for axis in ['X', 'Y', 'Z'] for band in bands]

    elif set == "all_features":
        if data_path is None:
            raise ValueError("For 'all_features', the data_path must be provided.")
        
        config = configuration[group]
        operations = config["OPERATIONS"]

        # Read feature columns dynamically from the file for the first operation
        operation_file = os.path.join(data_path, f"{operations[0]}.parquet")
        
        if not os.path.exists(operation_file):
            raise FileNotFoundError(f"Feature file for operation '{operations[0]}' not found at: {operation_file}")
        ## TODO improve memory manager
        # Read only column names
        df_columns = pd.read_parquet(operation_file).columns.tolist()
        
        # Select features starting with X, Y, Z, excluding specific axis columns
        exclude_columns = {'X_axis', 'Y_axis', 'Z_axis'}
        base_features = ['Time']
        features = base_features + [col for col in df_columns if col.startswith(('X', 'Y', 'Z')) and col not in exclude_columns]

    else:
        raise ValueError(f"Unknown feature set: {set}")

    return operations, features

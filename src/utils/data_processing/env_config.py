import os
import pandas as pd
def configure_environment(group, set: str = "frequency_features", data_path: str = None):
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
    CONFIG = {
        1: {
            "OPERATIONS": ['OP01', 'OP03', 'OP04', 'OP06', 'OP08', 'OP09', 'OP10', 'OP11', 'OP12', 'OP14'],
            "FREQ": '235-265Hz',
        },
        2: {
            "OPERATIONS": ['OP05', 'OP02', 'OP07'],
            "FREQ": '185-215Hz',
        },
        'OP01': {
            "OPERATIONS": ['OP01'],
            "FREQ": '235-265Hz',
        },
        'OP02': {
            "OPERATIONS": ['OP02'],
            "FREQ": '185-215Hz',
        },
        'OP03': {
            "OPERATIONS": ['OP03'],
            "FREQ": '235-265Hz',
        },
        'OP04': {
            "OPERATIONS": ['OP04'],
            "FREQ": '235-265Hz',
        },
        'OP05': {
            "OPERATIONS": ['OP05'],
            "FREQ": '185-215Hz',
        },
        'OP06': {
            "OPERATIONS": ['OP06'],
            "FREQ": '235-265Hz',
        },
        'OP07': {
            "OPERATIONS": ['OP07'],
            "FREQ": '185-215Hz',
        },
        'OP08': {
            "OPERATIONS": ['OP08'],
            "FREQ": '235-265Hz',
        },
        'OP09': {
            "OPERATIONS": ['OP09'],
            "FREQ": '235-265Hz',
        },
        'OP10': {
            "OPERATIONS": ['OP10'],
            "FREQ": '235-265Hz',
        },
        'OP11': {
            "OPERATIONS": ['OP11'],
            "FREQ": '235-265Hz',
        },
        'OP12': {
            "OPERATIONS": ['OP12'],
            "FREQ": '235-265Hz',
        },
        'OP14': {
            "OPERATIONS": ['OP14'],
            "FREQ": '235-265Hz',
        },
    }

    if set == "frequency_features":
        config = CONFIG[group]
        operations = config["OPERATIONS"]
        freq = config["FREQ"]

        # Feature creation
        base_features = ['Time']
        bands = [freq, '385-405Hz']  # Add other bands if needed
        features = base_features + [f'{axis}_{band}' for axis in ['X', 'Y', 'Z'] for band in bands]

    elif set == "all_features":
        if data_path is None:
            raise ValueError("For 'all_features', the data_path must be provided.")
        
        config = CONFIG[group]
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

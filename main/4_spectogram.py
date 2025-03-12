import os
import gc
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from phm_feature_lab.features.custom_cwt import CustomCWT
from phm_feature_lab.utils.logger import Logger
from phm_feature_lab.utils.data_processing.data_scaler import DataScaler
from phm_feature_lab.utils.data_processing.operation_filter import OperationFilter
from phm_feature_lab.utils.data_processing.load_files import LoadFiles

logger = Logger().get_logger()


# Constants and configuration
CURRENT_DIR = os.getcwd()
SAVING_PATH = os.path.join(CURRENT_DIR, "data", "processed", "spectogram")
DATA_PATH = os.path.join(CURRENT_DIR, "data", "processed", "ETL", "ETL_final.parquet")

os.makedirs(SAVING_PATH, exist_ok=True)

# Parameters
FREQUENCIES = [600, 100]  # Desired frequencies
SAMPLE_RATE = 2000  # Sampling rate
OPERATIONS = ["OP06", "OP07"]
COLUMNS = ["Time", "X_axis", "Y_axis", "Z_axis", "Process", "Unique_Code"]
WAVELET = "cmor0.5-1.5"


# Main script
def main() -> None:

    cwt_transform = CustomCWT(
        frequencies=FREQUENCIES, wavelet=WAVELET, sampling_rate=SAMPLE_RATE
    )

    # Load the data
    data_loader = LoadFiles(DATA_PATH)
    df = data_loader.load(format="parquet")

    op_filter = OperationFilter(df)
    df = op_filter.filter(COLUMNS, OPERATIONS)

    scale = DataScaler(scaler=StandardScaler(), exclude_columns=["Time", "Unique_Code"]
    )

    for op in tqdm(OPERATIONS, desc="Processing operations"):
        
        df_filtered = op_filter.filter_by_operation(df, operation=op)
        df_filtered = scale.fit_transform(df_filtered)

        unique_codes = op_filter.get_unique_codes(df_filtered)
        for code in tqdm(
            unique_codes, desc=f"Processing unique codes for {op}", leave=False
        ):
            subset = op_filter.filter_by_unique_code(df_filtered, unique_code=code)

            for axis in ["X_axis", "Y_axis", "Z_axis"]:
                signal = subset[axis]
                path = os.path.join(SAVING_PATH, f"{op}", f"{code}_{axis}.png")
                cwt_transform.save_cwt_img(
                    time=subset.Time, signal=signal.values, save_path=path
                )
            del subset
            gc.collect()
            
        del df_filtered
        gc.collect()


if __name__ == "__main__":
    main()

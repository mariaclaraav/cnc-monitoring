import time
import os
import sys
import logging
from tqdm import tqdm

# Configure logging to output only to the console
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Outputs to the console only
    ]
)


current_dir = os.getcwd()
sys.path.append(current_dir)
print(f"Current directory: {current_dir}")

from src.utils.data_processing.make_dataset import DataTransform, UniqueCodeCorrector, DataSplitter

if __name__ == "__main__":
    # Start the timer for the entire script
    os.system('cls')
    start_time = time.time()
    
    path_to_dataset = os.path.join(current_dir, 'data', 'raw')
    machines = ["M01", "M02", "M03"]
    process_names = ["OP00", "OP01", "OP02", "OP03", "OP04", "OP05", "OP06", "OP07", "OP08", "OP09", "OP10", "OP11", "OP12", "OP13", "OP14"]
    labels = ["good", "bad"]
    path_to_save_parquet = os.path.join(current_dir, 'data', 'processed', 'ETL')



    ################ Load and Transform data  ##################
    logging.info("Starting data load and transformation...")
    load_transform_start = time.time()
    data_transform = DataTransform(path_to_dataset, machines, process_names, labels)
    transformed_file_path = data_transform.run(path_to_save_parquet)
    load_transform_end = time.time()
    logging.info(f"Data load and transformation completed in {load_transform_end - load_transform_start:.2f} seconds.")


    ############ Correct Unique Codes for each machine ############
    logging.info("Starting unique code correction for each machine...")
    unique_code_correction_start = time.time()    
    
    for machine in tqdm(machines, desc="Correcting Unique Codes\n"):
        path_to_save_machine_parquet = os.path.join(current_dir, 'data', 'processed', 'ETL', f'ETL_{machine}.parquet')        
        unique_code_corrector = UniqueCodeCorrector()
        unique_code_corrector.run(transformed_file_path, machine, path_to_save_machine_parquet)
    unique_code_correction_end = time.time()
    logging.info(f"Unique code correction completed in {unique_code_correction_end - unique_code_correction_start:.2f} seconds.")



    ############### Final Processing and Split data into Train and Test datasets ###############
    
    split_data = False # Set to False if you do not want to split the data
    test_year = 2020   # Define the test year

    # Create the dynamic start message
    if split_data:
        start_message = f"Starting final processing and Split = True - Test Period = {test_year}"
    else:
        start_message = "Starting final processing and Split = False"

    logging.info(start_message)

    data_split_start = time.time()

    # Define paths for the data
    paths = [os.path.join(current_dir, 'data', 'processed', 'ETL', f'ETL_{machine}.parquet') for machine in machines]

    # Define the base path for saving the dataset(s)
    base_path = os.path.join(current_dir, 'data', 'processed', 'ETL', 'ETL')

    # Initialize and run the DataSplitter
    data_splitter = DataSplitter(paths, base_path, test_year=test_year, split=split_data)
    data_splitter.split_data()

    data_split_end = time.time()
    logging.info(f"Data processing completed in {data_split_end - data_split_start:.2f} seconds.")

    # End the timer for the entire script
    end_time = time.time()
    logging.info(f"Total execution time: {end_time - start_time:.2f} seconds.")



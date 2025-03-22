import os
from typing import List

import numpy as np
import pandas as pd
from PIL import Image

class ImageProcessing:
    
    @staticmethod
    def get_files(path_to_files, size=(224, 224), mode='RGB'):
        """
        Loads and processes the image files from the specified path.
        
        Parameters:
        - path_to_files: The path to the image directory.
        - size: Desired image size (default is (224, 224)).
        - mode: The mode for image conversion (default is 'RGB'). Use "L" for grayscale.
        
        Returns:
        - imgs: A list of processed images.
        - img_names: A list of image file names.
        
        Raises:
        - FileNotFoundError: If the specified path does not exist.
        - ValueError: If no image files are found in the specified path.
        """
        if not os.path.exists(path_to_files):
            raise FileNotFoundError(f"The specified path {path_to_files} does not exist.")
        
        imgs = []
        img_names = []
        
        files = [file for file in os.listdir(path_to_files) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]  # Filter image files
        
        if not files:
            raise ValueError(f"No image files found in the specified path {path_to_files}.")
        
        for file in files:
            try:
                img = Image.open(os.path.join(path_to_files, file))
                img = img.convert(mode)  # Convert to RGB
                img = np.asarray(img.resize(size))
                imgs.append(img)
                img_names.append(file)
            except Exception as e:
                print(f"Error processing file {file}: {e}")
        
        return imgs, img_names

    @staticmethod
    def get_info(img_names: List[str], Label: pd.DataFrame) -> pd.DataFrame:
        
        """
        Preprocesses image names and associates metadata such as Unique_Code, Label, period, machine, and axis.
        
        Parameters:
        - img_names: List of image file names (list of strings).
        - Label: DataFrame with 'Unique_Code' and corresponding 'Label' columns.
        
        Returns:
        - img_info: DataFrame with processed image metadata.
        
        Raises:
        - ValueError: If img_names or Label are empty.
        - KeyError: If required columns are missing in the Label DataFrame.
        """
        if not img_names:
            raise ValueError("The img_names list is empty.")
        
        if Label.empty:
            raise ValueError("The Label DataFrame is empty.")
        
        required_columns = {'Unique_Code', 'Label'}
        if not required_columns.issubset(Label.columns):
            raise KeyError(f"The Label DataFrame must contain the following columns: {required_columns}")
        
        # Initialize the DataFrame with image names
        img_info = pd.DataFrame(img_names, columns=['image_name'])

        # Generate the Unique_Code by removing the suffixes from the image name (X, Y, Z)
        img_info['Unique_Code'] = img_info['image_name'].str.replace(r'_(X|Y|Z)\.png$', '', regex=True)

        # Map labels based on Unique_Code using a safe mapping approach
        label_mapping = Label.set_index('Unique_Code')['Label'].to_dict()
        img_info['Label'] = img_info['Unique_Code'].map(label_mapping).fillna('Unknown').astype(str)  # Default to 'Unknown' if no match

        # Extract the period using regex
        img_info['period'] = img_info['Unique_Code'].str.extract(r'_(\w{3}_\d{4})_')[0]
        
        # Extract year using regex
        img_info['year'] = img_info['Unique_Code'].str.extract(r'_(\d{4})_')[0]
        
        # Extract the machine using regex
        img_info['machine'] = img_info['Unique_Code'].str.extract(r'^(M\d+)_')[0]

        # Extract axis (X, Y, Z) from the image name
        img_info['axis'] = img_info['image_name'].str.extract(r'([X-Z])')[0]

        # Ensure all columns are correctly assigned (no NaN values)
        img_info = img_info.dropna(subset=['Label', 'period', 'machine', 'axis'])

        return img_info

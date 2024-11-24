import numpy as np
import matplotlib.pyplot as plt

def sliding_window_anomalies(anomaly_labels, window_size=2000):
    """
    Calculates the number of anomalies per non-overlapping sliding window.

    Parameters:
    anomaly_labels (np.array): Array of anomaly labels (1 for anomaly, 0 for normal).
    window_size (int): The size of the sliding window (equal to the sample rate, for example, 2000).

    Returns:
    list: A list with the proportion of anomalies detected per window.
    list: A list of starting indices for each window.
    """
    num_samples = len(anomaly_labels)
    anomaly_index = []
    windows = []

    # Loop through the data in steps of window_size
    for i in range(0, num_samples, window_size):
        window_data = anomaly_labels[i:i+window_size]
        # Calculate the proportion of anomalies in the window
        anomaly_proportion = np.sum(window_data) / window_size
        anomaly_index.append(anomaly_proportion)
        windows.append(i)  # Track the starting index of the window
    
    return anomaly_index, windows

def expand_anomaly_index_with_threshold(anomaly_index, windows, num_samples, window_size, threshold):
    """
    Expands the anomaly_index to match the size of the original sample data and applies a threshold.

    Parameters:
    - anomaly_index (list): List of anomaly index values for each window.
    - windows (list): List of starting indices for each window.
    - num_samples (int): Total number of samples in the original data.
    - window_size (int): The size of each window.
    - threshold (float): Threshold below which the anomaly index is replaced by 0.

    Returns:
    np.array: An expanded array of anomaly index values matching the size of the original data,
              with values below the threshold replaced by 0.
    """
    expanded_anomaly_index = np.zeros(num_samples)  # Initialize an array for the expanded anomaly index

    for i, start_idx in enumerate(windows):
        end_idx = min(start_idx + window_size, num_samples)  # Handle the last window if it's smaller
        # If the anomaly index is below the threshold, replace it with 0
        anomaly_value = anomaly_index[i] if anomaly_index[i] >= threshold else 0
        # Assign the same anomaly index for the entire window
        expanded_anomaly_index[start_idx:end_idx] = anomaly_value

    return expanded_anomaly_index



import pandas as pd
import numpy as np
import os

def load_pcg_data(file_path):
    """Loads PCG signal data from a CSV file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Based on the notebooks, signals are often transposed or need specific handling
    # hi.csv: (10000, 3829) -> (3829, 10000)
    if 'hi.csv' in file_path:
        return df.values.T
    return df.values

def load_labels(file_path):
    """Loads diagnostic labels from a CSV file."""
    if not os.path.exists(file_path):
        # Try to find common label file names in the same directory
        dir_name = os.path.dirname(file_path)
        common_names = ['labels.csv', 'diagnosis_labels.csv', 'targets_cleaned.csv']
        for name in common_names:
            alt_path = os.path.join(dir_name, name)
            if os.path.exists(alt_path):
                print(f"--- Info: Labels file not found at {file_path}. Using {alt_path} instead. ---")
                file_path = alt_path
                break
        else:
            raise FileNotFoundError(f"Labels file not found: {file_path}")
    
    df = pd.read_csv(file_path)
    # Return numeric values only
    labels = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    # Remove any rows with NaN caused by headers or string errors
    labels = labels.dropna()
    return labels.values

def ensure_dirs(dirs):
    """Ensures that the specified directories exist."""
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

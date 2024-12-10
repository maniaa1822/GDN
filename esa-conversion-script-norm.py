import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_to_range(data, feature_range=(-1, 1)):
    """
    Normalize data to specified range using MinMaxScaler.
    Handles division by zero cases.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    
    # Reshape for single column case
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        
    # Check if the column has any variation
    if np.all(data == data[0]):
        # If all values are the same, return the first range value
        return np.full_like(data, feature_range[0])
    
    normalized = scaler.fit_transform(data)
    return normalized

def convert_esa_format(input_file, output_directory, max_channels=None):
    """
    Convert ESA dataset format to create train.csv, test.csv and list.txt
    with normalized values between -1 and 1.
    
    Args:
        input_file (str): Path to input CSV with raw ESA format
        output_directory (str): Directory to save output files
        max_channels (int, optional): Maximum number of channels to include
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Read the input CSV file
    df = pd.read_csv(input_file)
    
    # Get only channel columns (excluding telecommand)
    channel_cols = [col for col in df.columns if col.startswith('channel_')]
    channel_names = [col.replace('channel_', '') for col in channel_cols]
    
    # Sort channels numerically
    channel_pairs = [(name, col) for name, col in zip(channel_names, channel_cols)]
    channel_pairs.sort(key=lambda x: int(x[0]))
    
    # Apply channel limit if specified
    if max_channels is not None:
        channel_pairs = channel_pairs[:max_channels]
        print(f"Limiting to first {max_channels} channels")
    
    # Unzip the sorted and potentially limited pairs
    channel_names, channel_cols = zip(*channel_pairs) if channel_pairs else ([], [])
    
    # Create new dataframe with normalized channels
    new_df = pd.DataFrame()
    new_df['timestamp'] = df['timestamp']
    
    # Add normalized channel columns
    for col, name in zip(channel_cols, channel_names):
        channel_data = df[col].values
        normalized_data = normalize_to_range(channel_data)
        new_df[name] = normalized_data
    
    # Create attack column from channel anomalies only
    anomaly_cols = [f'is_anomaly_{col}' for col in channel_cols]
    anomaly_cols = [col for col in anomaly_cols if col in df.columns]
    attack_series = df[anomaly_cols].any(axis=1).astype(int)
    
    # Create test.csv (with attack column)
    test_df = new_df.copy()
    test_df['attack'] = attack_series
    test_df.set_index('timestamp', inplace=True)
    test_output = os.path.join(output_directory, 'test.csv')
    test_df.to_csv(test_output)
    
    # Create train.csv (without attack column)
    train_df = new_df.copy()
    train_df.set_index('timestamp', inplace=True)
    train_output = os.path.join(output_directory, 'train.csv')
    train_df.to_csv(train_output)
    
    # Create list.txt with channel names
    list_output = os.path.join(output_directory, 'list.txt')
    with open(list_output, 'w') as f:
        for channel in channel_names:
            f.write(f"{channel}\n")
    
    print(f"\nCreated files in directory: {output_directory}")
    print(f"Number of channels: {len(channel_names)}")
    print(f"Number of timestamps: {len(df)}")
    print(f"Number of anomalies: {attack_series.sum()}")
    
    # Print some statistics about the normalized data
    print("\nValue ranges after normalization:")
    for name in list(channel_names)[:5]:  # First 5 channels
        values = new_df[name]
        print(f"Channel {name}: min={values.min():.3f}, max={values.max():.3f}")
    
    print("\nFiles created:")
    print(f"- {test_output}")
    print(f"- {train_output}")
    print(f"- {list_output}")

# Example usage
if __name__ == "__main__":
    input_file = "data/orginal ESA preprocessd/3_months.train.csv"
    output_directory = "data/esa_converted_norm"
    max_channels = 20  # Example: limit to first 50 channels
    convert_esa_format(input_file, output_directory, max_channels)
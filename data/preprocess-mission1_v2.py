import pandas as pd
import numpy as np
from scipy import stats
import os

def tanh_normalize(values):
    """
    Normalize using tanh to get smooth [-1,1] distribution
    Preserves zero-points and handles outliers naturally
    """
    if len(np.unique(values)) <= 1:
        return values
        
    # Scale data to zero mean and unit variance
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return values
    z_scores = (values - mean) / std
    
    # Apply tanh scaling
    scaled = np.tanh(z_scores)
    
    return scaled

def improve_channel(values):
    """
    Enhanced channel processing:
    1. Detect constant/near-constant features
    2. Handle outliers via robust scaling
    3. Apply tanh normalization
    """
    # Handle constant/near-constant
    unique_ratio = len(np.unique(values)) / len(values)
    if unique_ratio < 0.01:  # Near constant
        most_common = stats.mode(values, keepdims=True)[0][0]
        return most_common * np.ones_like(values)
    
    # Winsorize extreme outliers
    percentiles = np.percentile(values, [1, 99])
    values = np.clip(values, percentiles[0], percentiles[1])
    
    # Apply tanh normalization
    return tanh_normalize(values)

def filter_channels(input_file, output_directory):
    """Enhanced channel filtering with improved normalization"""
    os.makedirs(output_directory, exist_ok=True)
    
    # Read input
    df = pd.read_csv(input_file)
    
    # List of channels to keep (unchanged)
    channels = [
        '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', 
        '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', 
        '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', 
        '50', '51', '52', '53', '54'
    ]
    
    # Process channels
    all_columns = ['timestamp'] + channels + ['attack']
    filtered_df = df[all_columns].copy()
    
    # Enhanced processing
    feature_stats = {}
    for col in channels:
        values = filtered_df[col].values
        filtered_df[col] = improve_channel(values)
        
        # Collect statistics
        feature_stats[col] = {
            'mean': np.mean(filtered_df[col]),
            'std': np.std(filtered_df[col]),
            'zeros': np.mean(filtered_df[col] == 0),
            'unique_ratio': len(np.unique(filtered_df[col])) / len(filtered_df[col])
        }
    
    # Print analysis
    print("\nFeature Analysis:")
    for col, stats in feature_stats.items():
        print(f"Channel {col}:")
        print(f"  Range: [{filtered_df[col].min():.3f}, {filtered_df[col].max():.3f}]")
        print(f"  Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}")
        print(f"  Zero ratio: {stats['zeros']:.1%}")
        print(f"  Unique ratio: {stats['unique_ratio']:.1%}")
    
    # Save outputs
    filtered_df.to_csv(f'{output_directory}/test.csv', index=False)
    train_df = filtered_df.drop(columns=['attack'])
    train_df.to_csv(f'{output_directory}/train.csv', index=False)
    
    # Save feature list
    with open(f'{output_directory}/list.txt', 'w') as f:
        for channel in channels:
            f.write(f"{channel}\n")

if __name__ == '__main__':
    filter_channels('data/3_months.train_with_attack.csv', 'esa-sub-1')
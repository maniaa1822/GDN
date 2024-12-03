#%%
import pandas as pd
import os
#%%
def convert_data_format(input_file, train_output, test_output):
    """
    Convert 3_months.train.csv data format to two files:
    - Training file without attack column
    - Test file with attack column
    
    Args:t
        input_file (str): Pathh to input CSV file
        train_output (str):e Path to training CSV file (no attack column)
        test_output (str): Path to test CSV file (with attack column)
    """
    # Read the input CSV
    df = pd.read_csv(input_file)
    
    # Get only channel_ columns 
    channel_cols = [col for col in df.columns if col.startswith('channel_')]
    is_anomaly_channel_cols = [col for col in df.columns if col.startswith('is_anomaly_channel_')]
    
    # Create base dataframe with timestamp and channel columns
    new_df = pd.DataFrame()
    new_df['timestamp'] = range(len(df))
    
    # Add channel columns with their values
    for col in channel_cols:
        new_df[col.replace('channel_', '')] = df[col]
    
    # Create attack column
    attack = df[is_anomaly_channel_cols].apply(lambda x: x.any(), axis=1).astype(int)
    
    # Create training df (without attack)
    train_df = new_df.copy()
    
    # Create test df (with attack)
    test_df = new_df.copy()
    test_df['attack'] = attack
    
    # Save both files
    train_df.to_csv(train_output, index=False)
    test_df.to_csv(test_output, index=False)
    
    print(f"Converted {len(df)} rows from {len(channel_cols)} channels")
    print(f"Training data saved to {train_output}")
    print(f"Test data with attack labels saved to {test_output}")

def create_channel_list_from_df(input_file, output_file):
    """
    Create list.txt with channel names from dataframe, excluding timestamp
    
    Args:
        input_file (str): Path to input CSV file
        output_file (str): Path to output list file
    """
    # Read the CSV
    df = pd.read_csv(input_file)
    
    # Get channel columns (excluding timestamp)
    channel_cols = [col for col in df.columns if col.startswith('channel_')]
    channel_names = [col.replace('channel_', '') for col in channel_cols]
    
    # Write to file
    with open(output_file, 'w') as f:
        for name in channel_names:
            f.write(f"{name}\n")
            
    print(f"Created channel list with {len(channel_names)} channels in {output_file}")

#%% Example usage:
convert_data_format(
    'preprocessed/multivariate/ESA-Mission1-semi-supervised/3_months.train.csv',
    '3_months.train_no_attack.csv',
    '3_months.train_with_attack.csv'
)
#%%
def filter_channels(input_file,output_directory):
    #create the new folder named output_directory in the 
    #/home/matteo/AI_and_Robotics/EAI/GDN/data directory
    os.makedirs(output_directory, exist_ok=True)
    # Read the CSV input
    df = pd.read_csv(input_file)
    #list of channels
    channels = [
        '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', 
        '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', 
        '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '5', 
        '50', '51', '52', '53', '54'
    ]
    #convert the channels to string
    channels = [f'{ch}' for ch in channels]
    # Get channel columns
    # Filter the channels keep the first adn last columns
    channels = ['timestamp'] + channels + ['attack']
    df = df[channels]
    # Save the file as test.csv
    df.to_csv(f'{output_directory}/test.csv', index=False)
    print(f"Filtered channels and saved to {output_directory}/test.csv")
    #drop the attack column and save the file as train.csv
    df = df.drop(columns=['attack'])
    df.to_csv(f'{output_directory}/train.csv', index=False)
    print(f"Filtered channels and saved to {output_directory}/train.csv")
    
filter_channels('3_months.train_with_attack.csv',

                'esa-sub-1')
# %%

import pandas as pd
import os

def load_dataset(file_path):
    """
    Load the dataset from the specified file path.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    return pd.read_csv(file_path)

def preprocess_data(df):
    """
    Preprocess the dataset by selecting relevant features and handling missing values.

    Args:
        df (pd.DataFrame): Raw dataset as a pandas DataFrame.

    Returns:
        pd.DataFrame: Preprocessed dataset.
    """
    # Select relevant features for training
    relevant_features = [
        'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
        'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
        'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
        'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
        'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
        'Flow Pkts/s', 'Fwd IAT Tot', 'Bwd IAT Tot',
        'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
        'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
        'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt',
        'RST Flag Cnt', 'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt',
        'CWE Flag Count', 'ECE Flag Cnt', 'Pkt Size Avg',
        'Fwd Seg Size Avg', 'Bwd Seg Size Avg', 'Fwd Byts/b Avg',
        'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
        'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts',
        'Subflow Fwd Byts', 'Subflow Bwd Pkts', 'Subflow Bwd Byts',
        'Init Fwd Win Byts', 'Init Bwd Win Byts', 'Fwd Act Data Pkts',
        'Fwd Seg Size Min', 'Active Mean', 'Active Std', 'Active Max',
        'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
    ]

    # Select the relevant features from the dataset
    df = df[relevant_features]

    # Handle missing values by filling them with the mean of the column
    df.fillna(df.mean(), inplace=True)

    return df

def save_preprocessed_data(df, output_path):
    """
    Save the preprocessed dataset to the specified output path.

    Args:
        df (pd.DataFrame): Preprocessed dataset.
        output_path (str): Path to save the preprocessed dataset.
    """
    df.to_csv(output_path, index=False)

def main():
    # Define the directory containing the processed traffic data
    data_dir = '/home/ubuntu/cse-cic-ids2018/Processed Traffic Data for ML Algorithms'

    # Iterate over each CSV file in the directory
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.csv'):
            input_file_path = os.path.join(data_dir, file_name)
            output_file_path = os.path.join(data_dir, f'preprocessed_{file_name}')

            # Load the dataset
            df = load_dataset(input_file_path)

            # Preprocess the dataset
            preprocessed_df = preprocess_data(df)

            # Save the preprocessed dataset
            save_preprocessed_data(preprocessed_df, output_file_path)

            print(f"Preprocessed data saved to {output_file_path}")

if __name__ == "__main__":
    main()

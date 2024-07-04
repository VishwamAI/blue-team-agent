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
        'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
        'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
        'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
        'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
        'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
        'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
        'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
        'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
        'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length',
        'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length',
        'Max Packet Length', 'Packet Length Mean', 'Packet Length Std',
        'Packet Length Variance', 'FIN Flag Count', 'SYN Flag Count',
        'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count', 'URG Flag Count',
        'CWE Flag Count', 'ECE Flag Count', 'Down/Up Ratio', 'Average Packet Size',
        'Avg Fwd Segment Size', 'Avg Bwd Segment Size', 'Fwd Avg Bytes/Bulk',
        'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
        'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
        'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
        'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
        'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max',
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
    # Define the file paths
    input_file_path = 'cse-cic-ids2018/Original Network Traffic and Log data/Friday-16-02-2018/logs.csv'
    output_file_path = 'cse-cic-ids2018/preprocessed_data.csv'

    # Load the dataset
    df = load_dataset(input_file_path)

    # Preprocess the dataset
    preprocessed_df = preprocess_data(df)

    # Save the preprocessed dataset
    save_preprocessed_data(preprocessed_df, output_file_path)

    print(f"Preprocessed data saved to {output_file_path}")

if __name__ == "__main__":
    main()

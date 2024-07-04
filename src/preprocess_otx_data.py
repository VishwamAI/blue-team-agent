import json
import numpy as np

# Define the list of relevant features for state representation
relevant_features = [
    'Flow Duration', 'Tot Fwd Pkts', 'Tot Bwd Pkts',
    'TotLen Fwd Pkts', 'TotLen Bwd Pkts',
    'Fwd Pkt Len Max', 'Fwd Pkt Len Min', 'Fwd Pkt Len Mean',
    'Fwd Pkt Len Std', 'Bwd Pkt Len Max', 'Bwd Pkt Len Min',
    'Bwd Pkt Len Mean', 'Bwd Pkt Len Std', 'Flow Byts/s',
    'Flow Pkts/s', 'Fwd IAT Tot', 'Bwd IAT Tot',
    'Fwd Header Len', 'Bwd Header Len', 'Fwd Pkts/s', 'Bwd Pkts/s',
    'Pkt Len Min', 'Pkt Len Max', 'Pkt Len Mean', 'Pkt Len Std',
    'Pkt Len Var', 'FIN Flag Cnt', 'SYN Flag Cnt', 'RST Flag Cnt',
    'PSH Flag Cnt', 'ACK Flag Cnt', 'URG Flag Cnt', 'CWE Flag Count',
    'ECE Flag Cnt', 'Pkt Size Avg', 'Fwd Seg Size Avg', 'Bwd Seg Size Avg',
    'Fwd Byts/b Avg', 'Fwd Pkts/b Avg', 'Fwd Blk Rate Avg', 'Bwd Byts/b Avg',
    'Bwd Pkts/b Avg', 'Bwd Blk Rate Avg', 'Subflow Fwd Pkts', 'Subflow Fwd Byts',
    'Subflow Bwd Pkts', 'Subflow Bwd Byts', 'Init Fwd Win Byts', 'Init Bwd Win Byts',
    'Fwd Act Data Pkts', 'Fwd Seg Size Min', 'Active Mean', 'Active Std',
    'Active Max', 'Active Min', 'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min',
    'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min',
    'Fwd IAT Mean', 'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min',
    'Bwd IAT Mean', 'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min',
    'Fwd PSH Flags', 'Bwd PSH Flags', 'Fwd URG Flags', 'Bwd URG Flags',
    'Fwd Header Length', 'Bwd Header Length'
]

def preprocess_otx_data(input_file, output_file, num_samples=1000, max_indicators=10):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        preprocessed_data = []
        labels = []
        for i, pulse in enumerate(data):
            if i >= num_samples:
                break
            # Quantify indicators based on their type and presence
            feature_vector = []
            for j, indicator in enumerate(pulse.get('indicators', [])):
                if j >= max_indicators:
                    break
                feature_vector.append(len(indicator.get('indicator', '')))
                feature_vector.append(1 if indicator.get('type') == 'FileHash-SHA256' else 0)
                feature_vector.append(1 if indicator.get('type') == 'domain' else 0)
                feature_vector.append(1 if indicator.get('type') == 'IP' else 0)
                feature_vector.append(1 if indicator.get('type') == 'URL' else 0)
                feature_vector.append(1 if indicator.get('type') == 'CVE' else 0)
            # Pad the feature vector with zeros if there are fewer indicators than max_indicators
            while len(feature_vector) < max_indicators * 6:
                feature_vector.append(0)
            preprocessed_data.append(feature_vector)
            # Generate synthetic labels (1 if at least one active indicator, else 0)
            labels.append(1 if any(indicator.get('is_active', 0) == 1 for indicator in pulse.get('indicators', [])) else 0)

        preprocessed_data = np.array(preprocessed_data)
        labels = np.array(labels)
        np.save(output_file, preprocessed_data)
        np.save(output_file.replace('.npy', '_labels.npy'), labels)
        print(f"Preprocessed data and labels saved to {output_file} and {output_file.replace('.npy', '_labels.npy')}")
    except Exception as e:
        print(f"Error preprocessing data: {e}")

if __name__ == "__main__":
    input_file = "otx_data.json"
    output_file = "preprocessed_otx_data.npy"
    preprocess_otx_data(input_file, output_file)

import numpy as np
from rl_agent_model import convert_log_to_state, choose_action, execute_action

# Simulated cybersecurity log data for various test cases
test_cases = [
    {
        'Flow Duration': 1000, 'Total Fwd Packets': 10, 'Total Backward Packets': 5,
        'Total Length of Fwd Packets': 500, 'Total Length of Bwd Packets': 250,
        'Fwd Packet Length Max': 100, 'Fwd Packet Length Min': 50, 'Fwd Packet Length Mean': 75,
        'Fwd Packet Length Std': 10, 'Bwd Packet Length Max': 80, 'Bwd Packet Length Min': 40,
        'Bwd Packet Length Mean': 60, 'Bwd Packet Length Std': 8, 'Flow Bytes/s': 1000,
        'Flow Packets/s': 10, 'Flow IAT Mean': 100, 'Flow IAT Std': 10, 'Flow IAT Max': 200,
        'Flow IAT Min': 50, 'Fwd IAT Total': 500, 'Fwd IAT Mean': 100, 'Fwd IAT Std': 10,
        'Fwd IAT Max': 200, 'Fwd IAT Min': 50, 'Bwd IAT Total': 250, 'Bwd IAT Mean': 50,
        'Bwd IAT Std': 5, 'Bwd IAT Max': 100, 'Bwd IAT Min': 25, 'Fwd PSH Flags': 1,
        'Bwd PSH Flags': 0, 'Fwd URG Flags': 0, 'Bwd URG Flags': 0, 'Fwd Header Length': 20,
        'Bwd Header Length': 10, 'Fwd Packets/s': 5, 'Bwd Packets/s': 2.5, 'Min Packet Length': 40,
        'Max Packet Length': 100, 'Packet Length Mean': 70, 'Packet Length Std': 10,
        'Packet Length Variance': 100, 'FIN Flag Count': 0, 'SYN Flag Count': 1,
        'RST Flag Count': 0, 'PSH Flag Count': 1, 'ACK Flag Count': 1, 'URG Flag Count': 0,
        'CWE Flag Count': 0, 'ECE Flag Count': 0, 'Down/Up Ratio': 1.0, 'Average Packet Size': 70,
        'Avg Fwd Segment Size': 75, 'Avg Bwd Segment Size': 60, 'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0, 'Fwd Avg Bulk Rate': 0, 'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0, 'Bwd Avg Bulk Rate': 0, 'Subflow Fwd Packets': 10,
        'Subflow Fwd Bytes': 500, 'Subflow Bwd Packets': 5, 'Subflow Bwd Bytes': 250,
        'Init_Win_bytes_forward': 8192, 'Init_Win_bytes_backward': 4096, 'act_data_pkt_fwd': 10,
        'min_seg_size_forward': 20, 'Active Mean': 100, 'Active Std': 10, 'Active Max': 200,
        'Active Min': 50, 'Idle Mean': 1000, 'Idle Std': 100, 'Idle Max': 2000, 'Idle Min': 500
    },
    {
        'Flow Duration': 2000, 'Total Fwd Packets': 20, 'Total Backward Packets': 10,
        'Total Length of Fwd Packets': 1000, 'Total Length of Bwd Packets': 500,
        'Fwd Packet Length Max': 200, 'Fwd Packet Length Min': 100, 'Fwd Packet Length Mean': 150,
        'Fwd Packet Length Std': 20, 'Bwd Packet Length Max': 160, 'Bwd Packet Length Min': 80,
        'Bwd Packet Length Mean': 120, 'Bwd Packet Length Std': 16, 'Flow Bytes/s': 2000,
        'Flow Packets/s': 20, 'Flow IAT Mean': 200, 'Flow IAT Std': 20, 'Flow IAT Max': 400,
        'Flow IAT Min': 100, 'Fwd IAT Total': 1000, 'Fwd IAT Mean': 200, 'Fwd IAT Std': 20,
        'Fwd IAT Max': 400, 'Fwd IAT Min': 100, 'Bwd IAT Total': 500, 'Bwd IAT Mean': 100,
        'Bwd IAT Std': 10, 'Bwd IAT Max': 200, 'Bwd IAT Min': 50, 'Fwd PSH Flags': 1,
        'Bwd PSH Flags': 0, 'Fwd URG Flags': 0, 'Bwd URG Flags': 0, 'Fwd Header Length': 40,
        'Bwd Header Length': 20, 'Fwd Packets/s': 10, 'Bwd Packets/s': 5, 'Min Packet Length': 80,
        'Max Packet Length': 200, 'Packet Length Mean': 140, 'Packet Length Std': 20,
        'Packet Length Variance': 400, 'FIN Flag Count': 0, 'SYN Flag Count': 1,
        'RST Flag Count': 0, 'PSH Flag Count': 1, 'ACK Flag Count': 1, 'URG Flag Count': 0,
        'CWE Flag Count': 0, 'ECE Flag Count': 0, 'Down/Up Ratio': 1.0, 'Average Packet Size': 140,
        'Avg Fwd Segment Size': 150, 'Avg Bwd Segment Size': 120, 'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0, 'Fwd Avg Bulk Rate': 0, 'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0, 'Bwd Avg Bulk Rate': 0, 'Subflow Fwd Packets': 20,
        'Subflow Fwd Bytes': 1000, 'Subflow Bwd Packets': 10, 'Subflow Bwd Bytes': 500,
        'Init_Win_bytes_forward': 16384, 'Init_Win_bytes_backward': 8192, 'act_data_pkt_fwd': 20,
        'min_seg_size_forward': 40, 'Active Mean': 200, 'Active Std': 20, 'Active Max': 400,
        'Active Min': 100, 'Idle Mean': 2000, 'Idle Std': 200, 'Idle Max': 4000, 'Idle Min': 1000
    },
    {
        'Flow Duration': 500, 'Total Fwd Packets': 5, 'Total Backward Packets': 2,
        'Total Length of Fwd Packets': 250, 'Total Length of Bwd Packets': 100,
        'Fwd Packet Length Max': 50, 'Fwd Packet Length Min': 25, 'Fwd Packet Length Mean': 37.5,
        'Fwd Packet Length Std': 5, 'Bwd Packet Length Max': 40, 'Bwd Packet Length Min': 20,
        'Bwd Packet Length Mean': 30, 'Bwd Packet Length Std': 4, 'Flow Bytes/s': 500,
        'Flow Packets/s': 5, 'Flow IAT Mean': 50, 'Flow IAT Std': 5, 'Flow IAT Max': 100,
        'Flow IAT Min': 25, 'Fwd IAT Total': 250, 'Fwd IAT Mean': 50, 'Fwd IAT Std': 5,
        'Fwd IAT Max': 100, 'Fwd IAT Min': 25, 'Bwd IAT Total': 100, 'Bwd IAT Mean': 20,
        'Bwd IAT Std': 2, 'Bwd IAT Max': 40, 'Bwd IAT Min': 10, 'Fwd PSH Flags': 1,
        'Bwd PSH Flags': 0, 'Fwd URG Flags': 0, 'Bwd URG Flags': 0, 'Fwd Header Length': 10,
        'Bwd Header Length': 5, 'Fwd Packets/s': 2.5, 'Bwd Packets/s': 1.25, 'Min Packet Length': 20,
        'Max Packet Length': 50, 'Packet Length Mean': 35, 'Packet Length Std': 5,
        'Packet Length Variance': 25, 'FIN Flag Count': 0, 'SYN Flag Count': 1,
        'RST Flag Count': 0, 'PSH Flag Count': 1, 'ACK Flag Count': 1, 'URG Flag Count': 0,
        'CWE Flag Count': 0, 'ECE Flag Count': 0, 'Down/Up Ratio': 1.0, 'Average Packet Size': 35,
        'Avg Fwd Segment Size': 37.5, 'Avg Bwd Segment Size': 30, 'Fwd Avg Bytes/Bulk': 0,
        'Fwd Avg Packets/Bulk': 0, 'Fwd Avg Bulk Rate': 0, 'Bwd Avg Bytes/Bulk': 0,
        'Bwd Avg Packets/Bulk': 0, 'Bwd Avg Bulk Rate': 0, 'Subflow Fwd Packets': 5,
        'Subflow Fwd Bytes': 250, 'Subflow Bwd Packets': 2, 'Subflow Bwd Bytes': 100,
        'Init_Win_bytes_forward': 4096, 'Init_Win_bytes_backward': 2048, 'act_data_pkt_fwd': 5,
        'min_seg_size_forward': 10, 'Active Mean': 50, 'Active Std': 5, 'Active Max': 100,
        'Active Min': 25, 'Idle Mean': 500, 'Idle Std': 50, 'Idle Max': 1000, 'Idle Min': 250
    }
]

# Run tests for each test case
for i, log_data in enumerate(test_cases):
    print(f"Running test case {i+1}")

    # Convert log data to state representation
    state = convert_log_to_state(log_data)
    print(f"Converted state: {state}")

    # Choose an action based on the current state
    action = choose_action(state)
    print(f"Chosen action: {action}")

    # Execute the chosen action with dynamic parameters
    execute_action(
        action,
        ip_address=log_data.get('ip_address', '0.0.0.0'),
        rate_limit=log_data.get('rate_limit', 0),
        system_id=log_data.get('system_id', 'unknown'),
        message=log_data.get('message', ''),
        settings=log_data.get('settings', {}),
        query=log_data.get('query', '')
    )

print("All tests completed successfully.")

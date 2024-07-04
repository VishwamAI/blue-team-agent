import numpy as np
from rl_agent_model import convert_log_to_state, choose_action, execute_action

# Simulated cybersecurity log data for various test cases
test_cases = [
    {
        'Flow Duration': 1000, 'Tot Fwd Pkts': 10, 'Tot Bwd Pkts': 5,
        'TotLen Fwd Pkts': 500, 'TotLen Bwd Pkts': 250,
        'Fwd Pkt Len Max': 100, 'Fwd Pkt Len Min': 50, 'Fwd Pkt Len Mean': 75,
        'Fwd Pkt Len Std': 10, 'Bwd Pkt Len Max': 80, 'Bwd Pkt Len Min': 40,
        'Bwd Pkt Len Mean': 60, 'Bwd Pkt Len Std': 8, 'Flow Byts/s': 1000,
        'Flow Pkts/s': 10, 'Fwd IAT Tot': 500, 'Bwd IAT Tot': 250,
        'Fwd Header Len': 20, 'Bwd Header Len': 10, 'Fwd Pkts/s': 5, 'Bwd Pkts/s': 2.5,
        'Pkt Len Min': 40, 'Pkt Len Max': 100, 'Pkt Len Mean': 70, 'Pkt Len Std': 10,
        'Pkt Len Var': 100, 'FIN Flag Cnt': 0, 'SYN Flag Cnt': 1, 'RST Flag Cnt': 0,
        'PSH Flag Cnt': 1, 'ACK Flag Cnt': 1, 'URG Flag Cnt': 0, 'CWE Flag Count': 0,
        'ECE Flag Cnt': 0, 'Pkt Size Avg': 70, 'Fwd Seg Size Avg': 75, 'Bwd Seg Size Avg': 60,
        'Fwd Byts/b Avg': 0, 'Fwd Pkts/b Avg': 0, 'Fwd Blk Rate Avg': 0, 'Bwd Byts/b Avg': 0,
        'Bwd Pkts/b Avg': 0, 'Bwd Blk Rate Avg': 0, 'Subflow Fwd Pkts': 10, 'Subflow Fwd Byts': 500,
        'Subflow Bwd Pkts': 5, 'Subflow Bwd Byts': 250, 'Init Fwd Win Byts': 8192, 'Init Bwd Win Byts': 4096,
        'Fwd Act Data Pkts': 10, 'Fwd Seg Size Min': 20, 'Active Mean': 100, 'Active Std': 10,
        'Active Max': 200, 'Active Min': 50, 'Idle Mean': 1000, 'Idle Std': 100, 'Idle Max': 2000, 'Idle Min': 500,
        'Flow IAT Mean': 100, 'Flow IAT Std': 10, 'Flow IAT Max': 200, 'Flow IAT Min': 50,
        'Fwd IAT Mean': 100, 'Fwd IAT Std': 10, 'Fwd IAT Max': 200, 'Fwd IAT Min': 50
    },
    {
        'Flow Duration': 2000, 'Tot Fwd Pkts': 20, 'Tot Bwd Pkts': 10,
        'TotLen Fwd Pkts': 1000, 'TotLen Bwd Pkts': 500,
        'Fwd Pkt Len Max': 200, 'Fwd Pkt Len Min': 100, 'Fwd Pkt Len Mean': 150,
        'Fwd Pkt Len Std': 20, 'Bwd Pkt Len Max': 160, 'Bwd Pkt Len Min': 80,
        'Bwd Pkt Len Mean': 120, 'Bwd Pkt Len Std': 16, 'Flow Byts/s': 2000,
        'Flow Pkts/s': 20, 'Fwd IAT Tot': 1000, 'Bwd IAT Tot': 500,
        'Fwd Header Len': 40, 'Bwd Header Len': 20, 'Fwd Pkts/s': 10, 'Bwd Pkts/s': 5,
        'Pkt Len Min': 80, 'Pkt Len Max': 200, 'Pkt Len Mean': 140, 'Pkt Len Std': 20,
        'Pkt Len Var': 400, 'FIN Flag Cnt': 0, 'SYN Flag Cnt': 1, 'RST Flag Cnt': 0,
        'PSH Flag Cnt': 1, 'ACK Flag Cnt': 1, 'URG Flag Cnt': 0, 'CWE Flag Count': 0,
        'ECE Flag Cnt': 0, 'Pkt Size Avg': 140, 'Fwd Seg Size Avg': 150, 'Bwd Seg Size Avg': 120,
        'Fwd Byts/b Avg': 0, 'Fwd Pkts/b Avg': 0, 'Fwd Blk Rate Avg': 0, 'Bwd Byts/b Avg': 0,
        'Bwd Pkts/b Avg': 0, 'Bwd Blk Rate Avg': 0, 'Subflow Fwd Pkts': 20, 'Subflow Fwd Byts': 1000,
        'Subflow Bwd Pkts': 10, 'Subflow Bwd Byts': 500, 'Init Fwd Win Byts': 16384, 'Init Bwd Win Byts': 8192,
        'Fwd Act Data Pkts': 20, 'Fwd Seg Size Min': 40, 'Active Mean': 200, 'Active Std': 20,
        'Active Max': 400, 'Active Min': 100, 'Idle Mean': 2000, 'Idle Std': 200, 'Idle Max': 4000, 'Idle Min': 1000,
        'Flow IAT Mean': 200, 'Flow IAT Std': 20, 'Flow IAT Max': 400, 'Flow IAT Min': 100,
        'Fwd IAT Mean': 200, 'Fwd IAT Std': 20, 'Fwd IAT Max': 400, 'Fwd IAT Min': 100
    },
    {
        'Flow Duration': 500, 'Tot Fwd Pkts': 5, 'Tot Bwd Pkts': 2,
        'TotLen Fwd Pkts': 250, 'TotLen Bwd Pkts': 100,
        'Fwd Pkt Len Max': 50, 'Fwd Pkt Len Min': 25, 'Fwd Pkt Len Mean': 37.5,
        'Fwd Pkt Len Std': 5, 'Bwd Pkt Len Max': 40, 'Bwd Pkt Len Min': 20,
        'Bwd Pkt Len Mean': 30, 'Bwd Pkt Len Std': 4, 'Flow Byts/s': 500,
        'Flow Pkts/s': 5, 'Fwd IAT Tot': 250, 'Bwd IAT Tot': 100,
        'Fwd Header Len': 10, 'Bwd Header Len': 5, 'Fwd Pkts/s': 2.5, 'Bwd Pkts/s': 1.25,
        'Pkt Len Min': 20, 'Pkt Len Max': 50, 'Pkt Len Mean': 35, 'Pkt Len Std': 5,
        'Pkt Len Var': 25, 'FIN Flag Cnt': 0, 'SYN Flag Cnt': 1, 'RST Flag Cnt': 0,
        'PSH Flag Cnt': 1, 'ACK Flag Cnt': 1, 'URG Flag Cnt': 0, 'CWE Flag Count': 0,
        'ECE Flag Cnt': 0, 'Pkt Size Avg': 35, 'Fwd Seg Size Avg': 37.5, 'Bwd Seg Size Avg': 30,
        'Fwd Byts/b Avg': 0, 'Fwd Pkts/b Avg': 0, 'Fwd Blk Rate Avg': 0, 'Bwd Byts/b Avg': 0,
        'Bwd Pkts/b Avg': 0, 'Bwd Blk Rate Avg': 0, 'Subflow Fwd Pkts': 5, 'Subflow Fwd Byts': 250,
        'Subflow Bwd Pkts': 2, 'Subflow Bwd Byts': 100, 'Init Fwd Win Byts': 4096, 'Init Bwd Win Byts': 2048,
        'Fwd Act Data Pkts': 5, 'Fwd Seg Size Min': 10, 'Active Mean': 50, 'Active Std': 5,
        'Active Max': 100, 'Active Min': 25, 'Idle Mean': 500, 'Idle Std': 50, 'Idle Max': 1000, 'Idle Min': 250,
        'Flow IAT Mean': 50, 'Flow IAT Std': 5, 'Flow IAT Max': 100, 'Flow IAT Min': 25,
        'Fwd IAT Mean': 50, 'Fwd IAT Std': 5, 'Fwd IAT Max': 100, 'Fwd IAT Min': 25
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

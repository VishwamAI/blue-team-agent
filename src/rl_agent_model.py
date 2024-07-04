import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask, request, jsonify
import logging
import threading

# Configure logging
logging.basicConfig(filename='rl_agent_errors.log', level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

# Define the state representation and action space for cybersecurity data
num_inputs = 78  # Number of cybersecurity metrics
num_actions = 10  # Number of possible actions

# List of relevant features for state representation
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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the neural network model
model = tf.keras.Sequential([
    layers.Input(shape=(num_inputs,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(num_actions, activation='linear')
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
loss_function = tf.keras.losses.MeanSquaredError()

# Compile the model
model.compile(optimizer=optimizer, loss=loss_function)

# Define the training parameters
gamma = 0.99  # Discount factor for future rewards
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
batch_size = 32
memory = []

# Function to choose an action based on the current state
def choose_action(state):
    if np.random.rand() <= epsilon:
        return np.random.randint(num_actions)
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Function to train the model
def train_model():
    if len(memory) < batch_size:
        return
    batch = np.random.choice(len(memory), batch_size, replace=False)
    for i in batch:
        state, action, reward, next_state, done = memory[i]
        target = reward
        if not done:
            next_state = np.reshape(next_state, [1, num_inputs])
            target += gamma * np.amax(model.predict(next_state)[0])
        state = np.reshape(state, [1, num_inputs])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)

def run_training_loop():
    global epsilon, memory, model
    num_episodes = 1000
    save_interval = 100  # Save the model every 100 episodes
    for episode in range(num_episodes):
        state = np.random.rand(num_inputs)  # Initialize state with random values
        state = np.reshape(state, [1, num_inputs])
        total_reward = 0
        for time in range(500):
            action = choose_action(state)
            next_state = np.random.rand(num_inputs)  # Generate next state with random values
            reward = np.random.rand()  # Generate random reward
            done = np.random.choice([True, False])  # Randomly choose if the episode is done
            next_state = np.reshape(next_state, [1, num_inputs])
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {episode+1}/{num_episodes}, Score: {total_reward}, Epsilon: {epsilon:.2}")
                break
            train_model()
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if (episode + 1) % save_interval == 0:
            model.save(f'rl_agent_model_{episode + 1}.h5')
    # Save the trained model
    model.save('rl_agent_model.h5')

# Flask web server to receive log data from Logstash
app = Flask(__name__)

@app.route('/logs', methods=['POST'])
def receive_logs():
    try:
        log_data = request.json
        print(f"Received log data: {log_data}")
        logging.info(f"Received log data: {log_data}")

        # Convert log data to state representation for the RL agent
        state = convert_log_to_state(log_data)
        print(f"Converted state: {state}")
        logging.info(f"Converted state: {state}")

        # Choose an action based on the current state
        action = choose_action(state)
        print(f"Chosen action: {action}")
        logging.info(f"Chosen action: {action}")

        # Extract parameters from log data
        ip_address = log_data.get('ip_address')
        rate_limit = log_data.get('rate_limit')
        system_id = log_data.get('system_id')
        message = log_data.get('message')
        settings = log_data.get('settings')
        query = log_data.get('query')

        # Execute the chosen action with dynamic parameters
        execute_action(action, ip_address=ip_address, rate_limit=rate_limit, system_id=system_id, message=message, settings=settings, query=query)
        logging.info(f"Executed action: {action}")

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error processing log data: {e}")
        logging.error("Error processing log data", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

def convert_log_to_state(log_data):
    # Create state representation
    state = []
    for feature in relevant_features:
        value = log_data.get(feature, 0)
        print(f"Feature: {feature}, Value: {value}")
        state.append(value)
        print(f"State length after appending {feature}: {len(state)}")

    state = np.array(state)
    print(f"State before reshaping: {state}, Length: {len(state)}")
    state = np.reshape(state, [1, 76])  # Update to match the number of metrics
    return state

def execute_action(action, ip_address=None, rate_limit=None, system_id=None, message=None, settings=None, query=None):
    # Define actions based on the action space
    actions = [
        "block_ip",
        "allow_ip",
        "rate_limit",
        "isolate_system",
        "notify_admin",
        "run_malware_scan",
        "change_firewall_settings",
        "update_software",
        "search_logs",
        "generate_report"
    ]

    # Execute the chosen action
    chosen_action = actions[action]
    print(f"Executing action: {chosen_action}")

    # Implement the logic to interact with the blue team's infrastructure based on the chosen action
    if chosen_action == "block_ip":
        block_ip_address(ip_address)
    elif chosen_action == "allow_ip":
        allow_ip_address(ip_address)
    elif chosen_action == "rate_limit":
        apply_rate_limiting(ip_address, rate_limit)
    elif chosen_action == "isolate_system":
        isolate_compromised_system(system_id)
    elif chosen_action == "notify_admin":
        send_alert_to_admin(message)
    elif chosen_action == "run_malware_scan":
        trigger_malware_scan(system_id)
    elif chosen_action == "change_firewall_settings":
        update_firewall_settings(settings)
    elif chosen_action == "update_software":
        update_software_packages(system_id)
    elif chosen_action == "search_logs":
        perform_log_search(query)
    elif chosen_action == "generate_report":
        generate_security_report()

def block_ip_address(ip_address):
    import requests
    firewall_api_url = "http://localhost:5001/api/block_ip"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "ip_address": ip_address
    }
    try:
        response = requests.post(firewall_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully blocked IP address: {ip_address}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to block IP address: {ip_address}. Error: {e}")

def allow_ip_address(ip_address):
    import requests
    firewall_api_url = "http://localhost:5001/api/allow_ip"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "ip_address": ip_address
    }
    try:
        response = requests.post(firewall_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully allowed IP address: {ip_address}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to allow IP address: {ip_address}. Error: {e}")

def apply_rate_limiting(ip_address, rate_limit):
    import requests
    rate_limit_api_url = "http://localhost:5001/api/rate_limit"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "ip_address": ip_address,
        "rate_limit": rate_limit
    }
    try:
        response = requests.post(rate_limit_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully applied rate limiting to IP address: {ip_address}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to apply rate limiting to IP address: {ip_address}. Error: {e}")

def isolate_compromised_system(system_id):
    import requests
    isolate_api_url = "http://localhost:5001/api/isolate_system"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "system_id": system_id
    }
    try:
        response = requests.post(isolate_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully isolated system: {system_id}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to isolate system: {system_id}. Error: {e}")

def send_alert_to_admin(message):
    import requests
    alert_api_url = "http://localhost:5001/api/send_alert"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "message": message
    }
    try:
        response = requests.post(alert_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully sent alert: {message}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to send alert: {message}. Error: {e}")

def trigger_malware_scan(system_id):
    import requests
    malware_scan_api_url = "http://localhost:5001/api/trigger_malware_scan"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "system_id": system_id
    }
    try:
        response = requests.post(malware_scan_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully triggered malware scan on system: {system_id}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to trigger malware scan on system: {system_id}. Error: {e}")

def update_firewall_settings(settings):
    import requests
    firewall_settings_api_url = "http://localhost:5001/api/update_settings"
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(firewall_settings_api_url, headers=headers, json=settings)
        response.raise_for_status()
        print("Successfully updated firewall settings")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update firewall settings. Error: {e}")

def update_software_packages(system_id):
    import requests
    update_packages_api_url = "http://localhost:5001/api/update_packages"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "system_id": system_id
    }
    try:
        response = requests.post(update_packages_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully updated software packages on system: {system_id}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to update software packages on system: {system_id}. Error: {e}")

def perform_log_search(query):
    import requests
    log_search_api_url = "http://localhost:5001/api/search_logs"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "query": query
    }
    try:
        response = requests.post(log_search_api_url, headers=headers, json=data)
        response.raise_for_status()
        print(f"Successfully performed log search with query: {query}")
    except requests.exceptions.RequestException as e:
        print(f"Failed to perform log search with query: {query}. Error: {e}")

def generate_security_report():
    import requests
    report_api_url = "http://localhost:5001/api/generate_report"
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(report_api_url, headers=headers)
        response.raise_for_status()
        print("Successfully generated security report")
    except requests.exceptions.RequestException as e:
        print(f"Failed to generate security report. Error: {e}")

if __name__ == '__main__':
    print("Starting Flask server...")
    flask_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000))
    flask_thread.start()
    run_training_loop()

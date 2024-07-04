# blue-team-agent

## Overview
The Blue Team Agent is designed to automate various security tasks, including threat detection, incident response, system health monitoring, and patch management. It leverages reinforcement learning to make intelligent decisions based on real-time data and interactions with the environment.

## Components
### 1. SOC Automation
- **Tools**: XSOAR, Intezer, Torq, Swimlane
- **Capabilities**: Automated alert handling, incident triage, response actions

### 2. Incident Response Playbooks
- **Tools**: Societé Generale incident response playbooks
- **Capabilities**: Automated responses to common security incidents

### 3. Log Management and Analysis
- **Tools**: Wireshark, Splunk, Kibana, Logstash
- **Capabilities**: Network analysis, data visualization, data collection

### 4. Threat Hunting and Incident Response
- **Tools**: Seculyze, Caldera, Blue Team Training Toolkit
- **Capabilities**: Automated threat hunting, adversary emulation, realistic attack scenario creation

### 5. Security Monitoring
- **Tools**: Sysmon, maltrail, velociraptor
- **Capabilities**: System monitoring, malicious traffic detection

### 6. Vulnerability Management
- **Tools**: OpenVAS, Nessus Essentials, Nexpose
- **Capabilities**: Vulnerability scanning, management

## Architecture
### Data Flow
1. **Data Collection**: Logs and network data are collected using Logstash and Wireshark.
2. **Data Analysis**: Collected data is analyzed using Splunk and Kibana for visualization and threat detection.
3. **Threat Detection**: Automated threat hunting is performed using Seculyze and Caldera.
4. **Incident Response**: Detected threats trigger incident response playbooks from Societé Generale.
5. **System Monitoring**: Continuous monitoring is performed using Sysmon and maltrail.
6. **Vulnerability Management**: Regular vulnerability scans are conducted using OpenVAS and Nessus Essentials.

### Integration
- **Centralized Dashboard**: A centralized dashboard is created using Kibana to provide a unified view of all security events and actions.
- **Automation Orchestration**: XSOAR is used to orchestrate automation workflows and integrate various tools.

## Model Components
- **State Representation**: The state representation includes metrics extracted from log data and preprocessed OTX data, such as CPU usage, memory usage, disk usage, packet rate, connection count, anomaly score, intrusion alerts, firewall logs, and quantified indicators from the OTX data. Additionally, it includes binary features for the presence of IOCs (URLs, FQDNs, and IPv4 addresses). These metrics provide a comprehensive view of the system's current status and potential security threats. The number of inputs is set to 51.
- **Action Space**: The action space reflects possible actions a blue team agent might take in response to threats, such as blocking or allowing an IP address, applying rate limiting, isolating a system, sending alerts, running malware scans, changing firewall settings, updating software, searching logs, and generating reports. The `execute_action` function now accepts dynamic parameters, including `ip_address`, `rate_limit`, `system_id`, `message`, `settings`, and `query`, which are derived from the log data.
- **Neural Network**: A neural network model is used to predict the best action based on the current state. The model is defined with an input layer, two hidden layers with 128 and 64 neurons respectively, a third hidden layer with 32 neurons, and an output layer with 10 actions. The optimizer and loss function are defined, and the model is compiled.

## Training Loop
The main training loop involves the following steps:
1. **Reset Environment**: The environment is reset to its initial state, ensuring a consistent starting point for each training episode.
2. **Preprocess Data**: The OTX data is preprocessed using the `preprocess_otx_data.py` script to extract relevant features and format them into a structured numerical format suitable for training.
3. **Choose Action**: The agent chooses an action based on the current state using its neural network model. The action is selected to maximize the expected reward.
4. **Execute Action**: The chosen action is executed, and the environment provides feedback in the form of a reward or penalty.
5. **Update Model**: The agent updates its model based on the feedback received, adjusting its predictions to improve future performance.
6. **Repeat**: The loop repeats for a specified number of episodes, with the agent learning and improving over time. The training process continues until the agent achieves satisfactory performance.

## Usage Instructions
### Prerequisites
- Python 3.x
- Required Python libraries: Flask, gym, tensorflow, requests, numpy

### Setup
1. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```
2. Set the AlienVault OTX API key environment variable:
   ```bash
   export ALIENVAULT_OTX_API_KEY=your_api_key_here
   ```
3. Preprocess the OTX data:
   ```bash
   python3 preprocess_otx_data.py
   ```
4. Start the mock server:
   ```bash
   python3 mock_server.py
   ```
5. Run the RL agent script:
   ```bash
   python3 rl_agent_model.py
   ```

### Simulated Events and Actions
- The agent can handle various types of simulated security events, such as:
  - Simulated security event log data
  - Simulated intrusion attempt detected
- For each event, the agent processes the log data, converts it into a state, chooses an appropriate action, and executes it. The state representation now includes 57 input features, including binary features for the presence of IOCs (URLs, FQDNs, and IPv4 addresses).
- The actions the agent can take include:
  - Blocking or allowing an IP address (parameter: `ip_address`)
  - Applying rate limiting (parameter: `rate_limit`)
  - Isolating a system (parameter: `system_id`)
  - Sending alerts (parameter: `message`)
  - Running malware scans (parameter: `system_id`)
  - Changing firewall settings (parameter: `settings`)
  - Updating software (parameter: `system_id`)
  - Searching logs (parameter: `query`)
  - Generating reports

### Interpreting Output
- The agent's actions and decisions are printed to the console, providing real-time feedback on its performance.
- Success and error messages indicate the outcome of each action, helping users understand the agent's behavior.
- Performance metrics, such as total reward and epsilon value, are displayed for each episode, allowing users to monitor the agent's learning progress.

## Troubleshooting

### Common Issues
1. **Elasticsearch Connection Issues**:
   - **Symptom**: Logstash errors indicating connection issues with Elasticsearch.
   - **Solution**: Ensure Elasticsearch is running and accessible. Check the `logstash.conf` file for correct Elasticsearch host and port configuration. Verify network connectivity between Logstash and Elasticsearch.

2. **Grok Pattern Matching Failures**:
   - **Symptom**: Logstash fails to parse syslog messages.
   - **Solution**: Verify the grok patterns in the `logstash.conf` file. Use the `test_grok_patterns.py` script to test and validate the patterns. Ensure the syslog messages conform to the expected format.

3. **ModuleNotFoundError**:
   - **Symptom**: Missing Python modules when running scripts.
   - **Solution**: Install the required Python libraries using `pip install Flask gym tensorflow requests numpy`. Double-check the installation paths and Python environment.

### FAQ
1. **How do I start the Blue Team Agent?**
   - Follow the setup instructions in the "Usage Instructions" section to install the required libraries, start the mock server, and run the RL agent script.

2. **What actions can the RL agent perform?**
   - The RL agent can perform actions such as blocking or allowing an IP address, applying rate limiting, isolating a system, sending alerts, running malware scans, changing firewall settings, updating software, searching logs, and generating reports.

3. **How does the RL agent learn?**
   - The RL agent uses a reinforcement learning model to learn from interactions with the environment. It updates its model based on feedback received from executing actions and improves its decision-making over time.

4. **How can I monitor the agent's performance?**
   - The agent's actions, decisions, and performance metrics are printed to the console. You can monitor the total reward, epsilon value, and other metrics for each episode.

## Conclusion
The Blue Team Agent leverages reinforcement learning to automate security tasks and improve decision-making over time. By integrating various tools and best practices, the agent enhances the organization's security posture and streamlines incident response processes.

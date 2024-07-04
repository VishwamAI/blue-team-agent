# Blue Team Agent Documentation

## Overview
The Blue Team Agent is designed to automate various security tasks, including threat detection, incident response, system health monitoring, and patch management. It leverages reinforcement learning to make intelligent decisions based on real-time data and interactions with the environment.

## Components
### 1. SOC Automation
- **Tools**: XSOAR, Intezer, Torq, Swimlane
- **Capabilities**: Automated alert handling, incident triage, response actions
- **Description**: Implements a set of automation capabilities inspired by Tines' SOC Automation Capability Matrix.

### 2. Incident Response Playbooks
- **Tools**: Societé Generale incident response playbooks
- **Capabilities**: Automated responses to common security incidents
- **Description**: Utilizes playbooks to automate responses to incidents such as phishing, malware infections, and unauthorized access.

### 3. Log Management and Analysis
- **Tools**: Wireshark, Splunk, Kibana, Logstash
- **Capabilities**: Network analysis, data visualization, data collection
- **Description**: Integrates tools for comprehensive log management and analysis to detect and investigate security events.

### 4. Threat Hunting and Incident Response
- **Tools**: Seculyze, Caldera, Blue Team Training Toolkit
- **Capabilities**: Automated threat hunting, adversary emulation, realistic attack scenario creation
- **Description**: Incorporates tools for continuous threat hunting and security testing to identify and mitigate threats.

### 5. Security Monitoring
- **Tools**: Sysmon, maltrail, velociraptor
- **Capabilities**: System monitoring, malicious traffic detection
- **Description**: Sets up tools to monitor system activities and detect malicious behavior in real-time.

### 6. Vulnerability Management
- **Tools**: OpenVAS, Nessus Essentials, Nexpose
- **Capabilities**: Vulnerability scanning, management
- **Description**: Automates vulnerability scans to identify and remediate security weaknesses in the system.

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

## Reinforcement Learning Model
### Model Components
- **State Representation**: The state representation includes metrics extracted from log data, such as CPU usage, memory usage, disk usage, packet rate, connection count, anomaly score, intrusion alerts, and firewall logs. These metrics provide a comprehensive view of the system's current status and potential security threats.
- **Action Space**: The action space reflects possible actions a blue team agent might take in response to threats, such as blocking or allowing an IP address, applying rate limiting, isolating a system, sending alerts, running malware scans, changing firewall settings, updating software, searching logs, and generating reports. The `execute_action` function now accepts dynamic parameters, including `ip_address`, `rate_limit`, `system_id`, `message`, `settings`, and `query`, which are derived from the log data.
- **Neural Network**: A neural network model is used to predict the best action based on the current state. The model is trained using reinforcement learning techniques to improve its decision-making over time.

### Training Loop
The main training loop involves the following steps:
1. **Reset Environment**: The environment is reset to its initial state, ensuring a consistent starting point for each training episode.
2. **Choose Action**: The agent chooses an action based on the current state using its neural network model. The action is selected to maximize the expected reward.
3. **Execute Action**: The chosen action is executed, and the environment provides feedback in the form of a reward or penalty.
4. **Update Model**: The agent updates its model based on the feedback received, adjusting its predictions to improve future performance.
5. **Repeat**: The loop repeats for a specified number of episodes, with the agent learning and improving over time. The training process continues until the agent achieves satisfactory performance.

The training loop is implemented in the `run_training_loop` function, which runs for a specified number of episodes. It resets the environment, chooses actions, executes them, stores experiences in memory, trains the model, and updates the exploration rate. The model is saved periodically and at the end of training.

The training parameters include:
- **Discount Factor (gamma)**: 0.99
- **Exploration Rate (epsilon)**: 1.0 (decays over time)
- **Minimum Exploration Rate (epsilon_min)**: 0.01
- **Exploration Decay Rate (epsilon_decay)**: 0.995
- **Batch Size**: 32
- **Memory**: Experience replay memory

The neural network model is defined with an input layer, two hidden layers with 24 neurons each, and an output layer with the number of actions. The optimizer and loss function are defined, and the model is compiled.

Results from the training process indicate that the agent's performance improves over time, with the total reward increasing and the exploration rate decreasing as the agent learns to make better decisions based on the feedback received.

## Usage Instructions
### Prerequisites
- Python 3.x
- Required Python libraries: Flask, gym, tensorflow, requests, numpy

### Setup
1. Install the required Python libraries:
   ```bash
   pip install Flask gym tensorflow requests numpy
   ```
2. Start the mock server:
   ```bash
   python3 mock_server.py
   ```
3. Run the RL agent script:
   ```bash
   python3 rl_agent_model.py
   ```

### Simulated Events and Actions
- The agent can handle various types of simulated security events, such as:
  - Simulated security event log data
  - Simulated intrusion attempt detected
- For each event, the agent processes the log data, converts it into a state, chooses an appropriate action, and executes it.
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

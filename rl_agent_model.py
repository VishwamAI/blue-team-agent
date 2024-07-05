import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask, request, jsonify

# Define the environment
env = gym.make('CartPole-v1')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define the neural network model
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.n

model = tf.keras.Sequential([
    layers.Input(shape=(num_inputs,)),
    layers.Conv2D(32, (8, 8), strides=4, activation='relu'),
    layers.Conv2D(64, (4, 4), strides=2, activation='relu'),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
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
        return env.action_space.sample()
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

# Main training loop (commented out for testing Flask server)
# num_episodes = 1000
# save_interval = 100  # Save the model every 100 episodes
# for episode in range(num_episodes):
#     state = env.reset()
#     state = state[0] if isinstance(state, tuple) else state
#     print(f"Initial state shape: {state.shape}, state: {state}")
#     state = np.reshape(state, [1, num_inputs])
#     total_reward = 0
#     for time in range(500):
#         action = choose_action(state)
#         step_result = env.step(action)
#         print(f"Step result: {step_result}")
#         next_state, reward, done, _ = step_result[:4]
#         next_state = next_state[0] if isinstance(next_state, tuple) else next_state
#         print(f"Next state shape: {next_state.shape}, next_state: {next_state}")
#         next_state = np.reshape(next_state, [1, num_inputs])
#         memory.append((state, action, reward, next_state, done))
#         state = next_state
#         total_reward += reward
#         if done:
#             print(f"Episode: {episode+1}/{num_episodes}, Score: {total_reward}, Epsilon: {epsilon:.2}")
#             break
#         train_model()
#     if epsilon > epsilon_min:
#         epsilon *= epsilon_decay
#     if (episode + 1) % save_interval == 0:
#         model.save(f'rl_agent_model_{episode + 1}.h5')

# Save the trained model
model.save('rl_agent_model.h5')

# Flask web server to receive log data from Logstash
app = Flask(__name__)

@app.route('/logs', methods=['POST'])
def receive_logs():
    log_data = request.json
    # Process the log data (this is a placeholder, actual processing logic will be added)
    print(f"Received log data: {log_data}")
    return jsonify({"status": "success"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from flask import Flask, request, jsonify

# Initialize the training step counter
training_step_counter = 0  # Initialize the training step counter

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
    layers.Dense(24, activation='relu'),
    layers.Dense(24, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_actions, activation='linear')
])

# Define the optimizer and loss function
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # Increased learning rate
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
target_model = tf.keras.models.clone_model(model)
target_model.set_weights(model.get_weights())
update_target_frequency = 100  # Update target model every 100 training steps

# Function to choose an action based on the current state
def choose_action(state):
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    q_values = model.predict(state)
    return np.argmax(q_values[0])

# Function to train the model
def train_model():
    print("Entering train_model function.")
    print(f"Memory length: {len(memory)}")
    if len(memory) < batch_size:
        print("Not enough memory to train the model.")
        return
    batch = np.random.choice(len(memory), batch_size, replace=False)
    print(f"Sampled batch indices: {batch}")
    for i in batch:
        state, action, reward, next_state, done = memory[i]
        print(f"Experience {i}: state={state}, action={action}, reward={reward}, next_state={next_state}, done={done}")
        target = reward
        if not done:
            print(f"Reshaping next state: {next_state}")
            next_state = np.reshape(next_state, [1, num_inputs])
            print(f"Next state reshaped: {next_state}")
            print(f"Predicting target Q-values for next state: {next_state}")
            target_q_values = target_model.predict(next_state)
            print(f"Predicted target Q-values: {target_q_values}")
            target += gamma * np.amax(target_q_values[0])
            print(f"Updated target value: {target}")
        print(f"Reshaping current state: {state}")
        state = np.reshape(state, [1, num_inputs])
        print(f"Current state reshaped: {state}")
        print(f"Predicting Q-values for current state: {state}")
        q_values = model.predict(state)
        print(f"Predicted Q-values: {q_values}")
        target_f = q_values
        target_f[0][action] = target
        print(f"Updated Q-values: {target_f}")
        print(f"State shape: {state.shape}, Target_f shape: {target_f.shape}")  # Print shapes of state and target_f
        print(f"Fitting model with state: {state} and target_f: {target_f}")
        print("Before model.fit call")  # Print before model.fit call
        model.fit(state, target_f, epochs=1, verbose=0)
        print("After model.fit call")  # Print after model.fit call
        print("Model fit completed.")
    global training_step_counter
    print(f"Training step counter before increment: {training_step_counter}")
    training_step_counter += 1
    print(f"Training step counter after increment: {training_step_counter}")
    if training_step_counter % update_target_frequency == 0:
        print("Updating target model weights")
        target_model.set_weights(model.get_weights())
        print("Target model weights updated.")
    print("Exiting train_model function.")

# Function to run the main training loop
def run_training_loop():
    num_episodes = 1000
    save_interval = 100  # Save the model every 100 episodes
    for episode in range(num_episodes):
        state = env.reset()
        state = state[0] if isinstance(state, tuple) else state
        state = np.reshape(state, [1, num_inputs])
        total_reward = 0
        for time in range(500):
            action = choose_action(state)
            step_result = env.step(action)
            next_state, reward, done, _ = step_result[:4]
            next_state = next_state[0] if isinstance(next_state, tuple) else next_state
            next_state = np.reshape(next_state, [1, num_inputs])
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            if done:
                break
            train_model()
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        if (episode + 1) % save_interval == 0:
            model.save(f'rl_agent_model_{episode + 1}.h5')
        if (episode + 1) % update_target_frequency == 0:
            target_model.set_weights(model.get_weights())
            print(f"Target model weights updated at episode {episode + 1}")  # Print target model weight update
            print(f"Target model weights: {target_model.get_weights()}")  # Print target model weights

    # Save the trained model
    model.save('rl_agent_model.h5')

# Flask web server to receive log data from Logstash and interact with the RL model
app = Flask(__name__)

@app.route('/logs', methods=['POST'])
def receive_logs():
    log_data = request.json
    # Process the log data and make a prediction using the RL model
    state = np.array(log_data['state']).reshape(1, -1)
    action = choose_action(state)
    return jsonify({"status": "success", "action": int(action)}), 200

if __name__ == '__main__':
    run_training_loop()
    app.run(host='0.0.0.0', port=5000)

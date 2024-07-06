import sys
import os
import numpy as np
import tensorflow as tf

# Add the parent directory to the system path to resolve module import error
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_agent_model import choose_action, train_model, env, num_inputs, num_actions, model, gamma, epsilon, epsilon_min, epsilon_decay, batch_size, memory, target_model, update_target_frequency, training_step_counter

def test_choose_action():
    state = np.random.rand(1, num_inputs)
    action = choose_action(state)
    assert action in range(num_actions), f"Action {action} is not within the valid range of actions."

def test_train_model():
    # Create varied mock data for testing
    for _ in range(batch_size):
        state = np.random.rand(1, num_inputs)
        next_state = np.random.rand(1, num_inputs)
        action = np.random.randint(num_actions)
        reward = np.random.rand()
        done = np.random.choice([True, False])
        memory.append((state, action, reward, next_state, done))

    # Train the model with the mock data
    initial_weights = model.get_weights()
    for step in range(10):  # Train for multiple steps to ensure weight updates
        print(f"Train step {step}: Starting training step")  # Print starting step
        train_model()
        target_f = model.predict(state)  # Define target_f within the test function
        print(f"Train step {step}: Loss: {model.evaluate(state, target_f, verbose=0)}")  # Print loss
        print(f"Train step {step}: Updated weights: {model.get_weights()}")  # Print updated weights
    updated_weights = model.get_weights()

    for initial, updated in zip(initial_weights, updated_weights):
        assert not np.allclose(initial, updated, atol=1e-5), "Model weights have not been updated after training."

def test_epsilon_decay():
    global epsilon
    initial_epsilon = epsilon
    for _ in range(10):
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    assert epsilon < initial_epsilon, "Epsilon did not decay as expected."
    assert epsilon >= epsilon_min, "Epsilon decayed below the minimum threshold."

def test_target_model_update():
    global training_step_counter  # Ensure the global counter is used
    training_step_counter = 0  # Initialize the training step counter
    # Populate memory with enough samples
    for _ in range(batch_size):
        state = np.random.rand(1, num_inputs)
        next_state = np.random.rand(1, num_inputs)
        action = np.random.randint(num_actions)
        reward = np.random.rand()
        done = np.random.choice([True, False])
        memory.append((state, action, reward, next_state, done))
    initial_target_weights = target_model.get_weights()
    for step in range(update_target_frequency * 3):  # Ensure the update frequency is reached
        print(f"Train step {step}: Memory length before training: {len(memory)}")  # Print memory length before training
        train_model()
        print(f"Train step {step}: Memory length after training: {len(memory)}")  # Print memory length after training
        print(f"Train step {step}: Training step counter: {training_step_counter}")  # Print training step counter
        print(f"Train step {step}: Target model weights: {target_model.get_weights()}")  # Print target model weights
    updated_target_weights = target_model.get_weights()

    for initial, updated in zip(initial_target_weights, updated_target_weights):
        print(f"Initial weights: {initial}")
        print(f"Updated weights: {updated}")
        assert not np.allclose(initial, updated, atol=1e-3), "Target model weights have not been updated after the specified frequency."

def test_memory_replay():
    # Ensure memory is populated
    for _ in range(batch_size):
        state = np.random.rand(1, num_inputs)
        next_state = np.random.rand(1, num_inputs)
        action = np.random.randint(num_actions)
        reward = np.random.rand()
        done = np.random.choice([True, False])
        memory.append((state, action, reward, next_state, done))

    assert len(memory) >= batch_size, "Memory does not contain enough experiences for replay."

import sys
import os
import numpy as np
import tensorflow as tf

# Add the parent directory to the system path to resolve module import error
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rl_agent_model import choose_action, train_model, env, num_inputs, num_actions, model, gamma, epsilon, epsilon_min, epsilon_decay, batch_size, memory, target_model, update_target_frequency

def test_choose_action():
    state = np.random.rand(1, num_inputs)
    action = choose_action(state)
    assert action in range(num_actions), f"Action {action} is not within the valid range of actions."

def test_train_model():
    # Create mock data for testing
    state = np.random.rand(1, num_inputs)
    next_state = np.random.rand(1, num_inputs)
    action = np.random.randint(num_actions)
    reward = np.random.rand()
    done = np.random.choice([True, False])

    # Add mock data to memory
    memory.append((state, action, reward, next_state, done))

    # Train the model with the mock data
    train_model()

    # Check if the model's weights have been updated
    initial_weights = model.get_weights()
    train_model()
    updated_weights = model.get_weights()

    for initial, updated in zip(initial_weights, updated_weights):
        assert not np.array_equal(initial, updated), "Model weights have not been updated after training."

def test_epsilon_decay():
    global epsilon
    initial_epsilon = epsilon
    for _ in range(10):
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    assert epsilon < initial_epsilon, "Epsilon did not decay as expected."
    assert epsilon >= epsilon_min, "Epsilon decayed below the minimum threshold."

def test_target_model_update():
    initial_target_weights = target_model.get_weights()
    for _ in range(update_target_frequency):
        train_model()
    updated_target_weights = target_model.get_weights()

    for initial, updated in zip(initial_target_weights, updated_target_weights):
        assert np.array_equal(initial, updated), "Target model weights have not been updated after the specified frequency."

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

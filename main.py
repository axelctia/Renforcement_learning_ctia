import os
import gymnasium as gym
import numpy as np
import torch

env = gym.make('ALE/Skiing-v5', render_mode='rgb_array').env
env.metadata['render_fps'] = 30

# Observation Space
env.observation_space

action_size = env.action_space.n

# First pictures, first state
state, info = env.reset()


class SARSAAgent:
    """Here is an RL Agent using SARSA to solve our code"""

    def __init__(self, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.9995):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.all_state = []
        self.q_table = np.zeros((1, self.action_size), dtype=float)

    def choose_action(self, state_index):
        if np.random.uniform() < self.epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(self.q_table[state_index, :])
        return action

    def learn(self, state_index, action, reward, next_state_index, next_action, done):
        td_target = reward + self.discount_factor * self.q_table[next_state_index, next_action] * (1 - done)
        td_error = td_target - self.q_table[state_index, action]
        self.q_table[state_index, action] += self.learning_rate * td_error

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Define hyperparameters
episodes = 10000
max_steps = 500

# Define SARSA agentd
agent = SARSAAgent(action_size)


if not os.path.isfile('my_model.pt'):
    print("Le fichier n'existe pas.")

    # Train the agent using the SARSA algorithm
    for episode in range(episodes):
        step = 0
        state, info = env.reset()
        agent.all_state.append(state)
        print('Le nombre de state est', len(agent.all_state))
        action = agent.choose_action(step)
        done = False

        while not done and step < max_steps:
            # Take action and observe new state and reward
            next_state, reward, done, is_truncated, info = env.step(action)
            # Add one line to agent q_table
            if agent.q_table.shape[0] < step + 2:
                agent.q_table = np.append(agent.q_table, [[0, 0, 0]], axis=0)

            # Choose next action based on epsilon-greedy policy
            next_action = agent.choose_action(step + 1)
            # Update Q-table
            agent.learn(step, action, reward, step + 1, next_action, done)
            # Update state and action
            state = next_state
            action = next_action
            # Update step count
            step += 1

        # Decay epsilon
        agent.decay_epsilon()

    # Save the model's state dictionary to a file
    torch.save(agent.q_table, 'my_model.pt')


print("Le fichier existe maintenant.")
# Load the saved state dictionary into a new model object
loaded_q_table = torch.load('my_model.pt')

# Test the agent with the learned policy

# Search state in the all_state from the training
def find_array_index(arr_list, arr):
    for i, a in enumerate(arr_list):
        if np.array_equal(a, arr):
            return i
    return -1  # Si le tableau n'a pas été trouvé dans la liste, retourne -1



env = gym.make('ALE/Skiing-v5', render_mode='human').env
# Initialize the environment
state, info = env.reset()
done = False
while not done:
    index = find_array_index(agent.all_state, state)
    action = np.argmax(loaded_q_table[index, :])
    state, reward, done, is_truncated, info = env.step(action)
    env.render()

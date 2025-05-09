import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mcts import MCTS
from matrix_rl_env import MatrixMultiplicationEnv
import time

class PolicyNetwork(nn.Module):
    """
    A simple neural network for learning multiplication strategy probabilities.
    """
    def __init__(self, input_size, output_size, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

class RLAgent:
    """
    Reinforcement Learning agent using MCTS and Policy Gradient updates.
    """
    def __init__(self, env, learning_rate=0.001):
        self.env = env
        input_size = 2 * env.matrix_size[0] * env.matrix_size[1]  # Flattened matrix size
        output_size = env.action_space.n  # Number of multiplication strategies

        self.policy_net = PolicyNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.mcts = MCTS(self.env, self.policy_net)
        self.reward_history = []  # Initialize reward_history here


    def select_action(self, state_info):
        """
        Select an action using Monte Carlo Tree Search.
        Args:
            state_info: A tuple containing the flattened state and the info dictionary
                        as returned by the environment's reset() and step() methods.
        """
        state, info = state_info
        print(f"RLAgent.select_action: Received state type: {type(state)}, shape: {state.shape if hasattr(state, 'shape') else len(state)}")
        probabilities = self.mcts.search(state_info)
        probabilities = np.asarray(probabilities).flatten()
        action = int(np.random.choice(len(probabilities), p=probabilities))
        print(f"RLAgent.select_action: Chosen action: {action}")
        return action
    
    def estimate_baseline(self, state):
        if not self.reward_history:
            return 0  # Or some initial estimate
        return np.mean(self.reward_history)

    def learn(self, state, action, reward, entropy_weight=0.01):
        self.reward_history.append(reward) # Store the reward
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.int64)

        probabilities = self.policy_net(state_tensor)
        log_prob = torch.log(probabilities[0, action_tensor])

        baseline = self.estimate_baseline(state)  # Now this should work
        advantage = reward - baseline  # Stabilizes learning updates

        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
        loss = -log_prob * advantage - entropy_weight * entropy  # Adds entropy regularization

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def evaluate(self, episodes=10): # Remove matrix_sizes argument for now
        """
        Evaluate the agent's average reward, execution times, and action histories.
        """
        all_rewards = {}
        all_rl_action_histories = {}
        all_times = {}

        size = self.env.matrix_size[0] # Use the agent's environment size
        total_reward = 0
        rl_action_histories_for_size = []
        times_for_size = {"Standard": [], "Strassen": [], "RL_Discovered": []}

        for episode in range(episodes):
            state, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_action_history = []

            while not done:
                action = self.select_action((state, info))
                episode_action_history.append(action)

                start_time = time.time()
                next_state, reward, done, next_info = self.env.step(action)
                end_time = time.time()
                execution_time = end_time - start_time

                if info.get("algorithm_used") in times_for_size:
                    times_for_size[info["algorithm_used"]].append(execution_time)

                episode_reward += reward
                state = next_state
                info = next_info

            total_reward += episode_reward
            if any(a == 3 for a in episode_action_history):
                rl_action_histories_for_size.append(episode_action_history)

        avg_reward = total_reward / episodes
        all_rewards[size] = avg_reward
        all_rl_action_histories[size] = rl_action_histories_for_size
        all_times[size] = {alg: np.mean(times) if times else 0 for alg, times in times_for_size.items()}

        print("\n--- Evaluation Results ---")
        print(f"\nMatrix Size: {size}x{size}")
        print(f"  Average Reward: {all_rewards[size]:.2f}")
        print("  Average Execution Times (seconds):")
        for alg, avg_time in all_times[size].items():
            print(f"    {alg}: {avg_time:.4f}")
        if all_rl_action_histories[size]:
            print("  Action Histories when RL_Discovered was used:")
            for i, history in enumerate(all_rl_action_histories[size]):
                print(f"    Episode {i+1}: {history}")
        else:
            print("  RL_Discovered was not used in any evaluation episode for this size.")

        return all_rewards, all_times, all_rl_action_histories

    def train(self, episodes=1000, gamma=0.99):
        """
        Train the RL agent using policy gradient updates.
        """
        for episode in range(episodes):
            state, info = self.env.reset()
            print(f"Train Episode {episode}: Initial state shape: {state.shape}")
            trajectory = []
            done = False

            while not done:
                print(f"Train Episode {episode}: State before select_action shape: {state.shape}")
                action = self.select_action((state, info)) # Pass the full state tuple
                next_state, reward, done, next_info = self.env.step(action)
                print(f"Train Episode {episode}: Next state shape: {next_state.shape}")
                trajectory.append((state, action, reward))
                state = next_state
                info = next_info  # Update info here!

            # Compute discounted rewards
            G = 0
            discounted = []
            for _, _, reward in reversed(trajectory):
                G = reward + gamma * G
                discounted.insert(0, G)

            # Update policy
            for (state, action, _), Gt in zip(trajectory, discounted):
                self.learn(state, action, Gt)

            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}, Total Reward: {sum(r for _, _, r in trajectory)}")

            if (episode + 1) % 200 == 0:
                print(f"Train Episode {episode}: State before evaluate shape: {state.shape}")
                self.evaluate(episodes=10)

    def save_model(self, path="rl_agent.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print("Model saved.")

    def load_model(self, path="rl_agent.pth"):
        self.policy_net.load_state_dict(torch.load(path))
        print("Model loaded.")

if __name__ == "__main__":
    env = MatrixMultiplicationEnv()
    agent = RLAgent(env)
    agent.train(episodes=1000)
    agent.save_model()
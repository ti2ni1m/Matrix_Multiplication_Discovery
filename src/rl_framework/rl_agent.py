import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from mcts import MCTS
from matrix_rl_env import MatrixMultiplicationEnv

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

    def select_action(self, state_info):
        """
        Select an action using Monte Carlo Tree Search.
        Args:
            state_info: A tuple containing the flattened state and the info dictionary
                        as returned by the environment's reset() and step() methods.
        """
        state, info = state_info
        probabilities = self.mcts.search(state_info)
        probabilities = np.asarray(probabilities).flatten()
        action = int(np.random.choice(len(probabilities), p=probabilities))
        return action
    
    def learn(self, state, action, reward, entropy_weight=0.01):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.int64)

        probabilities = self.policy_net(state_tensor)
        log_prob = torch.log(probabilities[0, action_tensor])
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))

        loss = -log_prob * reward - entropy_weight * entropy

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def evaluate(self, episodes=10):
        """
        Evaluate the agent's average reward over a number of episodes.
        """
        total_reward = 0
        for _ in range(episodes):
            state, info = self.env.reset() # Unpack the tuple
            done = False
            episode_reward = 0
            while not done:
                action = self.select_action((state, info)) # Pass the full state tuple
                next_state, reward, done, next_info = self.env.step(action)
                episode_reward += reward
                state = next_state
                info = next_info
            total_reward += episode_reward
        avg_reward = total_reward / episodes
        print(f"Average reward over {episodes} episodes: {avg_reward:.2f}")
        return avg_reward

    def train(self, episodes=1000, gamma=0.99):
        """
        Train the RL agent using policy gradient updates.
        """
        for episode in range(episodes):
            state, info = self.env.reset()
            trajectory = []
            done = False

            while not done:
                action = self.select_action((state, info)) # Pass the full state tuple
                next_state, reward, done, next_info = self.env.step(action)
                trajectory.append((state, action, reward))
                state = next_state
                info = next_info

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
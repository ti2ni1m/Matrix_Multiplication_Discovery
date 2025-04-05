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
        input_size = env.matrix_size[0] * env.matrix_size[1]  # Flattened matrix size
        output_size = env.action_space.n  # Number of multiplication strategies

        self.policy_net = PolicyNetwork(input_size, output_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.mcts = MCTS(self.env, self.policy_net)

    def select_action(self, state):
        """
        Select an action using Monte Carlo Tree Search.
        """
        probabilities = self.mcts.search(state)
        action = np.random.choice(len(probabilities), p=probabilities)
        return action

    def train(self, episodes=1000):
        """
        Train the RL agent using policy gradient updates.
        """
        for episode in range(episodes):
            state_dict = self.env.reset()
            state = state_dict["matrix_a"].flatten()
            action = self.select_action(state_dict)
            _, reward, _, _ = self.env.step(action)

            # Convert state to tensor
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_tensor = torch.tensor([action], dtype=torch.int64)

            # Compute loss
            probabilities = self.policy_net(state_tensor)
            loss = -torch.log(probabilities[0, action]) * reward

            # Backpropagate
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (episode + 1) % 100 == 0:
                print(f"Episode {episode + 1}, Reward: {reward}")

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
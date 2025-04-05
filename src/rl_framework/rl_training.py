import os
import numpy as np
import pickle
import torch
import torch.nn as nn
from matrix_rl_env import MatrixMultiplicationEnv
from rl_agent import RLAgent  # Assuming rl_agent.py is properly defined
from mcts import MCTS  # Assuming mcts.py is properly defined

class RLTraining:
    def __init__(self, env, agent, mcts, num_episodes=1000, max_timesteps=100, model_dir="models"):
        self.env = env
        self.agent = agent
        self.mcts = mcts
        self.num_episodes = num_episodes
        self.max_timesteps = max_timesteps
        self.model_dir = model_dir

        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()  # Reset environment at the start of each episode
            total_reward = 0

            for t in range(self.max_timesteps):
                # Use MCTS to determine the best action for the agent
                action = self.mcts.search(state)

                print(f"Action received: {action} (Type: {type(action)})")

                # Check if action is a numpy array or tensor
                if isinstance(action, np.ndarray):
                    action = action.item()  # Convert to scalar if only one element

                elif isinstance(action, torch.Tensor):
                    action = action.item()  # Convert tensor to scalar
                
                print(f"Processed Action: {action} (Type: {type(action)})")

                # Take the action and observe the result
                try:
                    next_state, reward, done, _ = self.env.step(action)
                except ValueError as e:
                    print(f"Error during environment step: {e}")
                    break
                
                total_reward += reward

                # Use the agent to learn from the environment (e.g., update Q-values or policy)
                self.agent.learn(state, action, reward, next_state, done)

                # Move to the next state
                state = next_state

                if done:
                    break  # Exit the loop if the environment signals the end of an episode

            print(f"Episode {episode + 1}/{self.num_episodes} finished with total reward: {total_reward}")

            # Save the model periodically (every 100 episodes in this case)
            if (episode + 1) % 100 == 0:
                self.save_model(episode + 1)

            # Optionally log training metrics here (e.g., save total reward, loss, etc.)

    def save_model(self, episode):
        # Save the agent's model (e.g., Q-values, policy network, etc.)
        model_filename = os.path.join(self.model_dir, f"model_episode_{episode}.pkl")
        with open(model_filename, 'wb') as model_file:
            pickle.dump(self.agent, model_file)
        print(f"Model saved after episode {episode}.")

    def load_model(self, filename):
        # Load a previously saved model
        with open(filename, 'rb') as model_file:
            self.agent = pickle.load(model_file)
        print(f"Model loaded from {filename}.")

if __name__ == "__main__":
    # Initialize the environment, agent, and MCTS
    env = MatrixMultiplicationEnv()  # Replace with the actual environment initialization
    agent = RLAgent(env)  # Replace with the actual agent initialization
    policy_net = agent.policy_net  # Get the policy network from the agent (this will be passed to MCTS)
    mcts = MCTS(env, policy_net)  # Replace with the actual MCTS initialization

    # Initialize the training loop
    training = RLTraining(env, agent, mcts)

    # Train the agent
    training.train()

    # Optionally, load a saved model
    # training.load_model("models/model_episode_100.pkl")

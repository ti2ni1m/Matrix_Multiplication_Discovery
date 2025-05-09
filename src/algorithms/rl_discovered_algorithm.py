import numpy as np
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rl_framework.rl_generated_ops import apply_discovered_algorithm
from src.rl_framework.matrix_rl_env import MatrixMultiplicationEnv # Import environment

class PolicyNetwork(nn.Module):
    def __init__(self, observation_space_shape, action_space_n):
        super().__init__()
        hidden_size = 128
        self.fc1 = nn.Linear(32, hidden_size)  # Input size MUST be 32
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 4)   # Output size MUST be 4
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # You MUST ensure 'x' is processed to have a size of 32 here
        if x.numel() > 32:
            x = x.flatten()[:32].unsqueeze(0) # Example: take the first 32 elements
        x = self.relu(self.fc1(x))
        return self.softmax(self.fc2(x))

def load_trained_policy(observation_shape, action_n):
    model_path = "rl_agent.pth"
    if os.path.exists(model_path):
        policy_net = PolicyNetwork(observation_shape, action_n)
        policy_net.load_state_dict(torch.load(model_path))
        policy_net.eval()
        return policy_net
    else:
        print(f"Error: Policy network not found at {model_path}")
        return None

def rl_discovered_algorithm(matrix_a, matrix_b):
    temp_env = MatrixMultiplicationEnv(matrix_size=matrix_a.shape)
    observation_shape = temp_env.observation_space.shape
    action_n = temp_env.action_space.n
    policy = load_trained_policy(observation_shape, action_n)

    if policy is None:
        return np.zeros_like(np.dot(matrix_a, matrix_b))

    env = MatrixMultiplicationEnv(matrix_size=matrix_a.shape)
    state, _ = env.reset(state={"matrix_a": matrix_a, "matrix_b": matrix_b})
    done = False
    action_history = []
    max_steps = 50

    with torch.no_grad():
        for _ in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            policy_output = policy(state_tensor)
            action = torch.argmax(policy_output).item()
            action_history.append(action)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            if done:
                break

    print(f"Learned action sequence for size {matrix_a.shape}: {action_history}")
    env_eval = MatrixMultiplicationEnv(matrix_size=matrix_a.shape)
    eval_state, _ = env_eval.reset(state={"matrix_a": matrix_a, "matrix_b": matrix_b})
    for action in action_history:
        eval_state, reward, done, info = env_eval.step(action)
        if done:
            break
    return env_eval.current_result if env_eval.current_result is not None else np.zeros_like(np.dot(matrix_a, matrix_b))
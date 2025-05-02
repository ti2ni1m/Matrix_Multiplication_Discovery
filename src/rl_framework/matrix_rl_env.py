import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.algorithms.standard import standard_multiplication
from src.algorithms.strassen import strassen_multiplication
from src.algorithms.coppersmith import coppersmith_winograd

class MatrixMultiplicationEnv(gym.Env):
    """
    Custom environment for matrix multiplication using RL.
    The goal is to learn an optimal sequence of operations to multiply matrices efficiently.
    """
    def __init__(self, matrix_size=(4, 4)):
        super(MatrixMultiplicationEnv, self).__init__()

        self.matrix_size = matrix_size
        self.action_space = spaces.Discrete(3)  # 0 = Standard, 1 = Strassen, 2 = Coppersmith-Winograd
        self.observation_space = spaces.Box(low=-100, high=100, shape=(2 * matrix_size[0] * matrix_size[1],), dtype=np.float32)

        self.matrix_a = None
        self.matrix_b = None
        self.current_result = None
        self.action_history = []

    def step(self, action):
        """
        Execute an action and return (next_state, reward, done, info)
        """
        print(f"Action received: {action}")  # Debugging line
        print(f"Action type: {type(action)}")  # Print action type

        #Check if action is tensor
        if isinstance(action, torch.Tensor):
            action = action.item()
        
        elif isinstance(action, np.ndarray):
            if action.size == 1:
                action = action.item()
            else:
                raise ValueError(f"Invalid action shape: {action.shape}")

        if action == 0:
            self.current_result = self.standard_multiplication()
            print("Using standard multiplication.")
        elif action == 1:
            self.current_result = self.strassen_multiplication()
            print("Using Strassen multiplication.")
        elif action == 2:
            self.current_result = self.coppersmith_winograd_multiplication()
            print("Using Coppersmith-Winograd multiplication.")
        else:
            raise ValueError(f"Invalid action: {action}")  # Better error message

        reward = -self.compute_cost(action)
        done = True  # One-shot episode
        self.action_history.append(action)

        next_state = np.concatenate((self.matrix_a.flatten(), self.matrix_b.flatten())).astype(np.float32)

        return next_state, reward, done, {}

    def reset(self, seed=None, options=None, state=None):
        """
        Reset the environment state for a new episode.
        If a state dictionary is provided, restore from it.
        """
        super().reset(seed=seed)
        if state is not None:
            self.matrix_a = state.get("matrix_a")
            self.matrix_b = state.get("matrix_b")
            self.action_history = state.get("action_history", [])
            initial_state = np.concatenate((self.matrix_a.flatten(), self.matrix_b.flatten())).astype(np.float32)
            info = {
                "available_actions": [0, 1, 2],
                "action_history": self.action_history
            }
            return initial_state, info
        else:
            low = -10
            high = 10
            self.matrix_a = self.np_random.integers(low, high, size=self.matrix_size).astype(np.float32)
            self.matrix_b = self.np_random.integers(low, high, size=self.matrix_size).astype(np.float32)
            self.current_result = None
            self.action_history = []
            initial_state = np.concatenate((self.matrix_a.flatten(), self.matrix_b.flatten())).astype(np.float32)
            info = {
                "available_actions": [0, 1, 2],
                "action_history": self.action_history
            }
            return initial_state, info
    

    def render(self, mode="human"):
        print(f"Matrix A:\n{self.matrix_a}")
        print(f"Matrix B:\n{self.matrix_b}")
        if self.current_result is not None:
            print(f"Result:\n{self.current_result}")

    def standard_multiplication(self):
        return standard_multiplication(self.matrix_a, self.matrix_b)

    def strassen_multiplication(self):
        return strassen_multiplication(self.matrix_a, self.matrix_b)

    def coppersmith_winograd_multiplication(self):
        return coppersmith_winograd(self.matrix_a, self.matrix_b)

    def compute_cost(self, action):
        n = self.matrix_size[0]
        if action == 0:
            return n ** 3
        elif action == 1:
            return n ** 2.8
        elif action == 2:
            return n ** 2.37
        return float("inf")


if __name__ == "__main__":
    env = MatrixMultiplicationEnv()
    state = env.reset()
    env.render()
    next_state, reward, done, _ = env.step(1)
    print(f"Reward: {reward}")
    env.render()

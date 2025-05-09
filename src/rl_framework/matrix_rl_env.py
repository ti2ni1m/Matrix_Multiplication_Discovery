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
from src.rl_framework.rl_generated_ops import apply_discovered_algorithm
# Import the basic operation functions if you need them directly in the env
from src.rl_framework.rl_generated_ops import (
    basic_op_scalar_multiply,
    basic_op_elementwise_add,
    basic_op_transpose,
    basic_op_multiply_subblocks,
    basic_op_combine_subblocks,
    basic_op_extract_submatrix,
    basic_op_zero_matrix,
    basic_op_hadamard_product,
    basic_op_permute_rows,
    basic_op_permute_cols,
    basic_op_scale_row,
    basic_op_scale_col,
)

class MatrixMultiplicationEnv(gym.Env):
    """
    Custom environment for matrix multiplication using RL.
    The goal is to learn an optimal sequence of operations to multiply matrices efficiently.
    """
    def __init__(self, matrix_size=(4, 4), max_steps=50):
        super(MatrixMultiplicationEnv, self).__init__()

        self.matrix_size = matrix_size
        self.action_space = spaces.Discrete(4)  # Number of basic operations
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(matrix_size[0] * matrix_size[1] * (max_steps + 2),), dtype=np.float32)

        self.initial_matrix_a = None
        self.initial_matrix_b = None
        self.intermediate_matrices = {'A': None, 'B': None}
        self.steps_taken = 0
        self.max_steps = max_steps
        self.action_history = []
        self.current_result = None
    
    def _get_current_state_representation(self):
        all_matrices = list(self.intermediate_matrices.values())
        flattened = np.concatenate([m.flatten() for m in all_matrices]) if all_matrices else np.array([])
        padded = np.pad(flattened, (0, self.observation_space.shape[0] - flattened.size), constant_values=0)
        return padded[:self.observation_space.shape[0]]

    def step(self, action):
        """
        Execute an action and return (next_state, reward, done, info)
        """
        print(f"Action received: {action}")
        print(f"Action type: {type(action)}")

        if isinstance(action, torch.Tensor):
            action = action.item()
        elif isinstance(action, np.ndarray):
            if action.size == 1:
                action = action.item()
            else:
                raise ValueError(f"Invalid action shape: {action.shape}")

        reward = -0.1
        done = False
        info = {"action_history": self.action_history}
        self.action_history.append(action)
        self.steps_taken += 1

        matrices = self.intermediate_matrices

        try:
            if action == 0:
                matrices['A'] = basic_op_scalar_multiply(matrices['A'], 2)
            elif action == 1:
                matrices['Sum_AB_' + str(self.steps_taken)] = basic_op_elementwise_add(matrices['A'], matrices['B'])
            elif action == 2:
                matrices['B_T_' + str(self.steps_taken)] = basic_op_transpose(matrices['B'])
            elif action == 3:
                n = self.matrix_size[0]
                if n >= 2:
                    a_sub = matrices['A'][:n//2, :n//2]
                    b_sub = matrices['B'][:n//2, :n//2]
                    matrices['P1_' + str(self.steps_taken)] = basic_op_multiply_subblocks(a_sub, b_sub, n // 2)
            elif action == 4:
                keys = list(matrices.keys())
                if len(keys) >= 4:
                    subs = [matrices[keys[j]] for j in range(len(matrices) - 4, len(matrices))]
                    try:
                        matrices['Combined_' + str(self.steps_taken)] = basic_op_combine_subblocks(subs, (2, 2))
                    except ValueError as e:
                        print(f"Error combining sub-blocks: {e}")
            elif action == 5:
                matrices['A_sub_' + str(self.steps_taken)] = basic_op_extract_submatrix(matrices['A'], 0, 2, 0, 2)
            elif action == 6:
                matrices['Zero_' + str(self.steps_taken)] = basic_op_zero_matrix(self.matrix_size)
            elif action == 7:
                if matrices['A'].shape == matrices['B'].shape:
                    matrices['Hadamard_' + str(self.steps_taken)] = basic_op_hadamard_product(matrices['A'], matrices['B'])
            elif action == 8:
                if matrices['A'].shape[0] >= 2:
                    matrices['Perm_rows_A_' + str(self.steps_taken)] = basic_op_permute_rows(matrices['A'].copy(), 0, 1)
            elif action == 9:
                if matrices['B'].shape[1] >= 2:
                    matrices['Perm_cols_B_' + str(self.steps_taken)] = basic_op_permute_cols(matrices['B'].copy(), 0, 1)
            elif action == 10:  # Call apply_discovered_algorithm
                try:
                    self.current_result = apply_discovered_algorithm(
                        self.initial_matrix_a,
                        self.initial_matrix_b,
                        self.action_history,  # Pass the action history
                        self.intermediate_matrices.copy()
                    )
                except Exception as e:
                    print(f"Error applying discovered algorithm: {e}")
                    self.current_result = np.zeros_like(self.initial_matrix_a)

            # Check for a potential result (4x4 matrix)
            for name, matrix in matrices.items():
                if matrix.shape == (self.matrix_size[0], self.matrix_size[1]):
                    true_product = np.dot(self.initial_matrix_a, self.initial_matrix_b)
                    if np.allclose(matrix, true_product, atol=1e-5):
                        reward = 10
                        done = True
                        break

        except Exception as e:
            print(f"Error during step: {e}")
            reward = -1

        if self.steps_taken >= self.max_steps:
            done = True

        next_state = self._get_current_state_representation()
        info['available_actions'] = list(range(self.action_space.n))
        return next_state, reward, done, info

    def render(self, mode="human"):
        print("Intermediate Matrices:")
        for name, matrix in self.intermediate_matrices.items():
            print(f"{name}:\n{matrix}")

    def reset(self, seed=None, options=None, state=None):
        """
        Reset the environment state for a new episode.
        If a state dictionary is provided, restore from it.
        """
        super().reset(seed=seed)
        if state is None:
            self.initial_matrix_a = np.random.rand(*self.matrix_size).astype(np.float32)
            self.initial_matrix_b = np.random.rand(*self.matrix_size).astype(np.float32)
        else:
            self.initial_matrix_a = state['matrix_a'].copy()
            self.initial_matrix_b = state['matrix_b'].copy()
        self.intermediate_matrices = {'A': self.initial_matrix_a.copy(), 'B': self.initial_matrix_b.copy()}
        self.steps_taken = 0
        self.action_history = []
        self.current_result = None
        return self._get_current_state_representation(), {}
    

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
        elif action == 3:
            base_cost = 0.1 * (n ** 2) # Example: a cost proportional to matrix size squared
            cost = 0
            for historical_action in self.action_history:
                if historical_action == 0:
                    cost += n ** 3
                elif historical_action == 1:
                    cost += n ** 2.8
                elif historical_action == 2:
                    cost += n ** 2.37
            return cost + base_cost
        return float("inf")


if __name__ == "__main__":
    env = MatrixMultiplicationEnv()
    state = env.reset()
    env.render()
    next_state, reward, done, _ = env.step(1)
    print(f"Reward: {reward}")
    env.render()

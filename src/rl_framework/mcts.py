import numpy as np
import random
import math
from matrix_rl_env import MatrixMultiplicationEnv

class Node:
    """
    Represents a node in the MCTS search tree.
    """
    def __init__(self, state, parent=None, action=None):
        if state is None:
            raise ValueError("Node initialized with None state")
        self.state = state  # State of the environment (now a tuple)
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this state
        self.children = []  # Child nodes
        self.visit_count = 0  # Number of visits to this node
        self.value = 0  # Total value (reward) accumulated from this node

    def is_fully_expanded(self):
        """
        Checks if the node is fully expanded.
        The state is now a tuple: (flattened_array, info_dict).
        Available actions are in the info_dict.
        """
        if isinstance(self.state, tuple) and len(self.state) == 2:
            info = self.state[1]
            if 'available_actions' in info:
                return len(self.children) == len(info['available_actions'])
            else:
                raise ValueError("State info dictionary missing 'available_actions'")
        else:
            raise ValueError("Unsupported state structure (expected a tuple)")

    def best_child(self, exploration_weight=1.):
        """
        Returns the child with the best value (with exploration-exploitation balance).
        """
        best_value = -float('inf')
        best_child = None
        for child in self.children:
            # UCT formula for balancing exploration and exploitation
            uct_value = child.value / (child.visit_count + 1e-6) + exploration_weight * math.sqrt(math.log(self.visit_count + 1) / (child.visit_count + 1e-6))
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child



class MCTS:
    """
    Monte Carlo Tree Search for Matrix Multiplication RL Agent.
    """
    def __init__(self, env: MatrixMultiplicationEnv, policy_net, simulations=100):
        self.env = env
        self.policy_net = policy_net  # Used for generating probabilities (for action selection)
        self.simulations = simulations  # Number of MCTS simulations per action selection

    def simulate(self, node):
        """
        Simulates a game from the current node to get an estimate of its value.
        """
        # Get the flattened state (NumPy array) from the node
        if isinstance(node.state, tuple) and len(node.state) == 2:
            current_state_flat = node.state[0]
            info = node.state[1]
        else:
            raise ValueError("Unsupported state structure in simulate (expected a tuple)")

        n = self.env.matrix_size[0]
        matrix_a = current_state_flat[:n*n].reshape(self.env.matrix_size)
        matrix_b = current_state_flat[n*n:].reshape(self.env.matrix_size)

        # Temporarily set the environment's state for the simulation
        temp_env = MatrixMultiplicationEnv(matrix_size=self.env.matrix_size)
        temp_env.matrix_a = matrix_a.copy()
        temp_env.matrix_b = matrix_b.copy()
        temp_env.action_history = list(info.get('action_history', []))

        total_reward = 0
        done = False

        # Perform a few random rollouts
        for _ in range(5):
            if done:
                break
            action = temp_env.action_space.sample()
            next_state, reward, done, _ = temp_env.step(action)
            total_reward += reward

        return total_reward

    def expand(self, node):
        """
        Expands the node by adding a new child node for each available action.
        The state is now a tuple: (flattened_array, info_dict).
        Available actions are in the info_dict.
        """
        tried_actions = [child.action for child in node.children]
        if isinstance(node.state, tuple) and len(node.state) == 2:
            info = node.state[1]
            if 'available_actions' in info:
                available_actions = info['available_actions']
                for action in available_actions:
                    if action not in tried_actions:
                        next_state_flat, reward, _, _ = self.env.step(action)
                        next_info = {'available_actions': list(range(self.env.action_space.n)), 'action_history': info.get('action_history', []) + [action]}
                        child_node = Node(state=(next_state_flat, next_info), parent=node, action=action)
                        node.children.append(child_node)
                        break # Expand one child at a time
            else:
                raise ValueError("State info dictionary missing 'available_actions'")
        else:
            raise ValueError("Unsupported state structure (expected a tuple)")

    def backpropagate(self, node, reward):
        """
        Backpropagates the reward through the tree.
        """
        while node is not None:
            node.visit_count += 1
            node.value += reward
            node = node.parent

    def search(self, state):
        """
        Performs MCTS and returns a probability distribution over actions.
        """
        root = Node(state=state)
        if root is None:
            raise ValueError("Root node is None. Check initial state format.")
        
        for _ in range(self.simulations):
            node = root
            # Selection
            while node.is_fully_expanded() and node.children:
                node = node.best_child()

            # Expansion
            if not node.is_fully_expanded():
                self.expand(node)

            # Simulation
            reward = self.simulate(node)

            # Backpropagation
            self.backpropagate(node, reward)

        visit_counts = np.array([child.visit_count for child in root.children])
        probabilities = visit_counts / visit_counts.sum() 
        return probabilities

if __name__ == "__main__":
    # Test MCTS with a dummy environment
    env = MatrixMultiplicationEnv()
    policy_net = None  # Replace with actual policy network
    mcts = MCTS(env, policy_net)
    initial_state = env.reset()  # Example state, need to match the env's state
    print("Initial state:", initial_state)
    probabilities = mcts.search(initial_state)
    print("Probabilities:", probabilities)

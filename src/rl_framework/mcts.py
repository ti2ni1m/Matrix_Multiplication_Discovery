import numpy as np
import random
import math
from matrix_rl_env import MatrixMultiplicationEnv

class Node:
    """
    Represents a node in the MCTS search tree.
    """
    def __init__(self, state_info, parent=None, prior=0.0):
        self.state_info = state_info
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.total_reward = 0
        self.prior = prior  # Store the prior probability

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
    def __init__(self, env, policy_net, num_simulations=100, exploration_constant=1.0):
        self.env = env
        self.policy_net = policy_net
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.root = None  # Initialize the root node to None
        print(f"MCTS.__init__: env matrix_size: {self.env.matrix_size}")

    def select(self, node):
        """Select a leaf node in the MCTS tree using UCT."""
        while node.children:
            best_child = None
            best_uct = -float('inf')
            for child in node.children.values():
                uct = self.uct(child, self.exploration_constant)
                if uct > best_uct:
                    best_uct = uct
                    best_child = child
            node = best_child
        return node

    def uct(self, node, c):
        """Compute the Upper Confidence Bound for Trees (UCT) value."""
        if node.visit_count == 0:
            return float('inf')
        exploitation = node.total_reward / node.visit_count
        exploration = c * np.sqrt(np.log(node.parent.visit_count) / node.visit_count)
        return exploitation + exploration

    def simulate(self, node):
        """
        Simulates a game from the current node to get an estimate of its value.
        """
        # Get the flattened state (NumPy array) from the node
        if isinstance(node.state_info, tuple) and len(node.state_info) == 2:  # Changed node.state to node.state_info
            current_state_flat = node.state_info[0]
            info = node.state_info[1]
        else:
            raise ValueError("Unsupported state structure in simulate (expected a tuple)")

        n = self.env.matrix_size[0]
        print(f"MCTS.simulate: env matrix_size: {self.env.matrix_size}, n: {n}")
        print(f"MCTS.simulate: current_state_flat size: {len(current_state_flat)}")
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
                        node.children[action] = child_node  # Use action as key for children dict
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
            node.total_reward += reward
            node = node.parent

    def search(self, state_info):
        """Perform MCTS search for the best action."""
        state, info = state_info
        print(f"MCTS.search: Received state type: {type(state)}, shape: {state.shape if hasattr(state, 'shape') else len(state)}")
        root_node = self.root = Node(state_info, parent=None, prior=1.0)
        for _ in range(self.num_simulations):  # Changed from self.simulations
            node = self.select(root_node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.policy(root_node)

    def policy(self, node, temperature=1.0):
        """
        Returns the action probabilities as a NumPy array.
        """
        visits = {action: child.visit_count for action, child in node.children.items()}
        total_visits = sum(visits.values())
        num_actions = self.env.action_space.n
        probabilities = np.zeros(num_actions, dtype=np.float32)

        if total_visits > 0:
            for action in range(num_actions):
                if action in visits:
                    probabilities[action] = visits[action] ** (1 / temperature) / total_visits ** (1 / temperature)
                else:
                    probabilities[action] = 1e-6  # Small probability for unexplored actions

            # Normalize probabilities to sum to 1
            probabilities /= np.sum(probabilities)
        else:
            # Return uniform probabilities if no visits yet
            probabilities[:] = 1.0 / num_actions

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

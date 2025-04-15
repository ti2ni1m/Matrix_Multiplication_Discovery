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
        self.state = state  # State of the environment
        self.parent = parent  # Parent node
        self.action = action  # Action taken to reach this state
        self.children = []  # Child nodes
        self.visit_count = 0  # Number of visits to this node
        self.value = 0  # Total value (reward) accumulated from this node

    def is_fully_expanded(self):
        # Check if available_actions is an attribute of state
        if hasattr(self.state, 'available_actions'):
            return len(self.children) == len(self.state.available_actions)
        # If state is not dictionary-like, handle other structures
        elif isinstance(self.state, dict) and 'available_actions' in self.state:
            return len(self.children) == len(self.state['available_actions'])
        else:
            raise ValueError("Unsupported state structure")
    
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
        state = self.env.reset(state=node.state)
        total_reward = 0
        done = False

        # Perform a random rollout to simulate and get the reward.
        while not done:
            action = random.choice(state['available_actions'])
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            state = next_state

        return total_reward

    def expand(self, node):
        """
        Expands the node by adding a new child node for each available action.
        """
        tried_actions = [child.action for child in node.children]
        available_actions = node.state['available_actions']
        for action in available_actions:
            if action not in tried_actions:
                next_state, _, _, _ = self.env.step(action)
                child_node = Node(state=next_state, parent=node, action=action)
                node.children.append(child_node)
                break #Expand one child at a time

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

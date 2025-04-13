import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.rl_framework.rl_generated_ops import apply_discovered_algorithm

def rl_discovered_algorithm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Performs matrix multiplication using a discovered algorithm."""
    return apply_discovered_algorithm(A, B)


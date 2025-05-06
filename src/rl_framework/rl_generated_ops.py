import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.algorithms.standard import standard_multiplication
from src.algorithms.strassen import strassen_multiplication

def apply_discovered_algorithm(A: np.ndarray, B: np.ndarray, action_history: list = None) -> np.ndarray:
    """
    Applies a matrix multiplication algorithm "discovered" by the RL agent
    based on the sequence of actions in the action history.
    """
    if action_history is None or not action_history:
        return standard_multiplication(A, B)  # Default

    current_result = None
    matrix_x = A
    matrix_y = B

    for action in action_history:
        print(f"Applying historical action: {action}")  # Debugging

        if action == 0:
            current_result = standard_multiplication(matrix_x, matrix_y)
        elif action == 1:
            # For simplicity, we'll just apply Strassen to the original matrices
            # in this example. A more sophisticated approach might involve
            # applying it to sub-blocks if the agent learned a recursive strategy.
            current_result = strassen_multiplication(matrix_x, matrix_y)
        elif action == 2:
            current_result = standard_multiplication(matrix_x, matrix_y) # Placeholder for Coppersmith
        elif action == 3:
            # This action ideally shouldn't appear in the history
            # as it triggers this discovered algorithm. For now, default.
            return standard_multiplication(A, B)

        if current_result is not None:
            # For a sequence, we might update one of the matrices for the next step.
            # This is a very basic example and might not be semantically meaningful
            # for matrix multiplication. A more advanced approach would be needed
            # to learn meaningful sequences of operations.
            matrix_x = current_result
            matrix_y = B  # Or keep B the same, or alternate.

    return current_result if current_result is not None else standard_multiplication(A, B)

def strassen_multiply(A, B):
    # Your Strassen implementation (ensure it's correct and handles base cases)
    if A.shape == (2, 2):
        a, b, c, d = A.flatten()
        e, f, g, h = B.flatten()
        p1 = a * (f - h)
        p2 = (a + b) * h
        p3 = (c + d) * e
        p4 = d * (g - e)
        p5 = (a + d) * (e + h)
        p6 = (b - d) * (g + h)
        p7 = (a - c) * (e + f)
        return np.array([[p5 + p4 - p2 + p6, p1 + p2], [p3 + p4, p5 + p1 - p3 - p7]]).reshape(A.shape)
    else:
        n = A.shape[0]
        if n % 2 != 0:
            padding = ((0, 1), (0, 1))
            A = np.pad(A, padding, mode='constant')
            B = np.pad(B, padding, mode='constant')
            n += 1
        half = n // 2
        a = A[:half, :half]
        b = A[:half, half:]
        c = A[half:, :half]
        d = A[half:, half:]
        e = B[:half, :half]
        f = B[:half, half:]
        g = B[half:, :half]
        h = B[half:, half:]
        p1 = strassen_multiply(a, f - h)
        p2 = strassen_multiply(a + b, h)
        p3 = strassen_multiply(c + d, e)
        p4 = strassen_multiply(d, g - e)
        p5 = strassen_multiply(a + d, e + h)
        p6 = strassen_multiply(b - d, g + h)
        p7 = strassen_multiply(a - c, e + f)
        C = np.zeros((n, n), dtype=A.dtype)
        C[:half, :half] = p5 + p4 - p2 + p6
        C[:half, half:] = p1 + p2
        C[half:, :half] = p3 + p4
        C[half:, half:] = p5 + p1 - p3 - p7
        return C[:A.shape[0], :A.shape[1]]

def coppersmith_winograd(A, B):
    # Placeholder - replace with a real implementation if available
    return standard_multiplication(A, B)
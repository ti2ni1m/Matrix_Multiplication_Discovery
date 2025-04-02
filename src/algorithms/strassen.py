import numpy as np

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.algorithms.standard import standard_multiplication



def strassen_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Performs matrix multiplication using Strassen's algorithm."""
    if A.shape != B.shape:
        raise ValueError("Strassen's algorithm requires square matrices of the same size.")

    n = A.shape[0]

    # Base case: If small enough, use standard multiplication
    if n <= 2:
        return standard_multiplication(A, B)

    # Divide matrices into 4 sub-matrices
    mid = n // 2
    A11, A12, A21, A22 = A[:mid, :mid], A[:mid, mid:], A[mid:, :mid], A[mid:, mid:]
    B11, B12, B21, B22 = B[:mid, :mid], B[:mid, mid:], B[mid:, :mid], B[mid:, mid:]

    # Compute the 7 matrix multiplications (Strassen's key optimization)
    M1 = strassen_multiplication(A11 + A22, B11 + B22)
    M2 = strassen_multiplication(A21 + A22, B11)
    M3 = strassen_multiplication(A11, B12 - B22)
    M4 = strassen_multiplication(A22, B21 - B11)
    M5 = strassen_multiplication(A11 + A12, B22)
    M6 = strassen_multiplication(A21 - A11, B11 + B12)
    M7 = strassen_multiplication(A12 - A22, B21 + B22)

    # Compute sub-matrices of the result
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6

    # Reconstruct the full matrix
    C = np.zeros((n, n))
    C[:mid, :mid], C[:mid, mid:] = C11, C12
    C[mid:, :mid], C[mid:, mid:] = C21, C22

    return C

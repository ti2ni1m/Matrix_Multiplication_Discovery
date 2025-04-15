import numpy as np

def standard_multiplication(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Performs standard matrix multiplication (O(n^3) complexity)."""
    if A.shape[1] != B.shape[0]:
        raise ValueError("Matrix dimensions are incompatible for multiplication.")

    n, m = A.shape[0], B.shape[1]
    C = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            for k in range(A.shape[1]):
                C[i, j] += A[i, k] * B[k, j]

    return C

# Optional test
if __name__ == "__main__":
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C = standard_multiplication(A, B)
    print("Result of standard multiplication:\n", C)
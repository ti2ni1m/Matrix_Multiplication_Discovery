import numpy as np
import time
import pickle
import logging

def generate_random_matrix(rows: int, cols: int) -> np.ndarray:
    """Generates a random matrix of given dimensions."""
    return np.random.rand(rows, cols)

def time_function(func, *args, **kwargs) -> tuple[any, float]:
    """Times the execution of a given function and returns the result and the time taken."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time

def compare_matrices(matrix_a: np.ndarray, matrix_b: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Compares two matrices element-wise for approximate equality within a tolerance."""
    return np.allclose(matrix_a, matrix_b, atol=tolerance)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_data(data: any, filepath: str):
    """Saves Python data to a specified file using pickle."""
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Data saved successfully to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving data to {filepath}: {e}")

def load_data(filepath: str) -> any:
    """Loads Python data from a specified file using pickle."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logging.info(f"Data loaded successfully from: {filepath}")
        return data
    except FileNotFoundError:
        logging.warning(f"File not found: {filepath}")
        return None
    except Exception as e:
        logging.error(f"Error loading data from {filepath}: {e}")
        return None

def log_info(message: str):
    """Logs an informational message with a timestamp."""
    logging.info(message)

def log_warning(message: str):
    """Logs a warning message with a timestamp."""
    logging.warning(message)

def log_error(message: str):
    """Logs an error message with a timestamp."""
    logging.error(message)

def reshape_matrix(matrix: np.ndarray, new_shape: tuple[int, ...]) -> np.ndarray:
    """Reshapes a NumPy matrix to a new shape if possible."""
    try:
        return matrix.reshape(new_shape)
    except ValueError as e:
        logging.error(f"Error reshaping matrix of shape {matrix.shape} to {new_shape}: {e}")
        return matrix  # Return original matrix on error

def transpose_matrix(matrix: np.ndarray) -> np.ndarray:
    """Transposes a NumPy matrix."""
    return matrix.T

def flatten_matrix(matrix: np.ndarray) -> np.ndarray:
    """Flattens a NumPy matrix into a 1D array."""
    return matrix.flatten()

def print_matrix_info(matrix: np.ndarray, name: str = "Matrix"):
    """Prints the shape and data type of a NumPy matrix."""
    logging.info(f"Info for {name}: Shape = {matrix.shape}, Data Type = {matrix.dtype}")
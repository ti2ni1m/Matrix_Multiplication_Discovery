import numpy as np

def basic_op_scalar_multiply(matrix, scalar):
    return scalar * matrix

def basic_op_elementwise_add(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape for element-wise addition.")
    return matrix1 + matrix2

def basic_op_elementwise_subtract(matrix1, matrix2):
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape for element-wise subtraction.")
    return matrix1 - matrix2

def basic_op_transpose(matrix):
    return matrix.T

def basic_op_multiply_subblocks(matrix_a, matrix_b, block_size):
    """A very basic example for multiplying sub-blocks."""
    n_a_rows, n_a_cols = matrix_a.shape
    n_b_rows, n_b_cols = matrix_b.shape
    if n_a_cols != n_b_rows or n_a_rows % block_size != 0 or n_a_cols % block_size != 0 or n_b_cols % block_size != 0:
        raise ValueError("Matrices and block size are not compatible for sub-block multiplication.")

    n_blocks_row = n_a_rows // block_size
    n_blocks_col = n_b_cols // block_size
    result = np.zeros((n_a_rows, n_b_cols), dtype=matrix_a.dtype)

    for i in range(n_blocks_row):
        for j in range(n_blocks_col):
            for k in range(n_a_cols // block_size):
                sub_a = matrix_a[i * block_size:(i + 1) * block_size, k * block_size:(k + 1) * block_size]
                sub_b = matrix_b[k * block_size:(k + 1) * block_size, j * block_size:(j + 1) * block_size]
                result[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size] += np.dot(sub_a, sub_b)
    return result

def basic_op_combine_subblocks(sub_matrices, arrangement):
    """Combines sub-matrices based on the specified arrangement."""
    rows, cols = arrangement
    if len(sub_matrices) != rows * cols:
        raise ValueError(f"Number of sub-matrices does not match the arrangement ({rows}x{cols}).")

    sub_shapes = [sub.shape for sub in sub_matrices]
    row_shapes = [sub_shapes[i * cols:(i + 1) * cols] for i in range(rows)]

    if not row_shapes:
        return np.array([])

    first_row_height = row_shapes[0][0][0]
    first_col_width = row_shapes[0][0][1]

    if not all(shape[0] == first_row_height for shape_row in row_shapes for shape in shape_row):
        raise ValueError("Sub-matrices in each row must have the same height.")
    if not all(shape[1] == first_col_width for shape_row in row_shapes for shape in shape_row):
        raise ValueError("Sub-matrices in each column must have the same width.")

    block_rows = [np.hstack(sub_matrices[i * cols:(i + 1) * cols]) for i in range(rows)]
    return np.vstack(block_rows)

def basic_op_extract_submatrix(matrix, row_start, row_end, col_start, col_end):
    """Extracts a sub-matrix based on row and column indices."""
    return matrix[row_start:row_end, col_start:col_end].copy()

def basic_op_zero_matrix(shape, dtype=np.float32):
    """Creates a zero matrix of the specified shape."""
    return np.zeros(shape, dtype=dtype)

def basic_op_hadamard_product(matrix1, matrix2):
    """Performs element-wise (Hadamard) product."""
    if matrix1.shape != matrix2.shape:
        raise ValueError("Matrices must have the same shape for Hadamard product.")
    return matrix1 * matrix2

def basic_op_permute_rows(matrix, row_index1, row_index2):
    """Permutes two rows of a matrix."""
    temp = matrix[row_index1, :].copy()
    matrix[row_index1, :] = matrix[row_index2, :]
    matrix[row_index2, :] = temp
    return matrix

def basic_op_permute_cols(matrix, col_index1, col_index2):
    """Permutes two columns of a matrix."""
    temp = matrix[:, col_index1].copy()
    matrix[:, col_index1] = matrix[:, col_index2]
    matrix[:, col_index2] = temp
    return matrix

def basic_op_scale_row(matrix, row_index, scalar):
    """Scales a specific row of the matrix by a scalar."""
    matrix[row_index, :] = scalar * matrix[row_index, :]
    return matrix

def basic_op_scale_col(matrix, col_index, scalar):
    """Scales a specific column of the matrix by a scalar."""
    matrix[:, col_index] = scalar * matrix[:, col_index]
    return matrix

def apply_discovered_algorithm(matrix_a, matrix_b, action_history, intermediate_matrices=None):
    """
    Applies the sequence of basic operations learned by the RL agent.

    Args:
        matrix_a (np.ndarray): The first input matrix.
        matrix_b (np.ndarray): The second input matrix.
        action_history (list): The sequence of actions taken by the agent.
        intermediate_matrices (dict, optional): A dictionary to store intermediate results.
                                                Defaults to None, in which case a new one is created.

    Returns:
        np.ndarray: The resulting matrix after applying the discovered algorithm.
    """
    if intermediate_matrices is None:
        intermediate_matrices = {'A_init': matrix_a.copy(), 'B_init': matrix_b.copy()}

    current_matrices = intermediate_matrices.copy()

    for i, action in enumerate(action_history):
        try:
            if action == 0:
                scalar = 2  # Example parameter
                current_matrices[f'scaled_A_{i}'] = basic_op_scalar_multiply(current_matrices['A_init'], scalar)
            elif action == 1:
                current_matrices[f'sum_AB_{i}'] = basic_op_elementwise_add(current_matrices['A_init'], current_matrices['B_init'])
            elif action == 2:
                current_matrices[f'B_T_{i}'] = basic_op_transpose(current_matrices['B_init'])
            elif action == 3:
                block_size = 2  # Example parameter for 4x4 matrices
                current_matrices[f'sub_mult_{i}'] = basic_op_multiply_subblocks(current_matrices['A_init'], current_matrices['B_init'], block_size)
            elif action == 4:
                keys = list(current_matrices.keys())
                if len(keys) >= 4:
                    subs = [current_matrices[keys[j]] for j in range(len(current_matrices) - 4, len(current_matrices))]
                    current_matrices[f'combined_{i}'] = basic_op_combine_subblocks(subs, (2, 2))
            elif action == 5:
                current_matrices[f'A_sub_{i}'] = basic_op_extract_submatrix(current_matrices['A_init'], 0, 2, 0, 2)
            elif action == 6:
                current_matrices[f'zero_{i}'] = basic_op_zero_matrix(matrix_a.shape)
            elif action == 7:
                current_matrices[f'hadamard_AB_{i}'] = basic_op_hadamard_product(current_matrices['A_init'], current_matrices['B_init'])
            elif 8 <= action <= 9: # Example: Permute rows/cols (need to define parameters)
                if action == 8:
                    if 'A_init' in current_matrices and matrix_a.shape[0] >= 2:
                        current_matrices[f'perm_rows_A_{i}'] = basic_op_permute_rows(current_matrices['A_init'].copy(), 0, 1)
                elif action == 9:
                    if 'B_init' in current_matrices and matrix_b.shape[1] >= 2:
                        current_matrices[f'perm_cols_B_{i}'] = basic_op_permute_cols(current_matrices['B_init'].copy(), 0, 1)
            # Add more actions here based on your defined basic operations
        except ValueError as e:
            print(f"Error in apply_discovered_algorithm at step {i}, action {action}: {e}")
            return None  # Or handle the error as needed

    # The final result is likely the last matrix created or a matrix with a specific name
    final_result = None
    for key in current_matrices:
        if key.startswith('combined_') or key.startswith('result_'):
            final_result = current_matrices[key]
            break
    if final_result is None and current_matrices:
        final_result = list(current_matrices.values())[-1]

    return final_result
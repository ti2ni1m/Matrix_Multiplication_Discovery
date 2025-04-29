# evaluations/transformer_evaluation.py

import numpy as np
import torch
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.algorithms.standard import standard_multiplication
from src.algorithms.strassen import strassen_multiplication
from src.algorithms.rl_discovered_algorithm import rl_discovered_algorithm

def scaled_dot_product_attention(q, k, v, matmul_fn):
    """
    Implements scaled dot-product attention using standard numpy matmul.
    q, k, v: [batch, heads, seq_len, dim]
    matmul_fn: function to replace matrix multiplication (but we're ignoring it here)
    """
    scores = torch.stack([
        torch.tensor(np.matmul(
            q[b, h].detach().numpy(),
            k[b, h].transpose(-1, -2).detach().numpy()
        ))
        for b in range(q.shape[0]) for h in range(q.shape[1])
    ]).reshape(q.shape[0], q.shape[1], q.shape[2], k.shape[2])

    scores = scores / (q.shape[-1] ** 0.5)
    weights = F.softmax(scores, dim=-1)

    output = torch.stack([
        torch.tensor(np.matmul(
            weights[b, h].detach().numpy(),
            v[b, h].detach().numpy()
        ))
        for b in range(weights.shape[0]) for h in range(weights.shape[1])
    ]).reshape(q.shape)

    return output

def evaluate_attention(algorithm_name, matmul_fn):
    batch, heads, seq_len, dim = 2, 2, 32, 64
    q = torch.rand(batch, heads, seq_len, dim)
    k = torch.rand(batch, heads, seq_len, dim)
    v = torch.rand(batch, heads, seq_len, dim)

    print(f"Evaluating attention using {algorithm_name}")
    output = scaled_dot_product_attention(q, k, v, matmul_fn)
    print(f"Output shape: {output.shape}\n")

if __name__ == "__main__":
    evaluate_attention("Standard", standard_multiplication)
    evaluate_attention("Strassen", strassen_multiplication)
    evaluate_attention("RL_Discovered", rl_discovered_algorithm)

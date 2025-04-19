# evaluations/transformer_evaluation.py

import torch
import torch.nn.functional as F
from algorithms.standard import standard_multiplication
from algorithms.strassen import strassen_multiplication
from algorithms.rl_discovered_algorithm import rl_multiplication  # Update as needed

def scaled_dot_product_attention(q, k, v, matmul_fn):
    """
    Implements scaled dot-product attention using a custom matmul function.
    q, k, v: [batch, heads, seq_len, dim]
    matmul_fn: function to replace matrix multiplication
    """
    scores = torch.stack([
        torch.tensor(matmul_fn(q[i].detach().numpy(), k[i].transpose(-1, -2).detach().numpy()))
        for i in range(q.shape[0])
    ])
    scores = scores / (q.shape[-1] ** 0.5)
    weights = F.softmax(scores, dim=-1)
    
    output = torch.stack([
        torch.tensor(matmul_fn(weights[i].detach().numpy(), v[i].detach().numpy()))
        for i in range(weights.shape[0])
    ])
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
    evaluate_attention("RL_Discovered", rl_multiplication)

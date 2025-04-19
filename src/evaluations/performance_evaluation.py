# evaluations/performance_evaluation.py

import time
import csv
import numpy as np
import matplotlib.pyplot as plt
from algorithms.standard import standard_multiplication
from algorithms.strassen import strassen_multiplication
from algorithms.rl_discovered_algorithm import rl_multiplication  # Update as needed

def benchmark(algorithm_fn, A, B):
    start = time.time()
    C = algorithm_fn(A, B)
    end = time.time()
    return (end - start), C

def evaluate_algorithms(sizes=[64, 128, 256], save_csv=True, plot=True):
    algorithms = {
        "Standard": standard_multiplication,
        "Strassen": strassen_multiplication,
        "RL_Discovered": rl_multiplication
    }

    results = []

    for size in sizes:
        A = np.random.rand(size, size)
        B = np.random.rand(size, size)

        print(f"\nEvaluating size {size}x{size}")
        for name, fn in algorithms.items():
            try:
                duration, _ = benchmark(fn, A, B)
                print(f"{name}: {duration:.6f} sec")
                results.append((size, name, duration))
            except Exception as e:
                print(f"{name} failed at size {size}: {e}")
                results.append((size, name, None))

    if save_csv:
        with open("performance_results.csv", "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Size", "Algorithm", "Time(s)"])
            writer.writerows(results)
        print("\nSaved results to performance_results.csv")

    if plot:
        plt.figure(figsize=(8, 6))
        for name in algorithms:
            xs = [r[0] for r in results if r[1] == name and r[2] is not None]
            ys = [r[2] for r in results if r[1] == name and r[2] is not None]
            plt.plot(xs, ys, marker='o', label=name)

        plt.xlabel("Matrix Size (N x N)")
        plt.ylabel("Time (seconds)")
        plt.title("Matrix Multiplication Performance")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("performance_plot.png")
        plt.show()

if __name__ == "__main__":
    evaluate_algorithms()

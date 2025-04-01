import pandas as pd
import matplotlib.pyplot as plt
import os

print(os.path.exists("C:/Users/manan/OneDrive/Desktop/University/Year 4/Discovery Project/project-ti2ni1m/results/benchmarks.csv"))

df = pd.read_csv("C:/Users/manan/OneDrive/Desktop/University/Year 4/Discovery Project/project-ti2ni1m/results/benchmarks.csv")

from julia import Main
Main.include("test_algorithms.jl")
Main.run_matrix_test()

plt.plot(df["matrix_size"], df["execution_time"], label="RL Algorithm")
plt.plot(df["matrix_size"], df["strassen_time"], label="Strassen", linestyle="dashed")
plt.xlabel("Matrix Size")
plt.ylabel("Execution Time (ms)")
plt.legend()
plt.title("Performance of RL-Discovered Algorithms vs Strassen")
plt.show()
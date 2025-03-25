using PyCall
py"""
import json
def load_algorithms():
    with open("discovered.json", "r") as f:
        return json.load(f)
"""
algo = py"load_algorithms"()
println("Loaded from Python: ", algo)